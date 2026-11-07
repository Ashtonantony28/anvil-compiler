"""Tensor kernel (canonical form) -> anvil.mir with virtual registers, plus the args-block layout.

Input: a kernel from `Schedule.finalize()` (rules 1-7 of anvil.schedule). Output: `KernelCode` - a MIR function
for `anvil.backend.compile_mir`, the args slots it reads, and the launch geometry.

Machine mapping
---------------
* Threads. The block iv is %blockIdx (+ args[block_offset] when the block loop is split over several launches),
  the hardware warp w = %threadIdx div L and lane l = %threadIdx - L*w (L = THREADS_PER_BLOCK / WARPS_PER_CORE).
  Extent-1 levels are the constant 0. Warps w >= W RET at entry (a warp-uniform branch; BAR waits only for
  unfinished warps). Lanes l >= Lx are inactive: an implicit guard `l < Lx` around the body, predicated like any
  lane-varying guard (below), so they still execute every barrier.
* Loops. Every seq loop has constant, block-uniform trip counts: a counter register counts outer iterations
  0..n/u-1 and exits on equality (`ADD c,c,1; CMP c,n; BRn top`, CMP is equality-only); the body is emitted u
  times (unroll u) with iv = u*c + i. Fully unrolled loops and extent-1 loops emit no loop at all.
* Indices. Affine maps become linear forms over atoms (`anvil.codegen.affine`): trivial maps fold away, floordiv /
  mod by constants are decided from ranges where possible, and every partial sum is cached per region and placed
  at the outermost scope whose atoms it uses (loop-invariant code motion out of loops and guards).
* Memory. Global buffers are addressed from bases loaded once from the args block; smem buffers get fixed
  offsets (block-level buffers first, then warp-private buffers laid out at offset + w*size); reg buffers are
  virtual registers, one per element (every reg index is a constant after full unrolling).
* Guards (rule 7) - the predication strategy, chosen per region:
    static     the condition is decided from index ranges: no code, or the region is dropped;
    uniform    the condition does not depend on the lane: a branch around the region (block/warp-uniform);
    sync       lane-varying, SIMT_STACK=1: `SYNC end; CMP f,0; BRn end; body; end:` - masked-off lanes read and
               write nothing; nesting is limited so that at most 8 SIMT-stack entries are ever pushed (2 per level);
    mask       lane-varying otherwise: the body runs for every lane; global stores go to a per-thread sink word
               (args[sink] + global thread id) and smem stores to a spare smem word when the predicate is 0, and
               reg stores keep the old value (select). Loads and gathers of a lane whose predicate is 0 read a
               safe word instead (word 0 of the same buffer, or the args block for a buffer the kernel writes), and
               inactive lanes (l >= Lx) compute with l clamped to Lx - 1, so a masked-off lane never reads a word
               outside the buffers an active lane reads (canonical rule 2; no cross-core read/write races, A1-M1).
* Ops. requant / min / cmp / select via anvil.backend.idioms; requant with a register shift (rmsnorm's s + k)
  computes its rounding term 2^(S-1) branch-free (0 - SRA(INT_MIN, 31-S)) div 2; rsqrt_index is a counted,
  branch-free loop (K iterations, enough for any non-negative int32); gather is LDR(base + offset + v).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from xdsl.ir.affine import AffineMap

from ..backend import consts, idioms
from ..backend.lir import Ins, lift, lower as lir_lower
from ..ir import mir
from ..ir import tensor as t
from .affine import Affine, Atom, CodegenError, Lin

HW_MAX_THREADS = 255
SMEM_WORDS = 64
SIMT_STACK_DEPTH = 8
PREDICATION = ("auto", "mask", "sync")


@dataclass
class KernelCode:
    name: str
    func: mir.FuncOp                    # virtual registers, reads R12 (args pointer) and R13-R15
    slots: list                         # args-block slots: ("buf", sym) | ("block_offset", None) | ("sink", None)
                                        # | ("const", word)
    blocks: int                         # B (block loop extent)
    blocks_per_launch: int
    threads_per_block: int
    smem_words: int                     # incl. the mask sink word if used
    stats: dict = field(default_factory=dict)

    @property
    def launches(self) -> int:
        return -(-self.blocks // self.blocks_per_launch)

    def slot(self, kind, sym=None):
        return self.slots.index((kind, sym))


class _Scope:
    def __init__(self, kind, pmem=None, preg=None, sync=0):
        self.kind = kind
        self.ops: list = []
        self.cache: dict = {}
        self.pmem, self.preg, self.sync = pmem, preg, sync
        self.counter = None             # loop scopes: the iteration atom
        self.latch: list = []           # loop scopes: induction registers (reg, step) bumped before the back edge


def _ext(loop) -> int:
    lb, ub, _ = loop.bounds
    return ub - lb


def _bound(func):
    out = {}
    for op in func.walk():
        if isinstance(op, t.ForOp) and op.bind.data != "seq":
            out[op.bind.data] = op
    return out


def _shape(buf) -> list[int]:
    return list(buf.type.get_shape())


def _strides(shape):
    return [math.prod(shape[r + 1:]) for r in range(len(shape))]


def _rsqrt_trips(n: int) -> int:
    """Iterations of `idx >= n: idx >>= 2, k += 1` that suffice for every non-negative int32."""
    k, x = 0, (1 << 31) - 1
    while x >= n:
        x >>= 2
        k += 1
    return k


class _Gen(idioms.Builder):
    def __init__(self, kernel, params, predication, const_args=None):
        super().__init__()
        self.const_args = const_args
        if params.WORD_BITS != 32 or params.DATA_MEM_DATA_BITS != 32:
            raise CodegenError(f"anvil generates code for 32-bit words only (constants, requant rounding and the "
                               f"int32 sign idioms assume them); got WORD_BITS={params.WORD_BITS}, "
                               f"DATA_MEM_DATA_BITS={params.DATA_MEM_DATA_BITS}")
        if predication not in PREDICATION:
            raise CodegenError(f"predication {predication!r} not in {PREDICATION}")
        if predication == "sync" and not params.SIMT_STACK:
            raise CodegenError("predication='sync' needs SIMT_STACK=1")
        self.kernel, self.p, self.mode = kernel, params, predication
        self.simt = bool(params.SIMT_STACK) and predication != "mask"
        self.tpb, self.wpc = params.THREADS_PER_BLOCK, params.WARPS_PER_CORE
        self.lpw = self.tpb // self.wpc
        self.stack: list[_Scope] = []
        self._target = None
        self._target_depth = None
        self.vals: dict = {}             # tensor SSA value -> register | Lin (index values)
        self.elems: dict = {}            # (reg buffer SSA value, flat index) -> register
        self.smem_off: dict = {}         # smem buffer -> (offset, warp stride or 0)
        self.slots: list = []
        self.copies: set = set()
        self.labels = 0
        self.affine = Affine(self._div_atom)
        self.stats = {"regions": {"static": 0, "uniform": 0, "sync": 0, "mask": 0}, "sync_depth": 0,
                      "mask_global_stores": 0, "mask_smem_stores": 0}
        self.smem_used = 0
        self.smem_sink = None
        self.sink_used = False
        self.pred_depth: dict = {}       # load predicate register -> scope depth where it is defined
        self.written: set = set()        # buffers (SSA values) the kernel stores to

    # ------------------------------------------------------------------------------------- emission
    def emit(self, op):
        (self._target if self._target is not None else self.stack[-1].ops).append(op)
        return op

    class _At:
        def __init__(self, g, depth):
            self.g, self.depth = g, depth

        def __enter__(self):
            self.prev = self.g._target, self.g._target_depth
            self.g._target, self.g._target_depth = self.g.stack[self.depth].ops, self.depth

        def __exit__(self, *exc):
            self.g._target, self.g._target_depth = self.prev

    def at(self, depth):
        return self._At(self, depth)

    def label_name(self, what):
        self.labels += 1
        return f"{what}{self.labels}"

    def push(self, kind, **kw):
        top = self.stack[-1] if self.stack else None
        base = dict(pmem=top.pmem, preg=top.preg, sync=top.sync) if top else {}
        base.update(kw)
        s = _Scope(kind, **base)
        self.stack.append(s)
        return s

    def pop(self):
        s = self.stack.pop()
        self.stack[-1].ops.extend(s.ops)
        return s

    def cached(self, key):
        """A cached register visible where code is being emitted: only scopes up to the emission target (code
        hoisted into an outer scope must not use values computed later in inner ones)."""
        top = len(self.stack) if self._target_depth is None else self._target_depth + 1
        for s in reversed(self.stack[:top]):
            if key in s.cache:
                return s.cache[key]
        return None

    def const(self, value: int) -> str:
        """A word constant. Plain const32 (materialized, CSE'd and hoisted by the backend), except:
        * with `const_args`, a constant that takes at least that many instructions to build is an args-block word
          (`("const", word)` slot), loaded once where it is first needed, hoisted out of loops;
        * a multi-instruction constant in straight-line code outside every loop is built once per region into an
          ordinary register (`MAX r, c, c`): the backend hoists const32 registers to entry, where after a loop they
          are rematerialized at every use (an unrolled epilogue repeats requant M, the rounding term, the clamp
          bounds)."""
        w = consts.word(value)
        cost = consts.cost(w)
        if self.const_args is not None and cost >= self.const_args:
            key = ("karg", w)
            hit = self.cached(key)
            if hit is None:
                if ("const", w) not in self.slots:
                    self.slots.append(("const", w))
                slot = self.slots.index(("const", w))
                d = self.place(0)
                with self.at(d):
                    hit = self.ldr(self.args_addr(slot))
                self.stack[d].cache[key] = hit
            return hit
        if cost < 2 or any(s.kind == "loop" for s in self.stack):
            return super().const(value)
        key = ("k", consts.word(value))
        hit = self.cached(key)
        if hit is None:
            c = super().const(value)
            hit = self.reg()
            self.emit(mir.MaxOp(hit, c, c))
            self.stack[-1 if self._target_depth is None else self._target_depth].cache[key] = hit
        return hit

    def zero(self):
        return self.const(0)

    def copy(self, dst, src):
        op = self.emit(mir.AddOp(dst, src, self.zero()))
        self.copies.add(id(op))

    # ------------------------------------------------------------------------------------- index values
    def _div_atom(self, r: Lin, d: int) -> Lin:
        key = ("div", r.key, d)
        hit = self.cached(key)
        if hit is None:
            lo, hi = r.range()
            depth = self.place(r.depth)

            def make(g, r=r, d=d):
                return g.div(g.lin_reg(r), d)
            hit = Atom("div", lo // d, hi // d, depth, r.varying, make=make, key=key)
            self.stack[depth].cache[key] = hit
        return Lin.of(hit)

    def place(self, depth: int) -> int:
        """Where to emit a value whose operands are defined at scope `depth`: hoisted out of every loop scope
        above `depth` (loop-invariant code motion), otherwise at the current scope - never further out than a
        loop exit needs, so straight-line code is not moved ahead of earlier loops (register pressure)."""
        top = len(self.stack) - 1 if self._target_depth is None else self._target_depth
        loops = [i for i in range(depth + 1, top + 1) if self.stack[i].kind == "loop"]
        return loops[0] - 1 if loops else top

    def atom_reg(self, a: Atom) -> str:
        if a.reg is not None:
            return a.reg
        key = ("atom", a.uid)
        hit = self.cached(key)
        if hit is None:
            d = self.place(a.depth)
            with self.at(d):
                hit = a.make(self)
            self.stack[d].cache[key] = hit
        return hit

    def lin_reg(self, x: Lin) -> str:
        """A register holding x, emitted at the outermost open scope whose atoms x uses, cached there."""
        x = Lin.of(x)
        if x.is_const:
            return self.const(x.const)
        hit = self.cached(x.key)
        if hit is not None:
            return hit
        depth = self.place(x.depth)
        if x.const:
            base = Lin(x.terms)
            r0 = self.lin_reg(base)
            with self.at(depth):
                r = self.add(r0, x.const) if x.const > 0 else self.sub(r0, -x.const)
        elif len(x.terms) == 1:
            a, c = x.terms[0]
            ar = self.atom_reg(a)
            with self.at(depth):
                r = ar if c == 1 else (self.neg(ar) if c == -1 else self.mul(ar, c))
        elif (r := self.induction(x)) is not None:
            pass
        else:
            # peel the innermost term; the rest is placed (and reused) further out
            terms = sorted(x.terms, key=lambda tc: (tc[0].depth, tc[0].uid))
            a, c = terms[-1]
            rest = self.lin_reg(Lin(terms[:-1]))
            ar = self.atom_reg(a)
            with self.at(depth):
                if c == 1:
                    r = self.add(rest, ar)
                elif c == -1:
                    r = self.sub(rest, ar)
                elif c < 0:
                    r = self.sub(rest, self.mul(ar, -c))
                else:
                    r = self.add(rest, self.mul(ar, c))
        self.stack[depth].cache[x.key] = r
        return r

    def induction(self, x: Lin):
        """Strength reduction: x = rest + s*c with c the counter of the loop scope at depth d and rest invariant
        in that loop becomes an induction register p (p = rest before the loop, p += s at the latch)."""
        terms = sorted(x.terms, key=lambda tc: (tc[0].depth, tc[0].uid))
        a, c = terms[-1]
        d = a.depth
        if len(terms) < 2 or d >= len(self.stack) or self.stack[d].kind != "loop" or self.stack[d].counter is not a:
            return None
        rest = Lin(terms[:-1])
        if rest.depth >= d:
            return None
        key = ("iv", x.key)
        hit = self.stack[d].cache.get(key)
        if hit is None:
            r0 = self.lin_reg(rest)
            hit = self.reg()
            with self.at(d - 1):
                self.copy(hit, r0)
            self.stack[d].latch.append((hit, c))
            self.stack[d].cache[key] = hit
        return hit

    def index(self, op, mp: AffineMap | None = None) -> list[Lin]:
        dims = [self.vals[v] for v in op.indices]
        mp = mp or op.map.data
        return [self.affine.eval(e, dims) for e in mp.results]

    def flat(self, idx: list[Lin], shape) -> Lin:
        out = Lin()
        for x, s in zip(idx, _strides(shape)):
            out = out + x.scale(s)
        return out

    # ------------------------------------------------------------------------------------- predicates
    def false_indicator(self, conds) -> Lin:
        """Sum over (L, ub) of (L - m) floordiv (ub - m), m = min(lo(L), 0): >= 0, and 0 iff every L < ub."""
        f = Lin()
        for x, ub in conds:
            m = min(x.range()[0], 0)
            f = f + self.affine.floordiv(x - m, ub - m)
        return f

    def pred_from(self, f: Lin) -> str:
        """1 when f == 0 else 0 (f >= 0)."""
        lo, hi = f.range()
        if hi <= 1:
            return self.lin_reg(Lin.of(1) - f)
        return self.sub(1, self.nz(self.lin_reg(f)))

    def and_pred(self, a, b):
        if a is None:
            return b
        if b is None:
            return a
        return self.mul(a, b)

    def region(self, conds, body, *, lane_activity=False):
        """Emit body() under the conjunction of `conds` [(Lin, ub)] with the predication strategy of the module
        docstring."""
        live = []
        for x, ub in conds:
            lo, hi = x.range()
            if hi < ub:
                continue
            if lo >= ub:
                self.stats["regions"]["static"] += 1
                return                                  # statically dead: no code
            live.append((x, ub))
        if not live:
            if conds:
                self.stats["regions"]["static"] += 1
            body()
            return
        uni = [c for c in live if not c[0].varying]
        var = [c for c in live if c[0].varying]
        self.push("guard")
        end_u = None
        if uni:
            self.stats["regions"]["uniform"] += 1
            end_u = self.label_name("Lskip")
            fr = self.lin_reg(self.false_indicator(uni))
            self.emit(mir.CmpOp(fr, self.zero()))
            self.emit(mir.BrOp(0b100, end_u))           # skip when the false indicator is non-zero
        if var:
            self.push("guard")
            top = self.stack[-1]
            f = self.false_indicator(var)
            if self.simt and (top.sync + 1) * 2 <= SIMT_STACK_DEPTH:
                self.stats["regions"]["sync"] += 1
                end_v = self.label_name("Lrecon")
                fr = self.lin_reg(f)
                self.emit(mir.SyncOp(end_v))
                self.emit(mir.CmpOp(fr, self.zero()))
                self.emit(mir.BrOp(0b100, end_v))
                top.sync += 1
                self.stats["sync_depth"] = max(self.stats["sync_depth"], top.sync)
                self.push("plain")
                body()
                self.pop()
                self.emit(mir.LabelOp(end_v))
            else:
                if self.mode == "sync":
                    raise CodegenError(f"SIMT stack nesting above {SIMT_STACK_DEPTH // 2} needs mask predication")
                self.stats["regions"]["mask"] += 1
                p = self.pred_from(f)
                top.pmem = self.and_pred(top.pmem, p)
                if not lane_activity:
                    top.preg = self.and_pred(top.preg, p)
                    self.pred_depth[top.preg] = len(self.stack) - 1
                self.push("plain")
                body()
                self.pop()
            self.pop()
        else:
            self.push("plain")
            body()
            self.pop()
        if end_u is not None:
            self.emit(mir.LabelOp(end_u))
        self.pop()

    # ------------------------------------------------------------------------------------- memory
    def global_addr(self, buf, idx: list[Lin]) -> str:
        base = self.base_atom(buf)
        return self.lin_reg(Lin.of(base) + self.flat(idx, _shape(buf)))

    def smem_lin(self, buf, idx: list[Lin]) -> Lin:
        off, stride = self.smem_off[buf]
        x = self.flat(idx, _shape(buf)) + off
        if stride:
            x = x + self.warp_iv.scale(stride)
        return x

    def base_atom(self, buf) -> Atom:
        key = ("base", buf)
        a = self.stack[0].cache.get(key)
        if a is None:
            sym = buf.owner.sym.data
            slot = self.slots.index(("buf", sym))

            def make(g, slot=slot):
                return g.ldr(g.args_addr(slot))
            a = Atom(f"base:{sym}", 0, 1 << 16, 0, make=make, key=key)
            self.stack[0].cache[key] = a
        return a

    def args_addr(self, slot):
        return "R12" if slot == 0 else self.add("R12", slot)

    def sink_reg(self) -> str:
        if "sink" not in self.stack[0].cache:
            self.sink_used = True
            if ("sink", None) not in self.slots:
                self.slots.append(("sink", None))
            slot = self.slots.index(("sink", None))
            with self.at(0):
                base = self.ldr(self.args_addr(slot))
                hw = self.add(self.mul("R13", self.tpb), "R15")
                self.stack[0].cache["sink"] = self.add(base, hw)
        return self.stack[0].cache["sink"]

    def smem_sink_reg(self) -> str:
        if self.smem_sink is None:
            if self.smem_used >= SMEM_WORDS:
                raise CodegenError("masked shared-memory store needs a spare smem word, but all 64 are in use "
                                   "(use SIMT_STACK=1 for SYNC predication, or a schedule with fewer smem words)")
            self.smem_sink = self.smem_used
            self.smem_used += 1
        return self.const(self.smem_sink)

    def masked_addr(self, addr: str, sink: str, p: str) -> str:
        return self.select(p, addr, sink)

    def masked_load_addr(self, space, addr, buf) -> str:
        """Under a lane-varying mask guard (not the lane-activity guard: inactive lanes compute with a clamped
        lane index) a lane whose predicate is 0 must not load its out-of-range address (A1-M1, canonical rule 2):
        it loads `safe` instead, word 0 of the same buffer - a word an active lane may read in this phase, so no
        other thread writes it now - or the args block for a global buffer the kernel also writes (never written
        during a launch). addr' = p*addr + (1-p)*safe, with (1-p)*safe hoisted to the predicate's scope."""
        p = self.stack[-1].preg
        if p is None:
            return addr
        key = ("maddr", p, addr)
        hit = self.cached(key)
        if hit is not None:
            return hit
        qkey = ("mq", p, space, buf)
        q = self.cached(qkey)
        if q is None:
            d = self.pred_depth.get(p, len(self.stack) - 1)
            with self.at(d):
                if space == "smem":
                    zero = [Lin()] * len(_shape(buf))
                    safe = self.lin_reg(self.smem_lin(buf, zero))
                elif buf is None or buf in self.written:
                    safe = "R12"
                else:
                    safe = self.atom_reg(self.base_atom(buf))
                q = self.mul(self.sub(1, p), safe)
            self.stack[d].cache[qkey] = q
        hit = self.add(self.mul(p, addr), q)
        self.stack[-1].cache[key] = hit
        self.stats["mask_loads"] = self.stats.get("mask_loads", 0) + 1
        return hit

    def load_mem(self, space, addr, buf=None) -> str:
        """LDR/LDS with CSE: a word this thread already loaded since the last store to that space or barrier,
        within the current loop iteration, is reused (canonical kernels are race-free between barriers)."""
        addr = self.masked_load_addr(space, addr, buf)
        key = ("ld", space, addr)
        for s in reversed(self.stack):
            if key in s.cache:
                return s.cache[key]
            if s.kind == "loop":
                break
        r = self.reg()
        self.emit((mir.LdrOp if space == "global" else mir.LdsOp)(r, addr))
        self.stack[-1].cache[key] = r
        return r

    def forget_loads(self, space=None):
        for s in self.stack:
            for k in [k for k in s.cache if isinstance(k, tuple) and k[0] == "ld" and space in (None, k[1])]:
                del s.cache[k]

    def store_mem(self, space, addr, val):
        self.forget_loads(space)
        p = self.stack[-1].pmem
        if p is not None:
            if space == "global":
                self.stats["mask_global_stores"] += 1
                addr = self.masked_addr(addr, self.sink_reg(), p)
            else:
                self.stats["mask_smem_stores"] += 1
                addr = self.masked_addr(addr, self.smem_sink_reg(), p)
        self.emit((mir.StrOp if space == "global" else mir.StsOp)(addr, val))

    def reg_elem(self, buf, idx: list[Lin]) -> str:
        x = self.flat(idx, _shape(buf))
        if not x.is_const:
            raise CodegenError(f"register buffer indexed by a non-constant {x} (loop not fully unrolled)")
        key = (buf, x.const)
        if key not in self.elems:
            raise CodegenError("register buffer element used outside its allocation")
        return self.elems[key]

    # ------------------------------------------------------------------------------------- kernel
    def run(self) -> KernelCode:
        k = self.kernel
        f = k.func
        bnd = _bound(f)
        if set(bnd) != {"block", "warp", "lane"}:
            raise CodegenError(f"kernel {k.name} is not canonical (bound levels {sorted(bnd)}); use "
                               "Schedule.finalize()")
        Lb, Lw, Ll = bnd["block"], bnd["warp"], bnd["lane"]
        B, W, Lx = _ext(Lb), _ext(Lw), _ext(Ll)
        if W > self.wpc or Lx > self.lpw:
            raise CodegenError(f"{W} warps x {Lx} lanes do not fit WARPS_PER_CORE={self.wpc}, "
                               f"{self.lpw} lanes per warp")
        bpl = max(1, HW_MAX_THREADS // self.tpb)
        split = B > bpl
        for op in f.body.block.ops:
            if isinstance(op, t.AllocOp):
                self.slots.append(("buf", op.sym.data))
        if split:
            self.slots.append(("block_offset", None))
        self.order = {op: i for i, op in enumerate(f.walk())}
        self.reg_stores = {}
        self.written = {op.buf for op in f.walk() if isinstance(op, t.StoreOp)}
        for op in f.walk():
            if isinstance(op, t.StoreOp) and t.space_of(op.buf.type) == "reg":
                self.reg_stores.setdefault(op.buf, []).append(self.order[op])

        self.push("func")
        # thread coordinates (scope 0)
        if B == 1:
            blk = Lin()
        elif split:
            slot = self.slots.index(("block_offset", None))
            blk = Lin.of(Atom("block", 0, B - 1, 0, make=lambda g: g.add("R13", g.ldr(g.args_addr(slot)))))
        else:
            blk = Lin.of(Atom("block", 0, B - 1, 0, reg="R13"))
        self.vals[Lb.iv] = blk
        if self.wpc == 1:
            hw_warp = None
            raw_lane = Atom("lane", 0, self.lpw - 1, 0, varying=self.lpw > 1, reg="R15")
        else:
            hw_warp = Atom("warp", 0, self.wpc - 1, 0, make=lambda g: g.div("R15", g.lpw))
            raw_lane = Atom("lane", 0, self.lpw - 1, 0, varying=True,
                            make=lambda g: g.sub("R15", g.mul(g.atom_reg(hw_warp), g.lpw)))
        if W == 1:
            self.warp_iv = Lin()
        else:
            self.warp_iv = Lin.of(Atom("warp_iv", 0, W - 1, 0, make=lambda g: g.atom_reg(hw_warp)))
        self.vals[Lw.iv] = self.warp_iv
        if Lx == 1:
            lane_iv = Lin()
        elif Lx < self.lpw and not self.simt:
            # mask predication runs the body on inactive lanes l >= Lx too: they compute with l clamped to Lx - 1,
            # so every address they load is one an active lane loads (no out-of-bounds or racing reads, A1-M1);
            # their stores go to the sink (the lane-activity predicate still uses the raw lane)
            lane_iv = Lin.of(Atom("lane_iv", 0, Lx - 1, 0, varying=True,
                                  make=lambda g: g.min(g.atom_reg(raw_lane), Lx - 1)))
        else:
            lane_iv = Lin.of(Atom("lane_iv", 0, Lx - 1, 0, varying=True, make=lambda g: g.atom_reg(raw_lane)))
        self.vals[Ll.iv] = lane_iv

        # shared memory layout: block-level buffers, then warp-private ones (offset + w * size)
        for op in Lb.body.block.ops:
            if isinstance(op, t.AllocOp):
                n = math.prod(_shape(op.result))
                self.smem_off[op.result] = (self.smem_used, 0)
                self.smem_used += n
        for op in Lw.body.block.ops:
            if isinstance(op, t.AllocOp):
                n = math.prod(_shape(op.result))
                self.smem_off[op.result] = (self.smem_used, n if W > 1 else 0)
                self.smem_used += n * W
        if self.smem_used > SMEM_WORDS:
            raise CodegenError(f"{self.smem_used} smem words > {SMEM_WORDS}")

        body_ops = list(Ll.body.block.ops)

        def lanes():
            act = [] if Lx == self.lpw else [(Lin.of(raw_lane), Lx)]
            self.region(act, lambda: self.block(body_ops), lane_activity=True)

        if W < self.wpc:
            self.region([(Lin.of(hw_warp), W)], lanes)       # warp-uniform: inactive warps skip to RET
        else:
            lanes()
        self.emit(mir.RetOp())
        code = self.stack[0].ops
        self.stack.pop()
        code = _coalesce(code, self.copies)
        self.stats["sink"] = self.sink_used
        self.stats["smem_words"] = self.smem_used
        self.stats["simt_stack_entries"] = 2 * self.stats["sync_depth"]
        func = lir_lower(code, k.name)
        return KernelCode(k.name, func, list(self.slots), B, bpl if split else B, self.tpb, self.smem_used,
                          self.stats)

    # ------------------------------------------------------------------------------------- statements
    def block(self, ops):
        for op in ops:
            self.op(op)

    def loop(self, L: t.ForOp):
        n, u = _ext(L), int(L.unroll.value.data)
        if n % u:
            raise CodegenError(f"loop {L.tag}: unroll {u} does not divide {n}")
        trips = n // u
        ops = list(L.body.block.ops)

        def copies(o: Lin):
            for i in range(u):
                self.vals[L.iv] = o.scale(u) + i
                self.block(ops)

        self.counted(trips, copies)

    def counted(self, trips, body):
        """body(o: Lin) for o in 0..trips-1, as an equality-exit counted loop (uniform trip count)."""
        if trips <= 0:
            return
        if trips == 1:
            body(Lin())                 # straight-line: inline in the current scope (values stay reusable)
            return
        s = self.push("loop")
        c = self.reg()
        self.emit(mir.Const32Op(c, 0))
        top = self.label_name("Lloop")
        self.emit(mir.LabelOp(top))
        s.counter = Atom("iter", 0, trips - 1, len(self.stack) - 1, reg=c)
        body(Lin.of(s.counter))
        for p, step in s.latch:
            self.emit(mir.AddOp(p, p, self.const(step)))
        self.emit(mir.AddOp(c, c, self.const(1)))
        self.emit(mir.CmpOp(c, self.const(trips)))
        self.emit(mir.BrOp(0b100, top))
        assert self.stack[-1] is s
        self.pop()
        if not any(x.kind == "loop" for x in self.stack):
            # straight-line code after an outermost loop recomputes index values instead of keeping pre-loop
            # ones live across the loop (cheaper than the spills the extra pressure causes)
            for x in self.stack:
                x.cache = {k: v for k, v in x.cache.items() if k == "sink" or (isinstance(k, tuple) and k[0] == "base")}

    def r(self, v) -> str:
        x = self.vals[v]
        return self.lin_reg(x) if isinstance(x, Lin) else x

    def op(self, op):
        V = self.vals
        if isinstance(op, t.AllocOp):
            sp = t.space_of(op.result.type)
            if sp != "reg":
                raise CodegenError(f"{sp} alloc inside the lane loop")
            for i in range(math.prod(_shape(op.result))):
                e = self.reg()
                self.emit(mir.Const32Op(e, 0))
                self.elems[(op.result, i)] = e
        elif isinstance(op, t.ForOp):
            if op.bind.data != "seq":
                raise CodegenError(f"bound loop {op.tag} inside the lane loop")
            self.loop(op)
        elif isinstance(op, t.GuardOp):
            idx = self.index(op)
            ops = list(op.body.block.ops)
            self.region(list(zip(idx, op.ub_list)), lambda: self.block(ops))
        elif isinstance(op, t.LoadOp):
            sp = t.space_of(op.buf.type)
            idx = self.index(op)
            if sp == "reg":
                e = self.reg_elem(op.buf, idx)
                o = self.order[op]
                last = max((self.order[u.operation] for u in op.result.uses), default=o)
                if any(o < s < last for s in self.reg_stores.get(op.buf, ())):
                    c = self.reg()
                    self.copy(c, e)
                    e = c
                V[op.result] = e
            else:
                a = self.global_addr(op.buf, idx) if sp == "global" else self.lin_reg(self.smem_lin(op.buf, idx))
                V[op.result] = self.load_mem(sp, a, op.buf)
        elif isinstance(op, t.StoreOp):
            sp = t.space_of(op.buf.type)
            idx = self.index(op)
            val = self.r(op.value)
            if sp == "reg":
                e = self.reg_elem(op.buf, idx)
                p = self.stack[-1].preg
                if p is not None:
                    val = self.select(p, val, e)
                self.copy(e, val)
            elif sp == "global":
                self.store_mem("global", self.global_addr(op.buf, idx), val)
            else:
                self.store_mem("smem", self.lin_reg(self.smem_lin(op.buf, idx)), val)
        elif isinstance(op, t.GatherOp):
            off = int(op.offset.value.data)
            sp = t.space_of(op.buf.type)
            if sp == "global":
                a = self.add(self.lin_reg(Lin.of(self.base_atom(op.buf)) + off), self.r(op.idx))
                V[op.result] = self.load_mem("global", a, op.buf)
            elif sp == "smem":
                a = self.add(self.lin_reg(self.smem_lin(op.buf, [Lin.of(off)])), self.r(op.idx))
                V[op.result] = self.load_mem("smem", a, op.buf)
            else:
                raise CodegenError("gather from a register buffer")
        elif isinstance(op, t.ConstOp):
            V[op.result] = self.const(int(op.value.value.data))
        elif isinstance(op, t.CastOp):
            V[op.result] = self.r(op.x)
        elif isinstance(op, t.ApplyOp):
            V[op.result] = self.lin_reg(self.index(op)[0])
        elif isinstance(op, t.BarrierOp):
            self.forget_loads()
            self.emit(mir.BarOp())
        elif isinstance(op, (t.AddOp, t.SubOp, t.MulOp, t.MaxOp, t.DivOp, t.SraOp)):
            fn = {t.AddOp: self.add, t.SubOp: self.sub, t.MulOp: self.mul, t.MaxOp: self.max, t.DivOp: self.div,
                  t.SraOp: self.sra}[type(op)]
            V[op.result] = fn(self.r(op.lhs), self.r(op.rhs))
        elif isinstance(op, t.MinOp):
            V[op.result] = self.min(self.r(op.lhs), self.r(op.rhs))
        elif isinstance(op, t.CmpOp):
            fn = {"eq": self.eq, "ne": self.ne, "lt": self.lt, "le": self.le, "gt": self.gt, "ge": self.ge}
            V[op.result] = fn[op.pred.data](self.r(op.lhs), self.r(op.rhs))
        elif isinstance(op, t.SelectOp):
            c = self.r(op.cond)
            if not isinstance(op.cond.owner, t.CmpOp):
                c = self.nz(c)
            V[op.result] = self.select(c, self.r(op.a), self.r(op.b))
        elif isinstance(op, t.RequantOp):
            M, s = int(op.M.value.data), int(op.s.value.data)
            acc = self.r(op.acc)
            if op.extra is None:
                V[op.result] = self.requant(acc, M, s)
            else:
                S = self.add(self.r(op.extra), s) if s else self.r(op.extra)
                pw = self.sub(0, self.sra(1 << 31, self.sub(31, S)))       # 2^S as a word (S <= 31)
                rnd = self.div(pw, 2)                                        # 2^(S-1), 0 for S = 0
                V[op.result] = self.requant(acc, M, S, rnd=rnd)
        elif isinstance(op, t.RsqrtIndexOp):
            n = int(op.n.value.data)
            idx, kk = self.reg(), self.reg()
            self.copy(idx, self.r(op.ss))
            self.emit(mir.Const32Op(kk, 0))

            def step(_o):
                m = self.max(self.sub(idx, n - 1), 0)
                f = self.nz(m)                                   # idx >= n
                self.copy(kk, self.add(kk, f))
                self.copy(idx, self.sra(idx, self.add(f, f)))
            self.counted(_rsqrt_trips(n), step)
            V[op.idx], V[op.k] = idx, kk
        else:
            raise CodegenError(f"unsupported op {op.name}")


def _coalesce(ops, copies: set) -> list[Ins]:
    """Lift to Ins and remove the copies `add e, v, zero` made for reg-buffer stores when v is a single-definition
    register computed earlier in the same straight-line run and dead after the copy, and e is neither read nor
    written in between: v is renamed to e."""
    flags = [id(op) in copies for op in ops]
    code = lift(mir.FuncOp("tmp", [op.clone() for op in ops]))
    keep = list(flags)
    defs: dict[str, list[int]] = {}
    for k, i in enumerate(code):
        for r in i.writes():
            defs.setdefault(r, []).append(k)
    removed = set()
    for k, i in enumerate(code):
        if not keep[k]:
            continue
        e, v = i.d, i.a
        if e == v:
            removed.add(k)
            continue
        dv = defs.get(v, [])
        if len(dv) != 1 or dv[0] >= k or not v.startswith("v"):
            continue
        j = dv[0]
        between = range(j + 1, k)
        if any(code[m].op in ("label", "br", "sync", "ret") for m in between):
            continue
        if any(e in code[m].reads() or e in code[m].writes() for m in between if m not in removed):
            continue
        if any(v in code[m].reads() for m in range(k + 1, len(code)) if m not in removed):
            continue
        if any(v in code[m].reads() for m in range(0, j) if m not in removed):
            continue
        for m in [j, *between]:
            if m not in removed:
                code[m].rename({v: e})
        defs.setdefault(e, []).append(j)
        defs[v] = []
        removed.add(k)
    return [i for k, i in enumerate(code) if k not in removed]
