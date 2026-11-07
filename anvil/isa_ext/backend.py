"""Code generation for the extended ISA: MIR ops for the candidate instructions, the idiom switches and the MIR
peepholes that select them, and a context manager that turns them on.

    with enabled({"MIN", "SLT"}):          # any subset of anvil.isa_ext.isa.PROPOSALS
        ck = compile_kernel(...)           # the anvil codegen + backend, emitting the new ops where they pay

What each proposal changes (nothing else in the compiler is touched; with no proposal enabled the output is the base
compiler's, word for word):
  MIN  idioms.Builder.min -> MIN (clamp_i8 and requant's clamp become MAX + MIN).
  SLT  Builder.lt/gt -> SLT, ge/le -> 1 - SLT; the codegen predicate "f == 0" for a provably non-negative f
       (lower._Gen.pred_from) -> SLT f, 1 (base: 1 - (f div f)).
  SHL  Builder.mul by a power-of-two constant 2^k >= 256 -> SHL by k (k is one CONST, 2^k needs a const32 chain;
       below 256 MUL by a shared CONST is as cheap); the register-shift requant rounding term 2^(S-1) = (0 - SRA(INT_MIN, 31 - S)) div 2 ->
       SHL 1, (S - 1) (S - 1 wraps to 2^32 - 1 for S = 0, and SHL by >= 32 is 0: exactly the base value).
  AND  the predicate conjunction (Builder: and_pred) -> AND; MIR peephole `q = x div 2^k; t = q * 2^k; r = x - t`
       -> `r = x AND (2^k - 1)` (unsigned DIV, wrap-around MUL/SUB: identical for every word), e.g. the lane index
       %threadIdx mod lanes at WARPS_PER_CORE > 1.
  MAD  MIR peephole `t = a * b; d = d + t` (t used once, a, b, d not redefined in between, same straight-line
       run) -> `MAD d, a, b` (rd read and written: a tied accumulator). The register allocator sees rd as read
       (lir.Ins.reads patched); a program whose allocation spills while it contains a MAD is recompiled without
       MAD fusion (the spill rewriter does not handle a tied spilled accumulator).
Encoding: anvil.isa_ext.isa (5-bit opcodes); `anvil.backend.emit.encode`/`asm_text` are swapped while enabled.
"""
from __future__ import annotations

import contextlib

from ..backend import emit, idioms, lir
from ..ir import mir
from . import isa as X

# --------------------------------------------------------------------------------------------- MIR ops
MinOp, AndOp, ShlOp, SltOp, MadOp = (mir._alu(n, n.upper()) for n in ("min", "and", "shl", "slt", "mad"))
MadOp.READS = ("rs", "rt", "rd")         # tied accumulator (verify_ checks it like a read register)
EXT_OPS = {"min": MinOp, "and": AndOp, "shl": ShlOp, "slt": SltOp, "mad": MadOp}
NAME = {"MIN": "min", "AND": "and", "SHL": "shl", "SLT": "slt", "MAD": "mad"}
_MASK = 0xFFFFFFFF

_ACTIVE: frozenset = frozenset()


def active() -> frozenset:
    return _ACTIVE


def _fold(word_bits=32):
    sem = X.semantics(word_bits)
    return {NAME[m]: (lambda f: (lambda a, b: f(0, a, b)))(sem[m]) for m in ("MIN", "AND", "SHL", "SLT")}


# --------------------------------------------------------------------------------------------- idioms
def _is_pow2(c):
    return isinstance(c, int) and c > 1 and (c & _MASK) == c and c & (c - 1) == 0


def _builder_patches(props):
    B = idioms.Builder
    out = {}
    if "MIN" in props:
        def min_(self, a, b):
            return self._alu(MinOp, a, b)
        out["min"] = min_
    if "SLT" in props:
        def lt(self, a, b):
            return self._alu(SltOp, a, b)

        def gt(self, a, b):
            return self._alu(SltOp, b, a)

        def ge(self, a, b):
            return self.sub(1, self._alu(SltOp, a, b))

        def le(self, a, b):
            return self.sub(1, self._alu(SltOp, b, a))
        out.update(lt=lt, gt=gt, ge=ge, le=le)
    if "SHL" in props:
        base_mul = B.mul

        def mul(self, a, b, into=None):
            if _is_pow2(b) and b > 255 and isinstance(a, str):
                return self._alu(ShlOp, a, b.bit_length() - 1, into)
            if _is_pow2(a) and a > 255 and isinstance(b, str):
                return self._alu(ShlOp, b, a.bit_length() - 1, into)
            return base_mul(self, a, b, into)
        out["mul"] = mul
        base_sra = B.sra

        def sra(self, a, b, into=None):
            # INT_MIN as an inline const32 (not an args-block load) so the shl_round peephole can see the idiom
            if isinstance(a, int) and (a & _MASK) == 1 << 31 and isinstance(b, str):
                r = self.reg()
                self.emit(mir.Const32Op(r, 1 << 31))
                return self._alu(mir.SraOp, r, b, into)
            return base_sra(self, a, b, into)
        out["sra"] = sra
    return out


def _gen_patches(props):
    from ..codegen import lower as L
    out = {}
    if "SLT" in props:
        base = L._Gen.pred_from

        def pred_from(self, f):
            _, hi = f.range()
            if hi <= 1:
                return base(self, f)
            return self._alu(SltOp, self.lin_reg(f), 1)        # f >= 0 (false_indicator): f == 0 iff f < 1
        out["pred_from"] = pred_from
    if "AND" in props:
        base_and = L._Gen.and_pred

        def and_pred(self, a, b):
            if a is None or b is None:
                return base_and(self, a, b)
            return self._alu(AndOp, a, b)                       # predicates are 0 / 1
        out["and_pred"] = and_pred
    return out


# --------------------------------------------------------------------------------------------- peepholes
def _consts(code):
    from collections import Counter
    defs = Counter(r for i in code for r in i.writes())
    return {i.d: i.imm & _MASK for i in code if i.op in ("const", "const32") and defs[i.d] == 1}, defs


def _uses(code):
    from collections import Counter
    return Counter(r for i in code for r in i.reads())


def _redefined(code, lo, hi, regs):
    for i in code[lo + 1:hi]:
        if i.op in ("label", "br", "sync", "ret", "bar"):
            return True
        if any(w in regs for w in i.writes()):
            return True
    return False


def fuse_mad(code):
    """t = a*b; ...; d = d + t  ->  mad d, a, b (t single-use, single-def)."""
    _, defs = _consts(code)
    uses = _uses(code)
    pos = {i.d: k for k, i in enumerate(code) if i.op == "mul" and defs[i.d] == 1}
    drop, n = set(), 0
    for k, i in enumerate(code):
        if i.op != "add":
            continue
        for t, x in ((i.b, i.a), (i.a, i.b)):
            j = pos.get(t)
            if j is None or j in drop or j > k or uses[t] != 1 or x != i.d or t == x:
                continue
            m = code[j]
            if _redefined(code, j, k, {m.a, m.b, i.d}) or i.d in (m.a, m.b):
                continue
            code[k] = lir.Ins("mad", i.d, m.a, m.b)
            drop.add(j)
            n += 1
            break
    return [i for k, i in enumerate(code) if k not in drop], n


def and_mod(code, fresh):
    """q = x div 2^k; t = q * 2^k (or 2^k * q); r = x - t  ->  r = x AND (2^k - 1)."""
    cv, defs = _consts(code)
    divs = {i.d: i for i in code if i.op == "div" and defs[i.d] == 1 and _is_pow2(cv.get(i.b))}
    muls = {}
    for i in code:
        if i.op == "mul" and defs[i.d] == 1:
            for q, c in ((i.a, i.b), (i.b, i.a)):
                if q in divs and cv.get(c) == cv[divs[q].b]:
                    muls[i.d] = divs[q]
    out, n = [], 0
    for i in code:
        if i.op == "sub" and i.b in muls and muls[i.b].a == i.a and defs[i.a] == 1:
            m = fresh()
            out += [lir.Ins("const32", m, imm=cv[muls[i.b].b] - 1), lir.Ins("and", i.d, i.a, m)]
            n += 1
            continue
        out.append(i)
    return out, n


def shl_round(code, fresh):
    """u = 31 - S; v = SRA(INT_MIN, u); pw = 0 - v; rnd = pw div 2  ->  s1 = S - 1; rnd = SHL(1, s1)."""
    cv, defs = _consts(code)
    by = {i.d: i for i in code if i.d is not None and defs[i.d] == 1}
    uses = _uses(code)
    out, n = [], 0
    for i in code:
        if i.op == "div" and cv.get(i.b) == 2 and i.a in by:
            pw = by[i.a]
            v = by.get(pw.b) if pw.op == "sub" and cv.get(pw.a) == 0 else None
            u = by.get(v.b) if v is not None and v.op == "sra" and cv.get(v.a) == 1 << 31 else None
            if u is not None and u.op == "sub" and cv.get(u.a) == 31 and uses[pw.d] == 1:
                one, s1 = fresh(), fresh()
                out += [lir.Ins("const32", one, imm=1), lir.Ins("sub", s1, u.b, one), lir.Ins("shl", i.d, one, s1)]
                n += 1
                continue
        out.append(i)
    return out, n


def select(f, props):
    """MIR peepholes for the enabled proposals: (new func, {peephole: count})."""
    code = lir.lift(f)
    fresh = lir.Fresh(code)
    stats = {}
    if "AND" in props:
        code, stats["and_mod"] = and_mod(code, fresh)
    if "SHL" in props:
        code, stats["shl_round"] = shl_round(code, fresh)
    if "MAD" in props:
        code, stats["mad"] = fuse_mad(code)
    return lir.lower(code, f.sym_name.data), stats


# --------------------------------------------------------------------------------------------- encode / asm
def encode(code, name):
    f = lir.lower(code, name)
    lab, words = mir.labels(f), []
    for op in f.instrs:
        if isinstance(op, mir.LabelOp):
            continue
        fields = {fld: mir.phys_index(getattr(op, fld).data) for fld in dict.fromkeys(op.READS + op.WRITES)}
        if isinstance(op, mir.BrOp):
            fields.update(nzp=op.nzp.value.data, imm=lab[op.target.data])
        elif isinstance(op, mir.SyncOp):
            fields.update(nzp=0, imm=lab[op.target.data])
        elif isinstance(op, mir.ConstOp):
            fields["imm"] = op.imm.value.data
        words.append(X.encode(op.MNEMONIC, **fields))
    return words, f


_base_asm = emit.asm_text


def asm_text(code, name, header=""):
    text = _base_asm(code, name, header)
    regs = iter([i for i in code if i.op == "mad"])
    lines = []
    for ln in text.splitlines():
        if ln.strip() == "MAD":
            i = next(regs)
            ln = f"    MAD {i.d}, {i.a}, {i.b}"
        lines.append(ln)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------- enable
_base_reads = lir.Ins.reads


def _reads(self):
    r = _base_reads(self)
    return r + [self.d] if self.op == "mad" and self.d is not None else r


@contextlib.contextmanager
def enabled(props):
    """Compile for the base ISA + `props` (iterable of PROPOSALS) inside the block. Not re-entrant across threads."""
    global _ACTIVE
    props = frozenset(props)
    bad = props - set(X.PROPOSALS)
    if bad:
        raise ValueError(f"unknown proposals {sorted(bad)}")
    import anvil.backend as be
    import anvil.codegen as cg

    from ..codegen import lower as L
    names = tuple(NAME[m] for m in X.PROPOSALS)
    saved = []

    def patch(obj, attr, val):
        saved.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, val)

    base_compile = cg.compile_mir

    def compile_mir(module, **kw):
        f = lir.get_func(module)
        g, st = select(f, props)
        prog = base_compile(g, **kw)
        if st.get("mad") and prog.stats["spilled_vregs"]:
            g, st = select(f, props - {"MAD"})
            st["mad_fallback"] = 1
            prog = base_compile(g, **kw)
        prog.stats["isa_ext"] = {"props": sorted(props), **st}
        return prog

    try:
        _ACTIVE = props
        if props:
            patch(lir, "ALU", lir.ALU + names)
            patch(lir, "PURE", lir.PURE + names)
            patch(lir, "_OPS", {**lir._OPS, **EXT_OPS})
            patch(be, "PURE", lir.PURE)
            patch(emit, "ALU", emit.ALU + tuple(n for n in names if n != "mad"))
            patch(emit, "_FOLD", {**emit._FOLD, **_fold()})
            patch(emit, "encode", encode)
            patch(emit, "asm_text", asm_text)
            patch(lir.Ins, "reads", _reads)
            patch(cg, "compile_mir", compile_mir)
            for k, v in _builder_patches(props).items():
                patch(idioms.Builder, k, v)
            for k, v in _gen_patches(props).items():
                patch(L._Gen, k, v)
        yield props
    finally:
        for obj, attr, val in reversed(saved):
            setattr(obj, attr, val)
        _ACTIVE = frozenset()
