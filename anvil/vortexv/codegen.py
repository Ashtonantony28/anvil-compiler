"""anvil.graph -> one RV64IM + Xmlacc program for Vortex-V.

Memory: every value gets its own buffer and none is reused, so a buffer's padding stays zero. A tensor is stored
row-major over its last axis; an int8 tensor's rows are padded to a multiple of 8 bytes, so any int8 row can feed
``mlacc.macc`` eight lanes at a time (the zero padding adds nothing to a dot product). Wider tensors (i16, i32)
are dense. A matmul reads its right operand transposed (one padded row per output column): constant weights are
transposed when the image is built, ``b = transpose(x)`` of the last two axes reads ``x`` in place, and anything
else gets a transpose kernel into a scratch buffer.

Arithmetic: Anvil's requant is ``clamp((acc*M + 2^(s-1)) >> s, -128, 127)`` with every intermediate in int32
(``xf.golden.requant``). Vortex-V's ``mlacc.rqs`` is ``clamp((x + 2^(s-1)) >> s, -128, 127)`` on a 64-bit x, so
``requant(acc, M, s) == rqs(acc*M, s)`` whenever the golden model accepts the inputs: the product is exact in 64
bits. Every other op is plain RV64IM integer code, see the kernels below.
"""
from dataclasses import dataclass, field
import math

import numpy as np

from ..ir import graph as g
from . import _vv

LANES = 8                           # int8 lanes per 64-bit register (Vortex-V XMLACC-SPEC-v1)
ESIZE = {"i8": 1, "i16": 2, "i32": 4}
CODE_BASE = 0x8001_0000             # tohost sits at 0x8000_1000: the program lives above it, a stub jumps there
DATA_BASE = 0x8010_0000
MAX_UNROLL_WORDS = 8                # dot products of up to 64 int8 lanes are unrolled
I16 = (-32768, 32767)

# registers (RISC-V numbers). x31 is the assembler temporary, x0..x4 are left alone.
AT = 31
T0, T1, T2, T3, T4, T5 = 5, 6, 7, 28, 29, 30
A0, A1, A2, A3, A4, A5, A6, A7 = 10, 11, 12, 13, 14, 15, 16, 17
S0, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11 = 8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27


def pad8(n):
    return -(-n // LANES) * LANES


@dataclass
class Buf:
    """A tensor in memory: `rows` rows of `cols` elements, each row `stride` elements apart."""
    name: str
    dims: list
    dtype: str
    addr: int = 0

    @property
    def esize(self):
        return ESIZE[self.dtype]

    @property
    def cols(self):
        return self.dims[-1] if self.dims else 1

    @property
    def rows(self):
        return math.prod(self.dims[:-1]) if self.dims else 1

    @property
    def stride(self):
        return pad8(self.cols) if self.dtype == "i8" else self.cols

    @property
    def nbytes(self):
        return pad8(self.rows * self.stride * self.esize)

    def elem_strides(self):
        """Element stride of every axis (the last axis 1, the row axis `stride`)."""
        r = len(self.dims)
        st = [1] * r
        if r >= 2:
            st[r - 2] = self.stride
            for a in range(r - 3, -1, -1):
                st[a] = st[a + 1] * self.dims[a + 1]
        return st

    def pack(self, arr):
        """Bytes of `arr` (shape dims) in this buffer's layout."""
        arr = np.asarray(arr, dtype=np.int64).reshape(self.rows, self.cols)
        out = np.zeros((self.rows, self.stride), dtype=np.int64)
        out[:, :self.cols] = arr
        dt = {"i8": "<i1", "i16": "<i2", "i32": "<i4"}[self.dtype]
        raw = out.astype(dt).tobytes()
        return raw + b"\0" * (self.nbytes - len(raw))

    def unpack(self, raw):
        dt = {"i8": "<i1", "i16": "<i2", "i32": "<i4"}[self.dtype]
        a = np.frombuffer(bytes(raw[:self.rows * self.stride * self.esize]), dtype=dt).astype(np.int64)
        return a.reshape(self.rows, self.stride)[:, :self.cols].reshape(self.dims)


@dataclass
class Layout:
    bufs: dict = field(default_factory=dict)        # name -> Buf
    next: int = DATA_BASE

    def new(self, name, dims, dtype):
        assert name not in self.bufs, name
        b = Buf(name, list(dims), dtype, self.next)
        self.bufs[name] = b
        self.next += max(b.nbytes, LANES)
        return b


class Emitter:
    """Thin layer over Vortex-V's assembler: constants, loops, typed loads and stores."""

    def __init__(self, mutant=None):
        self.rv = _vv.rvasm()
        self.a = self.rv.Asm(base=CODE_BASE)
        self.n = 0
        self.mutant = mutant

    def lbl(self, p):
        self.n += 1
        return f"{p}_{self.n}"

    def li(self, rd, v):
        v = int(v)
        a = self.a
        if -2048 <= v < 2048:
            a.addi(rd, 0, v)
        elif -(1 << 31) <= v < (1 << 31):
            hi = ((v + 0x800) >> 12) & 0xFFFFF
            lo = v - ((((hi << 12) + (1 << 31)) & 0xFFFF_FFFF) - (1 << 31))
            a.lui(rd, hi)
            if lo:
                a.addiw(rd, rd, lo)
        else:
            lo = ((v & 0xFFF) ^ 0x800) - 0x800
            self.li(rd, (v - lo) >> 12)
            a.slli(rd, rd, 12)
            if lo:
                a.addi(rd, rd, lo)

    def addimm(self, rd, rs, v):
        if v == 0:
            if rd != rs:
                self.a.addi(rd, rs, 0)
        elif -2048 <= v < 2048:
            self.a.addi(rd, rs, v)
        else:
            self.li(AT, v)
            self.a.add(rd, rs, AT)

    def loop(self, cnt, n, body):
        """body() n times (n >= 1 known at compile time), counting `cnt` down."""
        if n <= 0:
            return
        if n == 1:
            body()
            return
        top = self.lbl("L")
        self.li(cnt, n)
        self.a.label(top)
        body()
        self.a.addi(cnt, cnt, -1)
        self.a.bne(cnt, 0, top)

    def loop_reg(self, cnt, nreg, body):
        """body() nreg times (a register holding a count >= 1)."""
        top = self.lbl("R")
        self.a.addi(cnt, nreg, 0)
        self.a.label(top)
        body()
        self.a.addi(cnt, cnt, -1)
        self.a.bne(cnt, 0, top)

    def load(self, rd, base, dtype, off=0):
        {"i8": self.a.lb, "i16": self.a.lh, "i32": self.a.lw}[dtype](rd, base, off)

    def store(self, rs, base, dtype, off=0):
        {"i8": self.a.sb, "i16": self.a.sh, "i32": self.a.sw}[dtype](rs, base, off)

    def clamp(self, r, lo, hi):
        a = self.a
        l1, l2 = self.lbl("ch"), self.lbl("cl")
        self.li(AT, hi)
        a.bge(AT, r, l1)
        a.addi(r, AT, 0)
        a.label(l1)
        self.li(AT, lo)
        a.bge(r, AT, l2)
        a.addi(r, AT, 0)
        a.label(l2)

    def requant_reg(self, r, sreg, dtype, s_const=None):
        """r = clamp_dtype((r + 2^(s-1)) >> s) where r already holds acc*M (64-bit exact)."""
        if dtype == "i8" and self.mutant != "floor":
            self.a.rqs(r, r, sreg)
            return
        if dtype == "i8":                      # mutant: arithmetic shift, no rounding, no clamp
            self.a.sra(r, r, sreg)
            return
        # wide result: round term from the shift register (2^(s-1), 0 when s == 0)
        if s_const is not None:
            if s_const:
                self.addimm(r, r, 1 << (s_const - 1))
        else:
            skip = self.lbl("rz")
            self.a.beq(sreg, 0, skip)
            self.a.addi(AT, sreg, -1)
            self.a.addi(T5, 0, 1)
            self.a.sll(AT, T5, AT)
            self.a.add(r, r, AT)
            self.a.label(skip)
        self.a.sra(r, r, sreg)
        self.clamp(r, *I16) if dtype == "i16" else None


def _int(attr):
    return int(attr.value.data)


def _is_swap_last2(perm):
    r = len(perm)
    return r >= 2 and list(perm) == list(range(r - 2)) + [r - 1, r - 2]


class Compiler:
    def __init__(self, func, consts, mutant=None):
        self.func, self.consts = func, consts or {}
        self.e = Emitter(mutant)
        self.lay = Layout()
        self.buf = {}          # SSA value -> Buf (aliases share a Buf)
        self.view = {}         # sym -> (Buf, transform) for values that are not materialised
        self.bt = {}           # SSA value -> Buf holding its transpose for matmul right operands
        self.const_data = {}   # buf name -> packed bytes
        self.inputs = {}       # sym -> Buf
        self.syms = {}         # sym -> Buf
        self.outputs = []
        self.users = {}
        for op in func.body.block.ops:
            for i, x in enumerate(op.operands):
                self.users.setdefault(x, []).append((op, i))

    # ------------------------------------------------------------------ values
    def _const_value(self, op):
        if op.value is not None:
            return np.asarray(op.value.get_values(), dtype=np.int64).reshape(g.qt(op.result).dims)
        return np.asarray(self.consts[op.sym_str], dtype=np.int64)

    def _new(self, op, suffix=""):
        t = g.qt(op.results[0])
        name = (op.sym_str or f"v{len(self.lay.bufs)}") + suffix
        return self.lay.new(name, t.dims, t.dtype)

    def _bt(self, bval):
        """Buffer holding b's transpose (..., N, K) with padded K rows, made on first use."""
        if bval in self.bt:
            return self.bt[bval]
        owner = bval.owner
        tb = g.qt(bval)
        dims = tb.dims[:-2] + [tb.dims[-1], tb.dims[-2]]
        if isinstance(owner, g.ConstOp):
            b = self.lay.new(f"{owner.sym_str}.T", dims, "i8")
            v = self._const_value(owner)
            self.const_data[b.name] = b.pack(np.swapaxes(v, -1, -2))
        elif isinstance(owner, g.TransposeOp) and _is_swap_last2(owner.perm_list):
            b = self.buf[owner.x]
        else:
            src = self.buf[bval]
            b = self.lay.new(f"{src.name}.T", dims, "i8")
            self.k_transpose(src, b, list(range(len(dims) - 2)) + [len(dims) - 1, len(dims) - 2])
        self.bt[bval] = b
        return b

    def _elide_transpose(self, op):
        """A last-two-axes transpose read only as matmul right operands is never materialised."""
        if not _is_swap_last2(op.perm_list):
            return False
        us = self.users.get(op.results[0], [])
        return bool(us) and all(isinstance(u, (g.QMatmulOp, g.QLinearOp)) and i == 1 for u, i in us)

    # ------------------------------------------------------------------ driver
    def run(self):
        e = self.e
        for op in self.func.body.block.ops:
            if isinstance(op, g.InputOp):
                b = self._new(op)
                self.inputs[op.sym_str] = b
            elif isinstance(op, g.ConstOp):
                b = self._new(op)
                self.const_data[b.name] = b.pack(self._const_value(op))
            elif isinstance(op, (g.QMatmulOp, g.QLinearOp)):
                b = self._new(op)
                lut = None if op.lut is None else self.buf[op.lut]
                self.k_matmul(self.buf[op.a], self._bt(op.b), b, _int(op.M), _int(op.s), lut,
                              batched_b=len(g.qt(op.b).dims) > 2)
            elif isinstance(op, g.LutOp):
                b = self._new(op)
                self.k_lut(self.buf[op.x], self.buf[op.table], b)
            elif isinstance(op, g.LutMulOp):
                b = self._new(op)
                self.k_lut_mul(self.buf[op.x], self.buf[op.y], self.buf[op.table], b, _int(op.M), _int(op.s))
            elif isinstance(op, g.RmsNormIntOp):
                b = self._new(op)
                self.k_rmsnorm(self.buf[op.x], self.buf[op.w], self.buf[op.lut], b, _int(op.M), _int(op.s),
                               op.ss_shift_int, op.ss_init_int)
            elif isinstance(op, g.SoftmaxLutOp):
                b = self._new(op)
                self.k_softmax(self.buf[op.x], self.buf[op.lut], b, _int(op.one), bool(op.causal.value.data))
            elif isinstance(op, g.AddRequantOp):
                b = self._new(op)
                self.k_add_requant(self.buf[op.a], self.buf[op.b], b, _int(op.Ma), _int(op.Mb), _int(op.s))
            elif isinstance(op, g.TransposeOp):
                src = self.buf[op.x]
                if self._elide_transpose(op):
                    if op.sym_str:
                        self.view[op.sym_str] = (src, "swap")
                    self.buf[op.results[0]] = None
                    continue
                b = self._new(op)
                self.k_transpose(src, b, op.perm_list)
            elif isinstance(op, g.ReshapeOp):
                src = self.buf[op.x]
                t = g.qt(op.result)
                if t.dims[-1:] == src.dims[-1:] and len(t.dims) >= 1:
                    b = Buf(src.name, t.dims, t.dtype, src.addr)        # same rows, same padding: a view
                    if op.sym_str:
                        self.view[op.sym_str] = (b, None)
                    self.buf[op.results[0]] = b
                    continue
                b = self._new(op)
                self.k_reshape(src, b)
            elif isinstance(op, g.OutputOp):
                for x in op.values:
                    if self.buf.get(x) is None:              # an elided transpose is an output: materialise
                        raise NotImplementedError("graph output of an elided transpose")
                    self.outputs.append(self.buf[x])
                continue
            else:
                raise NotImplementedError(op.name)
            self.buf[op.results[0]] = b
            if op.sym_str:
                self.syms[op.sym_str] = b
        e.a.la(T0, self.e.rv.TOHOST_ADDR)
        e.a.addi(T1, 0, 1)
        e.a.sw(T1, T0, 0)
        spin = e.lbl("spin")
        e.a.label(spin)
        e.a.jal(0, spin)

    # ------------------------------------------------------------------ kernels
    def k_matmul(self, A, BT, out, m, s, lut, batched_b):
        """out[..., i, j] = requant(sum_k A[..., i, k] * BT[..., j, k], m, s) [-> lut[. + 128]]."""
        e, a = self.e, self.e.a
        kp = A.stride
        assert BT.stride == kp, (A, BT)
        batch, mr, n = math.prod(A.dims[:-2]), A.dims[-2], BT.dims[-2]
        kw = kp // LANES
        e.li(S8, m)
        e.li(S9, s)
        if lut is not None:
            a.la(S10, lut.addr + 128)
        a.la(S2, A.addr)
        a.la(S4, out.addr)
        a.la(S3, BT.addr)

        def dot():
            a.addi(T0, 0, 0)
            a.addi(T1, S2, 0)
            a.addi(T2, A0, 0)
            if kw <= MAX_UNROLL_WORDS:
                for w in range(kw):
                    a.ld(T3, T1, LANES * w)
                    a.ld(T4, T2, LANES * w)
                    self._macc(T0, T3, T4)
            else:
                def step():
                    a.ld(T3, T1, 0)
                    a.ld(T4, T2, 0)
                    self._macc(T0, T3, T4)
                    a.addi(T1, T1, LANES)
                    a.addi(T2, T2, LANES)
                e.loop(A1, kw, step)
            a.mul(T0, T0, S8)
            e.requant_reg(T0, S9, out.dtype, s_const=s)
            if lut is not None:
                a.add(T0, T0, S10)
                a.lb(T0, T0, 0)
            e.store(T0, S11, out.dtype)
            e.addimm(S11, S11, out.esize)
            e.addimm(A0, A0, kp)

        def row():
            a.addi(A0, A2, 0)
            a.addi(S11, S4, 0)
            e.loop(S7, n, dot)
            e.addimm(S2, S2, kp)
            e.addimm(S4, S4, out.stride * out.esize)

        def bat():
            a.addi(A2, S3, 0)
            e.loop(S6, mr, row)
            if batched_b:
                e.addimm(S3, S3, n * kp)

        e.loop(S5, batch, bat)

    def _macc(self, acc, x, y):
        if self.e.mutant == "scalar":           # RV64IM-only baseline: 8 x (byte extract, multiply, add)
            a = self.e.a
            for lane in range(LANES):
                a.slli(T5, x, 56 - 8 * lane)
                a.srai(T5, T5, 56)
                a.slli(AT, y, 56 - 8 * lane)
                a.srai(AT, AT, 56)
                a.mul(T5, T5, AT)
                a.add(acc, acc, T5)
        else:
            self.e.a.macc(acc, x, y, acc)

    def _elementwise(self, ins, out, body):
        """body(in_regs, out_reg) per element; in_regs hold the loaded elements, the result goes in T0."""
        e, a = self.e, self.e.a
        ptrs = [S2, S3, S4][:len(ins)]
        regs = [T1, T2, T3][:len(ins)]
        for p, b in zip(ptrs, ins):
            a.la(p, b.addr)
        a.la(S5, out.addr)

        def elem():
            for p, r, b in zip(ptrs, regs, ins):
                e.load(r, p, b.dtype)
            body(regs)
            e.store(T0, S5, out.dtype)
            for p, b in zip(ptrs, ins):
                e.addimm(p, p, b.esize)
            e.addimm(S5, S5, out.esize)

        def row():
            e.loop(S7, out.cols, elem)
            for p, b in zip(ptrs, ins):
                e.addimm(p, p, (b.stride - b.cols) * b.esize)
            e.addimm(S5, S5, (out.stride - out.cols) * out.esize)

        e.loop(S6, out.rows, row)

    def k_lut(self, x, table, out):
        e, a = self.e, self.e.a
        a.la(S10, table.addr + 128)

        def body(r):
            a.add(T0, r[0], S10)
            a.lb(T0, T0, 0)
        self._elementwise([x], out, body)

    def k_lut_mul(self, x, y, table, out, m, s):
        """requant(lut[x + 128] * y, m, s)."""
        e, a = self.e, self.e.a
        a.la(S10, table.addr + 128)
        e.li(S8, m)
        e.li(S9, s)

        def body(r):
            a.add(T0, r[0], S10)
            a.lb(T0, T0, 0)
            a.mul(T0, T0, r[1])
            a.mul(T0, T0, S8)
            e.requant_reg(T0, S9, out.dtype, s_const=s)
        self._elementwise([x, y], out, body)

    def k_add_requant(self, x, y, out, ma, mb, s):
        """clamp_dtype((x*Ma + y*Mb + 2^(s-1)) >> s)."""
        e, a = self.e, self.e.a
        e.li(S8, ma)
        e.li(S9, mb)
        e.li(S10, s)

        def body(r):
            a.mul(T0, r[0], S8)
            a.mul(T4, r[1], S9)
            a.add(T0, T0, T4)
            e.requant_reg(T0, S10, out.dtype, s_const=s)
        self._elementwise([x, y], out, body)

    def k_rmsnorm(self, x, w, lut, out, m, s, c, init):
        """Per row: ss = init + sum((x^2 + 2^(c-1)) >> c); idx, k = rsqrt_index(ss, len(lut));
        out = requant(x * w * lut[idx], m, s + k)."""
        e, a = self.e, self.e.a
        n = lut.cols
        a.la(S2, x.addr)
        a.la(S4, out.addr)
        e.li(S8, m)
        e.li(S9, n)
        e.li(S10, s)
        fast = x.dtype == "i8" and c == 0

        def row():
            a.addi(T0, 0, 0)                       # ss
            a.addi(T1, S2, 0)
            if fast:                               # sum of squares on macc: the zero padding adds nothing
                kw = x.stride // LANES
                for wd in range(kw):
                    a.ld(T3, T1, LANES * wd)
                    self._macc(T0, T3, T3)
            else:
                def sq():
                    e.load(T3, T1, x.dtype)
                    a.mul(T3, T3, T3)
                    if c:
                        e.addimm(T3, T3, 1 << (c - 1))
                        a.srai(T3, T3, c)
                    a.add(T0, T0, T3)
                    e.addimm(T1, T1, x.esize)
                e.loop(S7, x.cols, sq)
            if init:
                e.addimm(T0, T0, init)
            a.addi(T2, S10, 0)                     # shift = s + k
            top, done = e.lbl("rsq"), e.lbl("rsqd")
            a.label(top)
            a.blt(T0, S9, done)
            a.srai(T0, T0, 2)
            a.addi(T2, T2, 1)
            a.jal(0, top)
            a.label(done)
            a.slli(T0, T0, 2)                      # r = lut[idx]
            a.la(AT, lut.addr)
            a.add(T0, T0, AT)
            a.lw(T0, T0, 0)
            a.mul(S11, T0, S8)                     # r*M (wraps like the exact product: the result fits int32)
            a.addi(T1, S2, 0)
            a.la(A0, w.addr)
            a.addi(A1, S4, 0)

            def elem():
                e.load(T0, T1, x.dtype)
                a.lb(T3, A0, 0)
                a.mul(T0, T0, T3)
                a.mul(T0, T0, S11)
                e.requant_reg(T0, T2, "i8")
                a.sb(T0, A1, 0)
                e.addimm(T1, T1, x.esize)
                a.addi(A0, A0, 1)
                a.addi(A1, A1, 1)
            e.loop(S6, x.cols, elem)
            e.addimm(S2, S2, x.stride * x.esize)
            e.addimm(S4, S4, out.stride * out.esize)

        e.loop(S5, x.rows, row)

    def k_softmax(self, x, lut, out, one, causal):
        """Per row i of (..., T1, T2): over kept j (j <= i when causal), e_j = lut[x_j - rowmax + 255],
        p_j = (e_j*one + S//2) // S with S = sum e_j; masked p = 0 (the buffer is zero)."""
        e, a = self.e, self.e.a
        t1, t2 = x.dims[-2], x.dims[-1]
        batch = x.rows // t1
        a.la(S2, x.addr)
        a.la(S4, out.addr)
        a.la(S10, lut.addr + 4 * 255)
        e.li(S8, one)
        e.li(S9, t2)

        def lut_e(dst, src):                    # dst = lut[src - rowmax + 255] (rowmax in T0)
            a.sub(dst, src, T0)
            a.slli(dst, dst, 2)
            a.add(dst, dst, S10)
            a.lw(dst, dst, 0)

        def row():
            if causal:                          # kept = min(row + 1, T2), row counter in S11 (1-based)
                a.addi(S11, S11, 1)
                a.addi(A2, S11, 0)
                l = e.lbl("kc")
                a.bge(S9, A2, l)
                a.addi(A2, S9, 0)
                a.label(l)
            else:
                a.addi(A2, S9, 0)
            a.addi(T0, 0, -128)                 # rowmax
            a.addi(T1, S2, 0)

            def mx():
                a.lb(T3, T1, 0)
                l = e.lbl("mx")
                a.bge(T0, T3, l)
                a.addi(T0, T3, 0)
                a.label(l)
                a.addi(T1, T1, 1)
            e.loop_reg(S7, A2, mx)
            a.addi(T2, 0, 0)                    # S
            a.addi(T1, S2, 0)

            def tot():
                a.lb(T3, T1, 0)
                lut_e(T3, T3)
                a.add(T2, T2, T3)
                a.addi(T1, T1, 1)
            e.loop_reg(S7, A2, tot)
            a.srli(T4, T2, 1)                   # S // 2
            a.addi(T1, S2, 0)
            a.addi(A1, S4, 0)

            def p():
                a.lb(T3, T1, 0)
                lut_e(T3, T3)
                a.mul(T3, T3, S8)
                a.add(T3, T3, T4)
                a.divu(T3, T3, T2)
                e.store(T3, A1, out.dtype)
                a.addi(T1, T1, 1)
                e.addimm(A1, A1, out.esize)
            e.loop_reg(S7, A2, p)
            e.addimm(S2, S2, x.stride * x.esize)
            e.addimm(S4, S4, out.stride * out.esize)

        def bat():
            if causal:
                a.addi(S11, 0, 0)
            e.loop(S6, t1, row)

        e.loop(S5, batch, bat)

    def k_transpose(self, src, out, perm):
        """out = transpose(src, perm), any rank: one pointer and one counter per output axis."""
        e, a = self.e, self.e.a
        r = len(perm)
        st = [v * src.esize for v in src.elem_strides()]
        step = [st[p] for p in perm]
        ptr = [A0, A1, A2, A3, A4, A5][:r]
        cnt = [S5, S6, S7, S8, S9, S10][:r]
        assert r <= 6
        a.la(S2, src.addr)
        a.la(S4, out.addr)

        def level(k, base):
            if k == r - 1:
                a.addi(T1, base, 0)
                a.addi(T2, S4, 0)

                def el():
                    e.load(T0, T1, src.dtype)
                    e.store(T0, T2, out.dtype)
                    e.addimm(T1, T1, step[k])
                    e.addimm(T2, T2, out.esize)
                e.loop(cnt[k], out.dims[k], el)
                e.addimm(S4, S4, out.stride * out.esize)
                return
            a.addi(ptr[k], base, 0)

            def it():
                level(k + 1, ptr[k])
                e.addimm(ptr[k], ptr[k], step[k])
            e.loop(cnt[k], out.dims[k], it)

        level(0, S2)

    def k_reshape(self, src, out):
        """Element-order copy between two row layouts."""
        e, a = self.e, self.e.a
        total = math.prod(out.dims)
        a.la(S2, src.addr)
        a.la(S4, out.addr)
        e.li(S8, src.cols)
        e.li(S9, out.cols)
        a.addi(S6, S8, 0)
        a.addi(S7, S9, 0)

        def el():
            e.load(T0, S2, src.dtype)
            e.store(T0, S4, out.dtype)
            e.addimm(S2, S2, src.esize)
            e.addimm(S4, S4, out.esize)
            for c, n, p, b in ((S6, S8, S2, src), (S7, S9, S4, out)):
                l = e.lbl("rs")
                a.addi(c, c, -1)
                a.bne(c, 0, l)
                a.addi(c, n, 0)
                e.addimm(p, p, (b.stride - b.cols) * b.esize)
                a.label(l)
        e.loop(S5, total, el)
