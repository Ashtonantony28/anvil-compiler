"""Idiom library: MIR builders for operations the ISA lacks.

The ISA has ADD SUB MUL DIV(unsigned) SRA(signed) MAX(signed) and an equality-only CMP; no AND/OR/XOR/SHL/MIN,
no signed compare and no select. Every idiom below is branch-free (so it is uniform across a warp by
construction) and uses only those ops on 32-bit words:

    neg(x)          0 - x
    min(a, b)       a + b - max(a, b)                      exact for every pair of words (no -x overflow)
    clamp(x,lo,hi)  min(max(x, lo), hi)                    clamp_i8 = clamp(x, -128, 127)
    abs(x)          max(x, 0 - x)                          wraps for INT32_MIN, like C
    ne(a, b)        d = a - b; d / d                       unsigned DIV, and DIV by 0 is 0: 1 iff a != b
    eq(a, b)        1 - ne(a, b)
    lt(a, b)        m = max(a, b); d = m - a; d / d        signed a < b  (m == a iff a >= b)
    le gt ge        from lt with swapped operands / 1 - .
    select(c,x,y)   y + c * (x - y)                        c in {0, 1}
    mask(c)         0 - c                                  0 or all ones
    mul_pow2(x,k)   x * 2^k                                (no SHL)
    requant(acc, m, s[, rnd])   clamp_i8((acc * m + rnd) >>> s), rnd = 2^(s-1) (0 when s = 0)
    addr(base, (idx, stride), ...)   base + sum(idx * stride): wide address arithmetic via const32 strides

Operands are registers (str) or Python ints (materialized with `const32`; constants are CSE'd by the backend).
"""
from __future__ import annotations

from ..ir import mir


class Builder:
    """Appends MIR ops to `self.ops`, allocating virtual registers from `start`."""

    def __init__(self, start: int = 0):
        self.ops: list = []
        self.n = start

    # ----------------------------------------------------------------------------------------- basics
    def reg(self) -> str:
        self.n += 1
        return f"v{self.n - 1}"

    def emit(self, op):
        self.ops.append(op)
        return op

    def val(self, x) -> str:
        return x if isinstance(x, str) else self.const(x)

    def const(self, value: int) -> str:
        r = self.reg()
        self.emit(mir.Const32Op(r, int(value) & 0xFFFFFFFF))
        return r

    def sreg(self, which: str) -> str:
        r = self.reg()
        self.emit(mir.SregOp(r, which))
        return r

    def _alu(self, cls, a, b, into=None) -> str:
        a, b = self.val(a), self.val(b)
        r = into or self.reg()
        self.emit(cls(r, a, b))
        return r

    def add(self, a, b, into=None):
        return self._alu(mir.AddOp, a, b, into)

    def sub(self, a, b, into=None):
        return self._alu(mir.SubOp, a, b, into)

    def mul(self, a, b, into=None):
        return self._alu(mir.MulOp, a, b, into)

    def div(self, a, b, into=None):
        return self._alu(mir.DivOp, a, b, into)

    def sra(self, a, b, into=None):
        return self._alu(mir.SraOp, a, b, into)

    def max(self, a, b, into=None):
        return self._alu(mir.MaxOp, a, b, into)

    def ldr(self, addr) -> str:
        r = self.reg()
        self.emit(mir.LdrOp(r, self.val(addr)))
        return r

    def str_(self, addr, value):
        self.emit(mir.StrOp(self.val(addr), self.val(value)))

    def label(self, sym):
        self.emit(mir.LabelOp(sym))

    def global_tid(self) -> str:
        """blockIdx * blockDim + threadIdx."""
        return self.add(self.mul("R13", "R14"), "R15")

    # ----------------------------------------------------------------------------------------- idioms
    def neg(self, x):
        return self.sub(0, x)

    def min(self, a, b):
        a, b = self.val(a), self.val(b)
        return self.sub(self.add(a, b), self.max(a, b))

    def clamp(self, x, lo, hi):
        return self.min(self.max(x, lo), hi)

    def clamp_i8(self, x):
        return self.clamp(x, -128, 127)

    def abs(self, x):
        x = self.val(x)
        return self.max(x, self.neg(x))

    def nz(self, d):
        """1 if the word d != 0 else 0."""
        d = self.val(d)
        return self.div(d, d)

    def ne(self, a, b):
        return self.nz(self.sub(a, b))

    def eq(self, a, b):
        return self.sub(1, self.ne(a, b))

    def lt(self, a, b):
        a = self.val(a)
        return self.nz(self.sub(self.max(a, b), a))

    def gt(self, a, b):
        return self.lt(b, a)

    def ge(self, a, b):
        return self.sub(1, self.lt(a, b))

    def le(self, a, b):
        return self.sub(1, self.lt(b, a))

    def select(self, c, x, y):
        y = self.val(y)
        return self.add(y, self.mul(c, self.sub(x, y)))

    def mask(self, c):
        return self.neg(c)

    def mul_pow2(self, x, k: int):
        if not 0 <= k <= 31:
            raise ValueError(f"shift {k} outside 0..31")
        return x if k == 0 and isinstance(x, str) else self.mul(x, 1 << k)

    def requant(self, acc, m, s, rnd=None):
        """clamp_i8((acc * m + rnd) >>> s). With s an int, rnd defaults to 2^(s-1) (0 for s = 0); with s a
        register, pass rnd (a register or int) as the xf kernels do."""
        if rnd is None:
            if not isinstance(s, int):
                raise ValueError("requant with a register shift needs rnd")
            rnd = (1 << (s - 1)) if s else 0
        x = self.mul(acc, m)
        if not (isinstance(rnd, int) and rnd == 0):
            x = self.add(x, rnd)
        if not (isinstance(s, int) and s == 0):
            x = self.sra(x, s)
        return self.clamp_i8(x)

    def addr(self, base, *terms):
        """base + sum(idx * stride) for (idx, stride) terms; int strides of 1 add directly, powers of two and
        other constants (any width) multiply by a const32."""
        acc = self.val(base)
        for idx, stride in terms:
            t = idx if (isinstance(stride, int) and stride == 1) else self.mul(idx, stride)
            acc = self.add(acc, t)
        return acc
