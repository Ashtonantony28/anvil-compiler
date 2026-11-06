"""Index arithmetic for codegen: affine expressions as linear forms over *atoms* with known ranges.

An atom is a machine value an index depends on: a thread coordinate (block / warp / lane), a counted loop's
iteration register, a buffer base address, or a materialized `floordiv` of a linear form. A `Lin` is
`const + sum(coef * atom)`; every affine map result of a load / store / guard / apply becomes one, so

* trivial maps simplify away: `(x floordiv 1)`, `d mod 4` with d < 4, `(4 * o + i) floordiv 4` with i < 4 and so
  on are decided from atom ranges and divisibility, with no instruction emitted;
* unrolled loop ivs and extent-1 loops are constants, so their terms fold into the constant;
* a guard `L < ub` is decided statically when range(L) is entirely below or above ub;
* each atom records the scope depth where it is defined and whether it varies across the lanes of a warp, which
  drive loop-invariant hoisting and the predication strategy (uniform branch vs mask vs SYNC).

`x mod d` is `x - d * (x floordiv d)`; only `floordiv` creates new atoms (one unsigned DIV, after shifting the
operand to be non-negative when its range dips below 0).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr, AffineDimExpr

_uid = itertools.count()


class CodegenError(ValueError):
    """A canonical kernel the code generator cannot compile (with the reason)."""


@dataclass(eq=False)
class Atom:
    name: str
    lo: int
    hi: int
    depth: int                      # scope index where the value is defined / computed
    varying: bool = False           # differs between the lanes of one warp
    reg: str | None = None          # register holding it (None: materialized lazily, see `make`)
    make: object = None             # callable(emitter) -> register, emitted at `depth`
    key: object = None
    uid: int = field(default_factory=lambda: next(_uid))

    def __repr__(self):
        return f"<{self.name}#{self.uid} [{self.lo},{self.hi}] d{self.depth}{' v' if self.varying else ''}>"


class Lin:
    """const + sum(coef * atom), immutable; terms sorted by atom uid."""
    __slots__ = ("terms", "const")

    def __init__(self, terms=(), const=0):
        acc = {}
        for a, c in terms:
            if c:
                acc[a] = acc.get(a, 0) + c
        self.terms = tuple(sorted(((a, c) for a, c in acc.items() if c), key=lambda t: t[0].uid))
        self.const = int(const)

    @staticmethod
    def of(x) -> "Lin":
        if isinstance(x, Lin):
            return x
        if isinstance(x, Atom):
            return Lin(((x, 1),))
        return Lin((), int(x))

    # -------------------------------------------------------------------------------- arithmetic
    def __add__(self, o):
        o = Lin.of(o)
        return Lin(self.terms + o.terms, self.const + o.const)

    __radd__ = __add__

    def __neg__(self):
        return Lin(((a, -c) for a, c in self.terms), -self.const)

    def __sub__(self, o):
        return self + (-Lin.of(o))

    def scale(self, k: int) -> "Lin":
        return Lin(((a, c * k) for a, c in self.terms), self.const * k)

    # -------------------------------------------------------------------------------- queries
    @property
    def is_const(self) -> bool:
        return not self.terms

    @property
    def key(self):
        return (tuple((a.uid, c) for a, c in self.terms), self.const)

    def __eq__(self, o):
        return isinstance(o, Lin) and self.key == o.key

    def __hash__(self):
        return hash(self.key)

    def range(self) -> tuple[int, int]:
        lo = hi = self.const
        for a, c in self.terms:
            x, y = c * a.lo, c * a.hi
            lo += min(x, y)
            hi += max(x, y)
        return lo, hi

    @property
    def depth(self) -> int:
        return max((a.depth for a, _ in self.terms), default=0)

    @property
    def varying(self) -> bool:
        return any(a.varying for a, _ in self.terms)

    def __repr__(self):
        s = " + ".join(f"{c}*{a!r}" for a, c in self.terms)
        return f"Lin({s} + {self.const})" if s else f"Lin({self.const})"


class Affine:
    """Evaluates xdsl AffineExprs over Lin operands; `div_atom(lin, d)` supplies (memoized) floordiv atoms."""

    def __init__(self, div_atom):
        self.div_atom = div_atom

    def floordiv(self, x: Lin, d: int) -> Lin:
        if d <= 0:
            raise CodegenError(f"floordiv by {d}")
        if d == 1:
            return x
        if x.is_const:
            return Lin.of(x.const // d)
        q = Lin(((a, c // d) for a, c in x.terms if c % d == 0))
        r = Lin(((a, c) for a, c in x.terms if c % d), 0)
        qc, rc = divmod(x.const, d)
        r = r + rc
        lo, hi = r.range()
        if lo // d == hi // d:                           # remainder part stays within one multiple of d
            return q + (qc + lo // d)
        # materialize floor(r / d) with r shifted non-negative
        shift = -(lo // d) if lo < 0 else 0
        a = self.div_atom(r + shift * d, d)
        return q + a + (qc - shift)

    def mod(self, x: Lin, d: int) -> Lin:
        return x - self.floordiv(x, d).scale(d)

    def eval(self, e, dims: list[Lin]) -> Lin:
        if isinstance(e, AffineConstantExpr):
            return Lin.of(e.value)
        if isinstance(e, AffineDimExpr):
            return dims[e.position]
        if isinstance(e, AffineBinaryOpExpr):
            if e.kind == AffineBinaryOpKind.Add:
                return self.eval(e.lhs, dims) + self.eval(e.rhs, dims)
            a, b = self.eval(e.lhs, dims), self.eval(e.rhs, dims)
            if e.kind == AffineBinaryOpKind.Mul:
                if b.is_const:
                    return a.scale(b.const)
                if a.is_const:
                    return b.scale(a.const)
                raise CodegenError(f"non-affine product {e}")
            if not b.is_const:
                raise CodegenError(f"division by a non-constant in {e}")
            if e.kind == AffineBinaryOpKind.FloorDiv:
                return self.floordiv(a, b.const)
            if e.kind == AffineBinaryOpKind.Mod:
                if b.const <= 0:
                    raise CodegenError(f"mod by {b.const}")
                return self.mod(a, b.const)
            if e.kind == AffineBinaryOpKind.CeilDiv:
                return -self.floordiv(-a, b.const)
        raise CodegenError(f"unsupported affine expression {e!r}")
