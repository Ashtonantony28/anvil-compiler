"""The backend's working form of anvil.mir: a flat list of small mutable `Ins` records, plus CFG, liveness and
loop depth over it.

The xDSL ops are immutable-ish data holders; the backend rewrites registers and splices code many times per
compile, so it lifts a `mir.FuncOp` to `list[Ins]` once (`lift`) and lowers the result back (`lower`).

    Ins.op   mir op name without the dialect prefix: add sub mul div sra max ldr lds str sts cmp const const32
             sreg label br sync nop bar ret
    Ins.d    written register (rd), or None
    Ins.a    first read register (rs), or None
    Ins.b    second read register (rt), or None
    Ins.imm  const imm / const32 value / br nzp
    Ins.sym  label name / branch target / sreg which
"""
from __future__ import annotations

from dataclasses import dataclass

from ..ir import mir

ALU = ("add", "sub", "mul", "div", "sra", "max")
PURE = ALU + ("ldr", "lds", "const", "const32", "sreg")    # no side effect besides writing d
_OPS = {c.name.split(".")[-1]: c for c in mir.OPS if c is not mir.FuncOp}


class BackendError(ValueError):
    """A MIR function the backend cannot compile (bad input, program too large, ...)."""


@dataclass
class Ins:
    op: str
    d: str | None = None
    a: str | None = None
    b: str | None = None
    imm: int | None = None
    sym: str | None = None

    def reads(self) -> list[str]:
        return [r for r in (self.a, self.b) if r is not None]

    def writes(self) -> list[str]:
        return [self.d] if self.d is not None else []

    def rename(self, m: dict) -> None:
        if self.a in m:
            self.a = m[self.a]
        if self.b in m:
            self.b = m[self.b]
        if self.d in m:
            self.d = m[self.d]

    def __str__(self):
        regs = ", ".join(r for r in (self.d, self.a, self.b) if r is not None)
        extra = [str(x) for x in (self.imm, self.sym) if x is not None]
        return f"{self.op} {', '.join(filter(None, [regs, *extra]))}".strip()


def get_func(module) -> mir.FuncOp:
    if isinstance(module, mir.FuncOp):
        return module
    fs = [op for op in module.ops if isinstance(op, mir.FuncOp)]
    if len(fs) != 1:
        raise BackendError(f"expected one anvil.mir.func, found {len(fs)}")
    return fs[0]


def lift(f: mir.FuncOp) -> list[Ins]:
    out = []
    for op in f.instrs:
        n = op.name.split(".")[-1]
        g = lambda k: getattr(op, k).data  # noqa: E731
        if n in ALU:
            out.append(Ins(n, g("rd"), g("rs"), g("rt")))
        elif n in ("ldr", "lds"):
            out.append(Ins(n, g("rd"), g("rs")))
        elif n in ("str", "sts", "cmp"):
            out.append(Ins(n, None, g("rs"), g("rt")))
        elif n == "const":
            out.append(Ins(n, g("rd"), imm=op.imm.value.data))
        elif n == "const32":
            out.append(Ins(n, g("rd"), imm=op.value.value.data))
        elif n == "sreg":
            out.append(Ins(n, g("rd"), sym=g("which")))
        elif n == "label":
            out.append(Ins(n, sym=g("sym")))
        elif n == "br":
            out.append(Ins(n, imm=op.nzp.value.data, sym=g("target")))
        elif n == "sync":
            out.append(Ins(n, sym=g("target")))
        elif n in ("nop", "bar", "ret"):
            out.append(Ins(n))
        else:
            raise BackendError(f"unknown MIR op {op.name}")
    return out


def to_op(i: Ins):
    c = _OPS[i.op]
    if i.op in ALU:
        return c(i.d, i.a, i.b)
    if i.op in ("ldr", "lds"):
        return c(i.d, i.a)
    if i.op in ("str", "sts", "cmp"):
        return c(i.a, i.b)
    if i.op in ("const", "const32"):
        return c(i.d, i.imm)
    if i.op == "sreg":
        return c(i.d, i.sym)
    if i.op == "label":
        return c(i.sym)
    if i.op == "br":
        return c(i.imm, i.sym)
    if i.op == "sync":
        return c(i.sym)
    return c()


def lower(code: list[Ins], name: str) -> mir.FuncOp:
    return mir.func(name, [to_op(i) for i in code])


# ------------------------------------------------------------------------------------------------ CFG

def label_index(code: list[Ins]) -> dict[str, int]:
    out = {}
    for k, i in enumerate(code):
        if i.op == "label":
            if i.sym in out:
                raise BackendError(f"duplicate label {i.sym}")
            out[i.sym] = k
    return out


def successors(code: list[Ins]) -> list[list[int]]:
    """Successor indices per entry (labels are entries too and fall through). Every BR may fall through: the
    flags it tests are runtime values, so the CFG keeps both edges (sound for liveness)."""
    lab = label_index(code)
    succ = []
    for k, i in enumerate(code):
        if i.op in ("br", "sync") and i.sym not in lab:
            raise BackendError(f"branch to unknown label {i.sym!r}")
        s = [] if i.op == "ret" or k + 1 == len(code) else [k + 1]
        if i.op == "br" and lab[i.sym] not in s:
            s.append(lab[i.sym])
        succ.append(s)
    return succ


def liveness(code: list[Ins], track=lambda r: True) -> tuple[list[set], list[set]]:
    """(live_in, live_out) per entry, over registers for which `track(name)` holds."""
    succ = successors(code)
    n = len(code)
    use = [{r for r in i.reads() if track(r)} for i in code]
    dfn = [{r for r in i.writes() if track(r)} for i in code]
    lin, lout = [set() for _ in range(n)], [set() for _ in range(n)]
    changed = True
    while changed:
        changed = False
        for k in range(n - 1, -1, -1):
            o = set().union(*(lin[s] for s in succ[k])) if succ[k] else set()
            i = use[k] | (o - dfn[k])
            if o != lout[k] or i != lin[k]:
                lout[k], lin[k], changed = o, i, True
    return lin, lout


def loop_depth(code: list[Ins]) -> list[int]:
    """Nesting depth per entry: each backward branch (target at or before it) opens a loop over [target, branch]."""
    lab = label_index(code)
    depth = [0] * len(code)
    for k, i in enumerate(code):
        if i.op == "br" and lab[i.sym] <= k:
            for j in range(lab[i.sym], k + 1):
                depth[j] += 1
    return depth


def is_virtual(r: str | None) -> bool:
    return r is not None and r.startswith("v")


def max_vreg(code: list[Ins]) -> int:
    m = -1
    for i in code:
        for r in (i.d, i.a, i.b):
            if is_virtual(r):
                m = max(m, int(r[1:]))
    return m


class Fresh:
    """vN name supply above every register already in use."""

    def __init__(self, code: list[Ins]):
        self.n = max_vreg(code) + 1

    def __call__(self) -> str:
        self.n += 1
        return f"v{self.n - 1}"
