"""De-allocation: lift an allocated program (e.g. an xf kernel decoded with `anvil.ir.mir.decode`) back to virtual
registers, the form `compile_mir` takes.

Every R0-R12 is split into def-use *webs* (reaching definitions over the CFG; defs that reach a common use are
merged), and each web gets its own virtual register, so the allocator sees the program's real live ranges, not the
hand allocation. With `prologue=True` the first six instructions must be the xf ARGS prologue; they are dropped
and the web of R12 they define becomes the args-pointer read `R12` of the backend input. A web whose use can see
the reset value of a register (read before any write on some path) starts with `const vN, 0`, as the GPU zeroes
registers at reset.
"""
from __future__ import annotations

import random

from ..ir import mir
from .lir import BackendError, Ins, lift, lower, successors

PHYS = [f"R{k}" for k in range(13)]


def _is_prologue(code: list[Ins]) -> bool:
    if len(code) < 6 or any(i.op == "label" for i in code[:6]):
        return False
    c0, c1, m1, m2, c2, a = code[:6]
    x = c1.d
    return (c0.op == "const" and c0.d == "R12" and c1.op == "const" and c1.imm == 16 and x != "R12"
            and (m1.op, m1.d, m1.a, m1.b) == ("mul", "R12", "R12", x) == (m2.op, m2.d, m2.a, m2.b)
            and c2.op == "const" and c2.d == x and (a.op, a.d, a.a, a.b) == ("add", "R12", "R12", x))


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def devirtualize(f, *, prologue: bool = True, seed: int | None = None, name: str | None = None) -> mir.FuncOp:
    """Physical-register MIR -> virtual-register MIR with the same behaviour (given R12 = args_base when
    `prologue`). `seed` shuffles the virtual register numbering (for tests)."""
    code = lift(f)
    if prologue and not _is_prologue(code):
        raise BackendError("the first six instructions are not the ARGS prologue")
    succ = successors(code)
    n = len(code)
    # reaching definitions: def ids are instruction indices, or ("entry", r) for the reset value
    entry = {r: frozenset({("entry", r)}) for r in PHYS}
    rin = [None] * n
    rin[0] = dict(entry) if n else None
    work = [0] if n else []
    while work:
        k = work.pop()
        out = dict(rin[k])
        for r in code[k].writes():
            if r in entry:
                out[r] = frozenset({k})
        for s in succ[k]:
            if rin[s] is None:
                rin[s] = out
                work.append(s)
            else:
                merged = {r: rin[s][r] | out[r] for r in PHYS}
                if merged != rin[s]:
                    rin[s] = merged
                    work.append(s)
    uf = _UF()
    use_web = {}                         # (k, field) -> a def id of the web
    for k, i in enumerate(code):
        if rin[k] is None:
            continue                     # unreachable
        for fld in ("a", "b"):
            r = getattr(i, fld)
            if r in entry:
                defs = sorted(rin[k][r], key=str)
                for d in defs[1:]:
                    uf.union(defs[0], d)
                use_web[k, fld] = defs[0]
        for r in i.writes():
            if r in entry:
                uf.find(k)
    skip = 6 if prologue else 0
    members, uses = {}, {}
    for x in list(uf.p):
        members.setdefault(uf.find(x), []).append(x)
    for (k, _), d in use_web.items():
        uses.setdefault(uf.find(d), []).append(k)
    roots = sorted(members, key=str)
    names = list(range(len(roots)))
    if seed is not None:
        random.Random(seed).shuffle(names)
    web_name, zero_init = {}, []
    for root, num in zip(roots, names):
        ms = members[root]
        insts = [m for m in ms if not isinstance(m, tuple)]
        if prologue and 5 in ms:
            if ms != [5]:
                raise BackendError("R12 is reassigned while the args pointer may still be live")
            web_name[root] = "R12"
            continue
        if prologue and any(m < skip for m in insts):
            if any(k >= skip for k in uses.get(root, [])):
                raise BackendError("a prologue temporary is read after the prologue")
            continue                     # lives and dies inside the prologue
        web_name[root] = f"v{num}"
        if any(isinstance(m, tuple) for m in ms):
            zero_init.append(f"v{num}")
    out = [Ins("const", v, imm=0) for v in sorted(zero_init)]
    for k, i in enumerate(code):
        if k < skip:
            continue
        j = Ins(i.op, i.d, i.a, i.b, i.imm, i.sym)
        if rin[k] is not None:
            for fld in ("a", "b"):
                if (k, fld) in use_web:
                    setattr(j, fld, web_name[uf.find(use_web[k, fld])])
            if j.d in entry:
                j.d = web_name[uf.find(k)]
        elif j.d in entry or j.a in entry or j.b in entry:
            continue                     # unreachable code: drop rather than invent registers
        out.append(j)
    return lower(out, name or f.sym_name.data)
