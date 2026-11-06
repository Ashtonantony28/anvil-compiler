"""Constant materialization: `const32` (any 32-bit word) on an ISA whose only immediate is CONST's 8 bits.

A value is built from a small DAG of *steps*, each one instruction:

    ("const", imm)      CONST rd, #imm                 0 <= imm <= 255
    ("mul", x, y)       MUL rd, <x>, <y>               e.g. 256 = 16 * 16, h*256 (Horner), a*b with a, b < 256
    ("add", x, y)       ADD rd, <x>, <y>               Horner: v = (v - lo) + lo
    ("sub", x, y)       SUB rd, <x>, <y>               negatives: v = 0 - (2^32 - v); v = (v + 256 - lo) - (256 - lo)

`plan(v)` searches these decompositions and returns the one with the fewest distinct steps; every node of a plan
is itself canonical (plan(x)[x] is the step used for x everywhere), so plans of several values merge without
conflicts and share sub-values (256, 0, small bytes) - this is the CSE.

`canonicalize` turns every single-definition const/const32 virtual register into one canonical register per
value, defined once at function entry (constants are thereby CSE'd and hoisted out of loops). The register
allocator rematerializes a canonical constant at its uses (`emit_tree`) instead of spilling it to memory.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache

from .lir import Fresh, Ins, is_virtual

WORD = (1 << 32) - 1


def word(v: int) -> int:
    return int(v) & WORD


@lru_cache(maxsize=None)
def _plan(v: int) -> tuple[tuple[int, tuple], ...]:
    v = word(v)
    if v < 256:
        return ((v, ("const", v)),)
    cands = []

    def cand(step, *children):
        nodes = {}
        for c in children:
            nodes.update(_plan(c))
        nodes[v] = step
        cands.append(nodes)

    if v < 65536 or v & (v - 1) == 0:                           # products of two smaller values
        if v & (v - 1) == 0:
            k = v.bit_length() - 1
            for j in range(1, k // 2 + 1):                       # every split, so factors can share nodes
                cand(("mul", 1 << j, 1 << (k - j)), 1 << j, 1 << (k - j))   # (2^21 = 128 * 128 * 128)
        for a in range(2, 256):
            if v % a == 0 and a <= v // a < 256:
                cand(("mul", a, v // a), a, v // a)
                break
    hi, lo = v >> 8, v & 0xFF
    if lo == 0:
        if hi > 1:
            cand(("mul", hi, 256), hi, 256)
    else:
        cand(("add", v - lo, lo), v - lo, lo)                  # Horner: (hi * 256) + lo
        up = v + 256 - lo
        if up < 1 << 31:
            cand(("sub", up, 256 - lo), up, 256 - lo)           # (hi + 1) * 256 - (256 - lo)
    if v > 1 << 31:
        cand(("sub", 0, (1 << 32) - v), 0, (1 << 32) - v)      # 0 - |v|
    best = min(cands, key=lambda n: (len(n), sorted(n)))
    return tuple(best.items())


def plan(v: int) -> dict[int, tuple]:
    """value -> step for every node needed to build word(v), children before parents."""
    return dict(_plan(word(v)))


def cost(v: int) -> int:
    """Instructions to materialize v from nothing."""
    return len(_plan(word(v)))


def order(nodes: dict[int, tuple], roots=None) -> list[int]:
    """Topological order (children first) of a merged plan; the last of `roots` (default: every node) is last."""
    out, seen = [], set()

    def visit(x):
        if x in seen:
            return
        seen.add(x)
        for c in nodes[x][1:] if nodes[x][0] != "const" else ():
            visit(c)
        out.append(x)

    for x in sorted(nodes) if roots is None else roots:
        visit(x)
    return out


def step_ins(step: tuple, d: str, reg_of) -> Ins:
    if step[0] == "const":
        return Ins("const", d, imm=step[1])
    return Ins(step[0], d, reg_of(step[1]), reg_of(step[2]))


def emit_tree(v: int, fresh: Fresh) -> tuple[list[Ins], str, list[str]]:
    """Straight-line code building word(v) in fresh registers: (code, result register, all registers used)."""
    nodes = plan(v)
    reg = {}
    code = []
    for x in order(nodes, [word(v)]):
        reg[x] = fresh()
        code.append(step_ins(nodes[x], reg[x], reg.__getitem__))
    return code, reg[word(v)], list(reg.values())


def canonicalize(code: list[Ins], fresh: Fresh, keep=frozenset()) -> tuple[list[Ins], dict[str, int]]:
    """CSE + hoisting of constants. Returns (new code, {canonical register: value}).

    Virtual registers with exactly one definition, a `const` or `const32`, are replaced by one register per
    value; the canonical registers (and every sub-value their plans need) are defined at entry, children first.
    Registers in `keep` are left alone."""
    defs = Counter(r for i in code for r in i.writes())
    value = {}
    for i in code:
        if i.op in ("const", "const32") and is_virtual(i.d) and defs[i.d] == 1 and i.d not in keep:
            value[i.d] = word(i.imm)
    used = {value[r] for i in code for r in i.reads() if r in value}
    nodes = {}
    for v in sorted(used):
        nodes.update(plan(v))
    canon = {v: fresh() for v in order(nodes)}
    rename = {r: canon[v] for r, v in value.items() if v in canon}
    body = []
    for i in code:
        if i.d in value:
            continue
        i.rename(rename)
        if i.op == "const32":            # a constant into a register that is also written elsewhere
            tree, root, _ = emit_tree(i.imm, fresh)
            tree[-1].d = i.d
            body += tree
            continue
        body.append(i)
    entry = [step_ins(nodes[v], canon[v], canon.__getitem__) for v in order(nodes)]
    return entry + body, {canon[v]: v for v in canon}
