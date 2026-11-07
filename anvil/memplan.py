"""Static memory planner for a lowered Program's launch chain (SG4b). Replaces `codegen.bump_allocate` in the
compile driver's main path.

    mp = plan_program(prog, cks, params)            # liveness over the kernel order, then a layout
    mp.addresses, mp.footprint, mp.regions, mp.report()

Data memory layout (word addresses, one flat space of 2^DATA_MEM_ADDR_BITS words):

    [consts]       weights, LUTs and inline consts, packed in program order (read-only, live for the whole job)
    [activations]  graph inputs, intermediates, outputs and Kernel.temps: offsets assigned by liveness so a buffer
                   reuses the words of buffers dead before it is defined (greedy best-fit, largest first)
    [args]         one args block per launch, in launch order (anvil.codegen ABI v1)
    [scratch]      the shared sink/spill words: max over kernels of CompiledKernel.scratch_words

Liveness is over kernel indices 0..n-1 of the program. A buffer is live on the closed interval [def, last use]:
inputs and consts are defined at -1 (written into the image before the first launch), outputs (and every value
when `keep="all"`) stay live to n (read back after the last launch), a kernel temp lives at its kernel only, and a
view (reshape) extends the liveness of its source. Two buffers whose intervals intersect never overlap in memory;
the output of kernel i never aliases an input it reads (both are live at i).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


class MemoryPlanError(ValueError):
    """The planned job does not fit the data memory (or the 16-bit args-base immediate)."""


@dataclass(frozen=True)
class Buf:
    sym: str
    size: int                   # words
    start: int                  # first kernel index where live (-1: before the first launch)
    end: int                    # last kernel index where live (n: read after the last launch)
    kind: str = "act"           # "const" | "input" | "act" | "temp" | "output"

    def overlaps(self, o: "Buf") -> bool:
        return self.start <= o.end and o.start <= self.end


@dataclass
class MemPlan:
    addresses: dict                             # sym -> word address (consts, activations, temps)
    bufs: dict                                  # sym -> Buf
    regions: dict                               # name -> (base, words): consts, activations, args, scratch
    args_blocks: list = field(default_factory=list)   # [(kernel name, launch index, base, words)]
    footprint: int = 0                          # first word past the job's data
    mem_words: int = 0                          # 2^DATA_MEM_ADDR_BITS

    @property
    def unshared(self) -> int:
        """Activation words a bump allocator (no reuse) would need."""
        return sum(b.size for b in self.bufs.values() if b.kind != "const")

    def report(self) -> str:
        r = self.regions
        parts = [f"{k} {r[k][1]}" for k in ("consts", "activations", "args", "scratch")]
        return (f"footprint {self.footprint}/{self.mem_words} words ({', '.join(parts)}; activations without reuse "
                f"{self.unshared})")


def assign_offsets(bufs, align: int = 1) -> tuple[dict, int]:
    """Offsets for interval-lifetime buffers such that no two buffers with intersecting lifetimes overlap.
    Greedy: largest first (ties: earliest start, then sym), each at the lowest offset that fits between the
    already placed buffers it conflicts with. Returns ({sym: offset}, arena words)."""
    order = sorted(bufs, key=lambda b: (-b.size, b.start, b.sym))
    placed: list[tuple[int, Buf]] = []
    out, top = {}, 0
    for b in order:
        busy = sorted((o, o + p.size) for o, p in placed if p.overlaps(b) and p.size)
        off = 0
        for lo, hi in busy:
            if off + b.size <= lo:
                break
            off = max(off, -(-hi // align) * align)
        out[b.sym] = off
        placed.append((off, b))
        top = max(top, off + b.size)
    return out, top


def check_disjoint(bufs, addresses) -> list:
    """Pairs of simultaneously-live buffers whose address ranges intersect (empty when the plan is sound)."""
    bad, bl = [], list(bufs)
    for i, a in enumerate(bl):
        for b in bl[i + 1:]:
            if a.size and b.size and a.overlaps(b):
                x, y = addresses[a.sym], addresses[b.sym]
                if x < y + b.size and y < x + a.size:
                    bad.append((a.sym, b.sym))
    return bad


def liveness(program, kernel_buffers: dict | None = None, *, keep: str = "outputs") -> dict:
    """sym -> Buf for every buffer the program's job touches. `kernel_buffers` (kernel name -> {sym: (shape,
    dtype)}, e.g. CompiledKernel.buffers) adds buffers a kernel addresses beyond Program.buffers / Kernel.temps."""
    if keep not in ("outputs", "all"):
        raise ValueError(f"keep must be 'outputs' or 'all', not {keep!r}")
    n = len(program.kernels)
    root = {}

    def resolve(s):
        while s in program.views:
            s = program.views[s][0]
        return s

    sizes = {s: math.prod(sh) for s, (sh, _) in program.buffers.items()}
    consts = set(program.const_values) | {s for s in program.buffers if s not in program.inputs
                                          and not any(s in k.writes for k in program.kernels)}
    first, last, kind = {}, {}, {}
    for s in program.buffers:
        root[s] = resolve(s)
    for s in program.inputs:
        first[s], kind[s] = -1, "input"
    for s in consts:
        if s not in kind:
            first[s], last[s], kind[s] = -1, n, "const"
    for i, k in enumerate(program.kernels):
        for s in k.writes:
            s = resolve(s)
            first.setdefault(s, i)
            last[s] = max(last.get(s, i), i)
            kind.setdefault(s, "act")
        for s in k.reads:
            s = resolve(s)
            first.setdefault(s, i)
            last[s] = max(last.get(s, -1), i)
        temps = dict(k.temps)
        for s, (sh, _) in (kernel_buffers or {}).get(k.name, {}).items():
            if s not in program.buffers and s not in program.views:
                temps.setdefault(s, (sh, None))
        for s, (sh, _) in temps.items():
            if s in program.buffers:
                continue
            sizes[s] = math.prod(sh)
            first.setdefault(s, i)
            last[s] = max(last.get(s, i), i)
            kind.setdefault(s, "temp")
    for s in program.outputs:
        s = resolve(s)
        last[s] = n
        if kind.get(s) == "act":
            kind[s] = "output"
    if keep == "all":
        for s in list(first):
            if kind[s] != "temp":
                last[s] = n
    out = {}
    for s in first:
        out[s] = Buf(s, sizes[s], first[s], max(last.get(s, first[s]), first[s]), kind[s])
    return out


def plan_layout(bufs: dict, launches: list, scratch_words: int, mem_words: int, *, const_order=None) -> MemPlan:
    """Lay out consts, the liveness arena, args blocks and scratch. `launches`: [(kernel name, launch index,
    args words)] in launch order."""
    consts = [b for b in bufs.values() if b.kind == "const"]
    if const_order is not None:
        rank = {s: i for i, s in enumerate(const_order)}
        consts.sort(key=lambda b: rank.get(b.sym, len(rank)))
    addresses, a = {}, 0
    for b in consts:
        addresses[b.sym] = a
        a += b.size
    cbase, cwords = 0, a
    offs, arena = assign_offsets([b for b in bufs.values() if b.kind != "const"])
    for s, o in offs.items():
        addresses[s] = cwords + o
    abase = cwords + arena
    blocks, p = [], abase
    for name, i, words in launches:
        blocks.append((name, i, p, words))
        p += words
    sbase = p
    end = sbase + scratch_words
    plan = MemPlan(addresses, dict(bufs), {"consts": (cbase, cwords), "activations": (cwords, arena),
                                           "args": (abase, sbase - abase), "scratch": (sbase, scratch_words)},
                   blocks, end, mem_words)
    if end > mem_words:
        raise MemoryPlanError(f"job needs {end} words of data memory > {mem_words} (2^DATA_MEM_ADDR_BITS): "
                              + plan.report())
    if blocks and blocks[-1][2] >= 1 << 16:
        raise MemoryPlanError(f"args block at {blocks[-1][2]} is beyond the 16-bit args base")
    bad = check_disjoint(bufs.values(), addresses)
    if bad:                                              # unreachable unless assign_offsets is broken
        raise MemoryPlanError(f"live buffers overlap: {bad[:4]}")
    return plan


def plan_program(program, kernels: dict, params, *, keep: str = "outputs") -> MemPlan:
    """The memory plan of a lowered Program whose kernels are compiled (`kernels`: name -> CompiledKernel)."""
    order = [kernels[k.name] for k in program.kernels]
    bufs = liveness(program, {ck.name: ck.buffers for ck in order}, keep=keep)
    launches = [(ck.name, i, len(ck.slots)) for ck in order for i in range(ck.launches)]
    scratch = max([ck.scratch_words for ck in order] + [0])
    return plan_layout(bufs, launches, scratch, 1 << params.DATA_MEM_ADDR_BITS, const_order=list(program.buffers))
