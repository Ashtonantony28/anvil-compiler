"""Cross-block (hence cross-core) read/write race checker over an isasim address trace (AUDIT-A1 M1, design §3 R6).

    rr = check_races(job)            # isasim run of every launch, recording which block reads/writes each word
    rr.races                         # [(launch, addr, kind, blocks)], kind "write-write" | "read-write"
    rr.result                        # the anvil.runtime.RunResult of that run (memory, flags as run(job, "isasim"))

Blocks of one launch may run concurrently on different cores (and share the L1-D when DCACHE_LINES > 0), so a
global word that one block writes must be neither written nor read by any other block of the same launch: the
compiler's rule is "never read a word in the same launch another core writes". isasim runs blocks one after another
and cannot see such races in its result; this checker sees them in the address trace. Like isasim's store_races, a
word that several blocks write with one and the same value (tail='pad' recomputation) is not a write-write race. Within a block, cross-warp
STR races stay with isasim's own `store_races` flag (BAR-separated phases); shared memory is per block and is not
traced. Launches are sequential (the GPU resets between them), so only accesses within one launch are compared.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .api import WARP_POLICIES, Job, RunResult
from .isasim_backend import ISASIM_POLICY, isasim_kwargs


@dataclass
class RaceReport:
    races: list = field(default_factory=list)       # (launch, addr, "write-write" | "read-write", sorted blocks)
    result: RunResult | None = None

    def __bool__(self) -> bool:
        return bool(self.races)


class _Trace(list):
    """The isasim data memory, recording per word the blocks that read / write it."""

    def __init__(self, data):
        super().__init__(data)
        self.block = 0
        self.reads: dict = {}
        self.writes: dict = {}
        self.values: dict = {}

    def __getitem__(self, a):
        if isinstance(a, int):
            self.reads.setdefault(a, set()).add(self.block)
        return super().__getitem__(a)

    def __setitem__(self, a, v):
        if isinstance(a, int):
            self.writes.setdefault(a, set()).add(self.block)
            self.values.setdefault(a, set()).add(v)
        super().__setitem__(a, v)


def _policy(name: str, wpc: int, mem_ref: list):
    """A callable warp policy reproducing isasim's "lrr" / "gto" choice that also tells the trace the block."""
    st = {"b": None, "cur": 0}

    def pick(b, ready):
        if st["b"] != b:
            st["b"], st["cur"] = b, 0
        mem_ref[0].block = b
        if name == "lrr":
            c = next(i % wpc for i in range(st["cur"], st["cur"] + wpc) if i % wpc in ready)
            st["cur"] = (c + 1) % wpc
            return c
        c = st["cur"] if st["cur"] in ready else min(ready)
        st["cur"] = c
        return c
    return pick


def check_races(job: Job) -> RaceReport:
    """Run `job` on isasim with an address trace and report cross-block write-write and read-write races."""
    from xf import isasim
    from .api import LaunchResult
    kw = isasim_kwargs(job.params)
    pol = ISASIM_POLICY[WARP_POLICIES[job.params.WARP_POLICY]]
    mask = (1 << job.params.WORD_BITS) - 1
    memory = [int(w) & mask for w in job.data]
    flags = {"divergent": [], "stores_conflict": [], "store_races": [], "race_words": set()}
    launches, races = [], []
    real = isasim.Result
    mem_ref = [None]

    def traced_result(mem, *a, **k):
        mem_ref[0] = _Trace(mem)
        return real(mem_ref[0], *a, **k)
    try:
        isasim.Result = traced_result
        for i, ln in enumerate(job.launches):
            kw["warp_policy"] = _policy(pol, job.params.WARPS_PER_CORE, mem_ref)
            r = isasim.run([int(w) for w in ln.program], memory, ln.threads, **kw)
            tr = mem_ref[0]
            for a, ws in tr.writes.items():
                if len(ws) > 1 and len(tr.values[a]) > 1:
                    races.append((i, a, "write-write", sorted(ws)))
                elif tr.reads.get(a, set()) - ws:
                    races.append((i, a, "read-write", sorted(ws | tr.reads[a])))
            memory = list(r.memory)
            for k in ("divergent", "stores_conflict", "store_races"):
                flags[k] += [(i, *e) for e in getattr(r, k)]
            flags["race_words"] |= r.race_words
            launches.append(LaunchResult(None, {"warp_issues_total": r.steps}))
    finally:
        isasim.Result = real
    return RaceReport(sorted(races), RunResult(memory, launches, flags))
