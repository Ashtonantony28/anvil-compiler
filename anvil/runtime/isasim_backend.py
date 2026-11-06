"""isasim backend: `xf.isasim.run` once per launch, each starting from the previous launch's memory."""
from __future__ import annotations

from .api import WARP_POLICIES, GpuParams, Job, LaunchResult, RunResult

# xf.isasim implements "lrr" and "gto". GTLRR (WARP_POLICY=2) is greedy like GTO between BARs; the memory of a
# race-free program does not depend on issue order, and races are flagged either way.
ISASIM_POLICY = {"lrr": "lrr", "gto": "gto", "gtlrr": "gto"}


def isasim_kwargs(p: GpuParams) -> dict:
    """xf.isasim.run keywords for a GPU configuration (memory-system parameters do not change ISA results)."""
    if p.WORD_BITS != p.DATA_MEM_DATA_BITS:
        raise ValueError(f"isasim needs WORD_BITS == DATA_MEM_DATA_BITS, got {p.WORD_BITS} and {p.DATA_MEM_DATA_BITS}")
    return {"word_bits": p.WORD_BITS, "addr_bits": p.DATA_MEM_ADDR_BITS, "threads_per_block": p.THREADS_PER_BLOCK,
            "warps_per_core": p.WARPS_PER_CORE, "simt": bool(p.SIMT_STACK),
            "warp_policy": ISASIM_POLICY[WARP_POLICIES[p.WARP_POLICY]]}


def run_isasim(job: Job) -> RunResult:
    from xf import isasim
    kw = isasim_kwargs(job.params)
    mask = (1 << job.params.WORD_BITS) - 1
    memory = [int(w) & mask for w in job.data]
    flags = {"divergent": [], "stores_conflict": [], "store_races": [], "race_words": set()}
    launches = []
    for i, ln in enumerate(job.launches):
        r = isasim.run([int(w) for w in ln.program], memory, ln.threads, **kw)
        memory = r.memory
        # Entries are (launch, block, pc[, addr]); race_words is the union over launches.
        for k in ("divergent", "stores_conflict", "store_races"):
            flags[k] += [(i, *e) for e in getattr(r, k)]
        flags["race_words"] |= r.race_words
        launches.append(LaunchResult(None, {"warp_issues_total": r.steps}))
    return RunResult(list(memory), launches, flags)
