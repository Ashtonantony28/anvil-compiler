"""Running extended-ISA Jobs: the isasim fork with the cross-block race checker, a tier-1 profile and tier-1 cycles.

    r = run_job(job)        # ExtRun: memory, flags (as anvil.runtime's isasim backend), races, steps per launch,
                            # profile (anvil.cost.profile.Profile), cycles (tier-1 event model, anvil.cost.analytic)
    raw = execute(job, backend="isa_ext:...", timing=True)    # anvil.tune.measure.execute's contract

Timing (model only, the RTL was not changed): the tier-1 event model of anvil.cost.analytic replayed on the fork's
instruction streams. A new instruction is neither LDR/STR nor LDS/STS, so the model charges it what it charges
every ALU op (FETCH, DECODE unless PIPELINED, REQUEST, a 1-cycle WAIT, EXECUTE, UPDATE). Tier 1 is cycle-exact
with Verilator on 43,818 of 43,826 G4 rows (base ISA), and tests/codesign checks the fork's profile gives the
base model's cycles on base programs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..cost.analytic import predict_profile
from ..cost.profile import Block, LaunchProfile, Profile, ProfileError
from ..runtime.api import WARP_POLICIES, Job
from ..runtime.isasim_backend import ISASIM_POLICY, isasim_kwargs
from ..runtime.races import _policy, _Trace
from . import isasim as fork

FLAGS = ("divergent", "stores_conflict", "store_races", "block_races")


@dataclass
class ExtRun:
    memory: list
    flags: dict
    races: list
    steps: list
    profile: Profile | None = None
    cycles: int | None = None
    launch_cycles: list = field(default_factory=list)


def run_job(job: Job, *, timing: bool = True) -> ExtRun:
    p = job.params
    kw = isasim_kwargs(p)
    pol = ISASIM_POLICY[WARP_POLICIES[p.WARP_POLICY]]
    mask = (1 << p.WORD_BITS) - 1
    memory = [int(w) & mask for w in job.data]
    flags = {"divergent": [], "stores_conflict": [], "store_races": [], "race_words": set()}
    races, steps, launches = [], [], []
    real = fork.Result
    mem_ref = [None]

    def traced(mem, *a, **k):
        mem_ref[0] = _Trace(mem)
        return real(mem_ref[0], *a, **k)
    try:
        fork.Result = traced
        for i, ln in enumerate(job.launches):
            kw["warp_policy"] = _policy(pol, p.WARPS_PER_CORE, mem_ref)
            r = fork.run([int(w) for w in ln.program], memory, ln.threads, record=timing, **kw)
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
            steps.append(r.steps)
            if timing:
                lp = LaunchProfile(list(ln.program), ln.threads)
                lp.blocks = [Block(b, n, st) for b, n, st in r.streams]
                launches.append(lp)
    finally:
        fork.Result = real
    out = ExtRun(memory, flags, sorted(races), steps)
    if timing:
        if p.SIMT_STACK:
            raise ProfileError("SIMT_STACK=1 is not modelled by tier 1")
        out.profile = Profile(launches, memory[:len(job.data)], p.THREADS_PER_BLOCK, p.WARPS_PER_CORE)
        t = predict_profile(out.profile, p, job.mem_latency)
        out.cycles, out.launch_cycles = t.cycles, list(t.launch_cycles)
    return out


def execute(job, backend="isa_ext", *, timing=True, scratch=None) -> dict:
    """anvil.tune.measure.execute for extended programs: verdict inputs from the fork, cycles from tier 1."""
    t0 = time.time()
    r = run_job(job, timing=False)
    out = {"memory": r.memory, "flags": {k: len(r.flags.get(k, [])) for k in FLAGS[:3]}, "cycles": None,
           "counters": {}, "launch_cycles": None}
    out["flags"]["block_races"] = len(r.races)
    if timing and not any(out["flags"].values()):
        rt = run_job(job, timing=True)
        out["cycles"], out["launch_cycles"] = rt.cycles, rt.launch_cycles
        out["counters"] = {"warp_issues_total": sum(rt.steps)}
    out["seconds"] = time.time() - t0
    return out
