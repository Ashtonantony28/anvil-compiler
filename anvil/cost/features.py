"""Tier-2 features: the compiled program, its dynamic profile, the run configuration and the tier-1 predictions.

    pf = program_features(prof, stats)               # config-independent (per program and WARPS_PER_CORE class)
    cf = config_features(params, mem_latency)
    x  = vector(pf, cf, t1)                          # numpy row in FEATURES order (t1: tier1_features(...))
    x  = featurize(job, stats=built.features())      # all of the above from a Job

Every feature is a deterministic function of its inputs (same program, same config -> same vector, bit for bit).

Program features (dynamic counts are per launch summed, from `anvil.cost.profile`):
  instruction mix (warp instructions per opcode class), lane memory ops, distinct addresses per warp global
  instruction, the share of LDRs whose lanes all read one word, smem ops, BARs, blocks, issues per block (max/mean),
  trip counts (dynamic / static instructions, taken backward branches), loop body spans (for the I-cache), the global
  footprint and reuse, and the backend's static stats (words, registers, spills, smem words, launches, threads).
Config features: every GpuParams field that changes timing, mem_latency, and interactions (blocks per core, waves,
  code size and loop span vs ICACHE_LINES, footprint vs DCACHE_LINES).
Tier-1 features: the closed-form estimate, the event model's cycles and its breakdown (fetch cycles, I/L1-D hits,
  channel wait, yields, idle stalls).
"""
from __future__ import annotations

import math

import numpy as np

from .profile import (ADD, BAR, BR, CMP, CONST, DIV, LDR, LDS, MAX, MUL, NOP, RET, SRA, STR, STS, SUB)

CLASSES = {"alu": (ADD, SUB, MUL, DIV, SRA, MAX), "mul": (MUL, DIV), "const": (CONST,), "cmp": (CMP,), "br": (BR,),
           "ldr": (LDR,), "str": (STR,), "lds": (LDS,), "sts": (STS,), "bar": (BAR,), "ret": (RET,), "nop": (NOP,)}
STATIC = ("words", "threads", "launches", "smem_words", "regs_used", "spill_loads", "spill_stores", "const_values",
          "remat_instrs", "frame_words")

PROGRAM = (
    ["n_" + c for c in CLASSES] +
    ["issues", "blocks", "max_blocks_per_launch", "max_block_issues", "mean_block_issues", "max_warp_issues",
     "lane_global_ops", "lanes_per_global", "distinct_per_ldr", "distinct_per_str", "uniform_ldr_frac",
     "footprint", "reuse", "unique_pcs", "static_words", "trip_ratio", "back_branches", "loop_span_mean",
     "loop_span_max", "short_blocks"] +
    ["s_" + s for s in STATIC])
CONFIG = ["NUM_CORES", "ICACHE_LINES", "COALESCE", "PIPELINED", "DCACHE_LINES", "WARPS_PER_CORE", "pol_lrr",
          "pol_gto", "pol_gtlrr", "mem_latency", "waves", "blocks_per_core", "icache_fits_loops", "icache_fits_code",
          "dcache_covers"]
TIER1 = ["t1_cycles", "t1_closed", "t1_log", "t1_closed_log", "t1_fetch_frac", "t1_icache_hit_frac",
         "t1_dcache_hit_frac", "t1_grab_wait", "t1_yields", "t1_idle_stall", "t1_mem_wait"]
FEATURES = PROGRAM + CONFIG + TIER1


def program_features(prof, stats: dict | None = None) -> dict:
    """Config-independent features of a profiled program (anvil.cost.profile.Profile)."""
    cnt = {c: 0 for c in CLASSES}
    by_op = [0] * 16
    issues = blocks = max_bpl = 0
    block_issues, warp_issues = [], []
    lane_global = n_glob = 0
    dist_ldr = dist_str = n_ldr = n_str = uniform = 0
    touched = set()
    pcs = set()
    spans = []
    words = 0
    short = 0
    for li, lp in enumerate(prof.launches):
        words += len(lp.program)
        max_bpl = max(max_bpl, len(lp.blocks))
        for b in lp.blocks:
            blocks += 1
            short += b.threads < prof.tpb
            bi = 0
            for w in b.warps:
                warp_issues.append(len(w))
                bi += len(w)
                for pc, op, addrs in w:
                    by_op[op] += 1
                    pcs.add((li, pc))
                    if addrs is not None:
                        n = len(addrs)
                        d = len({a for _, a in addrs})
                        lane_global += n
                        n_glob += 1
                        for _, a in addrs:
                            touched.add(a)
                        if op == LDR:
                            n_ldr += 1
                            dist_ldr += d
                            uniform += d == 1 and n > 1
                        else:
                            n_str += 1
                            dist_str += d
            block_issues.append(bi)
            issues += bi
    # taken backward branches and their spans: from consecutive pcs of one warp stream (first block only, cheap)
    back = 0
    for lp in prof.launches:
        for b in lp.blocks[:1]:
            for w in b.warps:
                for (pc, op, _), (npc, _, _) in zip(w, w[1:]):
                    if op == BR and npc <= pc:
                        back += 1
                        spans.append(pc - npc + 1)
    nb = max(1, len(prof.launches[0].blocks)) if prof.launches else 1
    for c, ops in CLASSES.items():
        cnt[c] = sum(by_op[o] for o in ops)
    out = {"n_" + c: cnt[c] for c in CLASSES}
    out.update(
        issues=issues, blocks=blocks, max_blocks_per_launch=max_bpl,
        max_block_issues=max(block_issues or [0]), mean_block_issues=float(np.mean(block_issues or [0])),
        max_warp_issues=max(warp_issues or [0]),
        lane_global_ops=lane_global, lanes_per_global=lane_global / max(1, n_glob),
        distinct_per_ldr=dist_ldr / max(1, n_ldr), distinct_per_str=dist_str / max(1, n_str),
        uniform_ldr_frac=uniform / max(1, n_ldr), footprint=len(touched), reuse=lane_global / max(1, len(touched)),
        unique_pcs=len(pcs), static_words=words, trip_ratio=issues / max(1, len(pcs)),
        back_branches=back * nb, loop_span_mean=float(np.mean(spans)) if spans else 0.0,
        loop_span_max=max(spans or [0]), short_blocks=short)
    stats = stats or {}
    for s in STATIC:
        out["s_" + s] = float(stats.get(s, 0) or 0)
    return out


def config_features(params, mem_latency: int, pf: dict | None = None) -> dict:
    p = params
    out = {k: getattr(p, k) for k in ("NUM_CORES", "ICACHE_LINES", "COALESCE", "PIPELINED", "DCACHE_LINES",
                                       "WARPS_PER_CORE")}
    pol = p.WARP_POLICY if p.WARPS_PER_CORE > 1 else -1
    out.update(pol_lrr=int(pol == 0), pol_gto=int(pol == 1), pol_gtlrr=int(pol == 2), mem_latency=int(mem_latency))
    pf = pf or {}
    bpl = pf.get("max_blocks_per_launch", 0)
    out["waves"] = math.ceil(bpl / p.NUM_CORES) if bpl else 0
    out["blocks_per_core"] = pf.get("blocks", 0) / p.NUM_CORES
    span = pf.get("loop_span_mean", 0)
    out["icache_fits_loops"] = (min(1.0, p.ICACHE_LINES / span) if span else 1.0) if p.ICACHE_LINES else 0.0
    out["icache_fits_code"] = min(1.0, p.ICACHE_LINES / max(1, pf.get("s_words", 1))) if p.ICACHE_LINES else 0.0
    out["dcache_covers"] = min(1.0, p.DCACHE_LINES / max(1, pf.get("footprint", 1))) if p.DCACHE_LINES else 0.0
    return out


def tier1_features(t1, closed: float) -> dict:
    b = t1.breakdown
    fetches = max(1, b.get("icache_hits", 0) + b.get("icache_misses", 0))
    lookups = max(1, b.get("dcache_hits", 0) + b.get("dcache_misses", 0))
    return {"t1_cycles": float(t1.cycles), "t1_closed": float(closed), "t1_log": math.log(max(1, t1.cycles)),
            "t1_closed_log": math.log(max(1.0, closed)),
            "t1_fetch_frac": b.get("fetch_cycles", 0) / max(1, t1.cycles),
            "t1_icache_hit_frac": b.get("icache_hits", 0) / fetches,
            "t1_dcache_hit_frac": b.get("dcache_hits", 0) / lookups,
            "t1_grab_wait": float(b.get("grab_wait", 0)), "t1_yields": float(b.get("yields", 0)),
            "t1_idle_stall": float(b.get("idle_stall", 0)), "t1_mem_wait": float(b.get("mem_wait", 0))}


def vector(pf: dict, cf: dict, t1f: dict, names=FEATURES) -> np.ndarray:
    d = {**pf, **cf, **t1f}
    return np.array([float(d[n]) for n in names], dtype=np.float64)


def featurize(job, stats: dict | None = None, prof=None):
    """(feature vector, tier-1 Timing) of a Job at its own params and mem_latency."""
    from .analytic import closed_form, predict_profile
    from .profile import profile
    prof = prof if prof is not None else profile(job)
    pf = program_features(prof, stats)
    cf = config_features(job.params, job.mem_latency, pf)
    t1 = predict_profile(prof, job.params, job.mem_latency)
    return vector(pf, cf, tier1_features(t1, closed_form(prof, job.params, job.mem_latency))), t1
