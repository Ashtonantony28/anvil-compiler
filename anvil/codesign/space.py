"""The co-design configuration space (PLAN SG8) and the sampled design the sweep evaluates.

Factors: NUM_CORES 1/2/4, ICACHE_LINES 0/8/16/32, DCACHE_LINES 0/8/16, COALESCE 0/1, PIPELINED 0/1,
(WARPS_PER_CORE, WARP_POLICY) in {(1, -), (2, LRR/GTO/GTLRR), (4, LRR/GTO/GTLRR)}, SIMT_STACK 0/1, mem_latency 0/3:
4,032 points (WARP_POLICY is meaningless at one warp, so it is fixed to 0 there). THREADS_PER_BLOCK stays 4,
WORD_BITS 32. Every point is separately tuned, so the full product is out of reach; `design()` evaluates

  ofat     the baseline (xf build: 2 cores, no caches) and every one-factor change from it, at both latencies;
  combo    the cheap-but-fast region the one-factor results point at: PIPELINED=1, COALESCE=1 with NUM_CORES
           1/2/4 x ICACHE_LINES 16/32 x (1 warp | 2 warps GTO) at latency 0, and NUM_CORES 1/2/4 x ICACHE 32 at 3;
  random   six uniform draws from the full product (seeded), as a check that nothing outside the structured part
           beats it.
`attention_design()` drops the latency-3 one-factor points for the
attention head; `model_design()` is the reduced subset for the 40-kernel SG5 model (see the report).
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

FACTORS = {
    "NUM_CORES": (1, 2, 4),
    "ICACHE_LINES": (0, 8, 16, 32),
    "DCACHE_LINES": (0, 8, 16),
    "COALESCE": (0, 1),
    "PIPELINED": (0, 1),
    "WARPS": ((1, 0), (2, 0), (2, 1), (2, 2), (4, 0), (4, 1), (4, 2)),
    "SIMT_STACK": (0, 1),
    "mem_latency": (0, 3),
}
BASE = {"NUM_CORES": 2, "ICACHE_LINES": 0, "DCACHE_LINES": 0, "COALESCE": 0, "PIPELINED": 0, "WARPS": (1, 0),
        "SIMT_STACK": 0}
POLICY = {0: "LRR", 1: "GTO", 2: "GTLRR"}


@dataclass(frozen=True, order=True)
class Config:
    NUM_CORES: int = 2
    ICACHE_LINES: int = 0
    DCACHE_LINES: int = 0
    COALESCE: int = 0
    PIPELINED: int = 0
    WARPS_PER_CORE: int = 1
    WARP_POLICY: int = 0
    SIMT_STACK: int = 0
    mem_latency: int = 0

    @classmethod
    def make(cls, mem_latency=0, **f):
        w, pol = f.pop("WARPS", (1, 0))
        f.setdefault("WARPS_PER_CORE", w)
        f.setdefault("WARP_POLICY", pol)
        if f["WARPS_PER_CORE"] == 1:
            f["WARP_POLICY"] = 0
        return cls(mem_latency=mem_latency, **f)

    def overrides(self) -> dict:
        """GpuParams fields that differ from the anvil default (GpuParams())."""
        d = {k: getattr(self, k) for k in ("NUM_CORES", "ICACHE_LINES", "DCACHE_LINES", "COALESCE", "PIPELINED",
                                           "WARPS_PER_CORE", "WARP_POLICY", "SIMT_STACK")}
        base = {"NUM_CORES": 2, "WARPS_PER_CORE": 1}
        return {k: v for k, v in d.items() if v != base.get(k, 0)}

    def params(self, addr_bits: int = 12):
        from ..runtime import GpuParams
        return GpuParams(DATA_MEM_ADDR_BITS=addr_bits, **self.overrides())

    @property
    def id(self) -> str:
        """Short stable name, e.g. 'C2' (baseline) or 'C4.I32.D16.CO.PI.W2gto.S.L3'."""
        s = [f"C{self.NUM_CORES}"]
        if self.ICACHE_LINES:
            s.append(f"I{self.ICACHE_LINES}")
        if self.DCACHE_LINES:
            s.append(f"D{self.DCACHE_LINES}")
        if self.COALESCE:
            s.append("CO")
        if self.PIPELINED:
            s.append("PI")
        if self.WARPS_PER_CORE > 1:
            s.append(f"W{self.WARPS_PER_CORE}{POLICY[self.WARP_POLICY].lower()}")
        if self.SIMT_STACK:
            s.append("S")
        if self.mem_latency:
            s.append(f"L{self.mem_latency}")
        return ".".join(s)

    def valid(self) -> bool:
        try:
            self.params()
        except ValueError:
            return False
        return (self.NUM_CORES in FACTORS["NUM_CORES"] and self.ICACHE_LINES in FACTORS["ICACHE_LINES"]
                and self.DCACHE_LINES in FACTORS["DCACHE_LINES"] and self.COALESCE in (0, 1)
                and self.PIPELINED in (0, 1) and (self.WARPS_PER_CORE, self.WARP_POLICY) in FACTORS["WARPS"]
                and self.SIMT_STACK in (0, 1) and self.mem_latency in FACTORS["mem_latency"])


def full_space() -> list:
    keys = [k for k in FACTORS if k != "mem_latency"]
    out = []
    for vals in itertools.product(*(FACTORS[k] for k in keys)):
        for lat in FACTORS["mem_latency"]:
            out.append(Config.make(lat, **dict(zip(keys, vals))))
    return out


def ofat(lat: int) -> list:
    out = [Config.make(lat, **BASE)]
    for k, vals in FACTORS.items():
        if k == "mem_latency":
            continue
        for v in vals:
            if v != BASE[k]:
                out.append(Config.make(lat, **{**BASE, k: v}))
    return out


def combo() -> list:
    out = []
    for c, i, w in itertools.product((1, 2, 4), (16, 32), ((1, 0), (2, 1))):
        out.append(Config.make(0, **{**BASE, "NUM_CORES": c, "ICACHE_LINES": i, "WARPS": w, "COALESCE": 1,
                                     "PIPELINED": 1}))
    for c in (1, 2, 4):
        out.append(Config.make(3, **{**BASE, "NUM_CORES": c, "ICACHE_LINES": 32, "COALESCE": 1, "PIPELINED": 1}))
    return out


def sample(n: int = 6, seed: int = 20261107) -> list:
    space = full_space()
    return random.Random(seed).sample(space, n)


def _dedupe(cs):
    seen, out = set(), []
    for c in cs:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def design() -> list:
    """The sweep's configurations for the MLP block and the attention head (deterministic order)."""
    return _dedupe(ofat(0) + ofat(3) + combo() + sample())


def attention_design() -> list:
    """design() without the latency-3 one-factor points (the baseline at latency 3 and the latency-3 combinations
    stay): the attention head's points take ~2x the MLP block's tuning time, cut to keep the sweep near 2 h."""
    l3 = set(ofat(3)[1:])
    return [c for c in design() if c not in l3]


def model_design() -> list:
    """The reduced subset for the SG5 model (about 3 min of tuning per point): the baseline, the one-factor
    points with the largest effect on the blocks (cores, I-cache 32, pipelining, L1-D 16, two warps GTO), the
    blocks' fast-and-cheap combinations at 2 and 4 cores, and the baseline at latency 3."""
    b = dict(BASE)
    pts = [Config.make(0, **b)]
    for k, v in (("NUM_CORES", 1), ("NUM_CORES", 4), ("ICACHE_LINES", 32), ("PIPELINED", 1), ("DCACHE_LINES", 16),
                 ("WARPS", (2, 1))):
        pts.append(Config.make(0, **{**b, k: v}))
    for c in (2, 4):
        pts.append(Config.make(0, **{**b, "NUM_CORES": c, "ICACHE_LINES": 32, "COALESCE": 1, "PIPELINED": 1}))
    pts.append(Config.make(3, **b))
    return _dedupe(pts)
