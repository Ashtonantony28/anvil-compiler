"""Co-design sweep driver: every (workload, configuration) point separately tuned and verified.

    PYTHONPATH=sim python3 -m anvil.codesign.sweep [--workloads mlp_block,attention,model] [--limit N] [--force]

Per point: `anvil.compile(..., schedules=TunedPolicy(...))` with the tier2_static-guided ranker (anvil.codesign.ranker,
top-k = 2 per round, online model updates), Verilator timing, and the per-candidate isasim verdict of anvil.tune
(bit-exact vs the tensor interpreter, no divergence / race / store-conflict / cross-block race flags). Then the
whole tuned chain is run on Verilator and compared bit-exactly with the reference (MLP / attention: the xf fixture
goldens on every segment; model: the graph interpreter's L1.out), and on isasim with the cross-block race checker
(no flag of any kind). The chain's Verilator cycles are the point's cycles.

Caches (all regenerable, resumable: a point whose record exists is skipped unless --force)
    data/codesign/sweep/<workload>/<config id>.json   one record per point (committed: the report reads only these)
    data/codesign/best.json                           best schedule per kernel signature (anvil.tune.BestDB), so a
                                                      point replays without searching (tests/codesign, rtl)
    .anvil-cache/codesign/measure/                    every candidate measurement (anvil.tune.MeasureCache); each
                                                      kernel of a record names its best measurement by `key`
Search budget: BUDGET measured candidates per kernel (MODEL_BUDGET for the model's kernels), seed SEED. The best-
schedule store is shared by every point, so a search is seeded with the best schedules of the same kernel IR at
configurations swept earlier (TunedPolicy.transfer): results depend on the sweep order, which is fixed (design()).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from . import cost as hw
from .space import Config, attention_design, design, model_design
from .workloads import ROOT, WORKLOADS, load

DATA = ROOT / "data" / "codesign"
SWEEP = DATA / "sweep"
BEST = DATA / "best.json"
BUDGET, MODEL_BUDGET, SEED, TOPK = 24, 12, 0, 2


def measure_root() -> Path:
    from ..runtime.rtl import CACHE
    return Path(os.environ.get("ANVIL_CODESIGN_CACHE", CACHE / "codesign" / "measure"))


def configs_for(workload: str) -> list:
    return {"model": model_design, "attention": attention_design}.get(workload, design)()


def point_path(workload: str, cfg: Config, root=SWEEP) -> Path:
    return Path(root) / workload / f"{cfg.id}.json"


def _policy_class():
    from ..tune import TunedPolicy
    from ..tune.space import Candidate

    class SweepPolicy(TunedPolicy):
        """TunedPolicy whose transferred candidates are adapted to this point's sketch domain: the `pred` knob exists
        only at SIMT_STACK=1, and a candidate carried across SIMT_STACK classes without (or with) it breaks the
        tuner's crossover (KeyError 'pred' in anvil.tune.search._cross; an anvil.tune bug, worked around here)."""

        def transfer(self, task):
            simt = bool(getattr(task.params, "SIMT_STACK", 0))
            out = []
            for c in super().transfer(task):
                p = c.p
                if simt:
                    p.setdefault("pred", "auto")
                else:
                    p.pop("pred", None)
                c2 = Candidate.make(c.sketch, **p)
                if c2 not in out:
                    out.append(c2)
            return out
    SweepPolicy.__name__ = "TunedPolicy"      # anvil.driver recognizes a policy by its class name
    return SweepPolicy


def policy(params, mem_latency, *, budget, db, cache, search=True, workers=4):
    from .ranker import StaticRanker
    return _policy_class()(params, budget=budget, seed=SEED, workers=workers, search=search, db=db, cache=cache,
                       ranker=StaticRanker(k=TOPK), mem_latency=mem_latency, save=True)


def check_chain(cm, wl, backend="verilator"):
    """Run the tuned chain: bit-exact vs the reference, race-free on isasim. -> (cycles per kernel, info)."""
    from ..runtime import check_races
    keep = "outputs" if wl.name == "model" else "all"
    r = cm.run(wl.x, backend, quantized=True, keep=keep)
    for sym, want in wl.golden.items():
        if sym in r.ints and not np.array_equal(np.asarray(r.ints[sym]).reshape(np.shape(want)), want):
            raise AssertionError(f"{wl.name}: {sym} differs from the reference on {backend}")
    missing = [s for s in wl.golden if s not in r.ints]
    if wl.name == "model" and missing:
        raise AssertionError(f"model: outputs {missing} not readable")
    rr = check_races(cm.job(wl.x, quantized=True, keep=keep).job)
    flags = {k: len(v) for k, v in rr.result.flags.items()}
    if rr.races or any(flags.values()):
        raise AssertionError(f"{wl.name}: isasim flags {flags}, cross-block races {rr.races[:3]}")
    checked = sorted(s for s in wl.golden if s in r.ints)
    return dict(r.cycles), {"checked": checked, "flags": flags, "block_races": len(rr.races)}


def kernel_records(cm, pol, cache):
    """Per kernel: tuned candidate, words, the isolated measurement and its measurement-cache key."""
    from ..tune import Built, build
    from ..tune.space import Candidate
    out = {}
    for k in cm.program.kernels:
        info = cm.tuning.get(k.name, {})
        e = info.get("entry") or {}
        rec = {"how": info.get("how"), "candidate": e.get("candidate"), "words": cm.kernels[k.name].program.stats["words"],
               "isolated": e.get("cycles"), "signature": info.get("signature"), "key": None}
        if e.get("candidate"):
            b = build(pol.task(k), Candidate.from_dict(e["candidate"]))
            if isinstance(b, Built):
                rec["key"] = b.key
                hit = cache.get(b.key)
                rec["cached_cycles"] = hit.get("cycles") if hit else None
        res = info.get("result")
        if res is not None:
            rec.update(measured=res.measured, timed=res.timed, invalid=res.invalid, tune_seconds=round(res.seconds, 2))
        out[k.name] = rec
    return link_shared(out)


def link_shared(kernels: dict) -> dict:
    """A kernel that reused another kernel's best schedule (`how` == "db": same signature, e.g. L1.kh after L0.kh)
    is compiled at its own buffers, so its own key has no measurement; it names the measurement of the kernel the
    schedule was tuned on (`key_from`)."""
    for name, rec in kernels.items():
        if rec.get("cached_cycles") is None and rec.get("signature"):
            src = next((n for n, r in kernels.items() if n != name and r.get("signature") == rec["signature"]
                        and r.get("cached_cycles") is not None), None)
            if src is not None:
                rec.update(key=kernels[src]["key"], cached_cycles=kernels[src]["cached_cycles"], key_from=src)
    return kernels


def run_point(wl, cfg: Config, *, db, cache, budget, search=True, workers=4) -> dict:
    from ..driver import compile
    p = cfg.params(wl.addr_bits)
    t0 = time.time()
    pol = policy(p, cfg.mem_latency, budget=budget, db=db, cache=cache, search=search, workers=workers)
    cm = compile(wl.model, params=p, schedules=pol)
    t1 = time.time()
    per, info = check_chain_at(cm, wl, cfg)
    t2 = time.time()
    return {"workload": wl.name, "config": cfg.id, "fields": {**cfg.__dict__}, "params": p.as_dict(),
            "mem_latency": cfg.mem_latency, "cost_bits": hw.state_bits(p), "cost_breakdown": hw.breakdown(p),
            "cycles": int(sum(per.values())), "kernel_cycles": per, "verify": info, "budget": budget, "seed": SEED,
            "topk": TOPK, "kernels": kernel_records(cm, pol, cache), "tune_seconds": round(t1 - t0, 1),
            "run_seconds": round(t2 - t1, 1)}


def check_chain_at(cm, wl, cfg):
    """check_chain with the point's memory latency (the driver runs Jobs at mem_latency 0 by default)."""
    from .. import driver
    real = driver.run_job

    def with_latency(job, backend, **kw):
        job.mem_latency = cfg.mem_latency
        return real(job, backend, **kw)
    driver.run_job = with_latency
    try:
        return check_chain(cm, wl)
    finally:
        driver.run_job = real


def sweep(workloads=WORKLOADS, *, limit=None, force=False, root=SWEEP, best=BEST, workers=4, log=print):
    from ..tune import BestDB, MeasureCache
    db = BestDB(committed=False, write=best)
    cache = MeasureCache(measure_root())
    t_all = time.time()
    for name in workloads:
        wl = load(name)
        cfgs = configs_for(name)[:limit]
        budget = MODEL_BUDGET if name == "model" else BUDGET
        for i, cfg in enumerate(cfgs):
            path = point_path(name, cfg, root)
            if path.exists() and not force:
                continue
            rec = run_point(wl, cfg, db=db, cache=cache, budget=budget, workers=workers)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec, indent=1, sort_keys=True, default=int) + "\n")
            tmp.replace(path)
            log(f"[{time.time() - t_all:7.0f}s] {name} {i + 1}/{len(cfgs)} {cfg.id}: {rec['cycles']} cycles, "
                f"{rec['cost_bits']} bits (tune {rec['tune_seconds']}s)")


def load_points(root=SWEEP) -> dict:
    """workload -> [record] in design order."""
    out = {}
    for name in WORKLOADS:
        recs = []
        for cfg in configs_for(name):
            p = point_path(name, cfg, root)
            if p.exists():
                recs.append(json.loads(p.read_text()))
        out[name] = recs
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", default=",".join(WORKLOADS))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    sweep(a.workloads.split(","), limit=a.limit, force=a.force, workers=a.workers)


if __name__ == "__main__":
    main()
