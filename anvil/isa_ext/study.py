"""ISA proposal study (SG8 / G5): each candidate instruction alone and all together, on the MLP block, the attention
head and the SG5 model, against the base ISA through the same pipeline.

    PYTHONPATH=sim python3 -m anvil.isa_ext.study [--force]      # writes data/codesign/isa/<variant>/<workload>.json

Variants: base, MIN, AND, SHL, SLT, MAD, ALL. At the baseline configuration (GpuParams(): 2 cores, no caches; the
model at DATA_MEM_ADDR_BITS=16), mem_latency 0. Two measurements per (variant, workload):

  retuned  (MLP, attention) every kernel re-tuned for the variant: anvil.tune.tune_kernel with measure-all
           ranking, budget RETUNE_BUDGET, seed 0, generation 0 seeded with the sweep's baseline schedules; a candidate is verified on the extended isasim fork
           (bit-exact vs the tensor interpreter, no divergence / race / store-conflict / cross-block race flags) and
           timed by the tier-1 event model on the fork's instruction streams. The base variant runs the identical
           pipeline (no Verilator), so the deltas compare like with like.
  fixed    every workload (the model included: re-tuning 40 kernels x 7 variants is out of budget) compiled with the
           schedules the co-design sweep chose at the baseline point (data/codesign/best.json, Verilator-tuned on
           the base ISA) under each variant: the instruction-selection effect alone.
Each compiled chain runs end to end on the fork and must equal the reference bit for bit (xf goldens on every
segment; the model: the graph interpreter's L1.out) with no flag; cycles are tier 1's for the whole chain, `steps`
the fork's warp-instruction issues, `words` the summed program sizes. Measurement records are cached under
.anvil-cache/codesign/isa_measure/ (anvil.tune.MeasureCache, backend "isa_ext:<variant>").
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import numpy as np

from ..codesign.workloads import ROOT, load
from . import isa as X
from .backend import enabled

OUT = ROOT / "data" / "codesign" / "isa"
VARIANTS = {"base": (), **{m: (m,) for m in X.PROPOSALS}, "ALL": X.PROPOSALS}
RETUNE = ("mlp_block", "attention")
WORKLOADS = ("mlp_block", "attention", "model")
RETUNE_BUDGET, SEED = 24, 0


def cache_root() -> Path:
    from ..runtime.rtl import CACHE
    return Path(os.environ.get("ANVIL_ISA_CACHE", CACHE / "codesign" / "isa_measure"))


@contextlib.contextmanager
def ext_measurement():
    """anvil.tune measures through the fork + tier 1 for tasks whose backend is 'isa_ext:*'."""
    from ..tune import search
    from . import run as R
    real = search.execute

    def execute(job, backend="verilator", **kw):
        if str(backend).startswith("isa_ext"):
            return R.execute(job, backend, **kw)
        return real(job, backend, **kw)
    search.execute = execute
    try:
        yield
    finally:
        search.execute = real


def run_chain(cm, wl):
    """The whole chain on the fork: bit-exact + flag-free, tier-1 cycles."""
    from .run import run_job
    keep = "outputs" if wl.name == "model" else "all"
    jp = cm.job(wl.x, quantized=True, keep=keep)
    r = run_job(jp.job, timing=True)
    flags = {k: len(v) for k, v in r.flags.items()}
    if r.races or any(flags.values()):
        raise AssertionError(f"{wl.name}: flags {flags} races {r.races[:3]}")
    checked = []
    for s in cm.readable(keep):
        if s not in wl.golden:
            continue
        src, shape = (cm.program.views[s] if s in cm.program.views else (s, None))
        v = jp.read(r.memory, src)
        v = v.reshape(shape) if shape is not None else v
        if not np.array_equal(np.asarray(v).reshape(np.shape(wl.golden[s])), wl.golden[s]):
            raise AssertionError(f"{wl.name}: {s} differs from the reference on the isa_ext fork")
        checked.append(s)
    if wl.name == "model" and not checked:
        raise AssertionError("model output not checked")
    per, li = {}, 0
    for k in cm.program.kernels:
        n = cm.kernels[k.name].launches
        per[k.name] = int(sum(r.launch_cycles[li:li + n]))
        li += n
    return {"cycles": int(r.cycles), "steps": int(sum(r.steps)), "kernel_cycles": per,
            "words": {k: cm.kernels[k].program.stats["words"] for k in cm.kernels},
            "isa_ext": {k: cm.kernels[k].program.stats.get("isa_ext") for k in cm.kernels},
            "ops": _op_counts(cm), "checked": sorted(checked)}


def _op_counts(cm):
    from collections import Counter
    c = Counter()
    for ck in cm.kernels.values():
        for w in ck.program.words:
            c[X.BY_OPCODE[w >> 12]] += 1
    return {m: c[m] for m in X.PROPOSALS if c[m]}


def sweep_baseline_entries(workload) -> dict:
    """The sweep's best schedules of this workload's kernels at the baseline point (C2): seeds of every re-tuning
    search (TunedPolicy.transfer), so a re-tuned variant starts from the base ISA's tuned schedules."""
    from ..codesign.sweep import BEST, SWEEP
    from ..tune.db import _load
    rec = json.loads((SWEEP / workload / "C2.json").read_text())
    best = _load(BEST)
    return {k["signature"]: best[k["signature"]] for k in rec["kernels"].values() if k.get("signature") in best}


def retuned(variant, wl, params):
    from ..driver import compile
    from ..tune import BestDB, MeasureCache, TunedPolicy
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"anvil-isa-{os.getpid()}-{variant}-{wl.name}.json"
    db = BestDB(committed=False, write=tmp)
    db.entries = sweep_baseline_entries(wl.name)      # transfer seeds only: their signatures (Verilator) never hit
    pol = TunedPolicy(params, budget=RETUNE_BUDGET, seed=SEED, workers=4, db=db, cache=MeasureCache(cache_root()),
                      backend=f"isa_ext:{variant}", save=False)
    t0 = time.time()
    with enabled(VARIANTS[variant]), ext_measurement():
        cm = compile(wl.model, params=params, schedules=pol)
        out = run_chain(cm, wl)
    out["tune_seconds"] = round(time.time() - t0, 1)
    out["candidates"] = {k: (v.get("entry") or {}).get("candidate") for k, v in cm.tuning.items()}
    out["measured"] = {k: (v["result"].measured if v.get("result") else None) for k, v in cm.tuning.items()}
    return out


def fixed(variant, wl, params):
    from ..codesign.sweep import BEST
    from ..driver import compile
    from ..tune import BestDB, TunedPolicy
    db = BestDB(committed=False, paths=[BEST], write=Path("/nonexistent/never-written.json"))
    pol = TunedPolicy(params, search=False, db=db, save=False)
    with enabled(VARIANTS[variant]):
        cm = compile(wl.model, params=params, schedules=pol)
        hows = {k: v.get("how") for k, v in cm.tuning.items()}
        if any(h != "db" for h in hows.values()):
            raise RuntimeError(f"{wl.name}: no sweep schedule for {[k for k, h in hows.items() if h != 'db']}")
        return run_chain(cm, wl)


def study(variants=tuple(VARIANTS), workloads=WORKLOADS, *, force=False, log=print):
    t_all = time.time()
    for name in workloads:
        wl = load(name)
        params = __import__("anvil.runtime", fromlist=["GpuParams"]).GpuParams(DATA_MEM_ADDR_BITS=wl.addr_bits)
        for v in variants:
            path = OUT / v / f"{name}.json"
            if path.exists() and not force:
                continue
            t0 = time.time()
            rec = {"variant": v, "props": list(VARIANTS[v]), "workload": name, "params": params.as_dict(),
                   "fixed": fixed(v, wl, params)}
            if name in RETUNE:
                rec["retuned"] = retuned(v, wl, params)
            rec["seconds"] = round(time.time() - t0, 1)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
            log(f"[{time.time() - t_all:6.0f}s] {name} {v}: fixed {rec['fixed']['cycles']}"
                + (f", retuned {rec['retuned']['cycles']}" if "retuned" in rec else ""))


def load_results(root=OUT) -> dict:
    out = {}
    for v in VARIANTS:
        for name in WORKLOADS:
            p = Path(root) / v / f"{name}.json"
            if p.exists():
                out[(v, name)] = json.loads(p.read_text())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--workloads", default=",".join(WORKLOADS))
    a = ap.parse_args(argv)
    study(a.variants.split(","), a.workloads.split(","), force=a.force)


if __name__ == "__main__":
    main()
