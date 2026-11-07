"""G3 report (SG6): tuned schedules vs the driver's default/tiled presets vs the hand-written xf kernels.

    PYTHONPATH=sim python3 -m anvil.tune.g3_report [--budget N] [--seed S] [--configs default|all]
        rewrites docs/g3-report.md and anvil/tune/best/xf-bench.json (the committed best schedules)

Workloads: the mlp_block_0 and attention_0 fixtures as graph IR (anvil.frontend.from_image; the torch frontend
produces the same kernel IR, tests/tune checks it), and a 1x8x8 qlinear (the xf-bench matmul case). Every kernel
is tuned from scratch (an empty best-schedule store, the measurement cache may be warm) with one seed and budget,
at each of the four xf-bench configurations. Every chain runs on Verilator golden-equal to the fixture on every
segment (keep-all plan) and on the output (reuse plan); hand chains are `xf.mlp.launches` /
`xf.attention.launches`, golden-checked the same way (anvil.g1_report.hand_chain).
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ["", "COALESCE=1", "DCACHE_LINES=16", "COALESCE=1 DCACHE_LINES=16"]
BLOCKS = ("mlp_block", "attention")
BEST = Path(__file__).with_name("best") / "xf-bench.json"
BUDGET, SEED = 200, 0
HAND_TARGET, STRETCH = 40329, 30247


def config_params(cfg: str):
    from ..runtime import GpuParams
    return GpuParams(**{k: int(v) for k, v in (kv.split("=") for kv in cfg.split())})


def fixture_model(block, i=0):
    """((module, consts), fixture dict) of an xf fixture as graph IR."""
    from xf import fixture

    from ..frontend import from_image
    d = fixture.load(fixture.FIXTURE_DIR / f"{block}_{i}.json")
    return from_image.BLOCKS[block](d["image"]), d


def mm_model(M=1, N=8, K=8, seed=7):
    """A qlinear y = requant(x @ w) of shape MxNxK (the xf-bench '1x8x8' matmul case), its inputs and goldens."""
    from ..interp import graph as gi
    from ..ir.graph import GraphBuilder
    rng = np.random.default_rng(seed)
    g = GraphBuilder(f"mm{M}x{N}x{K}")
    y = g.qlinear("y", g.input("x", [M, K]), g.const("w", [K, N]), 17915, 22)
    g.output(y)
    m = g.module()
    consts = {"w": rng.integers(-128, 128, (K, N))}
    x = rng.integers(-128, 128, (M, K))
    golden = gi.run(m, {"x": x}, consts).values
    return (m, consts), {"inputs": {"x": x}, "golden": {"y": golden["y"]}}


def hand_matmul(case="1x8x8", cfg=""):
    """{kind: cycles} of the hand matmul kernels (sim/bench/xf-matmul.json, reproduced exactly by SG0)."""
    rows = json.loads((ROOT / "sim" / "bench" / "xf-matmul.json").read_text())["rows"]
    return {r["kind"]: r["cycles"] for r in rows if r["case"] == case and r["params"] == cfg}


def run_chain(cm, d, backend="verilator"):
    """{kernel: cycles}, golden-checked on every segment (keep-all) and the outputs (reuse plan)."""
    r = cm.run(d["inputs"]["x"], backend, quantized=True, keep="all")
    for seg, want in d["golden"].items():
        if seg in r.ints and not np.array_equal(r.ints[seg], want):
            raise AssertionError(f"{cm.program.name}: {seg} differs from the golden on {backend}")
    r2 = cm.run(d["inputs"]["x"], backend, quantized=True)
    for s in cm.outputs:
        if not np.array_equal(r2.ints[s], d["golden"][s]):
            raise AssertionError(f"{cm.program.name}: output {s} differs from the golden (reuse plan)")
    if r2.cycles != r.cycles:
        raise AssertionError(f"{cm.program.name}: cycles depend on the memory plan")
    return r.cycles


def tuned_model(model, params, *, budget=BUDGET, seed=SEED, db=None, search=True, workers=4, cache=None):
    from ..driver import compile
    from .policy import TunedPolicy
    pol = TunedPolicy(params, budget=budget, seed=seed, workers=workers, search=search, db=db, cache=cache)
    return compile(model, params=params, schedules=pol)


def collect(configs=CONFIGS, budget=BUDGET, seed=SEED, workers=4, cache=None):
    from ..driver import compile
    from ..g1_report import hand_chain
    from .db import BestDB
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="anvil-g3-")
    db = BestDB(committed=False, write=Path(tmp) / "best.json")
    rows = []
    workloads = [(b, *fixture_model(b)) for b in BLOCKS] + [("mm1x8x8", *mm_model())]
    for cfg in configs:
        p = config_params(cfg)
        for name, model, d in workloads:
            t1 = time.time()
            cm_t = tuned_model(model, p, budget=budget, seed=seed, db=db, workers=workers, cache=cache)
            tune_s = time.time() - t1
            row = {"config": cfg or "default", "workload": name, "tuned": run_chain(cm_t, d),
                   "default": run_chain(compile(model, params=p, schedules="default"), d),
                   "tiled": run_chain(compile(model, params=p, schedules="tiled"), d),
                   "tuning": {}, "tune_seconds": tune_s, "words": {}}
            for k, info in cm_t.tuning.items():
                res = info.get("result")
                e = info.get("entry") or {}
                row["tuning"][k] = {"candidate": e.get("candidate"), "measured": e.get("measured"),
                                    "timed": e.get("timed"), "invalid": res.invalid if res else None,
                                    "seconds": e.get("seconds"), "isolated": e.get("cycles"), "how": info["how"]}
                row["words"][k] = cm_t.kernels[k].program.stats["words"]
            if name in BLOCKS:
                row["hand_tiled"] = hand_chain(name, "tiled", p)
                row["hand_naive"] = hand_chain(name, "naive", p)
            else:
                row["hand_tiled"] = {"y": hand_matmul(cfg=cfg)["tiled"]}
                row["hand_naive"] = {"y": hand_matmul(cfg=cfg)["naive"]}
            rows.append(row)
    return rows, db, time.time() - t0


def _desc(c):
    if not c:
        return "default"
    p = c["params"]
    s = c["sketch"]
    if s == "mm":
        st = "" if p["stage"] == "none" else f", smem {p['stage'].upper()} kc={p['kc']}"
        return f"mm {p['ti']}x{p['tj']} reg tile, {p['order']}, {p['lanes']} lanes, k unroll {p['ku']}{st}"
    if s == "ew":
        return f"ew {p['v']}/thread {p['mode']}, {p['lanes']} lanes, {p['tail']}"
    if s == "rows":
        return f"rows {p['mode']}, r={p['r']}, {p['lanes']} lanes, {p['tail']}"
    return f"tpl {p['template']}"


def _knobs(c):
    if not c:
        return ""
    p = c["params"]
    return f"U={p.get('U')} ca={p.get('ca')}"


def render(rows, wall, budget, seed, fresh=False) -> str:
    out = [f"""# G3 report: autotuned schedules

Generated by `PYTHONPATH=sim python3 -m anvil.tune.g3_report` (Verilator, cycle-exact with the iverilog RTL per
ADR-001); do not edit. It also writes `anvil/tune/best/xf-bench.json`, the best schedules that
`anvil.compile(..., schedules="tuned")` replays without searching.

Every kernel is tuned from an empty best-schedule store{' and an empty measurement cache' if fresh else ''} with seed {seed} and a budget of {budget} measured
candidates (`anvil.tune.tune_kernel`: the driver's templates and random sketch points, then mutation/crossover;
pruned candidates do not count). Each candidate is bit-exact on isasim with no divergence / race / store-conflict
flags before its Verilator cycles count. Every chain below runs on Verilator golden-equal to the fixture on every
segment. `default` / `tiled` are the driver's presets; hand chains are the xf kernels (`matmul_naive.asm` /
`matmul_tiled.asm` chains). `THREADS_PER_BLOCK=4`, `NUM_CORES=2`, `WARPS_PER_CORE=1`, `WORD_BITS=32`.

Knobs: `U` unrolls innermost sequential loops up to that factor, `ca` is codegen `const_args` (None: constants
built inline; c: constants of >= c instructions read from the args block).

Total wall time {wall / 60:.1f} min on 4 workers (tuning, every chain and the hand chains on Verilator).
"""]
    out.append("## Chain totals (cycles)\n")
    out.append("| workload | config | hand naive | hand tiled | default | tiled | **tuned** | tuned / hand tiled | "
               "candidates measured | tuning wall s |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        hn = sum(r["hand_naive"].values()) if r["hand_naive"] else None
        ht = sum(r["hand_tiled"].values()) if r["hand_tiled"] else None
        t = sum(r["tuned"].values())
        meas = sum((v["measured"] or 0) for v in r["tuning"].values())
        out.append(f"| {r['workload']} | {r['config']} | {hn or '-'} | {ht or '-'} | {sum(r['default'].values())} | "
                   f"{sum(r['tiled'].values())} | **{t}** | {f'{t / ht:.2f}' if ht else '-'} | {meas} | "
                   f"{r['tune_seconds']:.0f} |")
    for r in rows:
        if r["config"] != "default":
            continue
        out.append(f"\n## {r['workload']}, default configuration: per kernel\n")
        out.append("| kernel | hand naive | hand tiled | default | tiled | **tuned** | tuned schedule | knobs | words | "
                   "measured (timed) | pruned |")
        out.append("|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|")
        for k, t in r["tuned"].items():
            info = r["tuning"][k]
            hn = (r["hand_naive"] or {}).get(k, "-")
            ht = (r["hand_tiled"] or {}).get(k, "-")
            out.append(f"| {k} | {hn} | {ht} | {r['default'][k]} | {r['tiled'][k]} | **{t}** | "
                       f"{_desc(info['candidate'])} | {_knobs(info['candidate'])} | {r['words'][k]} | "
                       f"{info['measured']} ({info['timed']}) | {info['invalid'] if info['invalid'] is not None else '-'} |")
    mlp = next(r for r in rows if r["workload"] == "mlp_block" and r["config"] == "default")
    t = sum(mlp["tuned"].values())
    out.append(f"""
## G3 acceptance (default configuration)

* Tuned MLP block chain: **{t}** cycles vs the hand tiled chain's {HAND_TARGET} (target < {HAND_TARGET}:
  {'met' if t < HAND_TARGET else 'NOT met'}; stretch <= {STRETCH}: {'met' if t <= STRETCH else 'NOT met'}, {t / HAND_TARGET:.2f}x).
* The 1x8x8 matmul: see its table: the tuner's choice and the hand naive (1,324) / tiled (3,848) kernels.
* Reproducibility, RTL re-verification of the whole tuned chain and the naive-style choice are tested in
  `tests/tune` (`make check`, and `-m "rtl or torch"`).
""")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--configs", choices=["default", "all"], default="all")
    ap.add_argument("--out", default=str(ROOT / "docs" / "g3-report.md"))
    ap.add_argument("--best", default=str(BEST))
    ap.add_argument("--fresh-cache", action="store_true", help="measure into an empty cache (honest wall times)")
    a = ap.parse_args(argv)
    from .db import save_entries
    configs = CONFIGS if a.configs == "all" else [""]
    from .measure import MeasureCache
    cache = MeasureCache(tempfile.mkdtemp(prefix="anvil-g3-measure-")) if a.fresh_cache else None
    rows, db, wall = collect(configs, a.budget, a.seed, cache=cache)
    save_entries(Path(a.best), db.entries, note=f"anvil.tune.g3_report seed={a.seed} budget={a.budget}: best schedules "
                 "of the xf-bench workloads (mlp_block_0, attention_0, 1x8x8) at the four xf-bench configs")
    text = render(rows, wall, a.budget, a.seed, a.fresh_cache)
    Path(a.out).write_text(text)
    print(text)
    return rows


if __name__ == "__main__":
    main()
