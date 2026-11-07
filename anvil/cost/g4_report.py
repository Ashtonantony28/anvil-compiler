"""G4 report (SG7): tier-1 and tier-2 cost models on the frozen held-out split, and model-guided vs random tuning.

    PYTHONPATH=sim python3 -m anvil.cost.g4_report [--no-train] [--online]
        trains the tier-2 models on train kernels x train configs (seeds 0-2, both variants) into
        data/cost/{tier2,tier2_static}-seed{S}.txt.gz, evaluates every tier, writes docs/g4-report.md,
        docs/g4-curves.svg and data/cost/g4-metrics.json. --no-train reuses the committed models.
        --online adds a live tune_kernel run (Verilator) with ActiveRanker vs the default tuner.

Metrics (PLAN SG7):
  Kendall tau between predicted and measured cycles on the frozen held-out split (held-out kernels x held-out
  configs), per config: pooled over all held-out kernels' candidates, and within each kernel (mean / min: the
  ranking a tuner needs). Rows that failed (verification, timeouts) are counted, never dropped silently.
  Tuning: pools = all measured candidates of one held-out kernel at one held-out config; the number of measurements
  until one is within 5 % of the pool's best, model-guided (top-1 per round, online update) vs random search
  (uniform without replacement: 3 seeds, and the exact expectation (N+1)/(m+1)); the G4 ratio is
  sum(model-guided) / sum(random expectation) over pools and seeds.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from . import active as A
from . import dataset as D
from .learned import VARIANTS, CostModel, design, kendall, tau_table

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs" / "g4-report.md"
SVG = ROOT / "docs" / "g4-curves.svg"
METRICS = D.DATA / "g4-metrics.json"
SEEDS = (0, 1, 2)
TAU_TARGET = 0.8
RATIO_TARGET = 0.10
CURVE_N = 40


def model_path(variant, seed):
    return D.DATA / f"{variant}-seed{seed}.txt.gz"


def train(ds, log=print):
    X, y, _ = design(ds, ds.select("train", "train"))
    for variant in VARIANTS:
        for s in SEEDS:
            t0 = time.time()
            CostModel(variant, s).fit(X, y).save(model_path(variant, s))
            log(f"trained {variant} seed {s} on {len(y)} rows in {time.time() - t0:.1f} s")


def load_models():
    return {(v, s): CostModel.load(model_path(v, s)) for v in VARIANTS for s in SEEDS}


def evaluate(ds, models):
    """tau tables per split slice and tier."""
    slices = {"heldout": ("test", "test"), "heldout_kernels": ("test", "train"),
              "heldout_configs": ("train", "test"), "train": ("train", "train")}
    out = {}
    for name, (k, c) in slices.items():
        rows = ds.select(k, c)
        X, y, meta = design(ds, rows)
        yt = np.exp(y)
        yi = np.array([r["cycles"] for r in rows], dtype=np.int64)
        preds = {"tier1_closed": np.array([r["closed"] for r in rows], float),
                 "tier1": np.array([r["t1"] for r in rows], float)}
        for v in VARIANTS:
            preds[v] = models[(v, 0)].predict(X)
        res = {}
        for tier, p in preds.items():
            res[tier] = {"tau": tau_table(meta, yt, p), "mape": float(np.mean(np.abs(p - yt) / yt)),
                         "exact": float(np.mean(np.round(p).astype(np.int64) == yi)),
                         "max_abs_err": float(np.max(np.abs(p - yi)))}
        out[name] = {"rows": len(rows), "tiers": res}
    return out


def failures(ds):
    """Per config: rows that could not be measured, and programs that failed verification."""
    per = {}
    for r in ds.rows:
        d = per.setdefault(r["cid"], {"rows": 0, "failed": 0, "errors": {}})
        d["rows"] += 1
        if r["cycles"] is None:
            d["failed"] += 1
            d["errors"][r["err"][:60]] = d["errors"].get(r["err"][:60], 0) + 1
    bad_prog = [p for p in ds.programs.values() if not p["ok"]]
    return per, bad_prog


def tuning(ds, models, budget=CURVE_N):
    pools = A.pools(ds, "test", "test")
    res = {"pools": [], "curves": {}}
    curves = {}
    for (kid, cid), rows in pools.items():
        X, y, _ = design(ds, rows)
        cyc = [r["cycles"] for r in rows]
        best = min(cyc)
        n = len(cyc)
        e = {"kid": kid, "cid": cid, "n": n, "best": best,
             "good": sum(c <= best * A.TARGET for c in cyc), "random_expected": A.expected_random(cyc)}
        runs = {"random": [A.random_curve(cyc, s) for s in SEEDS],
                "tier1": [A.guided_curve(None, X, cyc, budget, tier1=True)],
                "tier2": [A.guided_curve(models[("tier2", s)], X, cyc, budget) for s in SEEDS],
                "tier2_static": [A.guided_curve(models[("tier2_static", s)], X, cyc, budget) for s in SEEDS]}
        for m, cs in runs.items():
            e[m] = [A.to_target(c, best) for c in cs]
            curves.setdefault(m, []).extend(A.best_so_far(c, best, budget) for c in cs)
        res["pools"].append(e)
    res["curves"] = {m: np.mean(np.array(cs), axis=0).tolist() for m, cs in curves.items()}
    res["within5"] = {}
    for m in ("random", "tier1", "tier2", "tier2_static"):
        frac = []
        for i in range(budget):
            vals = [v <= A.TARGET for v in np.array([c for c in curves[m]])[:, i]]
            frac.append(float(np.mean(vals)))
        res["within5"][m] = frac
    # summary
    ps = res["pools"]
    rnd = sum(p["random_expected"] for p in ps)
    rnd_seeds = sum(np.mean([x if x is not None else budget + 1 for x in p["random"]]) for p in ps)
    summ = {"pools": len(ps), "random_expected_total": rnd, "random_seeds_total": float(rnd_seeds)}
    for m in ("tier1", "tier2", "tier2_static"):
        tot = sum(np.mean([x if x is not None else budget + 1 for x in p[m]]) for p in ps)
        miss = sum(x is None for p in ps for x in p[m])
        summ[m] = {"total": float(tot), "ratio": float(tot / rnd), "ratio_vs_seeds": float(tot / rnd_seeds),
                   "not_reached": miss,
                   "per_seed_ratio": [float(sum((p[m][s] if p[m][s] is not None else budget + 1)
                                                for p in ps) / rnd) for s in range(len(ps[0][m]))]}
    res["summary"] = summ
    return res


def online(ds, budget=12, k=2, log=print):
    """Live tune_kernel on held-out kernels at a held-out config: ActiveRanker(k) vs the default tuner (measure all),
    both with a fresh measurement cache; the reference best is the pool best from the dataset."""
    import tempfile

    from ..tune import MeasureCache, Task, tune_kernel
    from .active import ActiveRanker
    specs = {s.kid: s for s in D.suite(("xf", "syn"))}
    out = []
    for kid, cid in (("xf_attention.q", "h0"), ("xf_attention.p", "h1"), ("syn.qlinear_8x4x12", "h0"),
                     ("syn.softmax_12x12", "h1")):
        spec = specs[kid]
        params, lat = ds.params(cid)
        task = Task.from_program(spec.program, spec.name, spec.values, params, mem_latency=lat)
        pool = [r["cycles"] for r in ds.select("test", "test") if ds.programs[r["pid"]]["kid"] == kid and r["cid"] == cid]
        e = {"kid": kid, "cid": cid, "pool_best": min(pool) if pool else None}
        for name, ranker, b, gen in (("active", ActiveRanker(k=k), budget, 64), ("default", None, 200, 16)):
            with tempfile.TemporaryDirectory() as d:
                t0 = time.time()
                r = tune_kernel(task, budget=b, seed=0, workers=4, ranker=ranker, cache=MeasureCache(d), log=False,
                                generation=gen)
                hist = [c for _, c in r.history]
                best = min(c for c in hist if c is not None)
                e[name] = {"measured": r.measured, "best": best, "seconds": round(time.time() - t0, 1),
                           "history": hist}
        ref = min(x for x in (e["pool_best"], e["active"]["best"], e["default"]["best"]) if x)
        for name in ("active", "default"):
            e[name]["to_target"] = A.to_target(e[name]["history"], ref)
        e["reference_best"] = ref
        log(f"online {kid}@{cid}: active best {e['active']['best']} in {e['active']['measured']}, default "
            f"{e['default']['best']} in {e['default']['measured']} (ref {ref})")
        out.append(e)
    return out


# ------------------------------------------------------------------------------------------------- render

def _f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def svg(curves, path=None):
    W, H, L, B, T, R = 640, 360, 60, 40, 20, 150
    ymax = 1.6
    n = len(next(iter(curves.values())))

    def X(i):
        return L + (W - L - R) * i / (n - 1)

    def Y(v):
        return T + (H - T - B) * (1 - (min(v, ymax) - 1) / (ymax - 1))
    path = SVG if path is None else path
    colors = {"random": "#888888", "tier1": "#1f77b4", "tier2": "#d62728", "tier2_static": "#2ca02c"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" '
             f'font-size="11"><rect width="100%" height="100%" fill="white"/>']
    for v in (1.0, 1.05, 1.2, 1.4, 1.6):
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#ddd"/>'
                     f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>')
    for i in range(0, n, 5):
        parts.append(f'<text x="{X(i):.1f}" y="{H - B + 14}" text-anchor="middle">{i + 1}</text>')
    parts.append(f'<text x="{(L + W - R) / 2}" y="{H - 6}" text-anchor="middle">measurements</text>')
    parts.append(f'<text x="12" y="{T + 120}" transform="rotate(-90 12 {T + 120})" text-anchor="middle">'
                 f'best so far / pool best (mean)</text>')
    for k, (m, c) in enumerate(curves.items()):
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(c))
        parts.append(f'<polyline fill="none" stroke="{colors.get(m, "#000")}" stroke-width="2" points="{pts}"/>')
        parts.append(f'<text x="{W - R + 10}" y="{T + 16 + 16 * k}" fill="{colors.get(m, "#000")}">{m}</text>')
    parts.append("</svg>\n")
    path.write_text("".join(parts))


def render(ds, ev, fails, bad_prog, tun, onl, sizes):
    L = []
    w = L.append
    w("# G4 report: cost models (SG7)\n")
    w("Generated by `PYTHONPATH=sim python3 -m anvil.cost.g4_report` from the frozen dataset in `data/cost/` "
      "(regenerate it with `python3 -m anvil.cost.dataset`). Do not edit by hand.\n")
    nk = len(ds.kernels)
    ntk = sum(ds.is_test_kernel(k) for k in ds.kernels)
    w(headline(ds, ev, tun))
    w("## Dataset\n")
    w(f"- {len(ds.programs)} distinct programs of {nk} kernels ({ntk} held out), {len(ds.rows)} program x config rows "
      f"measured on Verilator ({sum(r['cycles'] is not None for r in ds.rows)} succeeded), {len(ds.configs)} "
      f"GpuParams configurations ({len(ds.split['heldout_configs'])} held out).")
    groups = {}
    for kid, k in ds.kernels.items():
        groups.setdefault(k["group"], [0, 0])[0] += 1
        groups[k["group"]][1] += ds.is_test_kernel(kid)
    w("- Kernels by group (total / held out): " + ", ".join(f"{g} {a}/{b}" for g, (a, b) in sorted(groups.items())))
    w(f"- Files: " + ", ".join(f"`{n}` {s / 1e6:.2f} MB" for n, s in sizes.items()))
    w("- Every program is verified like a tuner candidate (isasim bit-exact vs the tensor interpreter, no "
      "divergence/race/store-conflict flags, no word outside its outputs changed) and every Verilator run's memory "
      f"equals isasim's on every word of the job, scratch included. Programs failing verification: {len(bad_prog)}.")
    dd = ds.split.get("dedupe")
    if dd:
        w(f"- Split deduplicated (AUDIT-A2 M3): {len(dd['reassigned'])} held-out kernels with the same op and shapes "
          "as a training kernel were reassigned to train (" + ", ".join(f"{k} = {v}" for k, v in
                                                               dd["reassigned"].items()) +
          f"), and {dd['dropped_programs']} held-out programs whose feature vector equals a training program's were "
          "dropped (" + ", ".join(f"{k} {n}" for k, n in dd["dropped_by_kernel"].items()) + ").")
    w("")
    w("| config | params | mem_latency | split | rows | failed |")
    w("|---|---|---|---|---|---|")
    for cid, c in ds.configs.items():
        f = fails.get(cid, {"rows": 0, "failed": 0})
        p = " ".join(f"{k}={v}" for k, v in c["params"].items()) or "xf default"
        w(f"| {cid} | {p} | {c['mem_latency']} | {c['split']} | {f['rows']} | {f['failed']} |")
    w("")
    w("## Kendall tau on the frozen held-out split\n")
    w("Held-out kernels x held-out configs. *pooled*: all held-out kernels' candidates of the config ranked together; "
      "*within*: tau over the candidates of one kernel (mean and min over kernels), i.e. the ranking the tuner "
      f"uses. Target (G4): tau >= {TAU_TARGET}.\n")
    h = ev["heldout"]["tiers"]
    tiers = ["tier1_closed", "tier1", "tier2", "tier2_static"]
    w("| config | rows | " + " | ".join(f"{t} pooled | {t} within (mean/min)" for t in tiers) + " |")
    w("|---|---|" + "---|---|" * len(tiers))
    for cid in sorted(h["tier1"]["tau"]):
        cells = []
        for t in tiers:
            d = h[t]["tau"][cid]
            cells.append(f"{_f(d['pooled'])} | {_f(d['within_mean'])} / {_f(d['within_min'])}")
        w(f"| {cid} | {h['tier1']['tau'][cid]['n']} | " + " | ".join(cells) + " |")
    w("")
    w("Accuracy on the same rows (mean absolute percentage error, fraction cycle-exact):\n")
    w("| tier | MAPE | exact | max abs error (cycles) |")
    w("|---|---|---|---|")
    for t in tiers:
        w(f"| {t} | {100 * h[t]['mape']:.3f} % | {100 * h[t]['exact']:.2f} % | {h[t]['max_abs_err']:.0f} |")
    allrows = ds.ok_rows()
    mis = [r for r in allrows if r["t1"] != r["cycles"]]
    w("")
    w(f"Tier 1 over the whole dataset: {len(allrows) - len(mis)} of {len(allrows)} rows cycle-exact; the "
      f"{len(mis)} others (max error {max([abs(r['t1'] - r['cycles']) for r in mis] or [0])} cycles, "
      f"{max([abs(r['t1'] - r['cycles']) / r['cycles'] for r in mis] or [0]):.4%}) are: "
      + ", ".join(sorted({f"{r['pid'].split('|')[0]}@{r['cid']}" for r in mis})) + ".")
    w("")
    w("Other slices (tau pooled, per config; within-kernel mean in parentheses):\n")
    for sl in ("heldout_kernels", "heldout_configs", "train"):
        e = ev[sl]["tiers"]
        w(f"**{sl}** ({ev[sl]['rows']} rows)\n")
        w("| config | " + " | ".join(tiers) + " |")
        w("|---|" + "---|" * len(tiers))
        for cid in sorted(e["tier1"]["tau"]):
            w(f"| {cid} | " + " | ".join(f"{_f(e[t]['tau'][cid]['pooled'])} ({_f(e[t]['tau'][cid]['within_mean'])})"
                                         for t in tiers) + " |")
        w("")
    s = tun["summary"]
    w("## Model-guided vs random tuning\n")
    w(f"{s['pools']} pools (held-out kernel x held-out config, every measured candidate of the dataset). Measurements "
      "until one candidate is within 5 % of the pool best; model-guided = top-1 by the model each round, tier 2 "
      "boosted online on the measured points (seeds 0-2 = three independently trained tier-2 models); random = "
      "uniform without replacement (seeds 0-2) and its exact expectation. G4 target: model-guided total <= "
      f"{RATIO_TARGET:.0%} of random's.\n")
    w("| method | total measurements | ratio vs random E[.] | ratio vs random (seeds) | per-seed ratio | not reached |")
    w("|---|---|---|---|---|---|")
    w(f"| random (expected) | {s['random_expected_total']:.1f} | 1 | | | |")
    w(f"| random (3 seeds, mean) | {s['random_seeds_total']:.1f} | {s['random_seeds_total'] / s['random_expected_total']:.3f} | 1 | | |")
    for m in ("tier1", "tier2", "tier2_static"):
        d = s[m]
        w(f"| {m}-guided | {d['total']:.1f} | {d['ratio']:.3f} | {d['ratio_vs_seeds']:.3f} | "
          f"{', '.join(f'{x:.3f}' for x in d['per_seed_ratio'])} | {d['not_reached']} |")
    w("")
    w("Best-so-far / pool best, mean over pools and seeds (curve: `docs/g4-curves.svg`), and the fraction of pools "
      "within 5 %:\n")
    marks = [1, 2, 3, 5, 10, 20, 40]
    w("| measurements | " + " | ".join(str(m) for m in marks) + " |")
    w("|---|" + "---|" * len(marks))
    for m in ("random", "tier1", "tier2", "tier2_static"):
        c = tun["curves"][m]
        f5 = tun["within5"][m]
        w(f"| {m} | " + " | ".join(f"{c[i - 1]:.3f} ({100 * f5[i - 1]:.0f} %)" for i in marks if i <= len(c)) + " |")
    w("")
    w("Per pool (N candidates, m within 5 %, random expectation, model-guided measurements per seed):\n")
    w("| kernel | config | N | m | random E | tier1 | tier2 | tier2_static |")
    w("|---|---|---|---|---|---|---|---|")
    for p in tun["pools"]:
        w(f"| {p['kid']} | {p['cid']} | {p['n']} | {p['good']} | {p['random_expected']:.1f} | "
          f"{p['tier1'][0]} | {', '.join(map(str, p['tier2']))} | {', '.join(map(str, p['tier2_static']))} |")
    w("")
    if onl:
        w("## Online: tune_kernel with ActiveRanker\n")
        w("Live Verilator tuning with a fresh measurement cache: `ActiveRanker(k=2)` (tier 2, online update; "
          "generations of 64 built candidates, of which the ranker measures 2) with budget 12 vs the default tuner "
          "(generations of 16, every built candidate measured) with budget 200. Reference best: the best of the "
          "dataset pool and both runs. 'to 5 %' = measurements until within 5 % of the reference. The ranker "
          "featurizes every built candidate (profile + tier-1 event model), which costs about as much wall time "
          "per candidate as a Verilator run of the same program; the saving is in timing-backend runs.\n")
        w("| kernel | config | reference best | active: best (measured, to 5 %, s) | "
          "default: best (measured, to 5 %, s) |")
        w("|---|---|---|---|---|")
        for e in onl:
            a, d = e["active"], e["default"]
            w(f"| {e['kid']} | {e['cid']} | {e['reference_best']} | {a['best']} ({a['measured']}, {a['to_target']}, "
              f"{a['seconds']}) | {d['best']} ({d['measured']}, {d['to_target']}, {d['seconds']}) |")
        w("")
    w("## Notes\n")
    w("- Tier 1 (`anvil.cost.analytic.predict`) evaluates closed-form FSM state durations read off the RTL over each "
      "block's dynamic instruction stream, with edge-resolution arbitration of the shared program/data channels, the "
      "L1-D and the warp scheduler; it has no fitted parameter. `tier1_closed` is its contention-free closed form.")
    w("- Tier 2 learns log(cycles / prior) with LightGBM; `tier2` uses the event model as prior and feature, "
      "`tier2_static` only the closed form (no event-level evaluation).")
    w("- Cost of a prediction (rows.jsonl.gz): `t_model` = functional profile + closed form (what tier2_static "
      "needs), `t_tier1` = the event model on top, `t_sim` = the Verilator run; totals in the headline.")
    w("- Because tier 1 is (almost) cycle-exact, tier 2's residual target is ~0 and its trees are nearly empty; "
      "tier 2 only adds value over tier 1 where the event model is wrong (SIMT_STACK=1, MEDIC, future RTL changes). "
      "`tier2_static` shows what the learned model achieves without the event model.")
    w("- The tuning ratio is bounded below by the pools: a perfect model needs 1 measurement, so a pool where random "
      "needs E < 10 cannot meet 10 % on its own (e.g. pools with N = 8-12 candidates at the W=2/W=4 configs); the "
      "G4 ratio sums over pools. The pools are random samples of the sketch spaces, as random search would draw.")
    w("- Rows that timed out on Verilator (2M cycles) are failures, listed per config above, and excluded from tau.")
    return "\n".join(L) + "\n"


def costs(ds):
    """Aggregate wall time of predictions vs measurements over the measured rows (seconds)."""
    rows = ds.ok_rows()
    return {k: float(sum(r.get(k, 0.0) for r in rows)) for k in ("t_model", "t_tier1", "t_sim")}


def headline(ds, ev, tun) -> str:
    """The G4 result as stated after audit A: on tier2_static (no event model), which is cheaper than measuring;
    tier 1 is an RTL replay as slow as Verilator and is reported for reference only (AUDIT-A2 M2)."""
    h = ev["heldout"]["tiers"]["tier2_static"]
    taus = h["tau"]
    pooled = [d["pooled"] for d in taus.values()]
    wmin = {c: d["within_min"] for c, d in taus.items()}
    s = tun["summary"]["tier2_static"]
    c = costs(ds)
    t_ok = min(pooled) >= TAU_TARGET
    w_ok = min(wmin.values()) >= TAU_TARGET
    r_ok = s["ratio"] <= RATIO_TARGET
    worst = min(wmin, key=wmin.get)
    L = ["## Headline (G4, stated on tier2_static)\n",
         f"- Kendall tau pooled over held-out kernels x held-out configs: {_f(min(pooled))} - {_f(max(pooled))} "
         f"(target >= {TAU_TARGET}: {'met' if t_ok else 'MISSED'}).",
         f"- Within-kernel tau (the ranking the tuner consumes), worst kernel per config: "
         + ", ".join(f"{k} {_f(v)}" for k, v in sorted(wmin.items()))
         + f" (target >= {TAU_TARGET}: {'met' if w_ok else f'MISSED at {worst}'}).",
         f"- MAPE {100 * h['mape']:.2f} %. Model-guided tuning needs {s['total']:.1f} measurements vs random search's "
         f"expected {tun['summary']['random_expected_total']:.1f}: ratio {s['ratio']:.3f} (target <= "
         f"{RATIO_TARGET:.0%}: {'met' if r_ok else 'MISSED'}).",
         f"- Cost: tier2_static's inputs (profile + closed form) took {c['t_model']:.0f} s over the dataset vs "
         f"{c['t_sim']:.0f} s of Verilator runs ({c['t_model'] / max(c['t_sim'], 1e-9):.2f}x). Tier 1's event model "
         f"took {c['t_tier1']:.0f} s on top ({(c['t_model'] + c['t_tier1']) / max(c['t_sim'], 1e-9):.2f}x Verilator): "
         "its tau = 1 comes from replaying the RTL state machines and saves timing-backend runs, not time, so it is "
         "reported below for reference and does not carry G4.",
         ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--online", action="store_true")
    a = ap.parse_args(argv)
    ds = D.load()
    if not a.no_train:
        train(ds)
    models = load_models()
    ev = evaluate(ds, models)
    fails, bad_prog = failures(ds)
    tun = tuning(ds, models)
    onl = online(ds) if a.online else None
    sizes = {p.name: p.stat().st_size for p in sorted(D.DATA.glob("*")) if p.is_file()}
    REPORT.write_text(render(ds, ev, fails, bad_prog, tun, onl, sizes))
    svg(tun["curves"])
    slim = {"heldout": {t: {c: {k: v for k, v in d.items() if k != "per_kernel"} for c, d in e["tau"].items()}
                        for t, e in ev["heldout"]["tiers"].items()},
            "mape": {t: e["mape"] for t, e in ev["heldout"]["tiers"].items()},
            "tuning": tun["summary"], "online": onl, "costs": costs(ds), "dedupe": ds.split.get("dedupe")}
    METRICS.write_text(json.dumps(slim, indent=1, sort_keys=True, default=float) + "\n")
    print(f"wrote {REPORT}")
    s = tun["summary"]
    print("tier2 ratio", s["tier2"]["ratio"], "tier1 ratio", s["tier1"]["ratio"])


if __name__ == "__main__":
    main()
