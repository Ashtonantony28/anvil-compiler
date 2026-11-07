"""G5 report: Pareto fronts of cycles vs hardware cost per workload, and the ISA proposal table.

    PYTHONPATH=sim python3 -m anvil.codesign.report [--verify]     # writes docs/g5-report.md, docs/g5-pareto.svg

Reads only cached results: the sweep's point records (data/codesign/sweep/<workload>/<config>.json, written by
anvil.codesign.sweep) and the ISA study's records (data/codesign/isa/<variant>/<workload>.json, anvil.isa_ext.study).
Every number in the report is a field of one of those files (each table names its source); `--verify` also
re-reads every kernel's best measurement from the measurement cache (.anvil-cache/codesign/measure, by the `key`
the record stores) and checks its cycles against the record.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .sweep import BUDGET, MODEL_BUDGET, SWEEP, TOPK, configs_for, load_points, measure_root
from .workloads import ROOT, WORKLOADS

REPORT = ROOT / "docs" / "g5-report.md"
SVG = ROOT / "docs" / "g5-pareto.svg"
BASE_ID = "C2"


def pareto(points):
    """Non-dominated points (lower cost and lower cycles are better), sorted by cost."""
    pts = sorted(points, key=lambda r: (r["cost_bits"], r["cycles"]))
    front, best = [], None
    for r in pts:
        if best is None or r["cycles"] < best:
            front.append(r)
            best = r["cycles"]
    return front


def knee(front):
    """The front point farthest below the chord from the cheapest to the fastest point, both axes normalized to
    [0, 1] over the front (None for fewer than 3 points)."""
    if len(front) < 3:
        return front[-1] if front else None
    c0, c1 = front[0]["cost_bits"], front[-1]["cost_bits"]
    y0, y1 = front[0]["cycles"], front[-1]["cycles"]

    def nx(r):
        return (r["cost_bits"] - c0) / max(1, c1 - c0)

    def ny(r):
        return (r["cycles"] - y1) / max(1, y0 - y1)
    # chord from (0, 1) to (1, 0): distance below it is (1 - x - y) / sqrt 2
    return max(front[1:-1], key=lambda r: 1 - nx(r) - ny(r))


def groups(points):
    """(workload, mem_latency) pairs with points: mem_latency is the memory's, not the GPU's, so each latency gets
    its own front (groups of fewer than 3 points, e.g. the model at latency 3, have none)."""
    return [(w, lat) for w in WORKLOADS for lat in (0, 3)
            if sum(r["mem_latency"] == lat for r in points.get(w) or []) >= 3]


def verify_points(points) -> dict:
    from ..tune import MeasureCache
    cache = MeasureCache(measure_root())
    n = ok = missing = 0
    bad = []
    for recs in points.values():
        for r in recs:
            for k, kr in r["kernels"].items():
                if not kr.get("key"):
                    continue
                n += 1
                hit = cache.get(kr["key"])
                if hit is None:
                    missing += 1
                elif hit.get("cycles") == kr["isolated"]:
                    ok += 1
                else:
                    bad.append((r["workload"], r["config"], k))
    return {"kernels": n, "ok": ok, "missing": missing, "mismatch": bad}


def _fmt(n):
    return f"{n:,}"


def _desc(cid):
    return cid


# ------------------------------------------------------------------------------------------------- svg
def svg(points, fronts, knees) -> str:
    W, H, pad = 980, 400, 16
    panels = [w for w in WORKLOADS if points.get(w)]
    pw = (W - 2 * pad) / max(1, len(panels))
    ink, ink2, grid, surf, accent, accent2, other = ("#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6",
                                                    "#eb6834", "#a3a29d")
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'font-family="system-ui, sans-serif" font-size="11">',
           f'<rect width="{W}" height="{H}" fill="{surf}"/>',
           f'<text x="{pad}" y="20" font-size="14" font-weight="600" fill="{ink}">Cycles (log scale) vs hardware cost (state '
           f'bits), each point separately tuned; knees ringed</text>',
           f'<circle cx="{pad + 6}" cy="36" r="4.5" fill="{accent}"/><text x="{pad + 16}" y="40" fill="{ink}">'
           f'Pareto front, mem_latency 0</text>',
           f'<circle cx="{pad + 196}" cy="36" r="4.5" fill="{accent2}"/><text x="{pad + 206}" y="40" fill="{ink}">'
           f'Pareto front, mem_latency 3</text>',
           f'<circle cx="{pad + 386}" cy="36" r="4" fill="{other}"/><text x="{pad + 396}" y="40" fill="{ink}">'
           f'dominated point</text>']
    for pi, w in enumerate(panels):
        recs = points[w]
        x0 = pad + pi * pw + 56
        x1 = pad + (pi + 1) * pw - 14
        y0, y1 = H - 46, 78
        cmin = min(r["cost_bits"] for r in recs)
        cmax = max(r["cost_bits"] for r in recs)
        import math
        lo = math.log10(min(r["cycles"] for r in recs) / 1.15)
        hi = math.log10(max(r["cycles"] for r in recs) * 1.1)

        def X(c):
            return x0 + (c - cmin) / max(1, cmax - cmin) * (x1 - x0)

        def Y(v):
            return y0 - (math.log10(v) - lo) / (hi - lo) * (y0 - y1)
        out.append(f'<text x="{x0}" y="{y1 - 12}" font-size="12" font-weight="600" fill="{ink}">{w}</text>')
        ticks = [m * 10 ** e for e in range(int(lo) - 1, int(hi) + 2) for m in (1, 2, 3, 5)
                 if lo <= math.log10(m * 10 ** e) <= hi]
        for v in ticks:
            out.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{grid}"/>')
            lab = f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k"
            out.append(f'<text x="{x0 - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="{ink2}">{lab}</text>')
        for k in range(5):
            c = cmin + (cmax - cmin) * k / 4
            out.append(f'<text x="{X(c):.1f}" y="{y0 + 16}" text-anchor="middle" fill="{ink2}">{c / 1e3:.0f}k</text>')
        out.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{y0 + 32}" text-anchor="middle" fill="{ink2}">state bits</text>')
        front_ids = {r["config"] for lat in (0, 3) for r in fronts.get((w, lat), [])}
        for r in recs:
            if r["config"] not in front_ids:
                out.append(f'<circle cx="{X(r["cost_bits"]):.1f}" cy="{Y(r["cycles"]):.1f}" r="4" fill="{other}" '
                           f'stroke="{surf}" stroke-width="2"><title>{r["config"]}: {r["cycles"]:,} cycles, '
                           f'{r["cost_bits"]:,} bits</title></circle>')
        for lat, col in ((0, accent), (3, accent2)):
            f = fronts.get((w, lat))
            if not f:
                continue
            pts = " ".join(f"{X(r['cost_bits']):.1f},{Y(r['cycles']):.1f}" for r in f)
            out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
            for r in f:
                out.append(f'<circle cx="{X(r["cost_bits"]):.1f}" cy="{Y(r["cycles"]):.1f}" r="4.5" fill="{col}" '
                           f'stroke="{surf}" stroke-width="2"><title>{r["config"]}: {r["cycles"]:,} cycles, '
                           f'{r["cost_bits"]:,} bits (Pareto, latency {lat})</title></circle>')
            kn = knees.get((w, lat))
            if kn:
                kx, ky = X(kn["cost_bits"]), Y(kn["cycles"])
                out.append(f'<circle cx="{kx:.1f}" cy="{ky:.1f}" r="9" fill="none" stroke="{ink}" stroke-width="1.5"/>')
                dy = 22 if lat == 0 else -14             # latency-0 label below its ring, latency-3 above
                anchor, tx = ("end", kx - 4) if kx > x1 - 140 else ("start", kx + 4)
                out.append(f'<text x="{tx:.1f}" y="{ky + dy:.1f}" text-anchor="{anchor}" fill="{ink}">'
                           f'knee: {kn["config"]}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------- report
def render(points, isa, ver=None) -> str:
    L = []
    w_ = L.append
    w_("# G5 report: hardware/software co-design\n")
    w_("Generated by `PYTHONPATH=sim python3 -m anvil.codesign.report` from cached results only; do not edit. Sources: "
       "the co-design sweep's point records `data/codesign/sweep/<workload>/<config>.json` "
       "(`python3 -m anvil.codesign.sweep`) and the ISA study's records `data/codesign/isa/<variant>/<workload>.json` "
       "(`python3 -m anvil.isa_ext.study`). Every number below is a field of the record named in its row.\n")
    w_("## Method\n")
    w_(f"* **Configurations** (`anvil.codesign.space`): of the 4,032-point product (NUM_CORES 1/2/4, ICACHE_LINES "
       "0/8/16/32, DCACHE_LINES 0/8/16, COALESCE, PIPELINED, WARPS_PER_CORE 1/2/4 x WARP_POLICY, SIMT_STACK, "
       "mem_latency 0/3; THREADS_PER_BLOCK 4, WORD_BITS 32) the sweep evaluates the baseline `C2` (the xf build) "
       "and every one-factor change from it at both latencies, a grid of the cheap-but-fast region "
       "(PIPELINED + COALESCE x NUM_CORES x ICACHE 16/32 x 1 warp | 2 warps GTO) and six seeded random points: "
       f"{len(configs_for('mlp_block'))} points for the MLP block; the attention head (about twice as slow to tune) "
       f"skips the latency-3 one-factor points ({len(configs_for('attention'))} points). **The SG5 model** (40 kernels, "
       f"~3.5M cycles, the slowest to tune) is swept on a reduced subset of {len(configs_for('model'))} points "
       f"(`model_design()`) and with a reduced search budget ({MODEL_BUDGET} vs {BUDGET} measured candidates per "
       "kernel).")
    w_("* **Config names**: `C<cores>`, `I<icache lines>`, `D<dcache lines>`, `CO` coalescing, `PI` pipelined, "
       "`W<warps><policy>`, `S` SIMT stack, `L3` mem_latency 3.")
    w_(f"* **Tuning per point**: `anvil.compile(..., schedules=TunedPolicy)` with the tier2_static ranker "
       f"(`anvil.codesign.ranker.StaticRanker`: the committed `data/cost/tier2_static-seed0` model on profile + "
       f"closed-form features, top-{TOPK} per round, online updates), seed 0, Verilator timing. Every candidate is "
       "bit-exact on isasim with no divergence / race / store-conflict / cross-block race flag before its cycles "
       "count. A shared best-schedule store (`data/codesign/best.json`) seeds each search with the same kernel's "
       "best at configurations swept earlier, so results depend on the (fixed) sweep order.")
    w_("* **Every point's chain** is then run end to end on Verilator at the point's mem_latency and compared bit for "
       "bit with the reference (MLP/attention: the xf goldens on every segment; model: the graph interpreter's "
       "`L1.out` on a held-out input), and on isasim with the cross-block race checker (no flag). A point's cycles "
       "are that chain's Verilator cycles (`cycles` in the record).")
    w_("* **Hardware cost proxy** (`anvil.codesign.cost`, formula and RTL line citations in its docstring): declared "
       "state bits of `module gpu` (register files 16 x WORD_BITS per thread, LSU and ALU output registers, "
       "scheduler and per-warp state, SIMT stacks, 64-word shared memory per core, I-cache lines (valid + tag + "
       "16-bit word) per core, the shared L1-D (valid + tag + data per line plus per-lane port registers), "
       "the data/program memory controllers). Combinational logic and telemetry counters are not counted, so "
       "PIPELINED and WARP_POLICY are free and a multiplier-heavy ALU costs only its output register: a proxy, "
       "not synthesis.")
    w_("* **Pareto front**: points not dominated in (state bits, cycles), per workload and memory latency (latency is the "
       "memory's, not a GPU design choice). **Knee**: the front point farthest below "
       "the chord from the cheapest to the fastest front point, both axes normalized over the front.\n")

    fronts, knees = {}, {}
    w_("## Pareto summary\n")
    w_("| workload | points | baseline (bits / cycles) | fastest (bits / cycles, vs baseline) | knee (bits / cycles, vs baseline) | "
       "cheapest on front |")
    w_("|---|---:|---|---|---|---|")
    for w, lat in groups(points):
        recs = [r for r in points[w] if r["mem_latency"] == lat]
        f = pareto(recs)
        fronts[(w, lat)], knees[(w, lat)] = f, knee(f)
        base = next((r for r in points[w] if r["config"] == (BASE_ID if lat == 0 else f"{BASE_ID}.L{lat}")), None)
        fast, kn, cheap = f[-1], knees[(w, lat)], f[0]

        def vs(r):
            return f"{r['cycles'] / base['cycles']:.2f}x cycles, {r['cost_bits'] / base['cost_bits']:.2f}x bits" \
                if base else ""
        w_(f"| {w}, mem_latency {lat} | {len(recs)} | {_fmt(base['cost_bits'])} / {_fmt(base['cycles'])} | **{fast['config']}** "
           f"{_fmt(fast['cost_bits'])} / {_fmt(fast['cycles'])} ({vs(fast)}) | **{kn['config']}** "
           f"{_fmt(kn['cost_bits'])} / {_fmt(kn['cycles'])} ({vs(kn)}) | {cheap['config']} "
           f"{_fmt(cheap['cost_bits'])} / {_fmt(cheap['cycles'])} |")
    w_("\n![Pareto fronts](g5-pareto.svg)\n")

    for w, lat in groups(points):
        w_(f"### {w}, mem_latency {lat}: Pareto front\n")
        w_("| config | state bits | cycles | source |")
        w_("|---|---:|---:|---|")
        for r in fronts[(w, lat)]:
            mark = " (knee)" if r is knees[(w, lat)] else ""
            w_(f"| {r['config']}{mark} | {_fmt(r['cost_bits'])} | {_fmt(r['cycles'])} | "
               f"`data/codesign/sweep/{w}/{r['config']}.json` |")
        w_("")

    w_("## One-factor effects (latency 0, cycles relative to C2)\n")
    w_("| change from C2 | state bits (at 12 address bits) | " + " | ".join(w for w in WORKLOADS if points.get(w)) + " |")
    w_("|---|---:|" + "---:|" * len([w for w in WORKLOADS if points.get(w)]))
    by = {w: {r["config"]: r for r in points.get(w) or []} for w in WORKLOADS}
    from .space import ofat
    for c in ofat(0)[1:] + ofat(3)[:1]:
        cells = []
        for w in WORKLOADS:
            if not points.get(w):
                continue
            r, b = by[w].get(c.id), by[w].get(BASE_ID)
            cells.append(f"{r['cycles'] / b['cycles']:.3f}" if r and b else "-")
        bits = next((by[w][c.id]["cost_bits"] for w in ("mlp_block",) if c.id in by[w]), None)
        w_(f"| {c.id} | {_fmt(bits) if bits else '-'} | " + " | ".join(cells) + " |")
    w_("")

    w_("## ISA proposals (model-only)\n")
    w_("Candidate instructions in a fork of the ISA model (`anvil/isa_ext/`: 5-bit opcode, so a 17-bit instruction "
       "word: the base ISA already uses all 16 opcodes; base programs are unchanged word for word). Codegen idioms "
       "switch to the new instruction when it is enabled (`anvil.isa_ext.backend`). Timing is the tier-1 event "
       "model of `anvil.cost.analytic` on the fork's instruction streams, where a new op costs what every ALU op "
       "costs; **the RTL was not changed**, so these are model numbers, not Verilator measurements. Baseline "
       "configuration (C2, mem_latency 0). `fixed`: the sweep's C2 schedules (Verilator-tuned on the base ISA) "
       "compiled for each variant, all three workloads. `retuned`: every kernel re-tuned for the variant (fork + "
       "tier 1, measure-all, budget 24), MLP block and attention only; the `base` row runs the identical pipeline. "
       "Every chain is bit-exact against the reference on the fork with no flag.\n")
    variants = ["base", "MIN", "AND", "SHL", "SLT", "MAD", "ALL"]
    if isa:
        hdr = []
        for w in WORKLOADS:
            if ("base", w) in isa:
                hdr.append(w)
        w_("| proposal | replaces | " + " | ".join(f"{w} fixed: cycles saved" for w in hdr) + " | " +
           " | ".join(f"{w} retuned: cycles saved" for w in hdr if "retuned" in isa[("base", w)]) +
           " | code words saved (fixed; " + " / ".join(hdr) + ") | new ops emitted (fixed, all) |")
        w_("|---|---|" + "---:|" * (len(hdr) + len([w for w in hdr if "retuned" in isa[("base", w)]]) + 2))
        repl = {"base": "-", "MIN": "a+b-max(a,b)", "AND": "x - 2^k(x div 2^k)", "SHL": "MUL by 2^k>=256, "
                "requant 2^(S-1)", "SLT": "(max(a,b)-a) div itself", "MAD": "MUL + ADD acc", "ALL": "all five"}
        single = {}
        for v in variants:
            cells, rcells = [], []
            words_saved, ops = [], 0
            for w in hdr:
                b, r = isa.get(("base", w)), isa.get((v, w))
                if not r:
                    cells.append("-")
                    continue
                d = b["fixed"]["cycles"] - r["fixed"]["cycles"]
                cells.append(f"{_fmt(d)} ({d / b['fixed']['cycles'] * 100:.1f} %)")
                words_saved.append(sum(b["fixed"]["words"].values()) - sum(r["fixed"]["words"].values()))
                ops += sum(r["fixed"]["ops"].values())
                if v not in ("base", "ALL"):
                    single.setdefault(v, 0.0)
                    single[v] += d / b["fixed"]["cycles"] * 100 / len(hdr)
            for w in hdr:
                b, r = isa.get(("base", w)), isa.get((v, w))
                if b and "retuned" in b:
                    d = b["retuned"]["cycles"] - r["retuned"]["cycles"]
                    rcells.append(f"{_fmt(d)} ({d / b['retuned']['cycles'] * 100:.1f} %)")
            w_(f"| {v} | {repl[v]} | " + " | ".join(cells) + " | " + " | ".join(rcells) +
               f" | {' / '.join(map(str, words_saved))} | {ops} |")
        if single:
            top = max(single, key=single.get)
            w_(f"\n**The single instruction that pays most: {top}** (mean fixed-schedule saving over the workloads "
               f"{single[top]:.1f} %; ranking: " + ", ".join(f"{k} {v:.1f} %" for k, v in
                                                          sorted(single.items(), key=lambda kv: -kv[1])) + ").\n")
        w_("Base-ISA absolute numbers (tier 1 on the fork): " + "; ".join(
            f"{w} fixed {_fmt(isa[('base', w)]['fixed']['cycles'])}" +
            (f", retuned {_fmt(isa[('base', w)]['retuned']['cycles'])}" if "retuned" in isa[("base", w)] else "")
            for w in hdr) + ". Cross-check: the fixed base chain's tier-1 cycles vs its Verilator cycles in the "
            "sweep's C2 record: " + "; ".join(
                f"{w} {_fmt(isa[('base', w)]['fixed']['cycles'])} vs {_fmt(by[w][BASE_ID]['cycles'])}"
                for w in hdr if BASE_ID in by.get(w, {})) + ".\n")
    else:
        w_("(no ISA study records)\n")

    w_("## Verification and cost of the sweep\n")
    allr = [r for w in WORKLOADS for r in points.get(w) or []]
    w_(f"* {len(allr)} points, every chain bit-exact on Verilator and flag-free on isasim (`verify` in each record: "
       f"segments checked, isasim flags, cross-block races, all zero).")
    tune = sum(r["tune_seconds"] for r in allr) / 60
    runs = sum(r["run_seconds"] for r in allr) / 60
    isa_s = sum(r.get("seconds", 0) for r in isa.values()) / 60 if isa else 0
    w_(f"* Wall time on 4 workers (from the records): tuning {tune:.0f} min, chain runs + checks {runs:.1f} min, "
       f"ISA study {isa_s:.0f} min.")
    meas = sum(k.get("measured") or 0 for r in allr for k in r["kernels"].values())
    timed = sum(k.get("timed") or 0 for r in allr for k in r["kernels"].values())
    w_(f"* Candidates measured {_fmt(meas)} (Verilator-timed {_fmt(timed)}, the rest measurement-cache hits).")
    if ver is not None:
        w_(f"* `--verify`: {ver['ok']} of {ver['kernels']} kernel best measurements re-read from "
           f"`.anvil-cache/codesign/measure` equal the record; {ver['missing']} missing, "
           f"{len(ver['mismatch'])} mismatched.")
    w_("\n## Caveats\n")
    w_("* Each point is tuned with a small guided budget; a point can lose to a cheaper one through search noise "
       "rather than hardware (e.g. C4 vs C2 on the MLP block). Points are reproducible by replay "
       "(`tests/codesign`, rtl), not by re-searching in another order.")
    w_("* WARPS_PER_CORE > 1 at THREADS_PER_BLOCK = 4 splits a block into 2 or 4 narrower warps that share one issue "
       "slot, so with this block size it only costs cycles; warps would need a larger THREADS_PER_BLOCK to pay.")
    w_("* SIMT_STACK = 1 points are ranked by tier2_static on a SIMT_STACK = 0 profile (tier 1 does not model the "
       "stack); their Verilator measurements are exact.")
    w_("* The cost proxy counts state bits only (see Method); the ISA results are model-only (see above).")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", default=str(REPORT))
    ap.add_argument("--svg", default=str(SVG))
    a = ap.parse_args(argv)
    from ..isa_ext.study import load_results
    points = load_points(SWEEP)
    isa = load_results()
    ver = verify_points(points) if a.verify else None
    text = render(points, isa, ver)
    fronts = {g: pareto([r for r in points[g[0]] if r["mem_latency"] == g[1]]) for g in groups(points)}
    knees = {w: knee(f) for w, f in fronts.items()}
    Path(a.svg).write_text(svg(points, fronts, knees))
    Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
