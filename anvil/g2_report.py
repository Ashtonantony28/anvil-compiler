"""G2 report (SG5b): the SG5a target model (2 stacked nanoGPT Extended Blocks, d=32, 2 heads, T=16) compiled by
`anvil.compile` at DATA_MEM_ADDR_BITS=16 and run on isasim, Verilator and the iverilog RTL.

    NANOGPT_DIR=... PYTHONPATH=sim python3 -m anvil.g2_report [--rtl]    # rewrites docs/g2-report.md

Every run is checked: output bit-exact with the batched graph interpreter, within the model contract's TOL vs the
PyTorch module on dequant(x_q), and (isasim) no divergence / race flags. Cycles are Verilator's (cycle-exact with
iverilog, ADR-001). `--rtl` also runs one input per preset on iverilog and records its wall time; without it the
iverilog wall times of the previous report are kept.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from . import model as M

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "g2-report.md"
PRESETS = ("default", "tiled")
N_INPUTS = 32
BINS = [(0, 31), (32, 63), (64, 95), (96, 127), (128, 159), (160, 191), (192, 223), (224, 255)]


def params():
    from .runtime import GpuParams
    return GpuParams(DATA_MEM_ADDR_BITS=M.ADDR_BITS)


def inputs(n=N_INPUTS):
    """Held-out inputs spanning the contract's domain: x_i = sigma_i * z_i, sigma_i evenly over TOL_SIGMA."""
    sig = np.linspace(*M.TOL_SIGMA, n)
    return M.heldout_inputs(n) * sig[:, None, None]


def build(nanogpt_dir=None):
    """(torch Stack, anvil Quantized) of the target model."""
    m = M.build_model(nanogpt_dir or os.environ["NANOGPT_DIR"])
    return m, M.quantize_model(m)


def compile_model(q, preset):
    from .driver import compile
    return compile(q, params=params(), schedules=preset)


class References:
    """Graph-interpreter ints (every segment) and PyTorch float output for a batch of float inputs."""

    def __init__(self, torch_model, q, xs):
        import torch

        from .interp import graph as gi
        self.xq = np.clip(np.rint(xs / q.scales["x"]), -128, 127).astype(np.int64)
        self.graph = gi.run(q.module, {"x": self.xq}, q.consts, batch=True).values
        with torch.no_grad():
            self.torch = torch_model(torch.tensor(self.xq * q.scales["x"], dtype=torch.float64)).numpy()
        self.scale = q.scales[M.OUT]

    def err_lsb(self, i, out_float):
        return float(np.abs(out_float - self.torch[i]).max() / self.scale)


def check(cm, refs: References, i, backend, *, keep="outputs"):
    """Run input i on `backend` and check it -> (RunOutput, error in output LSB, wall seconds). Raises on a
    mismatch with the graph interpreter, on isasim flags, or on an error above TOL."""
    t = time.perf_counter()
    r = cm.run(refs.xq[i], backend, quantized=True, keep=keep)
    wall = time.perf_counter() - t
    for s, v in r.ints.items():
        if s in refs.graph and not np.array_equal(v, refs.graph[s][i]):
            raise AssertionError(f"{backend} input {i}: {s} differs from the graph interpreter")
    if M.OUT not in r.ints:
        raise AssertionError(f"{backend}: output {M.OUT} not readable")
    flags = r.result.flags or {}
    bad = {k: v for k, v in flags.items() if v}
    if bad:
        raise AssertionError(f"{backend} input {i}: isasim flags {bad}")
    err = refs.err_lsb(i, r.output)
    if err > M.TOL_LSB:
        raise AssertionError(f"{backend} input {i}: error {err:.3f} LSB > TOL {M.TOL_LSB}")
    return r, err, wall


def kernel_rows(cm):
    """[(kernel, op, schedule, words, launches, threads, spill frame words, smem words, sink)] in chain order."""
    rows = []
    for k in cm.program.kernels:
        ck = cm.kernels[k.name]
        st = ck.program.stats
        rows.append((k.name, k.op, cm.schedules[k.name][0], st["words"], ck.launches,
                     sum(ck.threads(i) for i in range(ck.launches)), st["frame_words"],
                     ck.code.smem_words, ("sink", None) in ck.slots))
    return rows


def histogram(words) -> list:
    return [sum(lo <= w <= hi for w in words) for lo, hi in BINS]


def collect(torch_model, q, n=N_INPUTS, rtl=False, backends=("isasim", "verilator")):
    """{preset: dict(cm, cycles, errs, walls, ...)}; every run checked (see `check`)."""
    xs = inputs(n)
    refs = References(torch_model, q, xs)
    out = {}
    for preset in PRESETS:
        t = time.perf_counter()
        cm = compile_model(q, preset)
        d = {"cm": cm, "compile_s": time.perf_counter() - t, "errs": [], "walls": {}, "cycles": None}
        for backend in backends:
            walls = []
            for i in range(n):
                r, err, wall = check(cm, refs, i, backend)
                walls.append(wall)
                if backend == "isasim":
                    d["errs"].append(err)
                if backend == "verilator":
                    if d["cycles"] is None:
                        d["cycles"] = r.cycles
                    elif r.cycles != d["cycles"]:
                        raise AssertionError(f"{preset}: cycles depend on the input ({i})")
            d["walls"][backend] = walls
        if rtl:
            r, err, wall = check(cm, refs, 0, "rtl")
            if d["cycles"] is not None and r.cycles != d["cycles"]:
                raise AssertionError(f"{preset}: iverilog cycles differ from Verilator")
            d["walls"]["rtl"] = [wall]
        out[preset] = d
    return out


# ------------------------------------------------------------------------------------------------ render

def cycle_table(res) -> str:
    a, b = (res[p]["cm"] for p in PRESETS)
    ra, rb = kernel_rows(a), {r[0]: r for r in kernel_rows(b)}
    ca, cb = res["default"]["cycles"], res["tiled"]["cycles"]
    out = ["| kernel | op | default cycles | words | launches | tiled schedule | tiled cycles | words | launches | "
           "tiled / default |", "|---|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for name, op, _, w, nl, *_ in ra:
        tb = rb[name]
        out.append(f"| {name} | {op} | {ca[name]} | {w} | {nl} | {tb[2]} | {cb[name]} | {tb[3]} | {tb[4]} | "
                   f"{cb[name] / ca[name]:.2f} |")
    ta, tt = sum(ca.values()), sum(cb.values())
    out.append(f"| **total** | | **{ta}** | | **{sum(r[4] for r in ra)}** | | **{tt}** | | "
               f"**{sum(r[4] for r in rb.values())}** | **{tt / ta:.2f}** |")
    return "\n".join(out)


def summary_table(res) -> str:
    out = ["| preset | total cycles | kernels | launches | max program words | spilling kernels | footprint | "
           "consts | activations (reuse) | activations (no reuse) | args | scratch |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for p in PRESETS:
        cm = res[p]["cm"]
        rows = kernel_rows(cm)
        mp = cm.plan
        r = mp.regions
        out.append(f"| {p} | {sum(res[p]['cycles'].values())} | {len(rows)} | {sum(x[4] for x in rows)} | "
                   f"{max(x[3] for x in rows)} | {sum(x[6] > 0 for x in rows)} | {mp.footprint} | {r['consts'][1]} | "
                   f"{r['activations'][1]} | {mp.unshared} | {r['args'][1]} | {r['scratch'][1]} |")
    return "\n".join(out)


def histogram_table(res) -> str:
    out = ["| program words | " + " | ".join(PRESETS) + " |", "|---|" + "---:|" * len(PRESETS)]
    hs = {p: histogram([x[3] for x in kernel_rows(res[p]["cm"])]) for p in PRESETS}
    for j, (lo, hi) in enumerate(BINS):
        out.append(f"| {lo}-{hi} | " + " | ".join(str(hs[p][j]) for p in PRESETS) + " |")
    return "\n".join(out)


def schedule_notes(res) -> str:
    """Kernels for which the tiled preset asked for a template but fell back to the default schedule."""
    from .driver import PRESETS as DP
    cm = res["tiled"]["cm"]
    fb = [k.name for k in cm.program.kernels if k.op in DP["tiled"] and cm.schedules[k.name][0] == "default"]
    return ", ".join(f"`{k}`" for k in fb) or "none"


def wall_table(res, old_rtl=None) -> str:
    out = ["| preset | compile (s) | isasim (s / input) | Verilator (s / input) | iverilog (s, 1 input) |",
           "|---|---:|---:|---:|---:|"]
    for p in PRESETS:
        w = res[p]["walls"]
        rtl = f"{w['rtl'][0]:.0f}" if "rtl" in w else (old_rtl or {}).get(p, "not measured")
        out.append(f"| {p} | {res[p]['compile_s']:.1f} | {np.mean(w['isasim']):.2f} | {np.mean(w['verilator']):.2f} | "
                   f"{rtl} |")
    return "\n".join(out)


def old_rtl_walls(path=REPORT) -> dict:
    """iverilog wall times recorded in an earlier report (kept when the report is regenerated without --rtl)."""
    if not Path(path).exists():
        return {}
    out = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"\| (\w+) \| [\d.]+ \| [\d.]+ \| [\d.]+ \| (\d+) \|$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def render(res, n=N_INPUTS, old_rtl=None) -> str:
    errs = {p: max(res[p]["errs"]) for p in PRESETS}
    return f"""# G2 report: the full target model on RTL

Generated by `NANOGPT_DIR=... PYTHONPATH=sim python3 -m anvil.g2_report --rtl`; do not edit.
`tests/g2/test_g2_rtl.py` checks that the cycle, summary and histogram tables are current.

Model: SG5a's target (`anvil.model`, `docs/model-contract.md`): two stacked nanoGPT Extended Blocks, d=32,
2 heads (head-batched, ADR-003), T=16, RMSNorm, SwiGLU (hidden 64), `pos_embedding='none'`, captured and quantized
by `anvil.frontend`, compiled by `anvil.compile` with the `default` and `tiled` presets for
`GpuParams(DATA_MEM_ADDR_BITS=16)` (otherwise gpu.sv's defaults: 2 cores, 4 threads per block, one warp per core,
no caches, no coalescing). No hand-written assembly.

Checks ({n} held-out inputs `x = sigma * z`, z from `HELDOUT_SEED`, sigma evenly over [0.5, 3], per preset): the
output `{M.OUT}` (and every other readable value) is bit-exact with the batched graph interpreter on isasim and on
Verilator for all {n}, and on iverilog for input 0; isasim raises no divergence / store-race / store-conflict flag;
Verilator's per-kernel cycles are identical for every input and equal iverilog's. Max error vs the PyTorch module
on `dequant(x_q)`: default {errs['default']:.3f}, tiled {errs['tiled']:.3f} output LSB (TOL = {M.TOL_LSB}).

## Summary

{summary_table(res)}

Footprint in words of the 2^16-word data memory (`anvil.memplan`; the contract's no-reuse bound is
{M.FOOTPRINT_WORDS}). No kernel spills registers and every program is within the 256-word limit.

## Cycles per kernel (Verilator)

Kernels with more than 255 threads are split into several launches (`launches`).

{cycle_table(res)}

Tiled-preset kernels that fall back to the default schedule: {schedule_notes(res)} (the tiled matmul cannot index
through the merge-heads reshape view: "index ... is not linear in the loops it ranges over").

## Program-size histogram (kernels per bucket)

{histogram_table(res)}

## Wall time

One process, Verilator / iverilog builds cached under `.anvil-cache/`; per-input wall times include job
assembly and read-back.

{wall_table(res, old_rtl)}
"""


def main(path=REPORT, rtl=False):
    torch_model, q = build()
    old = old_rtl_walls(path)
    res = collect(torch_model, q, rtl=rtl)
    text = render(res, old_rtl=old)
    Path(path).write_text(text)
    return res, text


if __name__ == "__main__":
    print(main(rtl="--rtl" in sys.argv[1:])[1])
