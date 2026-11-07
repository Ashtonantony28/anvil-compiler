"""Derivation of the target-model contract (docs/model-contract.md, v2: SG9 / ADR-004), as a script (A2-L2).

    NANOGPT_DIR=... PYTHONPATH=sim python3 -m anvil.model_derive [--n 200000] [--quick]

Every number in `anvil/model.py`'s contract block (OUT_UNIT, TOL_LSB, SEGMENTS, INPUT_SEGMENTS) and in the
contract's derivation table comes from here. Only DERIVE_SEED (and seeds derived from it) is used; CALIB_SEED
only through the model's own calibration; HELDOUT_SEED never (asserted below). The rules:

* Sets (contract model unless noted, errors in OUT_UNIT):
  - domain: N_DERIVE inputs N(0, 1) * sigma, sigma log-uniform over TOL_SIGMA (derive_inputs);
  - fixed sigma: 0.02, 0.05, 0.2, 1, 3 (n/5 each);
  - families at unit variance and at sigma = 0.02: Laplace, Student-t(5), N(0, 1) with one channel x4 (n/10 each);
  - matched calibration (the A2-H1 setting): the model re-calibrated on calib_inputs() * sigma, measured on
    N(0, 1) * sigma, sigma in 0.02, 0.05, 0.2, 1 (n/10 each);
  - hill climb: from the 4 worst domain inputs, CLIMB_ROUNDS rounds of CLIMB_CANDIDATES integer perturbations
    (a random eighth of the entries moved by N(0, 0.1 * row RMS + 1) LSB, kept inside int16), keeping the worst.
* TOL = max(1.5 x climbed max, 1.25 x worst sampled max over every set), rounded up to a multiple of 0.5.
* SEGMENTS[k] = (pooled least-squares gain, pooled RMS error) on the domain set (checked with contract 8's rule,
  GAIN_TOL / RMS_TOL); INPUT_SEGMENTS[k] = (1.5 x max per-input |gain projection| (model.gain_projection, against
  the rounded SEGMENTS gain), 1.25 x max per-input RMS) over the domain set: the per-input bounds compiled runs
  are checked with (A2-M1).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

from . import model as M

CLIMB_ROUNDS, CLIMB_CANDIDATES, CLIMB_STARTS = 60, 128, 4
TOL_CLIMB, TOL_SAMPLED, TOL_STEP = 1.5, 1.25, 0.5
IN_GAIN, IN_RMS = 1.5, 1.25
SIGMAS = (0.02, 0.05, 0.2, 1.0, 3.0)
MATCHED = (0.02, 0.05, 0.2, 1.0)


def _rng(*k):
    assert M.HELDOUT_SEED not in k, "the held-out seed is never used for derivation"
    return np.random.default_rng((M.DERIVE_SEED, *k))


def run_batch(q, xq):
    """(graph-interpreter ints, float references) of a batch of integer inputs."""
    from .interp import graph as gi
    got = gi.run(q.module, {"x": xq}, q.consts, batch=True).values
    return got, q.reference(xq * q.scales["x"])


def errors(q, xq, unit):
    got, ref = run_batch(q, xq)
    so = q.scales[M.OUT]
    return np.abs(got[M.OUT] * so - ref[M.OUT]).reshape(len(xq), -1).max(axis=1) / unit


def domain_pass(q, x, unit, chunk=4096):
    """Domain set: per-input errors, pooled segment stats, per-input segment extremes, peaks."""
    keys = M.segment_syms(q)
    num, den, sq, cnt = ({k: 0.0 for k in keys} for _ in range(4))
    gains = {k: [] for k in keys}
    norms = {k: [] for k in keys}
    rmax = {k: 0.0 for k in keys}
    peak = {k: 0.0 for k in keys}
    dts = M.value_dtypes(q.module)
    errs = []
    for lo in range(0, len(x), chunk):
        xq = M.quantize_x(q, x[lo:lo + chunk])
        got, ref = run_batch(q, xq)
        so = q.scales[M.OUT]
        errs.append(np.abs(got[M.OUT] * so - ref[M.OUT]).reshape(len(xq), -1).max(axis=1) / unit)
        per = M.input_segment_stats(q, got, ref, keys)
        for k in keys:
            s = q.scales[k]
            gq, r = got[k] * s, ref[k]
            if k.endswith(".scores"):
                msk = M._causal(r.shape)
                gq, r = gq[msk], r[msk]
            num[k] += float((gq * r).sum())
            den[k] += float((r * r).sum())
            sq[k] += float(((gq - r) ** 2).sum())
            cnt[k] += r.size
            peak[k] = max(peak[k], float(np.abs(r).max()) / M.unit_of(s, dts[k]))
            gains[k].append(per[k][0])
            norms[k].append(per[k][2])
            rmax[k] = max(rmax[k], float(per[k][1].max()))
    pooled = {k: (num[k] / den[k], math.sqrt(sq[k] / cnt[k]) / M.unit_of(q.scales[k], dts[k])) for k in keys}
    gdev = {k: float(np.abs((np.concatenate(gains[k]) - round(pooled[k][0], 4)) * np.concatenate(norms[k])).max())
            for k in keys}
    return np.concatenate(errs), pooled, gdev, rmax, peak


def families(n, sigma, rng):
    shape = (n, M.T, M.D)
    gain = np.ones((n, 1, M.D))
    gain[np.arange(n), 0, np.arange(n) % M.D] = 4
    return {"laplace": rng.laplace(size=shape) / math.sqrt(2) * sigma,
            "student_t5": rng.standard_t(5, size=shape) / math.sqrt(5 / 3) * sigma,
            "channel_x4": rng.standard_normal(shape) * gain * sigma}


def climb(q, x0, unit, rounds=CLIMB_ROUNDS, cands=CLIMB_CANDIDATES, seed=0):
    rng = _rng(100, seed)
    lo, hi = -32768, 32767
    best = M.quantize_x(q, x0[None])[0]
    e_best = float(errors(q, best[None], unit)[0])
    for _ in range(rounds):
        step = 0.1 * math.sqrt(float((best.astype(np.float64) ** 2).mean())) + 1
        mask = rng.random((cands,) + best.shape) < 1 / 8
        c = np.clip(best + np.rint(rng.standard_normal((cands,) + best.shape) * step) * mask, lo, hi).astype(np.int64)
        e = errors(q, c, unit)
        if e.max() > e_best:
            e_best, best = float(e.max()), c[int(e.argmax())]
    return e_best


OPTIONS = {   # ADR-004's comparison: (residual dtype, widen branch linears, calibration)
    "A: v1 (i8 residual, N(0,1) calibration)": ("i8", False, "unit"),
    "B: i8 residual, sigma-range calibration": ("i8", False, "range"),
    "C: i8 residual, matched calibration": ("i8", False, "matched"),
    "C2: F with matched calibration (the A2-H1 test's setting)": ("i16", True, "matched"),
    "D: i16 residual adds, N(0,1) calibration": ("i16", False, "unit"),
    "E: i16 residual adds, sigma-range calibration": ("i16", False, "range"),
    "F: i16 residual adds + branch outputs, sigma-range calibration (chosen)": ("i16", True, "range"),
    "G: F with N(0,1) calibration": ("i16", True, "unit"),
}


def options(n=4000):
    """Max error (OUT_UNIT, and the option's own output unit) of each ADR-004 option at fixed sigmas."""
    m = M.build_model(os.environ["NANOGPT_DIR"])
    unit = M.out_unit(M.quantize_model(m))
    print(f"| option | " + " | ".join(f"sigma {s}" for s in SIGMAS) + " |\n|---|" + "---|" * len(SIGMAS))
    for name, (res, br, cal) in OPTIONS.items():
        cells, q = [], None
        for j, s in enumerate(SIGMAS):
            if cal == "matched" or q is None:
                c = {"unit": M.calib_inputs(), "range": M.contract_calib(), "matched": M.calib_inputs() * s}[cal]
                q = M.quantize_model(m, calib=c, residual=res, residual_branches=br)
            e = errors_chunked(q, _rng(9, j).standard_normal((n, M.T, M.D)) * s, unit).max()
            cells.append(f"{e:.2f} ({e * unit / M.out_unit(q):.2f})")
        print(f"| {name} | " + " | ".join(cells) + " |", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=M.N_DERIVE)
    ap.add_argument("--quick", action="store_true", help="2k domain inputs, short climb (smoke test)")
    ap.add_argument("--options", action="store_true", help="ADR-004's option comparison table instead")
    ap.add_argument("--domain-only", action="store_true", help="only the domain set (SEGMENTS, INPUT_SEGMENTS)")
    a = ap.parse_args(argv)
    if a.options:
        return options()
    n = 2000 if a.quick else a.n
    rounds = 5 if a.quick else CLIMB_ROUNDS
    t0 = time.time()
    m = M.build_model(os.environ["NANOGPT_DIR"])
    q = M.quantize_model(m)
    unit = M.out_unit(q)
    print(f"OUT_UNIT = {unit!r}  (scale(L1.out) {q.scales[M.OUT]!r}), bounds max {max(q.bounds.values())}",
          flush=True)

    x = M.derive_inputs(n)
    err, pooled, gdev, rmax, peak = domain_pass(q, x, unit)
    sets = {f"domain sigma in {M.TOL_SIGMA}": err.max()}
    if a.domain_only:
        print_segments(pooled, gdev, rmax)
        return
    print(f"domain {n}: max {err.max():.3f}, 99.99% {np.quantile(err, 0.9999):.3f}, "
          f"peak {max(v for k, v in peak.items() if not k.endswith('.p')):.1f} ({time.time() - t0:.0f} s)", flush=True)
    for j, s in enumerate(SIGMAS):
        xs = _rng(1, j).standard_normal((n // 5, M.T, M.D)) * s
        sets[f"sigma {s}"] = float(errors_chunked(q, xs, unit).max())
        print(f"sigma {s} ({n // 5}): max {sets[f'sigma {s}']:.3f}", flush=True)
    for j, s in enumerate((1.0, 0.02)):
        for name, xs in families(n // 10, s, _rng(2, j)).items():
            sets[f"{name} sigma {s}"] = float(errors_chunked(q, xs, unit).max())
            print(f"{name} sigma {s} ({n // 10}): max {sets[f'{name} sigma {s}']:.3f}", flush=True)
    for j, s in enumerate(MATCHED):
        qm = M.quantize_model(m, calib=M.calib_inputs() * s)
        xs = _rng(3, j).standard_normal((n // 10, M.T, M.D)) * s
        e = errors_chunked(qm, xs, unit)
        own = e * unit / M.out_unit(qm)
        sets[f"matched calibration sigma {s}"] = float(e.max())
        print(f"matched calibration sigma {s} ({n // 10}): max {e.max():.3f} (own output LSB {own.max():.3f})",
              flush=True)
    worst = np.argsort(err)[-CLIMB_STARTS:]
    climbed = max(climb(q, x[i], unit, rounds, seed=j) for j, i in enumerate(worst))
    print(f"hill climb from the {CLIMB_STARTS} worst domain inputs: {climbed:.3f} ({time.time() - t0:.0f} s)")
    worst_sampled = max(sets.values())
    tol = math.ceil(max(TOL_CLIMB * climbed, TOL_SAMPLED * worst_sampled) / TOL_STEP) * TOL_STEP
    print(f"\nworst sampled {worst_sampled:.3f}, climbed {climbed:.3f} -> TOL = {tol}\n")
    print(f"OUT_UNIT = {unit!r}\nTOL_LSB = {tol}")
    print_segments(pooled, gdev, rmax)
    print(f"done in {time.time() - t0:.0f} s")


def print_segments(pooled, gdev, rmax):
    print("SEGMENTS = {")
    for k, (g, r) in pooled.items():
        print(f'    "{k}": ({g:.4f}, {r:.3f}),')
    print("}\nINPUT_SEGMENTS = {")
    for k in pooled:
        p, r = math.ceil(IN_GAIN * gdev[k] * 1e3) / 1e3, math.ceil(IN_RMS * rmax[k] * 1e3) / 1e3
        print(f'    "{k}": ({p:.3f}, {r:.3f}),')
    print("}")


def errors_chunked(q, x, unit, chunk=4096):
    return np.concatenate([errors(q, M.quantize_x(q, x[lo:lo + chunk]), unit) for lo in range(0, len(x), chunk)])


if __name__ == "__main__":
    sys.exit(main())
