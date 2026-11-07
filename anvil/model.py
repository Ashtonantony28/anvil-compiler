"""SG5a target model: two stacked nanoGPT Extended Blocks at d=32, 2 heads, T=16 (design D4), quantized by the
frontend, plus the measurement helpers behind its derived tolerance (docs/model-contract.md, v2 since SG9).

    m = build_model(nanogpt_dir)                 # torch Stack of N_LAYER Blocks, float64, eval, seeded
    q = quantize_model(m)                        # anvil Quantized: i16 residual stream, calibrated on contract_calib()
    err, stats, peak = measure(q, heldout_domain(n))  # per-input max error (OUT_UNIT), segment gain/RMS, peaks

Model (D4): `GPTConfig(n_embd=32, n_head=2, block_size=16, bias=False, pos_embedding='none', norm_type='rmsnorm',
activation='swiglu')`, Blocks seeded SEED + layer. Like xf/export.py::build_block, the SwiGLU hidden width is
overridden (HIDDEN = 2 * d = 64; nanoGPT's own rounding to a multiple of 256 would give 256, quadrupling the MLP
weights and pushing the image past 16-bit addresses) and the RMSNorm weights are drawn from U(0.5, 1.5) so they
are not all ones. Quantization (ADR-004): the residual stream (input, residual adds, the branch linears feeding
them) is i16, everything else int8. Data: calibration on 2^16 N(0, 1) inputs from CALIB_SEED each scaled by a
log-uniform sigma over [0.02, 3]; the contract derived on DERIVE_SEED inputs by anvil.model_derive; HELDOUT_SEED
never used for anything but the final check (AUDIT-6 L1).

Import does not need torch; the builders import it lazily.
"""
from __future__ import annotations

import math
import sys

import numpy as np

D, N_HEAD, T, N_LAYER, HIDDEN = 32, 2, 16, 2, 64
SEED = 0                      # torch.manual_seed(SEED + layer) before each Block
CALIB_SEED, N_CALIB = 0, 1 << 16
DERIVE_SEED, N_DERIVE = 2, 200_000     # contract v2 (SG9); v1 (SG5a) derived on seed 1
HELDOUT_SEED = 20261107       # fresh for contract v2: never used for calibration, design or derivation. (v1's
                              # 20261106 was also looked at while choosing the SG9 fix, so it was retired.)
HEADROOM = 1.5
RESIDUAL = "i16"              # residual stream dtype (ADR-004); "i8" with N(0, 1) calibration is contract v1
EPS = 1e-6

# per-layer IR syms "L<i>.<seg>": h1 q k v qh kh vh kT scores p o om y r1 h2 gate up act down out (names())


# --------------------------------------------------------------------------------- SG5b contract (frozen)
# docs/model-contract.md (v2, SG9 / ADR-004). Output = "L1.out" (the second Block's residual, i16). Errors are in
# OUT_UNIT, a fixed float: 1/127 of the contract model's calibrated L1.out range (its int8-equivalent LSB; with
# an int8 output it would be one LSB). They are input-quantization free (the float model runs on dequant(x_q)),
# on inputs x = sigma * z with sigma in TOL_SIGMA and z per element N(0, 1) (also Laplace / Student-t(5) at unit
# variance and any one channel x4). Reproduce the numbers with `python3 -m anvil.model_derive`.
OUT = "L1.out"
TOL_SIGMA = (0.02, 3.0)
CALIB_SIGMA = TOL_SIGMA            # calibration: N(0, 1) inputs scaled by sigma log-uniform over this range
OUT_UNIT = 0.15846072405721906    # out_unit(quantize_model(m)): 1/127 of the calibrated L1.out range (v1's TOL of 6
                                   # LSB = 6 x 0.0686 = 2.6 OUT_UNIT)
TOL_LSB = 5.0                      # max(1.5 x climbed 3.031, 1.25 x sampled 1.779), rounded up to 0.5 (model_derive)
FOOTPRINT_WORDS = 50_048           # footprint(lower(q.module))["total"]: no reuse, one element per word
ADDR_BITS = 16
GAIN_TOL, RMS_TOL = 0.0025, 1.1    # contract 8's: |gain - SEGMENTS[k][0]| <= GAIN_TOL, RMS <= SEGMENTS[k][1] * RMS_TOL
PEAK_LSB = 127.5                   # no non-softmax activation saturates on held-out inputs (int8-equivalent LSB)
# (least-squares gain, RMS error in the segment's int8-equivalent LSB) per segment, pooled over the 200k
# DERIVE_SEED domain inputs
SEGMENTS = {
    "L0.h1": (1.0008, 0.294),
    "L0.q": (1.0009, 0.466),
    "L0.k": (1.0012, 0.447),
    "L0.v": (1.0005, 0.465),
    "L0.qh": (1.0009, 0.466),
    "L0.kh": (1.0012, 0.447),
    "L0.vh": (1.0005, 0.465),
    "L0.kT": (1.0012, 0.447),
    "L0.scores": (1.0017, 0.543),
    "L0.p": (1.0001, 0.262),
    "L0.o": (0.9983, 0.391),
    "L0.om": (0.9983, 0.391),
    "L0.y": (0.9983, 0.320),
    "L0.r1": (1.0000, 0.046),
    "L0.h2": (0.9998, 0.532),
    "L0.gate": (0.9998, 0.733),
    "L0.up": (0.9999, 0.771),
    "L0.act": (0.9980, 0.382),
    "L0.down": (0.9979, 1.236),
    "L0.out": (1.0000, 0.081),
    "L1.h1": (0.9989, 0.844),
    "L1.q": (0.9989, 1.037),
    "L1.k": (0.9989, 1.114),
    "L1.v": (0.9988, 1.099),
    "L1.qh": (0.9989, 1.037),
    "L1.kh": (0.9989, 1.114),
    "L1.vh": (0.9988, 1.099),
    "L1.kT": (0.9989, 1.114),
    "L1.scores": (0.9976, 1.025),
    "L1.p": (1.0000, 0.366),
    "L1.o": (0.9992, 0.631),
    "L1.om": (0.9992, 0.631),
    "L1.y": (0.9994, 0.510),
    "L1.r1": (0.9999, 0.111),
    "L1.h2": (0.9989, 0.720),
    "L1.gate": (0.9986, 0.887),
    "L1.up": (0.9991, 0.906),
    "L1.act": (0.9990, 0.428),
    "L1.down": (0.9988, 1.275),
    "L1.out": (0.9999, 0.137),
}
# per-input bounds for single runs (compiled G2 checks, A2-M1): (max |gain_projection_i| in LSB, max RMS_i in
# LSB), 1.5x / 1.25x the extremes over the 200k derivation inputs
INPUT_SEGMENTS = {
    "L0.h1": (2.186, 0.400),
    "L0.q": (3.932, 0.665),
    "L0.k": (4.242, 0.644),
    "L0.v": (4.089, 0.663),
    "L0.qh": (3.932, 0.665),
    "L0.kh": (4.242, 0.644),
    "L0.vh": (4.089, 0.663),
    "L0.kT": (4.242, 0.644),
    "L0.scores": (5.592, 0.885),
    "L0.p": (1.799, 0.437),
    "L0.o": (5.906, 0.598),
    "L0.om": (5.906, 0.598),
    "L0.y": (6.966, 0.530),
    "L0.r1": (1.027, 0.076),
    "L0.h2": (7.234, 1.650),
    "L0.gate": (15.193, 2.170),
    "L0.up": (16.696, 2.328),
    "L0.act": (11.416, 0.877),
    "L0.down": (25.575, 3.099),
    "L0.out": (1.121, 0.181),
    "L1.h1": (10.016, 3.074),
    "L1.q": (15.718, 3.601),
    "L1.k": (20.919, 3.835),
    "L1.v": (24.568, 4.290),
    "L1.qh": (15.718, 3.601),
    "L1.kh": (20.919, 3.835),
    "L1.vh": (24.568, 4.290),
    "L1.kT": (20.919, 3.835),
    "L1.scores": (31.875, 4.161),
    "L1.p": (3.763, 1.117),
    "L1.o": (25.771, 2.252),
    "L1.om": (25.771, 2.252),
    "L1.y": (25.056, 1.818),
    "L1.r1": (3.912, 0.312),
    "L1.h2": (10.930, 3.184),
    "L1.gate": (24.455, 3.739),
    "L1.up": (25.500, 3.624),
    "L1.act": (19.439, 1.275),
    "L1.down": (46.921, 4.317),
    "L1.out": (3.799, 0.405),
}


def _torch():
    import torch

    return torch


def build_model(nanogpt_dir, *, layers=N_LAYER, d=D, n_head=N_HEAD, t=T, hidden=HIDDEN, seed=SEED):
    """N stacked nanoGPT Extended Blocks (x -> Block_i(x)[0]), float64, eval."""
    torch = _torch()
    from torch import nn

    if str(nanogpt_dir) not in sys.path:
        sys.path.insert(0, str(nanogpt_dir))
    from model import GPTConfig
    from models.gpt import Block

    cfg = GPTConfig(block_size=t, n_layer=layers, n_head=n_head, n_embd=d, dropout=0.0, bias=False,
                    pos_embedding="none", norm_type="rmsnorm", activation="swiglu")
    blocks = []
    for i in range(layers):
        torch.manual_seed(seed + i)
        blk = Block(cfg)
        if hidden is not None:
            blk.ff.gate_proj = nn.Linear(d, hidden, bias=False)
            blk.ff.up_proj = nn.Linear(d, hidden, bias=False)
            blk.ff.down_proj = nn.Linear(hidden, d, bias=False)
        for norm in (blk.ln_1, blk.ln_2):
            nn.init.uniform_(norm.weight, 0.5, 1.5)
        blocks.append(blk.double().eval())

    class Stack(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = nn.ModuleList(blocks)

        def forward(self, x):
            for b in self.b:
                x = b(x)[0]
            return x

    return Stack().eval()


def sigmas(seed, n, lo_hi=TOL_SIGMA):
    """Per-input scales, log-uniform over lo_hi, from a stream of `seed` distinct from its N(0, 1) stream."""
    lo, hi = lo_hi
    return np.exp(np.random.default_rng((seed, 1)).uniform(math.log(lo), math.log(hi), n))[:, None, None]


def calib_inputs(n=N_CALIB):
    """Unit-variance calibration inputs z (CALIB_SEED); the contract calibrates on contract_calib()."""
    return np.random.default_rng(CALIB_SEED).standard_normal((n, T, D))


def contract_calib(n=N_CALIB):
    """The contract model's calibration set: calib_inputs() * sigma, sigma log-uniform over CALIB_SIGMA."""
    return calib_inputs(n) * sigmas(CALIB_SEED, n, CALIB_SIGMA)


def derive_inputs(n=N_DERIVE):
    """Derivation set over the domain: N(0, 1) * sigma, sigma log-uniform over TOL_SIGMA (DERIVE_SEED)."""
    return np.random.default_rng(DERIVE_SEED).standard_normal((n, T, D)) * sigmas(DERIVE_SEED, n)


def heldout_inputs(n):
    """Unit-variance held-out inputs z (HELDOUT_SEED): check only."""
    return np.random.default_rng(HELDOUT_SEED).standard_normal((n, T, D))


def heldout_domain(n):
    """Held-out inputs over the domain: heldout_inputs(n) * sigma, sigma log-uniform over TOL_SIGMA."""
    return heldout_inputs(n) * sigmas(HELDOUT_SEED, n)


def names(fg, layers=N_LAYER) -> dict:
    """FGraph node -> IR sym "L<i>.<seg>", found structurally from each Block's named modules (the exported
    graph numbers unnamed nodes like view_3 / add_5 globally, so names are not used for those)."""
    users = {n.name: fg.users(n) for n in fg.nodes}

    def one(node, op, pred=lambda u: True):
        c = [u for u in users[node.name] if u.op == op and pred(u)]
        if len(c) != 1:
            raise ValueError(f"{node.name}: expected one {op} user, found {[u.name for u in c]}")
        return c[0]

    out = {}
    for i in range(layers):
        p = f"b.{i}."
        seg = {"h1": fg.node(p + "ln_1"), "y": fg.node(p + "attn.c_proj"), "h2": fg.node(p + "ln_2"),
               "gate": fg.node(p + "ff.gate_proj"), "up": fg.node(p + "ff.up_proj"),
               "down": fg.node(p + "ff.down_proj")}
        for j, k in enumerate("qkv"):
            seg[k] = fg.node(f"{p}attn.c_attn.{j}")
            heads = [u for u in users[seg[k].name] if u.op == "split_heads"]
            seg[k + "h"] = heads[0] if heads else seg[k]
        seg["kT"] = one(seg["kh"], "transpose")
        seg["scores"] = one(seg["kT"], "matmul")
        seg["p"] = one(seg["scores"], "softmax")
        seg["o"] = one(seg["p"], "matmul")
        merged = [u for u in users[seg["o"].name] if u.op == "merge_heads"]
        seg["om"] = merged[0] if merged else seg["o"]
        seg["r1"] = one(seg["y"], "add")
        seg["act"] = one(seg["up"], "mul")
        seg["out"] = one(seg["down"], "add")
        for k, n in seg.items():
            out.setdefault(n.name, f"L{i}.{k}")
    return out


def quantize_model(model, calib=None, *, headroom=HEADROOM, inline=False, residual=RESIDUAL, **kw):
    """The contract model (calib=None: contract_calib(), i16 residual stream) or a variant."""
    torch = _torch()
    from .frontend import capture, quantize

    example = torch.zeros(1, T, D, dtype=torch.float64)
    fg = capture(model, (example,))
    calib = contract_calib() if calib is None else calib
    return quantize(fg, calib, headroom, names=names(fg), inline=inline, sym_name="model", residual=residual, **kw)


# ------------------------------------------------------------------------------------------- measurement

def segment_syms(q) -> list:
    """Every non-input IR value with a float reference (the per-segment contract's keys), graph order."""
    return [s for s in q.syms.values() if s != "x" and q.scales.get(s) is not None]


def _causal(shape):
    return np.broadcast_to(np.tril(np.ones(shape[-2:], dtype=bool)), shape)


def value_dtypes(module) -> dict:
    """IR sym -> dtype of every graph value."""
    from .ir import graph as g

    return {op.sym_str: g.qt(op.results[0]).dtype for op in g.funcs(module)[0].body.block.ops
            if op.results and op.sym_str is not None}


def unit_of(scale, dtype) -> float:
    """The contract's error unit for a value: its int8-equivalent LSB, scale * qmax(dtype) / 127, i.e. 1/127 of
    its calibrated range. For an i8 value this is its LSB (contract v1); an i16 residual value (ADR-004) has
    the same unit an i8 value calibrated to the same range would have, so TOL keeps its meaning."""
    return scale * {"i8": 127, "i16": 32767}[dtype] / 127


def out_unit(q, out=OUT) -> float:
    """q's own int8-equivalent output LSB (= OUT_UNIT for the contract model)."""
    return unit_of(q.scales[out], value_dtypes(q.module)[out])


def quantize_x(q, x_float):
    """Float inputs -> integer inputs at the input's scale, saturating at its dtype (i8 or i16)."""
    lo, hi = {"i8": (-128, 127), "i16": (-32768, 32767)}[value_dtypes(q.module)["x"]]
    return np.clip(np.rint(np.asarray(x_float) / q.scales["x"]), lo, hi).astype(np.int64)


def measure(q, x_float, *, chunk=4096, out="L1.out", segments=True, unit=None):
    """Quantize x_float at the input scale, run the batched graph interpreter and the float graph on
    dequant(x_q) (input-quantization-free, as contract 8) chunk by chunk ->
    (per-input max |error| of `out` in `unit`, {segment: (gain, RMS)}, {segment: peak |ref|}).
    `unit` defaults to the contract's fixed OUT_UNIT for the contract output (so models calibrated differently
    are held to the same absolute accuracy), else q's own int8-equivalent LSB of `out`. Segment RMS and peaks
    are in each segment's own int8-equivalent LSB (unit_of: the LSB for i8 values, scale * 32767 / 127 for i16).
    Gain = sum(dequant * ref) / sum(ref^2); scores compared on causal entries only."""
    from .interp import graph as gi

    sx = q.scales["x"]
    keys = segment_syms(q) if segments else [out]
    dts = value_dtypes(q.module)
    if unit is None:
        unit = OUT_UNIT if out == OUT and OUT_UNIT is not None else out_unit(q, out)
    num = {k: 0.0 for k in keys}
    den = {k: 0.0 for k in keys}
    sq = {k: 0.0 for k in keys}
    cnt = {k: 0 for k in keys}
    peak = {k: 0.0 for k in keys}
    errs = []
    for lo in range(0, len(x_float), chunk):
        xq = quantize_x(q, x_float[lo:lo + chunk])
        got = gi.run(q.module, {"x": xq}, q.consts, batch=True).values
        ref = q.reference(xq * sx)
        so = q.scales[out]
        errs.append(np.abs(got[out] * so - ref[out]).reshape(len(xq), -1).max(axis=1) / unit)
        for k in keys:
            s = q.scales[k]
            gq, r = got[k] * s, ref[k]
            if k.endswith(".scores"):
                m = _causal(r.shape)
                gq, r = gq[m], r[m]
            num[k] += float((gq * r).sum())
            den[k] += float((r * r).sum())
            sq[k] += float(((gq - r) ** 2).sum())
            cnt[k] += r.size
            peak[k] = max(peak[k], float(np.abs(r).max()) / s)
    unit = {k: unit_of(q.scales[k], dts[k]) for k in keys}
    stats = {k: (num[k] / den[k], math.sqrt(sq[k] / cnt[k]) / unit[k]) for k in keys}
    return np.concatenate(errs), stats, {k: v * q.scales[k] / unit[k] for k, v in peak.items()}


def input_segment_stats(q, got, ref, keys=None) -> dict:
    """Per-input segment statistics of a batch: {segment: (gain_i, rms_i, norm_i)} arrays over the batch, from
    integer values `got` and float references `ref` (both {sym: array with a leading batch dim}): the
    least-squares gain, the RMS error and the reference's norm sqrt(sum ref^2), the last two in the segment's own
    int8-equivalent LSB. The single-input form of measure's pooled statistics; compiled runs (G1/G2) are checked
    with it against INPUT_SEGMENTS (A2-M1)."""
    dts = value_dtypes(q.module)
    out = {}
    for k in (keys if keys is not None else [k for k in segment_syms(q) if k in got]):
        s = q.scales[k]
        gq, r = np.asarray(got[k]) * s, np.asarray(ref[k])
        n = len(r)
        if k.endswith(".scores"):
            m = _causal(r.shape)
            gq, r = np.where(m, gq, 0.0), np.where(m, r, 0.0)
            cnt = m.reshape(n, -1).sum(axis=1)
        else:
            cnt = r[0].size
        gq, r = gq.reshape(n, -1), r.reshape(n, -1)
        den = (r * r).sum(axis=1)
        gain = np.where(den > 0, (gq * r).sum(axis=1) / np.where(den > 0, den, 1), 1.0)
        u = unit_of(s, dts[k])
        out[k] = (gain, np.sqrt(((gq - r) ** 2).sum(axis=1) / cnt) / u, np.sqrt(den) / u)
    return out


def gain_projection(gain, norm, k) -> np.ndarray:
    """(gain_i - SEGMENTS gain) * norm_i: the part of input i's error along its own reference, in LSB. Rounding
    noise is nearly uncorrelated with the reference, so this stays small; a gain error grows it with the
    signal (unlike the raw per-input gain, which is noisy on small-signal inputs)."""
    return (np.asarray(gain) - SEGMENTS[k][0]) * np.asarray(norm)


def input_segment_errors(stats) -> dict:
    """Per-input violations of INPUT_SEGMENTS: {segment: message} for any input whose gain projection
    |gain_projection| exceeds INPUT_SEGMENTS[k][0] or whose RMS error exceeds INPUT_SEGMENTS[k][1]."""
    bad = {}
    for k, (g, r, nrm) in stats.items():
        if k not in INPUT_SEGMENTS:
            raise AssertionError(f"segment {k} has no per-input bound")
        pmax, rmax = INPUT_SEGMENTS[k]
        p = np.abs(gain_projection(g, nrm, k))
        if p.max() > pmax or np.max(r) > rmax:
            i = int(np.argmax(np.maximum(p / pmax, np.asarray(r) / rmax)))
            bad[k] = (f"input {i}: gain {np.asarray(g)[i]:.4f} (contract {SEGMENTS[k][0]}; projection "
                      f"{p[i]:.2f} > {pmax} LSB?), RMS {np.asarray(r)[i]:.3f} (max {rmax})")
    return bad


def footprint(prog) -> dict:
    """Image words of a lowered program with one word per element and no buffer reuse (SG4b's planner can only
    shrink it): consts (weights + LUTs), activations (graph values with storage, views excluded), kernel temps."""
    consts = set(prog.const_values) | {s for s in prog.buffers if s not in prog.inputs and
                                       not any(s in k.writes for k in prog.kernels)}
    size = {s: math.prod(shape) for s, (shape, _) in prog.buffers.items()}
    temps = {s: math.prod(shape) for k in prog.kernels for s, (shape, _) in k.temps.items()}
    out = {"consts": sum(v for s, v in size.items() if s in consts),
           "activations": sum(v for s, v in size.items() if s not in consts),
           "temps": sum(temps.values())}
    out["total"] = sum(out.values())
    out["addr_bits"] = max(1, (out["total"] - 1).bit_length())
    return out


def segment_errors(stats) -> dict:
    """Segments whose gain or RMS error leaves the contract's bounds (contract 8's rule over SEGMENTS)."""
    if set(stats) != set(SEGMENTS):
        raise AssertionError(f"segments {sorted(set(stats) ^ set(SEGMENTS))} differ from the contract's")
    return {k: f"gain {g:.4f} (contract {g0}), RMS {r:.3f} LSB (contract {r0})" for k, (g, r) in stats.items()
            for g0, r0 in [SEGMENTS[k]] if abs(g - g0) > GAIN_TOL or r > r0 * RMS_TOL}


def nongaussian_sets(n, seed=HELDOUT_SEED + 1) -> dict:
    """The contract's other input families (never used for calibration or derivation): sigma at both ends of
    TOL_SIGMA, and Laplace, Student-t(5) and each channel in turn x4, at unit variance and at the small end."""
    rng = np.random.default_rng(seed)
    shape = (n, T, D)
    gain = np.ones((n, 1, D))
    gain[np.arange(n), 0, np.arange(n) % D] = 4
    out = {f"sigma{TOL_SIGMA[0]}": rng.standard_normal(shape) * TOL_SIGMA[0],
           f"sigma{TOL_SIGMA[1]}": rng.standard_normal(shape) * TOL_SIGMA[1]}
    for s in (1.0, TOL_SIGMA[0]):
        sfx = "" if s == 1.0 else f"_sigma{s}"
        out.update({"laplace" + sfx: rng.laplace(size=shape) / math.sqrt(2) * s,
                    "student_t5" + sfx: rng.standard_t(5, size=shape) / math.sqrt(5 / 3) * s,
                    "channel_x4" + sfx: rng.standard_normal(shape) * gain * s})
    return out


def main(argv=None):
    """python3 -m anvil.model [--heldout N]: quantize the target model and print the contract's held-out checks
    (the derivation itself is anvil.model_derive)."""
    import argparse
    import os

    from .lower.graph_to_tensor import lower

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--heldout", type=int, default=200_000)
    ap.add_argument("--other", type=int, default=20_000)
    a = ap.parse_args(argv)
    m = build_model(os.environ["NANOGPT_DIR"])
    q = quantize_model(m)
    print("footprint", footprint(lower(q.module)), "bounds max", max(q.bounds.values()),
          f"OUT_UNIT {out_unit(q)!r} (frozen {OUT_UNIT!r})")
    err, stats, peak = measure(q, heldout_domain(a.heldout))
    print(f"held-out domain {a.heldout} (sigma log-uniform over {TOL_SIGMA}): max {err.max():.3f} (TOL {TOL_LSB}), "
          f"over TOL {int((err > TOL_LSB).sum())}, peak {max(v for k, v in peak.items() if not k.endswith('.p')):.1f}, "
          f"segment violations {segment_errors(stats)}", flush=True)
    for s in (0.02, 0.05, 0.2, 1.0, 3.0):
        e = measure(q, heldout_inputs(a.other) * s, segments=False)[0]
        print(f"held-out sigma {s} {a.other}: max {e.max():.3f}", flush=True)
    for name, x in nongaussian_sets(a.other).items():
        e = measure(q, x, segments=False)[0]
        print(f"{name} {a.other}: max {e.max():.3f}", flush=True)
    for s in (0.02, 0.05):
        qm = quantize_model(m, calib=calib_inputs() * s)
        e = measure(qm, heldout_inputs(a.other) * s, segments=False)[0]
        print(f"matched calibration sigma {s} {a.other}: max {e.max():.3f} "
              f"(own output LSB {e.max() * OUT_UNIT / out_unit(qm):.3f})", flush=True)


if __name__ == "__main__":
    main()
