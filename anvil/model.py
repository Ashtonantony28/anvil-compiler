"""SG5a target model: two stacked nanoGPT Extended Blocks at d=32, 2 heads, T=16 (design D4), quantized by the
frontend, plus the measurement helpers behind its derived tolerance (docs/model-contract.md).

    m = build_model(nanogpt_dir)                 # torch Stack of N_LAYER Blocks, float64, eval, seeded
    q = quantize_model(m)                        # anvil Quantized (calibrated on calib_inputs())
    err, stats, peak = measure(q, heldout_inputs())   # per-input max error (LSB), segment gain/RMS, peaks

Model (D4): `GPTConfig(n_embd=32, n_head=2, block_size=16, bias=False, pos_embedding='none', norm_type='rmsnorm',
activation='swiglu')`, Blocks seeded SEED + layer. Like xf/export.py::build_block, the SwiGLU hidden width is
overridden (HIDDEN = 2 * d = 64; nanoGPT's own rounding to a multiple of 256 would give 256, quadrupling the MLP
weights and pushing the image past 16-bit addresses) and the RMSNorm weights are drawn from U(0.5, 1.5) so they
are not all ones. Data: calibration on 2^16 N(0, 1) inputs from CALIB_SEED (xf's attention recipe), the tolerance
derived on DERIVE_SEED inputs, and HELDOUT_SEED never used for anything but the final check (AUDIT-6 L1).

Import does not need torch; the builders import it lazily.
"""
from __future__ import annotations

import math
import sys

import numpy as np

D, N_HEAD, T, N_LAYER, HIDDEN = 32, 2, 16, 2, 64
SEED = 0                      # torch.manual_seed(SEED + layer) before each Block
CALIB_SEED, N_CALIB = 0, 1 << 16
DERIVE_SEED, N_DERIVE = 1, 200_000
HELDOUT_SEED = 20261106       # fresh: never used for calibration, headroom or tolerance derivation
HEADROOM = 1.5
EPS = 1e-6

# per-layer IR syms "L<i>.<seg>": h1 q k v qh kh vh kT scores p o om y r1 h2 gate up act down out (names())


# --------------------------------------------------------------------------------- SG5b contract (frozen)
# docs/model-contract.md. Output = "L1.out" (the second Block's residual), errors in its LSB, input-quantization
# free (the float model runs on dequant(x_q)), on inputs x ~ sigma * N(0, 1) per element with sigma in
# TOL_SIGMA (also Laplace / Student-t(5) at unit variance and any one channel x4).
OUT = "L1.out"
TOL_LSB = 6
TOL_SIGMA = (0.5, 3.0)
FOOTPRINT_WORDS = 50_048           # footprint(lower(q.module))["total"]: no reuse, one element per word
ADDR_BITS = 16
GAIN_TOL, RMS_TOL = 0.0025, 1.1    # contract 8's: |gain - SEGMENTS[k][0]| <= GAIN_TOL, RMS <= SEGMENTS[k][1] * RMS_TOL
PEAK_LSB = 127.5                   # no non-softmax activation saturates on held-out inputs
# (least-squares gain, RMS error in LSB) per segment on the 200k DERIVE_SEED inputs
SEGMENTS = {
    "L0.h1": (1.0009, 0.294),
    "L0.q": (1.0009, 0.465),
    "L0.k": (1.0013, 0.446),
    "L0.v": (1.0005, 0.465),
    "L0.qh": (1.0009, 0.465),
    "L0.kh": (1.0013, 0.446),
    "L0.vh": (1.0005, 0.465),
    "L0.kT": (1.0013, 0.446),
    "L0.scores": (1.0019, 0.539),
    "L0.p": (1.0001, 0.262),
    "L0.o": (0.9980, 0.390),
    "L0.om": (0.9980, 0.390),
    "L0.y": (0.9981, 0.430),
    "L0.r1": (1.0001, 0.326),
    "L0.h2": (1.0003, 0.396),
    "L0.gate": (1.0002, 0.581),
    "L0.up": (1.0004, 0.589),
    "L0.act": (1.0005, 0.358),
    "L0.down": (1.0004, 1.118),
    "L0.out": (1.0003, 0.464),
    "L1.h1": (1.0000, 0.508),
    "L1.q": (0.9999, 0.664),
    "L1.k": (1.0002, 0.681),
    "L1.v": (0.9999, 0.686),
    "L1.qh": (0.9999, 0.664),
    "L1.kh": (1.0002, 0.681),
    "L1.vh": (0.9999, 0.686),
    "L1.kT": (1.0002, 0.681),
    "L1.scores": (1.0001, 0.802),
    "L1.p": (1.0000, 0.315),
    "L1.o": (0.9999, 0.496),
    "L1.om": (0.9999, 0.496),
    "L1.y": (1.0001, 0.503),
    "L1.r1": (1.0002, 0.545),
    "L1.h2": (0.9993, 0.599),
    "L1.gate": (0.9992, 0.741),
    "L1.up": (0.9995, 0.751),
    "L1.act": (0.9965, 0.400),
    "L1.down": (0.9964, 1.157),
    "L1.out": (1.0002, 0.640),
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


def calib_inputs(n=N_CALIB):
    return np.random.default_rng(CALIB_SEED).standard_normal((n, T, D))


def derive_inputs(n=N_DERIVE):
    return np.random.default_rng(DERIVE_SEED).standard_normal((n, T, D))


def heldout_inputs(n):
    return np.random.default_rng(HELDOUT_SEED).standard_normal((n, T, D))


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


def quantize_model(model, calib=None, *, headroom=HEADROOM, inline=False):
    torch = _torch()
    from .frontend import capture, quantize

    example = torch.zeros(1, T, D, dtype=torch.float64)
    fg = capture(model, (example,))
    calib = calib_inputs() if calib is None else calib
    return quantize(fg, calib, headroom, names=names(fg), inline=inline, sym_name="model")


# ------------------------------------------------------------------------------------------- measurement

def segment_syms(q) -> list:
    """Every non-input IR value with a float reference (the per-segment contract's keys), graph order."""
    return [s for s in q.syms.values() if s != "x" and q.scales.get(s) is not None]


def _causal(shape):
    return np.broadcast_to(np.tril(np.ones(shape[-2:], dtype=bool)), shape)


def measure(q, x_float, *, chunk=4096, out="L1.out", segments=True):
    """Quantize x_float at the input scale, run the batched graph interpreter and the float graph on
    dequant(x_q) (input-quantization-free, as contract 8) chunk by chunk ->
    (per-input max |error| of `out` in its LSB, {segment: (gain, RMS LSB)}, {segment: peak |ref| in LSB}).
    Gain = sum(dequant * ref) / sum(ref^2); scores compared on causal entries only."""
    from .interp import graph as gi

    sx = q.scales["x"]
    keys = segment_syms(q) if segments else [out]
    num = {k: 0.0 for k in keys}
    den = {k: 0.0 for k in keys}
    sq = {k: 0.0 for k in keys}
    cnt = {k: 0 for k in keys}
    peak = {k: 0.0 for k in keys}
    errs = []
    for lo in range(0, len(x_float), chunk):
        xq = np.clip(np.rint(x_float[lo:lo + chunk] / sx), -128, 127).astype(np.int64)
        got = gi.run(q.module, {"x": xq}, q.consts, batch=True).values
        ref = q.reference(xq * sx)
        so = q.scales[out]
        errs.append(np.abs(got[out] * so - ref[out]).reshape(len(xq), -1).max(axis=1) / so)
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
    stats = {k: (num[k] / den[k], math.sqrt(sq[k] / cnt[k]) / q.scales[k]) for k in keys}
    return np.concatenate(errs), stats, peak


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
    TOL_SIGMA, Laplace and Student-t(5) at unit variance, and each channel in turn x4."""
    rng = np.random.default_rng(seed)
    shape = (n, T, D)
    gain = np.ones((n, 1, D))
    gain[np.arange(n), 0, np.arange(n) % D] = 4
    return {f"sigma{TOL_SIGMA[0]}": rng.standard_normal(shape) * TOL_SIGMA[0],
            f"sigma{TOL_SIGMA[1]}": rng.standard_normal(shape) * TOL_SIGMA[1],
            "laplace": rng.laplace(size=shape) / math.sqrt(2),
            "student_t5": rng.standard_t(5, size=shape) / math.sqrt(5 / 3),
            "channel_x4": rng.standard_normal(shape) * gain}


def main(argv=None):
    """python3 -m anvil.model [--heldout N]: quantize the target model and print the contract's measurements."""
    import argparse
    import os

    from .lower.graph_to_tensor import lower

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--heldout", type=int, default=200_000)
    ap.add_argument("--other", type=int, default=20_000)
    a = ap.parse_args(argv)
    q = quantize_model(build_model(os.environ["NANOGPT_DIR"]))
    print("footprint", footprint(lower(q.module)), "bounds max", max(q.bounds.values()))
    err, stats, peak = measure(q, heldout_inputs(a.heldout))
    print(f"held-out {a.heldout}: max {err.max():.3f} LSB (TOL {TOL_LSB}), over TOL {int((err > TOL_LSB).sum())}, "
          f"peak {max(v for k, v in peak.items() if not k.endswith('.p')):.1f} LSB, "
          f"segment violations {segment_errors(stats)}")
    for name, x in nongaussian_sets(a.other).items():
        e, _, pk = measure(q, x, segments=False)
        print(f"{name} {a.other}: max {e.max():.3f} LSB")


if __name__ == "__main__":
    main()
