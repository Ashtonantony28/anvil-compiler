"""SG4a acceptance: nanoGPT's Block, captured and quantized by anvil.frontend, reproduces the xf fixture images
(weights, LUTs, M, s, scales) and, through the graph interpreter, the fixture goldens bit-exactly; float
tolerance vs PyTorch and per-segment gain/RMS bounds (C4 level 6)."""
import types

import numpy as np
import pytest

from anvil.frontend import from_image
from anvil.interp import graph as gi
from anvil.ir import graph as g
from anvil.ir import parse_ir, print_ir
from xf import attention, export as xe, fixture

from conftest import GOLDEN, OUT

pytestmark = pytest.mark.torch
BLOCKS = ("mlp_block", "attention")
PROPS = ("M", "s", "Ma", "Mb", "one", "causal")


def _load(block, i=0):
    return fixture.load(fixture.FIXTURE_DIR / f"{block}_{i}.json")


def _ops(module):
    return list(gi._func(module).body.block.ops)


def _const(op, consts):
    if op.value is not None:
        return np.asarray(op.value.get_values(), dtype=np.int64).reshape(g.qt(op.result).dims)
    return np.asarray(consts[op.sym_str])


@pytest.mark.parametrize("block", BLOCKS)
def test_graph_equals_fixture_image(block, quantized):
    """Op for op, the torch-built graph equals the one built by hand from the fixture image: same op kinds,
    syms, types (shape, dtype, scale bit-exact), requant properties and constant values."""
    img = _load(block)["image"]
    ref_module, ref_consts = from_image.BLOCKS[block](img)
    q = quantized[block]
    mine, ref = _ops(q.module), _ops(ref_module)
    assert [op.name for op in mine] == [op.name for op in ref]
    for a, b in zip(mine, ref):
        if isinstance(a, g.OutputOp):
            continue
        assert g.qt(a.results[0]) == g.qt(b.results[0]), (a.sym_str, b.sym_str)
        if isinstance(a, g.ConstOp):
            assert np.array_equal(_const(a, q.consts), _const(b, ref_consts)), (a.sym_str, b.sym_str)
            continue
        assert a.sym_str == b.sym_str
        for p in PROPS:
            assert a.properties.get(p) == b.properties.get(p), (a.sym_str, p)
        if isinstance(a, g.TransposeOp):
            assert a.perm_list == b.perm_list
    for seg, s in img.segments.items():                       # every activation scale, bit-exact
        if seg in q.scales and s["scale"] is not None:
            assert q.scales[seg] == s["scale"], seg


@pytest.mark.parametrize("block", BLOCKS)
def test_constants_equal_image_segments(block, quantized):
    """Weights and LUTs by image segment name; (M, s) from the image's scalar segments."""
    img = _load(block)["image"]
    q = quantized[block]
    if block == "mlp_block":
        pairs = {"h.w": "norm_w", "h.rsqrt_lut": "rsqrt_lut", "gate.w": "w_gate", "up.w": "w_up",
                 "act.silu_lut": "silu_lut", "down.w": "w_down"}
    else:
        pairs = {"h.w": "norm_w", "h.rsqrt_lut": "rsqrt_lut", "q.w": "w_q", "k.w": "w_k", "v.w": "w_v",
                 "p.exp_lut": "exp_lut", "y.w": "w_o"}
    assert set(q.consts) == set(pairs)
    for mine, seg in pairs.items():
        assert np.array_equal(q.consts[mine], img.read(seg)), mine
    ops = {op.sym_str: op for op in _ops(q.module) if not isinstance(op, (g.OutputOp, g.ConstOp))}
    rq = {"h": "norm"} | ({"out": "res"} if block == "mlp_block" else {})
    for sym, op in ops.items():
        pre = rq.get(sym, sym)
        for p in ("M", "s"):
            if p in op.properties:
                assert op.properties[p].value.data == int(img.read(f"{pre}_{p}")[0]), (sym, p)
        if "Ma" in op.properties:
            assert [op.properties[k].value.data for k in ("Ma", "Mb", "s")] == \
                   [int(img.read(f"res_{k}")[0]) for k in ("Ma", "Mb", "s")]
        if "one" in op.properties:
            assert op.properties["one"].value.data == int(img.read("one")[0])


@pytest.mark.parametrize("block", BLOCKS)
@pytest.mark.parametrize("i", range(4))
def test_interprets_to_fixture_golden(block, i, quantized):
    d = _load(block, i)
    q = quantized[block]
    got = gi.run(q.module, {"x": d["inputs"]["x"]}, q.consts)
    for k, v in d["golden"].items():
        assert np.array_equal(got.values[k], v), k
    for k, v in GOLDEN[block](d["image"], d["inputs"]["x"]).items():
        assert np.array_equal(got.values[k], v), k
    assert np.array_equal(got.outputs[0], d["golden"][OUT[block]])


@pytest.mark.parametrize("block", BLOCKS)
def test_overflow_proof_equals_xf_check_bounds(block, quantized):
    img = _load(block)["image"]
    q = quantized[block]
    if block == "mlp_block":
        want = {{"norm": "h", "res": "out"}.get(k, k): v for k, v in xe.check_bounds(img).items()}
    else:
        want = {{"softmax": "p"}.get(k, k): v for k, v in attention.check_bounds(img).items()}
    assert q.bounds == want


@pytest.mark.parametrize("block", BLOCKS)
def test_inline_round_trip(block, modules):
    import torch

    from anvil.frontend import from_torch

    from conftest import NAMES, calib

    q = from_torch(modules[block], (torch.zeros(1, xe.T, xe.N_EMBD, dtype=torch.float64),), calib(block)[:512],
                   names=NAMES[block], inline=True)
    assert q.consts == {}
    text = print_ir(q.module)
    again = parse_ir(text)
    assert print_ir(again) == text
    x = np.random.default_rng(3).integers(-128, 128, (xe.T, xe.N_EMBD))
    assert np.array_equal(gi.run(again, {"x": x}).outputs[0], gi.run(q.module, {"x": x}).outputs[0])


def _run_batch(q, xq):
    vals = [gi.run(q.module, {"x": x}, q.consts).values for x in xq]
    return {k: np.stack([v[k] for v in vals]) for k in vals[0]}


@pytest.mark.parametrize("block", BLOCKS)
def test_float_tolerance_vs_pytorch(block, quantized, modules):
    """C4 level 6: |dequant(graph) - module(dequant(x_q))| <= the fixture tol on xf's seeded check inputs, and the
    captured float graph equals the module itself."""
    import torch

    q, d = quantized[block], _load(block)
    n = xe.N_CHECK if block == "mlp_block" else xe.ATTN_N_CHECK
    sx, out = q.scales["x"], OUT[block]
    xq = xe.quantize(np.random.default_rng(xe.SEED + 1).standard_normal((n, xe.T, xe.N_EMBD)), sx)
    with torch.no_grad():
        ref = modules[block](torch.as_tensor(xq * sx)).numpy()
    assert np.abs(q.reference(xq * sx)[out] - ref).max() <= 1e-12
    got = _run_batch(q, xq)[out] * q.scales[out]
    err = np.abs(got - ref).max(axis=(1, 2)) / q.scales[out]
    tol_lsb = d["tol"] / q.scales[out]
    assert err.max() <= tol_lsb, f"{block}: {err.max():.3f} LSB > {tol_lsb:.3f}"


@pytest.mark.parametrize("block", BLOCKS)
def test_segment_gain_and_rms(block, quantized):
    """Every non-input segment's least-squares gain and RMS error against its float activation stays within
    contract 8's bounds (xf export.SEGMENTS) on held-out inputs (the first 20k of xf's 200k set)."""
    q = quantized[block]
    sx = q.scales["x"]
    xq = xe.quantize(np.random.default_rng(xe.HELDOUT_SEED).standard_normal((xe.HELDOUT_N, xe.T, xe.N_EMBD))[:20_000],
                     sx)
    got = {k: v for k, v in _run_batch(q, xq).items() if k in xe.SEGMENTS[block]}
    ref = q.reference(xq * sx)
    img = types.SimpleNamespace(segments={k: {"scale": q.scales[k]} for k in got})
    stats = xe.segment_stats(img, got, ref)
    assert not xe.segment_errors(block, stats), stats
    peak = {k: float(np.abs(v).max()) / q.scales[k] for k, v in ref.items() if k in got and k != "p"}
    assert max(peak.values()) <= 127.5, peak
