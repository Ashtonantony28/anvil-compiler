"""SG5a: the target model (2 nanoGPT Extended Blocks, d=32, 2 heads, T=16) captured, quantized, proven
overflow-free, lowered with default schedules bit-exactly (C4 level 2), and within the derived contract
(docs/model-contract.md) on held-out inputs from a seed never used for calibration or derivation (C4 level 6)."""
import os

import numpy as np
import pytest

from anvil import model as M
from anvil.interp import graph as gi

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def target():
    d = os.environ.get("NANOGPT_DIR")
    if not d:
        pytest.fail("torch tests need NANOGPT_DIR (nanoGPT Extended @ 03dbab8)")
    m = M.build_model(d)
    return m, M.quantize_model(m)


def test_structure_overflow_and_footprint(target):
    from anvil.ir import graph as g
    from anvil.lower.graph_to_tensor import lower

    m, q = target
    ops = [n.op for n in q.graph.nodes]
    assert ops.count("split_heads") == 3 * M.N_LAYER and ops.count("merge_heads") == M.N_LAYER
    assert ops.count("add") == 2 * M.N_LAYER and ops.count("softmax") == M.N_LAYER
    assert M.HELDOUT_SEED not in (M.CALIB_SEED, M.DERIVE_SEED)
    assert m.b[0].ff.gate_proj.weight.shape == (M.HIDDEN, M.D) and m.b[0].attn.n_head == M.N_HEAD
    func = g.funcs(q.module)[0]
    requant = [op.sym_str for op in func.body.block.ops if hasattr(op, "M") or hasattr(op, "Ma")
               or isinstance(op, g.SoftmaxLutOp)]                   # every int32 intermediate is proven
    assert set(requant) == set(q.bounds) and max(q.bounds.values()) <= 2**31 - 1
    dims = {op.sym_str: g.qt(op.results[0]).dims for op in func.body.block.ops if getattr(op, "sym_str", None)}
    assert dims["L0.scores"] == [M.N_HEAD, M.T, M.T] and dims["L1.o"] == [M.N_HEAD, M.T, M.D // M.N_HEAD]
    assert set(M.segment_syms(q)) == set(M.SEGMENTS)
    fp = M.footprint(lower(q.module))
    assert fp["total"] == M.FOOTPRINT_WORDS and fp["addr_bits"] == M.ADDR_BITS
    assert fp["total"] <= 1 << 16 and fp["total"] > 1 << 12          # needs, and fits, 16-bit addresses


def test_float_graph_equals_module(target):
    m, q = target
    xs = torch.as_tensor(np.random.default_rng(5).standard_normal((64, M.T, M.D)))
    with torch.no_grad():
        got, want = q.graph.evaluate([xs])[q.graph.outputs[0].name], m(xs)
    assert torch.allclose(got, want, rtol=0, atol=1e-12)


def test_graph_equals_tensor_under_default_schedules(target):
    """C4 level 2 on the whole model: every kernel default-scheduled (block/lane, a per-thread seq loop beyond
    255 threads), finalized to canonical form, and interpreted with the race check; every value bit-exact."""
    from anvil.lower.graph_to_tensor import lower
    from anvil.schedule import Target, templates

    _, q = target
    prog = lower(q.module)
    kernels = {k.name: templates.default_schedule(k, Target()).finalize() for k in prog.kernels}
    rng = np.random.default_rng(31)
    sx = q.scales["x"]
    xs = [np.clip(np.rint(rng.standard_normal((M.T, M.D)) / sx), -128, 127).astype(np.int64),
          rng.integers(-128, 128, (M.T, M.D))]                     # in-distribution, and uniform full range
    for x in xs:
        ref = gi.run(q.module, {"x": x}, q.consts)
        got = prog.run({"x": x}, q.consts, kernels)
        bad = [k for k, v in ref.values.items() if not np.array_equal(got[k], v)]
        assert not bad, bad
        assert np.array_equal(got[prog.outputs[0]], ref.outputs[0])


def test_tolerance_and_segments_on_heldout(target):
    """20k fresh-seed N(0, 1) inputs: output error <= TOL, no saturation, every segment within its bounds."""
    _, q = target
    err, stats, peak = M.measure(q, M.heldout_inputs(20_000))
    assert err.max() <= M.TOL_LSB, f"{int((err > M.TOL_LSB).sum())} inputs over TOL, max {err.max():.3f} LSB"
    assert max(v for k, v in peak.items() if not k.endswith(".p")) <= M.PEAK_LSB, peak
    bad = M.segment_errors(stats)
    assert not bad, bad


def test_tolerance_on_other_input_families(target):
    _, q = target
    for name, x in M.nongaussian_sets(4000).items():
        err = M.measure(q, x, segments=False)[0]
        assert err.max() <= M.TOL_LSB, f"{name}: max {err.max():.3f} LSB"


def test_segment_check_catches_a_gain_error(target):
    """The segment bounds are not vacuous: a 1 % gain error on one segment fails them (AUDIT-7 H2 style)."""
    _, q = target
    _, stats, _ = M.measure(q, M.heldout_inputs(2000))
    g, r = stats["L1.act"]
    assert not M.segment_errors(stats)
    assert "L1.act" in M.segment_errors({**stats, "L1.act": (g * 1.01, r)})
