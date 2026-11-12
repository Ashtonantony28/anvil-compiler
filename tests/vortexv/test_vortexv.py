"""Anvil -> Vortex-V (RV64IM + Xmlacc): every value of every graph bit-exact to the golden interpreter on
Vortex-V's own instruction-set simulator. RTL runs are in test_vortexv_rtl.py (-m rtl)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent))
import vv_graphs as vg  # noqa: E402

from anvil import semantics as sem  # noqa: E402
from anvil import vortexv  # noqa: E402
from anvil.interp import graph as gi  # noqa: E402

pytestmark = pytest.mark.skipif(not vortexv.available(), reason="vortexv submodule not checked out")

RANDOM_SEEDS = range(24)
MIN_MACC_SPEEDUP = 1.5     # instructions without macc / with macc on the fixture blocks (measured 1.9 to 2.4x)


def _check(res, want):
    for k, v in want.values.items():
        assert k in res.values, f"value {k!r} not decoded"
        np.testing.assert_array_equal(res.values[k], np.asarray(v), err_msg=k)
    assert len(res.outputs) == len(want.outputs)
    for a, b in zip(res.outputs, want.outputs):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("block,i", vg.FIXTURES)
def test_fixture_blocks_bit_exact(block, i):
    m, c, x = vg.fixture_case(block, i)
    _check(vortexv.compile(m, c).run(x), gi.run(m, x, c))


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_random_graphs_bit_exact(seed):
    m, c, x, want = vg.random_case(seed)
    _check(vortexv.compile(m, c).run(x), want)


@pytest.mark.parametrize("block", vg.BLOCKS)
def test_floor_mutant_is_caught(block):
    """rqs replaced by a plain arithmetic shift (no rounding, no clamp) must change some value."""
    m, c, x = vg.fixture_case(block, 0)
    res, want = vortexv.compile(m, c, mutant="floor").run(x), gi.run(m, x, c)
    assert any(not np.array_equal(res.values[k], v) for k, v in want.values.items())


@pytest.mark.parametrize("block", vg.BLOCKS)
def test_macc_baseline_same_answers_more_instructions(block):
    m, c, x = vg.fixture_case(block, 0)
    fast, slow = vortexv.compile(m, c).run(x), vortexv.compile(m, c, mutant="scalar").run(x)
    _check(slow, gi.run(m, x, c))
    assert slow.instret >= MIN_MACC_SPEEDUP * fast.instret, (slow.instret, fast.instret)


def test_transposes_elided_or_materialised():
    """attention: kT is read in place by scores (no buffer); p @ v transposes v into a scratch buffer."""
    m, c, x = vg.fixture_case("attention", 0)
    p = vortexv.compile(m, c)
    assert "kT" not in p.syms and "kT" in p.views
    assert "v.T" in p.bufs and "w_q.T" in p.bufs


def test_image_matches_iss_layout():
    """The (stub, data) image for Vortex-V's RTL harness holds the program and the operands."""
    m, c, x = vg.fixture_case("mlp_block", 0)
    p = vortexv.compile(m, c)
    stub, data = p.image(x)
    mem = {}
    for a, n, v in data:
        for i in range(n):
            mem[a + i] = (v >> (8 * i)) & 0xFF
    word0 = sum(mem[vortexv.CODE_BASE + i] << (8 * i) for i in range(4))
    assert word0 == p.words[0] and len(stub) == 3
    xb = p.inputs["x"]
    assert np.array_equal(xb.unpack(bytes(mem.get(xb.addr + i, 0) for i in range(xb.nbytes))), x["x"])


I32 = (-(1 << 31), (1 << 31) - 1)


@settings(max_examples=3000, deadline=None)
@given(st.integers(*I32), st.integers(0, 1 << 31), st.booleans(), st.integers(0, 31))
def test_rqs_is_anvil_requant(acc, m_raw, neg, s):
    """Inside the golden model's int32 contract, mlacc.rqs(acc*M, s) == xf.golden.requant(acc, M, s).
    M is drawn so acc*M mostly stays inside int32 (the contract), including its edges."""
    rqs = vortexv._vv.xmlacc_model().rqs
    m = (m_raw % (I32[1] // max(1, abs(acc)) + 1)) * (-1 if neg else 1)
    try:
        want = int(sem.requant(np.int64(acc), m, s))
    except OverflowError:
        return
    got, _ = rqs((acc * m) & ((1 << 64) - 1), s)
    assert got == want, (acc, m, s)


def test_two_layer_model_bit_exact():
    """The SG5a model (2 Blocks, d=32, 2 heads, T=16, i16 residual stream; frozen graph in data/codesign):
    all 75 values bit-exact on the ISS."""
    from anvil.codesign import workloads
    w = workloads.load("model")
    m, c = w.model
    x = {"x": w.x}
    want = gi.run(m, x, c)
    assert len(want.values) == 75
    _check(vortexv.compile(m, c).run(x), want)
