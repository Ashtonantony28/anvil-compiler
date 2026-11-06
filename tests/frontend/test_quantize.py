"""SG4a quantization primitives without torch (runs in `make check`): fixed_point and the LUT builders equal
xf/export.py's, and the overflow proof equals xf's check_bounds on the fixture graphs and rejects overflow."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from anvil.frontend import from_image
from anvil.frontend.quantize import (exp_bits, exp_lut, fixed_point, prove_no_overflow, rmsnorm_lut, rsqrt_kmax,
                                     silu_lut)
from anvil.interp import graph as gi
from anvil.ir.graph import GraphBuilder
from xf import attention, export as xe, fixture


def _img(block, i=0):
    return fixture.load(fixture.FIXTURE_DIR / f"{block}_{i}.json")["image"]


def _xf(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except (OverflowError, ValueError) as e:
        return type(e)


@settings(max_examples=300, deadline=None)
@given(ms=st.lists(st.floats(1e-6, 4.0), min_size=1, max_size=3), data=st.data(), k=st.integers(0, 6))
def test_fixed_point_equals_xf(ms, data, k):
    bounds = data.draw(st.lists(st.integers(0, 1 << 26), min_size=len(ms), max_size=len(ms)))
    assert _xf(fixed_point, ms, bounds, k) == _xf(xe.fixed_point, ms, bounds, k)


def test_fixed_point_raises():
    with pytest.raises(OverflowError):
        fixed_point([1.0], [1 << 31])
    with pytest.raises(ValueError):
        fixed_point([1e-12, 1.0], [1 << 24, 1 << 24])            # first multiplier rounds to 0
    (m,), s = fixed_point([0.37], [1 << 20], max_shift=12)        # generalized: a narrower shift range
    assert s == 12 and m == round(0.37 * 4096)


@pytest.mark.parametrize("block", ["mlp_block", "attention"])
def test_rmsnorm_lut_equals_image(block):
    img = _img(block)
    seg = img.segments
    w = img.read("norm_w")
    lut, s = rmsnorm_lut(seg["x"]["scale"], seg["norm_w"]["scale"], seg["h"]["scale"], int(np.abs(w).max()),
                         seg["x"]["shape"][-1], xe.EPS)
    assert np.array_equal(lut, img.read("rsqrt_lut")) and s == int(img.read("norm_s")[0])
    assert rsqrt_kmax(8) == xe.rsqrt_kmax(8)


def test_silu_lut_equals_image():
    img = _img("mlp_block")
    lut, s = silu_lut(img.segments["gate"]["scale"])
    assert np.array_equal(lut, img.read("silu_lut")) and s == img.segments["silu_lut"]["scale"]


def test_exp_lut_equals_image():
    img = _img("attention")
    t = img.segments["scores"]["shape"][-1]
    assert np.array_equal(exp_lut(img.segments["scores"]["scale"], t), img.read("exp_lut"))
    assert all(exp_bits(t) == attention.exp_bits(t) for t in range(1, 64))


@pytest.mark.parametrize("block", ["mlp_block", "attention"])
def test_overflow_proof_equals_check_bounds(block):
    img = _img(block)
    module, consts = from_image.BLOCKS[block](img)
    got = prove_no_overflow(module, consts)
    if block == "mlp_block":
        want = {{"norm": "h", "res": "out"}.get(k, k): v for k, v in xe.check_bounds(img).items()}
    else:
        want = {{"softmax": "p"}.get(k, k): v for k, v in attention.check_bounds(img).items()}
    assert got == want
    inline, _ = from_image.BLOCKS[block](img, inline=True)
    assert prove_no_overflow(inline) == want


def _linear_graph(m, s, w=127):
    g = GraphBuilder("crafted")
    x = g.input("x", [4, 8], scale=0.03)
    y = g.qlinear("y", x, g.const("w", [8, 16], value=np.full((8, 16), w)), m, s, scale=0.05)
    g.output(y)
    return g.module()


def test_overflow_proof_rejects_crafted_overflow():
    """acc <= 128 * 8 * 127 = 130048: the largest safe M at s = 10 passes, one more fails, and the interpreter
    really overflows on the all -128 input exactly when the proof says it can."""
    acc = 128 * 8 * 127
    m_ok = ((1 << 31) - 1 - (1 << 9)) // acc
    x = np.full((4, 8), -128)
    assert prove_no_overflow(_linear_graph(m_ok, 10))["y"] == acc * m_ok + (1 << 9)
    gi.run(_linear_graph(m_ok, 10), {"x": x})
    with pytest.raises(OverflowError, match=r"y: acc\*M \+ round reaches"):
        prove_no_overflow(_linear_graph(m_ok + 1, 10))
    with pytest.raises(OverflowError):
        gi.run(_linear_graph(m_ok + 1, 10), {"x": x})


def test_overflow_proof_names_every_bad_op():
    g = GraphBuilder("crafted")
    x = g.input("x", [4, 8], scale=0.03)
    a = g.add_requant("res", x, x, 1 << 24, 1 << 24, 0, scale=0.05)
    sc = g.qmatmul("scores", a, g.transpose("aT", a, [1, 0]), 1 << 20, 3, scale=0.1)
    g.output(sc)
    with pytest.raises(OverflowError) as e:
        prove_no_overflow(g.module())
    assert "res:" in str(e.value) and "scores:" in str(e.value)


def test_frontend_imports_without_torch():
    code = "import sys; import anvil.frontend, anvil.frontend.capture, anvil.frontend.quantize; " \
           "assert 'torch' not in sys.modules; assert callable(anvil.frontend.capture)"
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root), str(root / "sim")])}
    subprocess.run([sys.executable, "-c", code], check=True, cwd=root, env=env)
