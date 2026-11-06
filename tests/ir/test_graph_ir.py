"""anvil.graph: verifier, textual round trip (hypothesis over random graphs), interpretation preserved by it."""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from xdsl.utils.exceptions import VerifyException

from anvil import semantics as sem
from anvil.interp import graph as gi
from anvil.ir import parse_ir, print_ir
from anvil.ir.graph import GraphBuilder, QTensorType, qt


@st.composite
def random_graph(draw):
    """A random chain of graph ops with consistent shapes and overflow-free requants, plus its numpy reference."""
    rng = np.random.default_rng(draw(st.integers(0, 2 ** 32 - 1)))
    t, d = draw(st.integers(1, 4)), draw(st.integers(1, 8))
    g = GraphBuilder("rand")
    x0 = rng.integers(-128, 128, (t, d))
    cur, ref = g.input("x", [t, d], scale=draw(st.floats(1e-4, 1.0))), x0
    for n in range(draw(st.integers(1, 6))):
        kind = draw(st.sampled_from(["qlinear", "qmatmul_t", "lut", "lut_mul", "add", "transpose", "rmsnorm"]))
        rows, cols = qt(cur).dims
        sym = f"v{n}"
        m, s = draw(st.integers(1, 64)), draw(st.integers(0, 14))
        if kind == "qlinear":
            k2 = draw(st.integers(1, 8))
            w = rng.integers(-128, 128, (cols, k2))
            lut = draw(st.booleans())
            tab = rng.integers(-128, 128, 256)
            lv = g.const(f"lut{n}", [256], value=tab) if lut else None
            cur = g.qlinear(sym, cur, g.const(f"w{n}", [cols, k2], value=w), m, s, lut=lv)
            ref = sem.qmatmul(ref, w, m, s, tab if lut else None)
        elif kind == "qmatmul_t":           # activation @ activation.T (attention scores)
            tr = g.transpose(sym + "T", cur)
            cur = g.qmatmul(sym, cur, tr, m, s)
            ref = sem.qmatmul(ref, ref.T, m, s)
        elif kind == "lut":
            tab = rng.integers(-128, 128, 256)
            cur = g.lut(sym, cur, g.const(f"lut{n}", [256], value=tab))
            ref = tab[ref + 128]
        elif kind == "lut_mul":
            tab = rng.integers(-128, 128, 256)
            cur = g.lut_mul(sym, cur, cur, g.const(f"lut{n}", [256], value=tab), m, s)
            ref = sem.swiglu_lut(ref, ref, tab, m, s)
        elif kind == "add":
            cur = g.add_requant(sym, cur, cur, m, draw(st.integers(1, 64)), s)
            ref = sem.residual_add(ref, ref, m, int(cur.owner.Mb.value.data), s)
        elif kind == "transpose":
            cur = g.transpose(sym, cur)
            ref = ref.T
        else:
            w = rng.integers(-128, 128, cols)
            lut = rng.integers(0, 1 << 12, 64)
            cur = g.rmsnorm_int(sym, cur, g.const(f"nw{n}", [cols], value=w), g.const(f"rl{n}", [64], "i32", value=lut),
                                1, s)
            ref = sem.rmsnorm_int(ref, w, lut, 1, s)
    g.output(cur)
    return g.module(), x0, ref


@settings(max_examples=60, deadline=None)
@given(random_graph())
def test_round_trip_random_graphs(case):
    module, x, ref = case
    text = print_ir(module)
    again = parse_ir(text)
    assert print_ir(again) == text
    assert again.is_structurally_equivalent(module)
    a, b = gi.run(module, {"x": x}), gi.run(again, {"x": x})
    assert np.array_equal(a.outputs[0], ref) and np.array_equal(b.outputs[0], ref)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.integers(1, 9), min_size=0, max_size=4), st.sampled_from(["i8", "i32"]),
       st.one_of(st.none(), st.floats(min_value=1e-12, max_value=1e6, allow_nan=False)))
def test_qtensor_type_round_trip_is_exact(shape, dtype, scale):
    g = GraphBuilder("t")
    v = g.input("x", shape or [1], dtype, scale)
    g.output(v)
    again = parse_ir(print_ir(g.module()))
    t = qt(next(iter(again.ops)).body.block.first_op.result)
    assert t == QTensorType(shape or [1], dtype, scale) and t.qscale == scale


def test_softmax_and_reshape_ops():
    g = GraphBuilder("sm")
    x = g.input("x", [2, 3, 3])
    lut = np.arange(256) * 7
    p = g.softmax_lut("p", x, g.const("e", [256], "i32", value=lut), 127, causal=False)
    r = g.reshape("r", p, [6, 3])
    g.output(r)
    xs = np.random.default_rng(1).integers(-128, 128, (2, 3, 3))
    got = gi.run(parse_ir(print_ir(g.module())), {"x": xs})
    assert np.array_equal(got.outputs[0], sem.softmax_lut(xs, lut, 127, causal=False).reshape(6, 3))


def test_verifier_rejects_bad_graphs():
    g = GraphBuilder("bad")
    x = g.input("x", [4, 8])
    with pytest.raises(VerifyException, match="contraction"):
        g.qmatmul("y", x, g.input("w", [4, 8]), 1, 1)
    with pytest.raises(VerifyException, match="const"):
        g.qlinear("y", x, g.input("w2", [8, 2]), 1, 1)
    with pytest.raises(VerifyException, match="outside"):
        g.qlinear("y", x, g.const("w3", [8, 2]), 1, 40)
    with pytest.raises(VerifyException, match="LUT"):
        g.lut("y", x, g.const("l", [128]))
    with pytest.raises(VerifyException, match="permutation"):
        g.transpose("y", x, [0, 0])
    with pytest.raises(VerifyException, match="fit"):
        g.softmax_lut("y", g.input("s", [4, 4]), g.const("e", [256], "i32"), 300)
    with pytest.raises(ValueError):
        QTensorType([4], "f32")


def test_interpreter_checks_declared_types():
    g = GraphBuilder("t")
    x = g.input("x", [2, 2])
    g.output(x)
    with pytest.raises(ValueError, match="shape"):
        gi.run(g.module(), {"x": np.zeros((3, 2))})
    with pytest.raises(ValueError, match="int8"):
        gi.run(g.module(), {"x": np.full((2, 2), 200)})
