"""SG9 / ADR-004 without torch: the wide (i16) residual stream. Semantics (rmsnorm with shifted squares and eps folded
into the sum, requant clamped to the result dtype) reduce to xf's golden ops on int8; the tight rmsnorm bound
holds on adversarial rows; a small residual graph (i16 input -> rmsnorm -> qlinear (i16 out) -> add_requant (i16)
-> rmsnorm) proves overflow-free, round-trips, lowers bit-exactly under default schedules (C4 level 2) and runs
bit-exactly on isasim (C4 level 4)."""
import importlib

import numpy as np
import pytest

from anvil import semantics as sem

from anvil.interp import graph as gi
from anvil.ir import graph as g
from anvil.lower.graph_to_tensor import lower

Q = importlib.import_module("anvil.frontend.quantize")      # the module (anvil.frontend.quantize is also a function)

RNG = np.random.default_rng(9)
T, D, H = 4, 32, 16
I16 = (-(1 << 15), (1 << 15) - 1)


def test_wide_semantics_reduce_to_golden_on_int8():
    x = RNG.integers(-128, 128, (5, T, D))
    w = RNG.integers(-128, 128, D)
    lut, s = Q.rmsnorm_lut(0.03, 0.01, 0.02, int(np.abs(w).max()), D, 1e-6)
    assert np.array_equal(sem.rmsnorm_wide(x, w, lut, 1, s), sem.rmsnorm_int(x, w, lut, 1, s))
    # the general path with c = e = 0 on int8 rows computes the same thing
    assert np.array_equal(sem.rmsnorm_wide(x, w, lut, 1, s, 0, 0, "i16"), sem.rmsnorm_int(x, w, lut, 1, s))
    a, b = RNG.integers(-128, 128, (2, 50))
    assert np.array_equal(sem.add_requant(a, b, 300, 500, 10), sem.residual_add(a, b, 300, 500, 10))
    assert np.array_equal(sem.requant_dtype(a * 1000, 7, 3, "i8"), sem.requant(a * 1000, 7, 3))


def test_requant_dtype_clamps_at_int16():
    acc = np.array([-(1 << 30), -5, 0, 5, 1 << 30])
    assert list(sem.requant_dtype(acc, 1, 4, "i16")) == [I16[0], 0, 0, 0, I16[1]]       # -5 + 8 >> 4 = 0
    a = np.array([I16[1], I16[0]])
    assert list(sem.add_requant(a, np.array([127, -128]), 1 << 14, 1 << 20, 14, "i16")) == [I16[1], I16[0]]
    with pytest.raises(ValueError):
        sem.add_requant(np.array([1 << 15]), np.array([0]), 1, 1, 0, "i16")             # operand outside int16


def test_rmsnorm_tight_bound_on_adversarial_rows():
    """Every x*w*r (+ round) the wide rmsnorm forms stays within the bound rms_wide_bounds proves, on rows built
    to maximise it: one loud entry, all entries equal, at every magnitude from 1 to 2^15."""
    sx, sw, sh = 4e-4, 0.01, 0.02
    lut, s, c, e = Q.rmsnorm_lut_wide(sx, sw, sh, 127, D, 1e-6, 1 << 15)
    xm = Q.rms_xmax_table(D, len(lut), c, 1 << 15, e)
    bound = Q.rms_wide_bounds(lut, 127, 1, s, xm)
    assert max(bound.values()) <= Q.INT32_MAX and s + max(xm) <= 31
    rows = []
    for mag in np.unique(np.geomspace(1, 1 << 15, 200).astype(np.int64)):
        one = np.zeros(D, np.int64)
        one[0] = min(mag, I16[1] + 1)
        rows += [one, -one, np.full(D, min(mag, I16[1] + 1)), np.r_[min(mag, I16[1] + 1), np.ones(D - 1, np.int64)]]
    x = np.clip(np.array(rows), -(1 << 15), I16[1])
    ss = sem.sq_shift(x, c).sum(axis=1) + e
    idx, k = sem.rsqrt_index(ss, len(lut))
    xwr = np.abs(x * 127) * lut[idx][:, None]
    for kv in np.unique(k):
        assert (xwr[k == kv] + sem.round_term(s + kv)).max() <= bound[int(kv)]
        assert (np.abs(x[k == kv]).max(axis=1) <= xm[int(kv)][idx[k == kv]]).all()
    out = sem.rmsnorm_wide(x, np.full(D, 127), lut, 1, s, c, e, "i16")      # raises on any int32 overflow
    assert np.abs(out).max() <= 128


def _graph(inline=False):
    """x (i16) -> h = rmsnorm(x) -> y = qlinear(h) (i16) -> r = add_requant(x, y) (i16) -> h2 = rmsnorm(r)."""
    sx, sy, sr, sh = 5e-4, 6e-4, 6e-4, 0.02
    b = g.GraphBuilder("wide")
    x = b.input("x", [T, D], "i16", sx)
    consts = {"n1.w": RNG.integers(-127, 128, D), "fc.w": RNG.integers(-127, 128, (D, D)),
              "n2.w": RNG.integers(-127, 128, D)}
    l1, s1, c1, e1 = Q.rmsnorm_lut_wide(sx, 0.01, sh, 127, D, 1e-6, 1 << 15)
    l2, s2, c2, e2 = Q.rmsnorm_lut_wide(sr, 0.01, sh, 127, D, 1e-6, 1 << 15)
    consts.update({"n1.lut": l1, "n2.lut": l2})
    h = b.rmsnorm_int("h", x, b.const("n1.w", [D]), b.const("n1.lut", [len(l1)], "i32"), 1, s1, sh, c1, e1)
    (m,), s = Q.fixed_point([sh * 0.01 / sy], [128 * int(np.abs(consts["fc.w"]).sum(axis=0).max())])
    y = b.qlinear("y", h, b.const("fc.w", [D, D], "i8", 0.01), m, s, sy, dtype="i16")
    (ma, mb), s = Q.fixed_point([sx / sr, sy / sr], [1 << 15, 1 << 15])
    r = b.add_requant("r", x, y, ma, mb, s, sr, dtype="i16")
    h2 = b.rmsnorm_int("h2", r, b.const("n2.w", [D]), b.const("n2.lut", [len(l2)], "i32"), 1, s2, sh, c2, e2)
    b.output(h2, r)
    return b.module(), consts


def _inputs():
    return [RNG.integers(-(1 << 15), 1 << 15, (T, D)), np.full((T, D), -(1 << 15)), np.full((T, D), I16[1]),
            np.rint(RNG.standard_normal((T, D)) * 40).astype(np.int64), np.zeros((T, D), np.int64)]


def test_wide_graph_verifies_round_trips_and_proves_no_overflow():
    from anvil.ir import parse_ir, print_ir
    mod, consts = _graph()
    assert print_ir(parse_ir(print_ir(mod))) == print_ir(mod)
    bounds = Q.prove_no_overflow(mod, consts)
    assert set(bounds) == {"h", "y", "r", "h2"} and max(bounds.values()) <= Q.INT32_MAX
    for x in _inputs():                                              # the interpreter checks every int32 term
        out = gi.run(mod, {"x": x}, consts)
        assert np.abs(out.values["r"]).max() <= 1 << 15


def test_wide_graph_rejects_bad_types():
    from xdsl.utils.exceptions import VerifyException
    b = g.GraphBuilder("bad")
    x = b.input("x", [T, D], "i16", 1.0)
    with pytest.raises(VerifyException):
        b.qlinear("q", x, b.const("w", [D, D]), 1, 0)                # matmul operands stay i8
    h = b.input("h", [T, D], "i8", 1.0)
    with pytest.raises(VerifyException):
        b.qlinear("q", h, b.const("w2", [D, D]), 1, 0, lut=b.const("l", [256]), dtype="i16")


def test_wide_graph_lowers_bit_exactly_under_default_schedules():
    from anvil.schedule import Target, templates
    mod, consts = _graph()
    prog = lower(mod)
    assert prog.buffers["x"][1] == "i32" and prog.buffers["r"][1] == "i32"
    kernels = {k.name: templates.default_schedule(k, Target()).finalize() for k in prog.kernels}
    for x in _inputs():
        ref = gi.run(mod, {"x": x}, consts).values
        got = prog.run({"x": x}, consts, kernels)
        assert all(np.array_equal(got[k], v) for k, v in ref.items()), [k for k in ref if not
                                                                         np.array_equal(got[k], ref[k])]


@pytest.mark.parametrize("preset", ["default", "tiled"])
def test_wide_graph_on_isasim(preset):
    import anvil
    mod, consts = _graph()
    cm = anvil.compile((mod, consts), schedules=preset)
    for x in _inputs():
        ref = gi.run(mod, {"x": x}, consts).values
        r = cm.run(x, quantized=True, keep="all")
        for k in ("h", "y", "r", "h2"):
            assert np.array_equal(r.ints[k], ref[k]), k
        assert not any(r.result.flags[f] for f in ("store_races", "stores_conflict", "divergent"))
    xf = RNG.standard_normal((T, D)) * 3                             # float input path quantizes to int16
    assert np.abs(cm.quantize_input(xf)).max() > 127


@pytest.mark.parametrize("template", ["rows_per_thread", "row_per_block"])
def test_wide_rmsnorm_takes_the_row_templates(template):
    """The eps term is added after the 0-initialised reduction, so the cross-lane reduction of row_per_block
    still applies to a wide rmsnorm (C4 level 2)."""
    from anvil.schedule import Schedule, Target, templates
    mod, consts = _graph()
    prog = lower(mod)
    for name in ("h", "h2"):
        k = templates.TEMPLATES[template](Schedule(prog.kernel(name), Target())).finalize()
        for x in _inputs():
            ref = gi.run(mod, {"x": x}, consts).values
            got = prog.run({"x": x}, consts, {name: k})
            assert np.array_equal(got[name], ref[name]), (name, template)
