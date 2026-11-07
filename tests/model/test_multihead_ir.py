"""SG5a without torch: head-batched graph IR (ADR-003) verifies, round-trips, interprets batched == per sample,
lowers to tensor IR and, under default schedules (including kernels with more iterations than one launch has
threads), interprets bit-exactly to the graph interpreter (C4 level 2)."""
import numpy as np
import pytest

from anvil.interp import graph as gi
from anvil.ir import parse_ir, print_ir
from anvil.ir.graph import GraphBuilder
from anvil.lower.graph_to_tensor import lower
from anvil.schedule import Target, templates
from xdsl.utils.exceptions import VerifyException

RNG = np.random.default_rng(23)


def _mha(T=4, D=8, nh=2):
    """x -> q, k, v -> split heads (reshape + transpose) -> batched scores -> causal softmax -> batched p @ v
    -> merge heads (transpose + reshape) -> c_proj -> residual, the shape quantize gives nanoGPT attention."""
    hd = D // nh
    g = GraphBuilder("mha")
    x = g.input("x", [T, D], scale=0.03)
    lut8 = g.const("lut8", [256])
    proj = {}
    for p in "qkvy":
        w = g.const(f"w{p}", [D, D])
        proj[p] = w
    heads = {}
    for p in "qkv":
        h = g.qlinear(p, x, proj[p], 600, 12)
        r = g.reshape(f"{p}h.r", h, [T, nh, hd])
        heads[p] = g.transpose(f"{p}h", r, [1, 0, 2])
    kT = g.transpose("kT", heads["k"], [0, 2, 1])
    sc = g.qmatmul("scores", heads["q"], kT, 300, 11)
    p = g.softmax_lut("p", sc, g.const("exp_lut", [256], "i32"), 127, True)
    o = g.qmatmul("o", p, heads["v"], 900, 10, lut=lut8)
    om = g.reshape("om", g.transpose("om.t", o, [1, 0, 2]), [T, D])
    y = g.qlinear("y", om, proj["y"], 500, 12)
    out = g.add_requant("out", x, y, 3, 5, 3)
    g.output(out)
    consts = {f"w{p}": RNG.integers(-128, 128, (D, D)) for p in "qkvy"}
    consts["lut8"] = RNG.integers(-128, 128, 256)
    consts["exp_lut"] = np.rint(np.exp(np.arange(-255, 1) * 0.05) * (1 << 20)).astype(np.int64)
    return g.module(), consts


def test_head_batched_qmatmul_verifies_and_round_trips():
    module, _ = _mha()
    text = print_ir(module)
    assert "anvil.graph.qmatmul" in text and print_ir(parse_ir(text)) == text
    g = GraphBuilder("bad")
    a = g.input("a", [2, 4, 3])
    b = g.input("b", [3, 3, 5])                     # batch dims differ from a's
    with pytest.raises(VerifyException, match="need"):
        g.qmatmul("c", a, b, 1, 0)
    g = GraphBuilder("bad2")
    a = g.input("a", [2, 4, 3])
    w = g.const("w", [2, 3, 5])
    with pytest.raises(VerifyException, match=r"\(K, N\)"):
        g.qlinear("c", a, w, 1, 0)                   # qlinear weights stay 2-D


def test_batched_interp_equals_per_sample():
    module, consts = _mha()
    xs = RNG.integers(-128, 128, (6, 4, 8))
    got = gi.run(module, {"x": xs}, consts, batch=True)
    for i, x in enumerate(xs):
        one = gi.run(module, {"x": x}, consts)
        for k, v in one.values.items():
            assert np.array_equal(got.values[k] if k in consts else got.values[k][i], v), k


def _schedule_all(prog, target):
    return {k.name: templates.default_schedule(k, target).finalize() for k in prog.kernels}


@pytest.mark.parametrize("T,D,nh", [(4, 8, 2), (4, 8, 4), (8, 16, 2)])
def test_multihead_lowering_bit_exact_under_default_schedules(T, D, nh):
    module, consts = _mha(T, D, nh)
    prog = lower(module)
    assert {"qh.r", "kh.r", "vh.r", "om"} <= set(prog.views)
    assert prog.kernel("scores").op == "qmatmul" and prog.kernel("p").op == "softmax_lut"
    target = Target()
    ks = _schedule_all(prog, target)
    for _ in range(2):
        x = RNG.integers(-128, 128, (T, D))
        ref = gi.run(module, {"x": x}, consts)
        for kernels in (None, ks):
            got = prog.run({"x": x}, consts, kernels)
            for k, v in ref.values.items():
                assert np.array_equal(got[k], v), k


@pytest.mark.parametrize("n,cap_ok", [(255, True), (1024, True), (13 * 37, True)])
def test_default_schedule_beyond_one_launch(n, cap_ok):
    """An elementwise kernel with more elements than 255 threads gets a per-thread seq loop and stays bit-exact."""
    g = GraphBuilder("wide")
    x = g.input("x", [n])
    y = g.lut("y", x, g.const("t", [256]))
    g.output(y)
    consts = {"t": RNG.integers(-128, 128, 256)}
    prog = lower(g.module())
    target = Target()
    s = templates.default_schedule(prog.kernels[0], target)
    k = s.finalize()
    blocks = [x for x in s.loops() if x[2] == "block"][0][1]
    assert blocks * target.threads_per_block <= target.max_threads
    xs = RNG.integers(-128, 128, n)
    got = k.run({"x": xs, **consts})
    assert np.array_equal(got["y"], gi.run(g.module(), {"x": xs}, consts).outputs[0])


def test_per_thread_choice():
    from anvil.schedule.templates import _per_thread

    assert _per_thread(512, 252) == 4 and _per_thread(1024, 252) == 8
    assert _per_thread(13 * 37, 252) == 2                         # 481 = 13 * 37: 37 more than doubles 2
    assert all(-(-n // _per_thread(n, 252)) <= 252 for n in range(253, 5000))
