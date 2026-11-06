"""anvil.tensor interpreter on hand-written loop nests: each equals anvil.semantics (= xf.golden) bit for bit."""
import numpy as np
import pytest

from anvil import semantics as sem
from anvil.interp import tensor as ti
from anvil.ir import parse_ir, print_ir
from anvil.ir.tensor import TensorBuilder
from xf import fixture, golden

RNG = np.random.default_rng(7)


def _both(module, buffers):
    """Run the module and its parsed textual form; they must agree."""
    a = ti.run(module, buffers)
    b = ti.run(parse_ir(print_ir(module)), buffers)
    assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)
    return a


def matmul_requant(m, k, n, M, s, bind=("block", "lane")):
    b = TensorBuilder("matmul")
    a, w, c = b.alloc("a", [m, k], "i8"), b.alloc("w", [k, n], "i8"), b.alloc("c", [m, n], "i8")
    with b.loop(0, m, bind=bind[0]) as i:
        with b.loop(0, n, bind=bind[1]) as j:
            acc = b.alloc(None, [1], "i32", "reg")
            with b.loop(0, k, unroll=2) as kk:
                p = b.mul(b.load(a, [i, kk]), b.load(w, [kk, j]))
                b.store(b.add(b.load(acc, [], lambda: (0,)), p), acc, [], lambda: (0,))
            b.store(b.requant(b.load(acc, [], lambda: (0,)), M, s), c, [i, j])
    return b.module()


@pytest.mark.parametrize("bind", [("block", "lane"), ("seq", "seq"), ("lane", "warp")])
def test_matmul_requant(bind):
    a, w = RNG.integers(-128, 128, (4, 8)), RNG.integers(-128, 128, (8, 4))
    out = _both(matmul_requant(4, 8, 4, 1234, 17, bind), {"a": a, "w": w})
    assert np.array_equal(out["c"], sem.qmatmul(a, w, 1234, 17))


def test_matmul_on_fixture_weights():
    d = fixture.load(fixture.FIXTURE_DIR / "mlp_block_0.json")
    img, h = d["image"], d["golden"]["h"]
    M, s = int(img.read("gate_M")[0]), int(img.read("gate_s")[0])
    out = ti.run(matmul_requant(4, 8, 16, M, s), {"a": h, "w": img.read("w_gate")})
    assert np.array_equal(out["c"], d["golden"]["gate"])


def test_swiglu_gather():
    t, hdim = 4, 16
    b = TensorBuilder("swiglu")
    gate, up = b.alloc("gate", [t, hdim], "i8"), b.alloc("up", [t, hdim], "i8")
    lut, out = b.alloc("lut", [256], "i8"), b.alloc("out", [t, hdim], "i8")
    with b.loop(0, t * hdim, bind="lane") as e:          # flattened elementwise: map splits e into (row, col)
        rc = lambda e: (e // hdim, e % hdim)  # noqa: E731
        sl = b.gather(lut, b.load(gate, [e], rc), offset=128)
        b.store(b.requant(b.mul(sl, b.load(up, [e], rc)), 300, 12), out, [e], rc)
    g, u, l = RNG.integers(-128, 128, (t, hdim)), RNG.integers(-128, 128, (t, hdim)), RNG.integers(-128, 128, 256)
    assert np.array_equal(_both(b.module(), {"gate": g, "up": u, "lut": l})["out"], golden.swiglu_lut(g, u, l, 300, 12))


def test_transpose_via_maps():
    b = TensorBuilder("transpose")
    x, y = b.alloc("x", [3, 5], "i8"), b.alloc("y", [5, 3], "i8")
    with b.loop(0, 3) as i:
        with b.loop(0, 5) as j:
            b.store(b.load(x, [i, j]), y, [i, j], lambda i, j: (j, i))
    xs = RNG.integers(-128, 128, (3, 5))
    assert np.array_equal(_both(b.module(), {"x": xs})["y"], xs.T)


def rmsnorm_prog(t, d, n, M, s):
    b = TensorBuilder("rmsnorm")
    x, w, lut, out = b.alloc("x", [t, d], "i8"), b.alloc("w", [d], "i8"), b.alloc("lut", [n]), b.alloc("out", [t, d], "i8")
    z = lambda *_: (0,)  # noqa: E731
    with b.loop(0, t, bind="block") as i:
        ss = b.alloc(None, [1], "i32", "reg")
        with b.loop(0, d) as j:
            xv = b.load(x, [i, j])
            b.store(b.add(b.load(ss, [], z), b.mul(xv, xv)), ss, [], z)
        idx, k = b.rsqrt_index(b.load(ss, [], z), n)
        r = b.gather(lut, idx)
        with b.loop(0, d, bind="lane") as j:
            xw = b.mul(b.load(x, [i, j]), b.load(w, [j]))
            b.store(b.requant(b.mul(xw, r), M, s, extra=k), out, [i, j])
    return b.module()


def test_rmsnorm_on_fixture():
    d = fixture.load(fixture.FIXTURE_DIR / "attention_1.json")
    img = d["image"]
    lut = img.read("rsqrt_lut")
    M, s = int(img.read("norm_M")[0]), int(img.read("norm_s")[0])
    xs = [d["inputs"]["x"], np.full((4, 8), -128), np.zeros((4, 8), int), RNG.integers(-128, 128, (4, 8))]
    for x in xs:
        out = ti.run(rmsnorm_prog(4, 8, len(lut), M, s), {"x": x, "w": img.read("norm_w"), "lut": lut})
        assert np.array_equal(out["out"], golden.rmsnorm_int(x, img.read("norm_w"), lut, M, s))
    assert np.array_equal(ti.run(rmsnorm_prog(4, 8, len(lut), M, s), {"x": xs[0], "w": img.read("norm_w"), "lut": lut})
                          ["out"], d["golden"]["h"])


def softmax_prog(t, one, causal=True):
    b = TensorBuilder("softmax")
    sc, lut, p = b.alloc("scores", [t, t], "i8"), b.alloc("lut", [256]), b.alloc("p", [t, t], "i8")
    z = lambda *_: (0,)  # noqa: E731
    with b.loop(0, t, bind="lane") as i:
        mx, tot = b.alloc(None, [1], "i32", "reg"), b.alloc(None, [1], "i32", "reg")
        lo = b.const(-128)
        b.store(lo, mx, [], z)

        def keep(j):
            return b.cmp("le", b.cast(j), b.cast(i)) if causal else b.const(1)

        def exp(j):
            kp = keep(j)
            dv = b.add(b.sub(b.load(sc, [i, j]), b.load(mx, [], z)), b.const(255))
            return b.select(kp, b.gather(lut, b.select(kp, dv, b.const(255))), b.const(0))

        with b.loop(0, t) as j:
            b.store(b.max(b.load(mx, [], z), b.select(keep(j), b.load(sc, [i, j]), lo)), mx, [], z)
        with b.loop(0, t) as j:
            b.store(b.add(b.load(tot, [], z), exp(j)), tot, [], z)
        with b.loop(0, t) as j:
            total = b.load(tot, [], z)
            num = b.add(b.mul(exp(j), b.const(one)), b.div(total, b.const(2)))
            b.store(b.div(num, total), p, [i, j])
    return b.module()


@pytest.mark.parametrize("causal", [True, False])
def test_softmax_on_fixture(causal):
    d = fixture.load(fixture.FIXTURE_DIR / "attention_0.json")
    img = d["image"]
    lut, one = img.read("exp_lut"), int(img.read("one")[0])
    for s in (d["golden"]["scores"], RNG.integers(-128, 128, (4, 4)), np.full((4, 4), -128)):
        out = _both(softmax_prog(4, one, causal), {"scores": s, "lut": lut})
        assert np.array_equal(out["p"], golden.softmax_lut(s, lut, one, causal)), s
    if causal:
        out = ti.run(softmax_prog(4, one), {"scores": d["golden"]["scores"], "lut": lut})
        assert np.array_equal(out["p"], d["golden"]["p"])


def test_smem_staged_matmul_with_barrier():
    """cache_read(a, smem) shape: lanes stage a row of `a` into SMEM, BAR, then every lane reads the whole row."""
    m, k, n = 4, 8, 4
    b = TensorBuilder("staged")
    a, w, c = b.alloc("a", [m, k], "i8"), b.alloc("w", [k, n], "i8"), b.alloc("c", [m, n], "i8")
    with b.loop(0, m, bind="block") as i:
        tile = b.alloc(None, [k], "i8", "smem")
        with b.loop(0, n, bind="lane") as j:
            with b.loop(0, k // n) as kk:
                b.store(b.load(a, [i, j, kk], lambda i, j, kk: (i, j * (k // n) + kk)), tile, [j, kk],
                        lambda j, kk: (j * (k // n) + kk,))
            b.barrier()
            acc = b.alloc(None, [1], "i32", "reg")
            with b.loop(0, k) as kk:
                p = b.mul(b.load(tile, [kk]), b.load(w, [kk, j]))
                b.store(b.add(b.load(acc, [], lambda: (0,)), p), acc, [], lambda: (0,))
            b.store(b.requant(b.load(acc, [], lambda: (0,)), 99, 10), c, [i, j])
    av, wv = RNG.integers(-128, 128, (m, k)), RNG.integers(-128, 128, (k, n))
    assert np.array_equal(_both(b.module(), {"a": av, "w": wv})["c"], sem.qmatmul(av, wv, 99, 10))


def test_barrier_phases_make_neighbour_writes_visible():
    b = TensorBuilder("rotate")
    out = b.alloc("out", [4])
    buf = b.alloc(None, [4], "i32", "smem")
    with b.loop(0, 4, bind="lane") as j:
        b.store(b.mul(b.add(b.cast(j), b.const(1)), b.const(10)), buf, [j])
        b.barrier()
        b.store(b.load(buf, [j], lambda j: ((j + 1) % 4,)), out, [j])
    assert _both(b.module(), {})["out"].tolist() == [20, 30, 40, 10]


def test_barrier_below_nested_warp_lane_and_seq_loops():
    """SG1 open issue 1: barriers inside seq loops under warp x lane loops sync every thread of the block."""
    b = TensorBuilder("nested")
    out = b.alloc("out", [3, 8])
    buf = b.alloc(None, [8], "i32", "smem")
    with b.loop(0, 2, bind="warp") as w:
        with b.loop(0, 4, bind="lane") as l:
            with b.loop(0, 3) as r:                     # each round: write own slot, BAR, read neighbour, BAR
                tid = lambda w, l: (w * 4 + l,)  # noqa: E731
                v = b.add(b.apply([w, l], lambda w, l: (w * 4 + l,)), b.mul(b.cast(r), b.const(100)))
                b.store(v, buf, [w, l], tid)
                b.barrier()
                b.store(b.load(buf, [w, l], lambda w, l: ((w * 4 + l + 3) % 8,)), out, [r, w, l],
                        lambda r, w, l: (r, w * 4 + l))
                b.barrier()
    got = _both(b.module(), {})["out"]
    assert got.tolist() == [[(t + 3) % 8 + 100 * r for t in range(8)] for r in range(3)]


def test_missing_barrier_is_a_race():
    b = TensorBuilder("racy")
    out = b.alloc("out", [4])
    buf = b.alloc(None, [4], "i32", "smem")
    with b.loop(0, 4, bind="lane") as j:
        b.store(b.add(b.cast(j), b.const(1)), buf, [j])
        b.store(b.load(buf, [j], lambda j: ((j + 1) % 4,)), out, [j])
    with pytest.raises(ti.TensorInterpError, match="race"):
        ti.run(b.module(), {})
    ti.run(b.module(), {}, check_races=False)       # sequential reading, no check


def test_conflicting_global_stores_are_a_race():
    b = TensorBuilder("conflict")
    out = b.alloc("out", [1])
    with b.loop(0, 2, bind="lane") as j:
        b.store(b.cast(j), out, [j], lambda j: (0,))
    with pytest.raises(ti.TensorInterpError, match="different values"):
        ti.run(b.module(), {})
    b = TensorBuilder("same")                         # identical values from several lanes are fine (pad tails)
    out = b.alloc("out", [1])
    with b.loop(0, 2, bind="lane") as j:
        b.store(b.const(5), out, [j], lambda j: (0,))
    assert ti.run(b.module(), {})["out"].tolist() == [5]


def test_barrier_in_guard_diverges():
    b = TensorBuilder("diverge")
    with b.loop(0, 4, bind="lane") as j:
        with b.guard([j], lambda j: (j,), [2]):
            b.barrier()
    with pytest.raises(ti.TensorInterpError, match="divergence"):
        ti.run(b.module(), {})


def test_guard_and_apply():
    b = TensorBuilder("guard")
    out = b.alloc("out", [6])
    with b.loop(0, 2) as o:
        with b.loop(0, 4) as i:
            with b.guard([o, i], lambda o, i: (o * 4 + i,), [6]):
                b.store(b.apply([o, i], lambda o, i: (o * 40 + i * 10,)), out, [o, i], lambda o, i: (o * 4 + i,))
    assert _both(b.module(), {})["out"].tolist() == [0, 10, 20, 30, 40, 50]


def test_errors_surface():
    b = TensorBuilder("ovf")
    o = b.alloc("o", [1])
    with b.loop(0, 1) as i:
        b.store(b.mul(b.const(1 << 20), b.const(1 << 20)), o, [i])
    with pytest.raises(OverflowError):
        ti.run(b.module(), {})
    b = TensorBuilder("i8")
    o = b.alloc("o", [1], "i8")
    with b.loop(0, 1) as i:
        b.store(b.const(200), o, [i])
    with pytest.raises(ValueError):
        ti.run(b.module(), {})
    b = TensorBuilder("oob")
    o = b.alloc("o", [2])
    with b.loop(0, 3) as i:
        b.store(b.const(1), o, [i])
    with pytest.raises(IndexError):
        ti.run(b.module(), {})
