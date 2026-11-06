"""SG3a: graph -> tensor lowering. Every op of both xf fixtures (and synthetic graphs for the ops the fixtures do not
use) lowers to default loop nests that interpret bit-exactly to the graph interpreter (C4 level 2)."""
import numpy as np
import pytest

from anvil.frontend import from_image
from anvil.interp import graph as gi
from anvil.ir import parse_ir, print_ir
from anvil.ir import tensor as t
from anvil.ir.graph import GraphBuilder
from anvil.lower.graph_to_tensor import lower
from xf import fixture

RNG = np.random.default_rng(11)
CASES = [(b, i) for b in ("mlp_block", "attention") for i in range(4)]


@pytest.mark.parametrize("block,i", CASES)
def test_fixture_programs_bit_exact(block, i):
    d = fixture.load(fixture.FIXTURE_DIR / f"{block}_{i}.json")
    module, consts = from_image.BLOCKS[block](d["image"])
    prog = lower(module)
    ref = gi.run(module, {"x": d["inputs"]["x"]}, consts)
    got = prog.run({"x": d["inputs"]["x"]}, consts)
    for k, v in ref.values.items():
        assert np.array_equal(got[k], v), k
    for k, v in d["golden"].items():
        assert np.array_equal(got[k], v), k
    assert [k.name for k in prog.kernels] == [s for s in ref.values if s in prog.buffers and s not in consts
                                              and s not in prog.inputs]


@pytest.mark.parametrize("block", ["mlp_block", "attention"])
def test_kernels_round_trip_and_are_unscheduled(block):
    d = fixture.load(fixture.FIXTURE_DIR / f"{block}_0.json")
    module, _ = from_image.BLOCKS[block](d["image"], inline=True)
    prog = lower(module)
    got = prog.run({"x": d["inputs"]["x"]})             # inline consts travel with the program
    ref = gi.run(module, {"x": d["inputs"]["x"]})
    assert all(np.array_equal(got[k], v) for k, v in ref.values.items())
    for k in prog.kernels:
        text = print_ir(k.module)
        assert print_ir(parse_ir(text)) == text
        loops = [op for op in k.func.walk() if isinstance(op, t.ForOp)]
        assert all(op.bind.data == "seq" and op.bounds[0] == 0 and op.bounds[2] == 1 for op in loops)
        tags = [op.tag.data for op in loops]
        assert len(set(tags)) == len(tags) and set(tags) == set(k.kinds)
        tops = [op for op in k.func.body.block.ops if not isinstance(op, t.AllocOp)]
        assert all(isinstance(op, t.ForOp) for op in tops)


def _synthetic():
    """Ops and forms the fixtures lack: lut, reshape views, batched matmul with a LUT epilogue, 3-D transpose,
    non-causal softmax with i32 output, batched rmsnorm."""
    g = GraphBuilder("synth")
    x = g.input("x", [2, 3, 8])
    lut8 = g.const("lut8", [256])
    w = g.const("w", [8, 5])
    y = g.lut("y", x, lut8)
    z = g.qmatmul("z", y, w, 700, 12, lut=lut8)                    # [2, 3, 5], batched + fused LUT
    zt = g.transpose("zt", z, [2, 0, 1])                          # [5, 2, 3]
    r = g.reshape("r", zt, [10, 3])                               # view
    r2 = g.reshape("r2", r, [5, 6])                               # view of a view
    s = g.qmatmul("s", r2, g.const("w2", [6, 6]), 900, 13)        # reads through the view chain
    sm = g.softmax_lut("sm", g.reshape("s3", s, [5, 6]), g.const("elut", [256], "i32"), 1000, False, dtype="i32")
    sc = g.softmax_lut("sc", g.reshape("sq", s, [5, 6]), g.const("elut2", [256], "i32"), 100, True)
    nw = g.const("nw", [3])
    rl = g.const("rsq", [64], "i32")
    hn = g.rmsnorm_int("hn", g.reshape("xr", zt, [2, 5, 3]), nw, rl, 30, 10)
    m = g.lut_mul("m", sc, sc, lut8, 200, 9)
    a = g.add_requant("a", m, sc, 3, 5, 3)
    g.output(sm, a, hn, r2)
    consts = {"lut8": RNG.integers(-128, 128, 256), "w": RNG.integers(-128, 128, (8, 5)),
              "w2": RNG.integers(-128, 128, (6, 6)), "elut": RNG.integers(1, 4000, 256),
              "elut2": RNG.integers(1, 4000, 256), "nw": RNG.integers(-128, 128, 3), "rsq": RNG.integers(0, 100, 64)}
    return g.module(), consts


def test_synthetic_ops_and_views():
    module, consts = _synthetic()
    prog = lower(module)
    assert set(prog.views) == {"r", "r2", "s3", "sq", "xr"}
    assert not any(k.name in prog.views for k in prog.kernels)
    for _ in range(3):
        x = RNG.integers(-128, 128, (2, 3, 8))
        ref = gi.run(module, {"x": x}, consts)
        got = prog.run({"x": x}, consts)
        for k, v in ref.values.items():
            assert np.array_equal(got[k], v), k
        assert all(np.array_equal(a, b) for a, b in zip([got[s] for s in prog.outputs], ref.outputs))


def test_view_loads_compose_reshape_maps():
    module, _ = _synthetic()
    k = lower(module).kernel("s")
    a = [op for op in k.func.body.block.ops if isinstance(op, t.AllocOp)]
    assert [op.sym.data for op in a][:1] == ["zt"]            # reads the view chain's base buffer
    assert k.reads[0] == "zt"
