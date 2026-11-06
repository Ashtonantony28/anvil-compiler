"""C4 level 1: the graph IR of the xf workloads interprets bit-exactly to xf.golden and to the fixture goldens."""
import numpy as np
import pytest

from anvil.frontend import from_image
from anvil.interp import graph as gi
from anvil.ir import parse_ir, print_ir
from xf import attention, export, fixture

GOLDEN = {"mlp_block": export.mlp_block_int, "attention": attention.attention_int}
CASES = [(b, i) for b in GOLDEN for i in range(4)]


def _load(block, i):
    return fixture.load(fixture.FIXTURE_DIR / f"{block}_{i}.json")


@pytest.mark.parametrize("block,i", CASES)
def test_fixture_bit_exact(block, i):
    d = _load(block, i)
    img, x = d["image"], d["inputs"]["x"]
    module, consts = from_image.BLOCKS[block](img)
    got = gi.run(module, {"x": x}, consts)
    want = GOLDEN[block](img, x)
    assert set(want) <= set(got.values)
    for k, v in want.items():
        assert np.array_equal(got.values[k], v), k
    for k, v in d["golden"].items():          # the fixture's own stored goldens
        assert np.array_equal(got.values[k], v), k
    out = "out" if block == "mlp_block" else "y"
    assert len(got.outputs) == 1 and np.array_equal(got.outputs[0], want[out])


@pytest.mark.parametrize("block", GOLDEN)
def test_inline_graph_round_trips_and_interprets(block):
    d = _load(block, 0)
    module, _ = from_image.BLOCKS[block](d["image"], inline=True)
    text = print_ir(module)
    again = parse_ir(text)
    assert print_ir(again) == text
    assert again.is_structurally_equivalent(module)
    got = gi.run(again, {"x": d["inputs"]["x"]})           # self-contained: no consts needed
    for k, v in GOLDEN[block](d["image"], d["inputs"]["x"]).items():
        assert np.array_equal(got.values[k], v), k


@pytest.mark.parametrize("block", GOLDEN)
def test_batched_inputs(block):
    """Graph ops broadcast over leading batch dims like golden does (many inputs in one run)."""
    d = _load(block, 0)
    img = d["image"]
    xs = np.random.default_rng(0).integers(-128, 128, (5, *img.segments["x"]["shape"]))
    module, consts = from_image.BLOCKS[block](img)
    for x in xs:
        got = gi.run(module, {"x": x}, consts)
        for k, v in GOLDEN[block](img, x).items():
            assert np.array_equal(got.values[k], v), k


def test_saturating_extremes():
    d = _load("mlp_block", 0)
    img = d["image"]
    module, consts = from_image.mlp_block(img)
    for fill in (-128, 127, 0):
        x = np.full(img.segments["x"]["shape"], fill)
        got = gi.run(module, {"x": x}, consts)
        assert np.array_equal(got.values["out"], export.mlp_block_int(img, x)["out"])
