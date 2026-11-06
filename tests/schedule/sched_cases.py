"""Shared cases: every graph op kind, lowered from the xf fixtures (plus a lut op), with its input buffers."""
import functools

import numpy as np

from anvil.frontend import from_image
from anvil.interp import graph as gi
from anvil.ir.graph import GraphBuilder
from anvil.lower.graph_to_tensor import lower
from xf import fixture


@functools.cache
def block_case(block, i=0):
    d = fixture.load(fixture.FIXTURE_DIR / f"{block}_{i}.json")
    module, consts = from_image.BLOCKS[block](d["image"])
    prog = lower(module)
    ref = gi.run(module, {"x": d["inputs"]["x"]}, consts).values
    return prog, {**consts, "x": d["inputs"]["x"], **ref}, ref


@functools.cache
def lut_case():
    rng = np.random.default_rng(5)
    g = GraphBuilder("lut")
    y = g.lut("y", g.input("x", [3, 5]), g.const("t", [256]))
    g.output(y)
    m = g.module()
    bufs = {"x": rng.integers(-128, 128, (3, 5)), "t": rng.integers(-128, 128, 256)}
    ref = gi.run(m, {"x": bufs["x"]}, {"t": bufs["t"]}).values
    return lower(m), {**bufs, **ref}, ref


# (block, kernel name) covering every lowered op kind on both fixtures
OPS = [("mlp_block", "h"), ("mlp_block", "gate"), ("mlp_block", "act"), ("mlp_block", "down"), ("mlp_block", "out"),
       ("attention", "kT"), ("attention", "scores"), ("attention", "p"), ("attention", "o"), ("lut", "y")]


def case(block, name):
    prog, bufs, ref = lut_case() if block == "lut" else block_case(block)
    return prog.kernel(name), bufs, ref[name]


