"""The sweep's workloads as (model, inputs, reference) triples, independent of torch at sweep time.

    wl = load("mlp_block")         # Workload: name, model (module, consts), x (quantized ints), golden {sym: ints}
    wl = load("attention")
    wl = load("model")             # the SG5 model (2 Blocks, d=32, 2 heads, T=16, i16 residual, contract v2)

mlp_block / attention: the xf fixtures `mlp_block_0` / `attention_0` as graph IR (anvil.frontend.from_image), the
fixture input and the fixture goldens (= xf.golden, C4 level 1).

model: the SG5a/SG9 model as `anvil.model.quantize_model(build_model(NANOGPT_DIR))` produces it (what `anvil.g2_report`
compiles), frozen once to data/codesign/model_graph.json.gz (graph IR text + consts) by `freeze_model()` so the
sweep and its tests run without torch. Its reference is the graph interpreter on one held-out input
(`anvil.model.heldout_domain(1)`, quantized by the input scale stored with the graph): every compiled chain must
reproduce it bit-exactly.
"""
from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_GRAPH = ROOT / "data" / "codesign" / "model_graph.json.gz"
WORKLOADS = ("mlp_block", "attention", "model")


@dataclass
class Workload:
    name: str
    model: tuple = field(repr=False)            # (graph module, consts)
    x: np.ndarray = field(repr=False)           # quantized input of the single graph input
    golden: dict = field(repr=False)            # sym -> ints every compiled run must reproduce
    addr_bits: int = 12                         # DATA_MEM_ADDR_BITS the workload needs


def freeze_model(path=MODEL_GRAPH, nanogpt_dir=None):
    """Capture + quantize the SG5 model from torch (NANOGPT_DIR) and freeze its graph IR, consts and one held-out
    input (quantized)."""
    from .. import model as M
    from ..ir import print_ir
    nd = nanogpt_dir or os.environ.get("NANOGPT_DIR")
    if not nd:
        raise RuntimeError("NANOGPT_DIR is not set")
    q = M.quantize_model(M.build_model(nd))
    x = M.quantize_x(q, M.heldout_domain(1)[0])
    body = {"ir": print_ir(q.module),
            "consts": {k: np.asarray(v).astype(int).ravel().tolist() for k, v in q.consts.items()},
            "shapes": {k: list(np.shape(v)) for k, v in q.consts.items()},
            "x": np.asarray(x).astype(int).tolist(),
            "note": "anvil.codesign.workloads.freeze_model: quantize_model(build_model(NANOGPT_DIR)), contract v2; "
                    "x = quantize_x(q, heldout_domain(1)[0])"}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(body, f, sort_keys=True)


def _model():
    from ..interp import graph as gi
    from ..ir import parse_ir
    with gzip.open(MODEL_GRAPH, "rt") as f:
        body = json.load(f)
    module = parse_ir(body["ir"])
    consts = {k: np.asarray(v, dtype=np.int64).reshape(body["shapes"][k]) for k, v in body["consts"].items()}
    x = np.asarray(body["x"], dtype=np.int64)
    vals = gi.run(module, {"x": x}, consts).values
    from .. import model as M
    return Workload("model", (module, consts), x, {M.OUT: np.asarray(vals[M.OUT])}, addr_bits=M.ADDR_BITS)


def load(name: str) -> Workload:
    if name == "model":
        return _model()
    if name not in ("mlp_block", "attention"):
        raise KeyError(f"unknown workload {name!r}: {WORKLOADS}")
    from ..tune.g3_report import fixture_model
    model, d = fixture_model(name)
    return Workload(name, model, np.asarray(d["inputs"]["x"]), {k: np.asarray(v) for k, v in d["golden"].items()})
