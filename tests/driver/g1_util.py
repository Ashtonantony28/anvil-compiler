"""Shared helpers for the G1 tests: the xf Block from nanoGPT, compiled by anvil.compile (cached per process).
Torch tests fail (not skip) without NANOGPT_DIR."""
import os

import numpy as np
import pytest

from xf import fixture

OUT = {"mlp_block": "out", "attention": "y"}
BLOCKS = ("mlp_block", "attention")
_MODULES, _COMPILED = {}, {}


def fx(block, i=0):
    return fixture.load(fixture.FIXTURE_DIR / f"{block}_{i}.json")


def modules():
    pytest.importorskip("torch")
    if not os.environ.get("NANOGPT_DIR"):
        pytest.fail("torch tests need NANOGPT_DIR (nanoGPT Extended @ 03dbab8)")
    if not _MODULES:
        from anvil import g1_report
        _MODULES.update(g1_report.torch_modules())
    return _MODULES


def compiled(block, preset="default", params=None):
    from anvil import g1_report
    key = (block, preset, params)
    if key not in _COMPILED:
        _COMPILED[key] = g1_report.compile_block(modules()[block], block, params, preset)
    return _COMPILED[key]


def assert_golden(cm, block, i, backend="isasim"):
    """Run fixture case i with the keep-all plan (every segment) and the reuse plan (the output)."""
    d = fx(block, i)
    full = cm.run(d["inputs"]["x"], backend, quantized=True, keep="all")
    for seg, want in d["golden"].items():
        assert np.array_equal(full.ints[seg], want), (block, i, seg, backend)
    reuse = cm.run(d["inputs"]["x"], backend, quantized=True)
    out = OUT[block]
    assert np.array_equal(reuse.ints[out], d["golden"][out]), (block, i, backend)
    if backend == "isasim":
        for r in (full, reuse):
            assert not any(r.result.flags[k] for k in ("store_races", "stores_conflict", "divergent"))
    return full, reuse
