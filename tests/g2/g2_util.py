"""Shared state for the G2 tests (cached per process): the target model, its quantization, the held-out inputs
with their graph-interpreter and PyTorch references, and the compiled model per preset at 16-bit addresses.
Torch tests fail (not skip) without NANOGPT_DIR."""
import os

import pytest

from anvil import g2_report as R

_STATE = {}


def state():
    pytest.importorskip("torch")
    if not os.environ.get("NANOGPT_DIR"):
        pytest.fail("torch tests need NANOGPT_DIR (nanoGPT Extended @ 03dbab8)")
    if not _STATE:
        m, q = R.build()
        _STATE.update(model=m, q=q, refs=R.References(m, q, R.inputs()), cm={}, verilator={})
    return _STATE


def compiled(preset):
    s = state()
    if preset not in s["cm"]:
        s["cm"][preset] = R.compile_model(s["q"], preset)
    return s["cm"][preset]


def verilator_run(preset, i=0):
    """Checked Verilator run of input i (cached)."""
    s = state()
    key = (preset, i)
    if key not in s["verilator"]:
        s["verilator"][key] = R.check(compiled(preset), s["refs"], i, "verilator")[0]
    return s["verilator"][key]
