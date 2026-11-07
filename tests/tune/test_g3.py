"""SG6 Accept (G3), on the Verilog simulators (marked rtl; torch tests also need NANOGPT_DIR).

* The tuned MLP block chain (committed best schedules, replayed) at the xf-bench default params: < 40,329 cycles
  (the hand tiled chain), stretch <= 30,247; the tuned attention chain beats its hand tiled chain (39,498).
* A real, seeded search (fresh measurement cache) is reproducible and beats the driver's presets.
* The tuner picks a naive-style schedule for the 1x8x8 matmul on its own (no smem staging, one output per thread).
* The whole tuned chain is re-verified on iverilog RTL: golden-equal, cycles equal to Verilator's.
"""
import json
import os

import pytest

from anvil.driver import compile
from anvil.runtime import GpuParams
from anvil.tune import BestDB, MeasureCache, tune_kernel
from anvil.tune.g3_report import HAND_TARGET, STRETCH, run_chain
from tune_util import program, task

pytestmark = pytest.mark.rtl
HAND_TILED = {"mlp_block": 40329, "attention": 39498}
_CM = {}


def tuned(block):
    if block not in _CM:
        _, _, model, d = program(block)
        cm = compile(model, schedules="tuned", tune={"search": False, "db": BestDB(write="/nonexistent/x.json")})
        assert {v["how"] for v in cm.tuning.values()} == {"db"}, \
            "committed best schedules are stale: rerun python3 -m anvil.tune.g3_report"
        _CM[block] = cm
    return _CM[block]


def test_g3_tuned_mlp_chain():
    _, _, _, d = program("mlp_block")
    cycles = run_chain(tuned("mlp_block"), d, "verilator")
    total = sum(cycles.values())
    print("tuned mlp chain", cycles, total)
    assert total < HAND_TARGET
    assert total <= STRETCH


def test_tuned_attention_chain():
    _, _, _, d = program("attention")
    total = sum(run_chain(tuned("attention"), d, "verilator").values())
    print("tuned attention chain", total)
    assert total < HAND_TILED["attention"]


def test_isolated_cycles_match_chain():
    """A kernel's cycles measured alone (the tuner) equal its cycles in the chain (default params: no caches)."""
    _, _, _, d = program("mlp_block")
    cm = tuned("mlp_block")
    cycles = run_chain(cm, d, "verilator")
    for k, info in cm.tuning.items():
        assert info["cycles"] == cycles[k], k


def test_real_seeded_search_reproducible(tmp_path):
    """A real short search on Verilator from an empty cache, twice: the same choice; better than both presets."""
    t = task("mlp_block", "gate")
    a = tune_kernel(t, budget=48, seed=0, workers=4, cache=MeasureCache(tmp_path / "a"))
    b = tune_kernel(t, budget=48, seed=0, workers=4, cache=MeasureCache(tmp_path / "b"))
    assert a.timed > 0 and a.failed == 0
    assert (a.best, a.cycles, a.history) == (b.best, b.cycles, b.history)
    assert a.default_cycles == 11511                    # matmul_naive (docs/g1-report.md)
    assert a.cycles < 7627                              # matmul_tiled


def test_1x8x8_picks_naive_style(tmp_path):
    t = task("mm1x8x8", "y")
    r = tune_kernel(t, budget=96, seed=0, workers=4, cache=MeasureCache(tmp_path))
    trace = json.loads(r.trace)
    print("1x8x8", r.best, r.cycles)
    assert not any(step[0] == "cache_read" for step in trace)          # operands straight from global memory
    p = r.best.p
    assert (r.best.sketch == "tpl" and p["template"] == "matmul_naive") or \
           (r.best.sketch == "mm" and p["ti"] == p["tj"] == 1)        # one output per thread
    assert r.cycles < 3848                              # the hand tiled kernel (sim/bench/xf-matmul.json)


@pytest.mark.parametrize("block", ["mlp_block", "attention"])
def test_tuned_chain_on_iverilog_rtl(block):
    _, _, _, d = program(block)
    cm = tuned(block)
    rtl = run_chain(cm, d, "rtl")
    assert rtl == run_chain(cm, d, "verilator")


@pytest.mark.torch
@pytest.mark.parametrize("block", ["mlp_block", "attention"])
def test_tuned_from_torch(block):
    """From the nanoGPT Block (G1 path) the kernels are the same IR: every one replays the committed schedule,
    and the chain is golden-equal on iverilog RTL with the same cycles as the graph-IR path."""
    if not os.environ.get("NANOGPT_DIR"):
        pytest.fail("torch tests need NANOGPT_DIR (nanoGPT Extended @ 03dbab8)")
    pytest.importorskip("torch")
    import torch
    from anvil import g1_report
    from xf import export as xe
    mod = g1_report.torch_modules()[block]
    ex = torch.zeros(1, xe.T, xe.N_EMBD, dtype=torch.float64)
    cm = compile(mod, (ex,), g1_report.calib(block), GpuParams(), "tuned", headroom=xe.HEADROOM,
                 names=g1_report.NAMES[block], sym_name=block,
                 tune={"search": False, "db": BestDB(write="/nonexistent/x.json")})
    assert {v["how"] for v in cm.tuning.values()} == {"db"}
    _, _, _, d = program(block)
    assert run_chain(cm, d, "rtl") == run_chain(tuned(block), d, "verilator")
