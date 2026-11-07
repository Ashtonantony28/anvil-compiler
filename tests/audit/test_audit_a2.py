"""Audit A2 (numerics, performance-claim honesty, API/docs): strict-xfail repros for every critical/high/medium
finding in AUDIT-A2.md. Each test states the behaviour the project should have and passes once the finding is
fixed (then strict xfail turns it into a failure, so the marker must be removed with the fix).

Fast tests (no torch, no RTL) run in `make check`; the torch ones need NANOGPT_DIR:
    NANOGPT_DIR=... python3 -m pytest -q -m torch tests/audit/test_audit_a2.py
"""
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "cost"


# ------------------------------------------------------------------------------------------------ helpers

def _fixture_graph(block="mlp_block"):
    from anvil.frontend import from_image
    from xf import fixture
    fx = fixture.load(fixture.FIXTURE_DIR / f"{block}_0.json")
    mod, consts = from_image.BLOCKS[block](fx["image"])
    return mod, consts, fx


def _op(mod, sym):
    from anvil.ir import graph as g
    for op in g.funcs(mod)[0].body.block.ops:
        if getattr(op, "sym_str", None) == sym:
            return op
    raise KeyError(sym)


def _scale_prop(op, key, f):
    a = op.properties[key]
    op.properties[key] = type(a)(int(round(a.value.data * f)), a.type)


def _nanogpt():
    d = os.environ.get("NANOGPT_DIR")
    if not d:
        pytest.fail("torch tests need NANOGPT_DIR (nanoGPT Extended @ 03dbab8)")
    return d


_MODEL = {}


def _target():
    pytest.importorskip("torch")
    if not _MODEL:
        from anvil import model as M
        m = M.build_model(_nanogpt())
        _MODEL.update(m=m, q=M.quantize_model(m))
    return _MODEL["m"], _MODEL["q"]


# ------------------------------------------------------------------------------------------------ A: numerics

@pytest.mark.torch
def test_h1_tolerance_holds_for_small_inputs_with_matched_calibration():
    from anvil import model as M
    m, _ = _target()
    sigma = 0.05
    q = M.quantize_model(m, calib=M.calib_inputs() * sigma)          # calibrated for the input scale it sees
    err = M.measure(q, M.heldout_inputs(4000) * sigma, segments=False)[0]
    assert err.max() <= M.TOL_LSB, f"max {err.max():.3f} LSB > TOL {M.TOL_LSB}"


@pytest.mark.torch
def test_m1_compiled_acceptance_catches_a_3pct_output_gain_error():
    from anvil import g2_report as R
    from anvil import model as M
    from anvil.frontend.quantize import Quantized
    m, q0 = _target()
    mod = q0.module.clone()
    op = _op(mod, M.OUT)
    _scale_prop(op, "Ma", 1.03)
    _scale_prop(op, "Mb", 1.03)
    q = Quantized(mod, q0.consts, q0.scales, q0.bounds, q0.graph, q0.syms)
    _, stats, _ = M.measure(q, M.heldout_inputs(2000))
    assert M.OUT in M.segment_errors(stats)                        # the graph-level contract does see it
    refs = R.References(m, q, R.inputs(8))
    cm = R.compile_model(q, "default")
    caught = 0
    for i in range(8):
        try:
            R.check(cm, refs, i, "isasim")
        except AssertionError:
            caught += 1
    assert caught, "every compiled run of a model with a 3 % output gain error passed the G2 acceptance check"


# ------------------------------------------------------------------------------------------------ B: performance

def test_m2_headline_cost_model_is_cheaper_than_the_measurement_it_replaces():
    """A2-M2: G4 rested on tier 1, an RTL replay that costs more wall time than the Verilator run it replaces.
    Fixed by stating G4 on tier2_static (docs/g4-report.md headline); its inputs (functional profile + closed form,
    the rows' `t_model`) must cost well under the measurement, and tier 1's own time (`t_tier1`) is reported
    separately rather than hidden."""
    rows = [json.loads(line) for line in gzip.open(DATA / "rows.jsonl.gz", "rt")]
    t_model = sum(r["t_model"] for r in rows)
    t_sim = sum(r["t_sim"] for r in rows)
    assert all("t_tier1" in r for r in rows)
    assert t_model <= 0.5 * t_sim, f"headline-model wall {t_model:.0f} s vs Verilator {t_sim:.0f} s"
    report = (ROOT / "docs" / "g4-report.md").read_text()
    assert "## Headline (G4, stated on tier2_static)" in report


def test_m3_heldout_kernels_have_no_feature_identical_twin_in_train():
    kern = json.loads((DATA / "kernels.json").read_text())
    progs = [json.loads(line) for line in gzip.open(DATA / "programs.jsonl.gz", "rt")]

    def key(p):
        pf = {k: v for k, v in p["pf"].items() if k != "footprint"}
        return json.dumps(pf, sort_keys=True), p["W"], p["stats"].get("words")

    train = {key(p) for p in progs if p["ok"] and kern[p["kid"]]["split"] == "train"}
    dup = [p["pid"] for p in progs if p["ok"] and kern[p["kid"]]["split"] == "test" and key(p) in train]
    assert not dup, f"{len(dup)} held-out programs duplicate a training program, e.g. {dup[:3]}"


# ------------------------------------------------------------------------------------------------ C: API

def test_m4_compile_of_graph_ir_proves_no_overflow():
    import anvil
    mod, consts, _ = _fixture_graph()
    _scale_prop(_op(mod, "gate"), "M", 4096)                      # acc*M reaches ~7.3e12
    with pytest.raises(OverflowError):
        anvil.compile((mod, consts))


def test_m5_run_rejects_out_of_range_integer_inputs():
    import anvil
    mod, consts, fx = _fixture_graph()
    cm = anvil.compile((mod, consts))
    x = np.asarray(fx["inputs"]["x"]) * 50
    with pytest.raises(ValueError):
        cm.run(x, quantized=True)


def test_m6_compile_without_consts_is_an_error():
    import anvil
    mod, _, _ = _fixture_graph()
    with pytest.raises((ValueError, KeyError)):
        anvil.compile(mod)


def test_m7_unknown_keyword_is_an_error():
    import anvil
    mod, consts, _ = _fixture_graph()
    with pytest.raises(TypeError):
        anvil.compile((mod, consts), shedules="tiled")


# ------------------------------------------------------------------------------------------- C: clean checkout

def test_m8_masked_split_then_compute_at_is_rejected_or_bit_exact():
    import sys
    sys.path.insert(0, str(ROOT / "tests" / "schedule"))
    from anvil.schedule import Schedule, ScheduleError, Target
    from sched_cases import case
    k, bufs, ref = case("mlp_block", "h")
    s = Schedule(k, Target())
    s.apply("split", "i", 3, "mask")
    try:
        s.apply("compute_at", "stat.i", "i.o")
    except ScheduleError:
        return
    assert np.array_equal(s.run(bufs)["h"], ref)
