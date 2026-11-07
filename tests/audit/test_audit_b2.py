"""Audit B2 (honesty of the G1-G5 claims, numerics of model contract v2): strict-xfail repros for every
critical/high/medium finding in AUDIT-B2.md. Each test states the behaviour the project should have and passes
once the finding is fixed (strict xfail then turns it into a failure, so the marker goes with the fix).

Fast tests (no torch, no RTL) run in `make check`; the torch one needs NANOGPT_DIR (about 2 min):
    NANOGPT_DIR=... python3 -m pytest -q -m torch tests/audit/test_audit_b2.py
"""
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

V1_OUT_LSB = 0.0686          # contract v1's output LSB (docs/model-contract.md "Unit"; ADR-004)
V1_TOL_LSB = 6.0             # contract v1's TOL, in that LSB


def _nanogpt():
    d = os.environ.get("NANOGPT_DIR")
    if not d:
        pytest.fail("torch tests need NANOGPT_DIR (nanoGPT Extended @ 03dbab8)")
    return d


# ------------------------------------------------------------------------- B2-M1: contract v2's TOL vs A2-H1

@pytest.mark.xfail(strict=True, reason="B2-M1: v2's TOL (5 OUT_UNIT = 0.79) is 1.9x looser than v1's (6 LSB = 0.41)")
def test_m1_v2_tol_is_not_looser_than_v1_in_absolute_terms():
    from anvil import model as M
    v2_abs = M.TOL_LSB * M.OUT_UNIT
    v1_abs = V1_TOL_LSB * V1_OUT_LSB
    assert v2_abs <= v1_abs * 1.001, (f"v2 TOL {v2_abs:.3f} absolute vs v1 TOL {v1_abs:.3f}: the contract got "
                                      f"{v2_abs / v1_abs:.2f}x looser while A2-H1 was being fixed")


@pytest.mark.torch
@pytest.mark.xfail(strict=True, reason="B2-M1: the unfixed v1 model (A2-H1's defect) passes the v2 TOL at sigma 0.02")
def test_m1_v2_tol_rejects_the_v1_model_at_small_sigma():
    pytest.importorskip("torch")
    from anvil import model as M
    m = M.build_model(_nanogpt())
    q1 = M.quantize_model(m, calib=M.calib_inputs(), residual="i8")       # contract v1's recipe (ADR-004 option A)
    err = M.measure(q1, M.heldout_inputs(4000) * 0.02, segments=False)[0]  # OUT_UNIT, as the A2-H1 test measures
    assert err.max() > M.TOL_LSB, (f"v1 model at sigma 0.02: max {err.max():.3f} OUT_UNIT <= TOL {M.TOL_LSB}: "
                                   "the contract cannot tell the A2-H1 defect from its fix")


# ------------------------------------------------------------------- B2-M2: G3 ratio vs the hand naive chain

@pytest.mark.xfail(strict=True, reason="B2-M2: PLAN says G3 ratios are reported against both hand chains; "
                                       "g3-report.md has only 'tuned / hand tiled' (mm1x8x8 0.30)")
def test_m2_g3_report_states_ratio_against_hand_naive():
    text = (ROOT / "docs" / "g3-report.md").read_text()
    header = next(line for line in text.splitlines() if line.startswith("| workload | config"))
    assert re.search(r"tuned / (hand naive|best hand)", header), header
