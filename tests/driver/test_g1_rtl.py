"""SG4b Accept (G1, RTL part): the compiled MLP block and attention head from nanoGPT's Block on the iverilog RTL
(fixture case 0, both schedule presets, both memory plans) and on Verilator (cases 1-3), golden-equal on every
segment and equal to isasim; docs/g1-report.md's default-configuration rows are current."""
import re
from pathlib import Path

import numpy as np
import pytest

from anvil import g1_report

from g1_util import BLOCKS, OUT, assert_golden, compiled, fx, modules

pytestmark = [pytest.mark.rtl, pytest.mark.torch]
REPORT = Path(__file__).resolve().parents[2] / "docs" / "g1-report.md"


@pytest.mark.parametrize("preset", ["default", "tiled"])
@pytest.mark.parametrize("block", BLOCKS)
def test_case0_on_iverilog(block, preset):
    cm = compiled(block, preset)
    full, reuse = assert_golden(cm, block, 0, "rtl")
    model = cm.run(fx(block)["inputs"]["x"], quantized=True, keep="all")
    assert full.result.memory[:full.job.end] == model.result.memory[:model.job.end]
    assert all(c > 0 for c in full.cycles.values()) and full.cycles == reuse.cycles
    vl = cm.run(fx(block)["inputs"]["x"], "verilator", quantized=True, keep="all")
    assert vl.cycles == full.cycles                        # the report's Verilator cycles are the RTL's
    tol = fx(block)["tol"]
    assert np.abs(reuse.output - fx(block)["reference"][OUT[block]]).max() <= tol


@pytest.mark.parametrize("preset", ["default", "tiled"])
@pytest.mark.parametrize("block", BLOCKS)
def test_cases_1_to_3_on_verilator(block, preset):
    cm = compiled(block, preset)
    for i in (1, 2, 3):
        assert_golden(cm, block, i, "verilator")


def _report_rows(text):
    rows = {}
    for line in text.splitlines():
        m = re.match(r"\| (\w+) \| ([^|]+) \| (\w+) \| (\w+) \| \**(\w+)\** \| \**(\d+)\** \| \**(\d+)\** \|", line)
        if m:
            rows[m.groups()[:5]] = (int(m.group(6)), int(m.group(7)))
    return rows


def test_report_is_current():
    """Re-measure the default configuration on Verilator: docs/g1-report.md's rows and footprints match."""
    rows, plans = g1_report.collect([""], modules=modules())
    want = _report_rows(g1_report.cycle_table(rows))
    have = _report_rows(REPORT.read_text())
    assert want and all(have.get(k) == v for k, v in want.items()), "regenerate: python3 -m anvil.g1_report"
    assert g1_report.footprint_table(plans) in REPORT.read_text()
    for block in BLOCKS:                                  # compiled chains beat the hand chains in total
        for preset in ("default", "tiled"):
            hand, comp = want[(block, "default", preset, g1_report.SCHEDULES[preset], "total")]
            assert comp < hand, (block, preset)
