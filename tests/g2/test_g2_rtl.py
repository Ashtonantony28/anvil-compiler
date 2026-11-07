"""SG5b Accept (G2), RTL part: the compiled target model (both presets, DATA_MEM_ADDR_BITS=16) on Verilator for 32
held-out inputs and on the iverilog reference RTL for one, bit-exact with the graph interpreter and isasim, within
TOL of PyTorch, with input-independent cycles equal between the two simulators; docs/g2-report.md is current."""
import pytest

from anvil import g2_report as R
from anvil import model as M

from g2_util import compiled, state, verilator_run

pytestmark = [pytest.mark.rtl, pytest.mark.torch]
PRESETS = R.PRESETS


@pytest.mark.parametrize("preset", PRESETS)
def test_verilator_bit_exact_on_heldout(preset):
    s, cm = state(), compiled(preset)
    n = len(s["refs"].xq)
    assert n >= 32
    first = verilator_run(preset, 0)
    for i in range(1, n):
        r = verilator_run(preset, i)
        assert r.cycles == first.cycles, f"cycles depend on input {i}"
    iso = cm.run(s["refs"].xq[0], "isasim", quantized=True)
    assert first.result.memory[:first.job.end] == iso.result.memory[:iso.job.end]
    assert all(c > 0 for c in first.cycles.values())


@pytest.mark.slow
@pytest.mark.parametrize("preset", PRESETS)
def test_iverilog_one_input(preset):
    """The reference RTL (about 10 minutes per preset): memory equal to isasim, cycles equal to Verilator."""
    s, cm = state(), compiled(preset)
    r = R.check(cm, s["refs"], 0, "rtl")[0]
    iso = cm.run(s["refs"].xq[0], "isasim", quantized=True)
    assert r.result.memory[:r.job.end] == iso.result.memory[:iso.job.end]
    assert r.cycles == verilator_run(preset, 0).cycles


def test_report_is_current():
    text = R.REPORT.read_text()
    res = {p: {"cm": compiled(p), "cycles": verilator_run(p, 0).cycles} for p in PRESETS}
    for table in (R.cycle_table(res), R.summary_table(res), R.histogram_table(res)):
        assert table in text, "regenerate: python3 -m anvil.g2_report --rtl"
    assert f"{M.OUT}" in text
