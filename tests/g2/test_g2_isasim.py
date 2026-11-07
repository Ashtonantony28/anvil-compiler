"""SG5b Accept (G2), isasim part: the SG5a target model compiled at DATA_MEM_ADDR_BITS=16 with the default and
tiled presets fits the address space and the 256-word program limit without spills, and on 32 held-out inputs
spanning the contract's domain its output (and every readable value) is bit-exact with the graph interpreter,
within TOL of PyTorch, with no isasim divergence / race flags (C4 levels 4 and 6)."""
import numpy as np
import pytest

from anvil import g2_report as R
from anvil import model as M
from anvil.memplan import MemoryPlanError
from anvil.runtime import GpuParams

from g2_util import compiled, state

pytestmark = pytest.mark.torch
PRESETS = R.PRESETS


@pytest.mark.parametrize("preset", PRESETS)
def test_compiles_within_limits(preset):
    cm = compiled(preset)
    rows = R.kernel_rows(cm)
    assert len(rows) == 40
    assert all(0 < r[3] <= 256 for r in rows)                       # program words
    assert all(r[6] == 0 for r in rows), [r[0] for r in rows if r[6]]    # no spill frames
    assert cm.plan.footprint <= min(1 << M.ADDR_BITS, M.FOOTPRINT_WORDS)
    assert cm.plan.regions["consts"][1] == 25_728
    assert sum(r[4] for r in rows) == sum(cm.kernels[k.name].launches for k in cm.program.kernels)
    if preset == "tiled":                       # every tiled-preset matmul but the merge-heads one is tiled
        mm = [r for r in rows if r[1] in ("qlinear", "qmatmul")]
        assert {r[0] for r in mm if r[2] != "matmul_tiled"} == {"L0.y", "L1.y"}


def test_needs_16_bit_addresses():
    from anvil.driver import compile
    with pytest.raises(MemoryPlanError):
        compile(state()["q"], params=GpuParams(DATA_MEM_ADDR_BITS=14))


@pytest.mark.parametrize("preset", PRESETS)
def test_isasim_bit_exact_on_heldout(preset):
    s, cm = state(), compiled(preset)
    errs = [R.check(cm, s["refs"], i, "isasim")[1] for i in range(len(s["refs"].xq))]
    assert len(errs) >= 32 and max(errs) <= M.TOL_LSB


@pytest.mark.parametrize("preset", PRESETS)
def test_isasim_every_segment_with_keep_all(preset):
    """keep="all": every graph value (40 segments per layer pair) equals the graph interpreter."""
    s, cm = state(), compiled(preset)
    r = R.check(cm, s["refs"], 1, "isasim", keep="all")[0]
    segs = [k for k in M.SEGMENTS if k in r.ints]
    assert len(segs) == len(M.SEGMENTS)


def test_presets_agree():
    s = state()
    a = compiled("default").run(s["refs"].xq[2], quantized=True)
    b = compiled("tiled").run(s["refs"].xq[2], quantized=True)
    assert np.array_equal(a.ints[M.OUT], b.ints[M.OUT])
