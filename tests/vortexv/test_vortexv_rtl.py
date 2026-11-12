"""Anvil-compiled graphs on every Vortex-V RTL core configuration (cocotb + Icarus), lockstep with the ISS."""
import os
import shutil
import sys
from pathlib import Path

import pytest

from anvil import vortexv
from anvil.vortexv import _vv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

pytestmark = [pytest.mark.rtl,
              pytest.mark.skipif(not vortexv.available(), reason="vortexv submodule not checked out"),
              pytest.mark.skipif(shutil.which("iverilog") is None, reason="needs iverilog")]


def _cores():
    if not vortexv.available():
        return []
    _vv.rvasm()                                    # puts verif/core on sys.path
    import cores
    return list(cores.CORES)


@pytest.mark.parametrize("core", _cores())
def test_anvil_graphs_on_rtl(core):
    import cores
    out = HERE / "sim_build" / f"{core}_cycles.json"
    out.parent.mkdir(exist_ok=True)
    cores.build_and_test(core, "vv_rtl_cocotb", HERE,
                         extra_pythonpath=[ROOT, ROOT / "sim", _vv.VV_CORE, _vv.VV],
                         env={"VV_CYCLES_OUT": str(out), "VV_MODEL": "0"})
    assert out.exists()


@pytest.mark.slow
@pytest.mark.parametrize("core", _cores())
def test_model_on_rtl(core):
    """The SG5a 2-layer model (462k instructions) in lockstep on the RTL: minutes per core."""
    import cores
    out = HERE / "sim_build" / f"{core}_model_cycles.json"
    out.parent.mkdir(exist_ok=True)
    cores.build_and_test(core, "vv_rtl_cocotb", HERE,
                         extra_pythonpath=[ROOT, ROOT / "sim", _vv.VV_CORE, _vv.VV],
                         env={"VV_CYCLES_OUT": str(out), "VV_MODEL": "1"})
    assert out.exists()
