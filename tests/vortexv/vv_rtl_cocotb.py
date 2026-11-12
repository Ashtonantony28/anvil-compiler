"""cocotb body for test_vortexv_rtl.py: Anvil-compiled graphs on a Vortex-V RTL core, in lockstep with the ISS.

Each graph runs through Vortex-V's own co-simulation harness (verif/core/harness.py): every retired instruction
is compared with the ISS, the whole data region must match byte for byte at the end, and the decoded values must
equal Anvil's golden interpreter. Cycle counts go to $VV_CYCLES_OUT (JSON) for docs/vortexv-backend.md.
"""
import json
import os

import cocotb
import numpy as np

import vv_graphs as vg
from anvil import vortexv
from anvil.interp import graph as gi

RANDOM_SEED = int(os.environ.get("VV_RANDOM_SEED", "11"))
MAX_CYCLES = 2_000_000


def _cases():
    if os.environ.get("VV_MODEL") == "1":            # the 2-layer d=32 model (data/codesign, torch-free)
        from anvil.codesign import workloads
        w = workloads.load("model")
        m, c = w.model
        x = {"x": w.x}
        return [("model", m, c, x, gi.run(m, x, c))]
    out = []
    for block in vg.BLOCKS:
        m, c, x = vg.fixture_case(block, 0)
        out.append((f"{block}_0", m, c, x, gi.run(m, x, c)))
    m, c, x, want = vg.random_case(RANDOM_SEED)
    out.append((f"random_{RANDOM_SEED}", m, c, x, want))
    return out


@cocotb.test()
async def anvil_graphs_cosim(dut):
    from harness import cosim_lockstep
    record = {}
    for name, m, c, x, want in _cases():
        p = vortexv.compile(m, c)
        stub, data = p.image(x)
        retires, state, iss = await cosim_lockstep(dut, name, stub, data=data, max_cycles=MAX_CYCLES,
                                                   mem_check=(vortexv.DATA_BASE, p.data_end - vortexv.DATA_BASE))
        outs, values = p.decode(iss.mem.load)      # mem_check: ISS memory == RTL memory over the data region
        for k, v in want.values.items():
            assert np.array_equal(values[k], np.asarray(v)), f"{name}: value {k} differs from the golden model"
        for a, b in zip(outs, want.outputs):
            assert np.array_equal(a, b), f"{name}: output differs"
        record[name] = {"instret": len(retires), "cycles": state["cycles"]}
        dut._log.info("%s: %d instructions, %d cycles", name, len(retires), state["cycles"])
    path = os.environ.get("VV_CYCLES_OUT")
    if path:
        with open(path, "w") as f:
            json.dump(record, f, indent=1)
