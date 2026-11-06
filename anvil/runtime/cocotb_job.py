"""Generic cocotb test module for the RTL backend: runs one Job (contract C2) on the DUT.

Environment: ANVIL_JOB names the Job JSON (api.Job.to_dict plus "timeout_cycles"), ANVIL_RESULT the result JSON this
writes: {"launches": [{cycles, counters}], "memory": [...]} or, on an exception, {"error": [type, message]}.

The loop is sim/test/helpers/runner.py::run_launches with the simulator's own helpers (launch, _run, Memory): one
clock, one data memory loaded once, a fresh program memory and a GPU reset per launch. It also reads the per-launch
warp_issues_total and bar_stall_cycles_total, which run_launches does not return.
"""
import json
import os
import traceback

import cocotb
from cocotb.clock import Clock

from test.helpers.memory import Memory
from test.helpers.runner import _run, _safe_int, data_bus
from test.helpers.setup import launch

COUNTERS = ("data_mem_requests", "data_mem_requests_coalesced", "icache_hits", "icache_misses", "dcache_hits",
            "dcache_misses")


async def run_job(dut, job: dict) -> dict:
    mem_latency, timeout = int(job["mem_latency"]), int(job["timeout_cycles"])
    p = job["params"]
    bits, addr = data_bus(dut)
    if (bits, addr) != (p["DATA_MEM_DATA_BITS"], p["DATA_MEM_ADDR_BITS"]):
        raise ValueError(f"DUT data bus is {bits}-bit x 2**{addr}, the job's params say "
                         f"{p['DATA_MEM_DATA_BITS']}-bit x 2**{p['DATA_MEM_ADDR_BITS']}")
    for name in ("THREADS_PER_BLOCK", "WARPS_PER_CORE", "NUM_CORES"):
        assert int(getattr(dut, name).value) == p[name], f"DUT {name} {int(getattr(dut, name).value)} != {p[name]}"
    channels = len(dut.data_mem_read_ready)
    data_memory = Memory(dut, addr, bits, channels, "data", latency=mem_latency)
    data_memory.load(job["data"])
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    results = []
    for ln in job["launches"]:
        program_memory = Memory(dut, 8, 16, 1, "program", latency=mem_latency)
        data_memory._pending_read = [0] * data_memory.channels
        data_memory._pending_write = [0] * data_memory.channels
        data_memory._served_read = [False] * data_memory.channels
        data_memory._served_write = [False] * data_memory.channels
        await launch(dut, program_memory, ln["program"], data_memory, None, ln["threads"])
        m = await _run(dut, program_memory, data_memory, ln["threads"], timeout)
        counters = {k: m[k] for k in COUNTERS}
        counters["warp_issues_total"] = _safe_int(dut.warp_issues_total.value)
        counters["bar_stall_cycles_total"] = _safe_int(dut.bar_stall_cycles_total.value)
        results.append({"cycles": m["cycles"], "counters": counters})
    dut.start.value = 0
    return {"launches": results, "memory": list(data_memory.memory)}


@cocotb.test()
async def anvil_job(dut):
    with open(os.environ["ANVIL_JOB"]) as f:
        job = json.load(f)
    out = os.environ["ANVIL_RESULT"]
    try:
        res = await run_job(dut, job)
    except Exception as e:
        with open(out, "w") as f:
            json.dump({"error": [type(e).__name__, str(e), traceback.format_exc()]}, f)
        raise
    with open(out, "w") as f:
        json.dump(res, f)
