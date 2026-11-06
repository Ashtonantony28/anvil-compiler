"""Verilator backend (SG0 spike, docs/decisions/ADR-001-verilator.md): the sv2v output built with Verilator and a C++
harness (verilator_harness.cpp) that replays the cocotb testbench's timing, cached per GpuParams like the RTL build."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .api import GpuParams, Job, LaunchResult, RunResult
from .rtl import CACHE, SimulationError, _locked, _sh, build_key, sv2v_output

HARNESS = Path(__file__).with_name("verilator_harness.cpp")
# The unroll limits let Verilator flatten the controller's channel x consumer x consumer loops at COALESCE=1 (else
# BLKLOOPINIT: "Delayed assignment to array inside for loops").
VERILATOR_FLAGS = ["--cc", "--exe", "--build", "--top-module", "gpu", "-O3", "--x-assign", "fast", "--x-initial", "fast",
                   "-Wno-fatal", "-Wno-lint", "-Wno-style", "-Wno-UNOPTFLAT", "--unroll-count", "1024",
                   "--unroll-stmts", "1000000"]
CSC = "channel_serving_consumer"


def verilator_source(text: str) -> str:
    """Rewrite sv2v output so Verilator 5.020 simulates it like iverilog.

    controller.sv writes channel_serving_consumer with both blocking (`= 1`, claim a consumer, seen by later loop
    iterations) and non-blocking (`<= 0`, release) assignments in one always block. Verilator rejects that
    (BLKANDNBLK), and waiving it loses the blocking writes' visibility (extra and missed coalesced requests). The
    rewrite keeps Verilog semantics with blocking shadows: csc_b is the variable as the block sees it, csc_nm/csc_nv
    collect the non-blocking writes, and one non-blocking write at the end applies them over csc_b. tests/runtime
    builds iverilog on this text too and checks it against the unmodified build."""
    start = text.index("module controller (")
    end = text.index("endmodule", start)
    body = text[start:end]
    decl = f"reg [NUM_CONSUMERS - 1:0] {CSC};"
    always = "always @(posedge clk)\n\t\tif (reset) begin"
    rules = [
        (decl, decl + "\n\treg [NUM_CONSUMERS - 1:0] csc_b, csc_nm, csc_nv;", 1),
        (always, "always @(posedge clk) begin\n\t\tcsc_b = " + CSC + ";\n\t\tcsc_nm = 0;\n\t\tcsc_nv = 0;\n"
                 "\t\tif (reset) begin", 1),
        (f"{CSC} <= 0;", "begin csc_nm = {NUM_CONSUMERS {1'b1}}; csc_nv = 0; end", 1),
    ]
    for old, new, n in rules:
        if body.count(old) != n:
            raise ValueError(f"verilator_source: expected {n} x {old!r} in module controller, found {body.count(old)}")
        body = body.replace(old, new)
    body = re.sub(CSC + r"\[(\w+|\w+\[\w+\])\] <= 1'b0;", r"begin csc_nm[\1] = 1'b1; csc_nv[\1] = 1'b0; end", body)
    body = re.sub(CSC + r"\[(\w+)\] = 1'b1;", r"csc_b[\1] = 1'b1;", body)
    body = re.sub(r"!" + CSC + r"\[(\w+)\]", r"!csc_b[\1]", body)
    left = body.count(CSC) - body.count(f"csc_b = {CSC};") - body.count(decl)
    if left:
        raise ValueError(f"verilator_source: {left} uses of {CSC} not rewritten")
    body = body.rstrip() + f"\n\t\t{CSC} <= (csc_b & ~csc_nm) | (csc_nv & csc_nm);\n\tend\n"
    return text[:start] + body + text[end:]


def build_verilator(params: GpuParams) -> Path:
    text = verilator_source(sv2v_output().read_text())
    d = CACHE / "verilator" / build_key(params, "verilator", "\n".join([HARNESS.read_text(), *VERILATOR_FLAGS, text]))
    exe = d / "Vgpu"
    if exe.exists():
        return exe
    with _locked(d.with_suffix(".lock")):
        if not exe.exists():
            d.mkdir(parents=True, exist_ok=True)
            gpu_v = d / "gpu.v"
            gpu_v.write_text(text)
            obj = d / "obj"
            shutil.rmtree(obj, ignore_errors=True)
            obj.mkdir(parents=True)
            cflags = f"-O2 -DADDR_BITS={params.DATA_MEM_ADDR_BITS} -DDATA_BITS={params.DATA_MEM_DATA_BITS}"
            _sh(["verilator", *VERILATOR_FLAGS, "-j", str(os.cpu_count() or 1), "-Mdir", str(obj), "-o", "Vgpu",
                 "-CFLAGS", cflags, *(f"-G{k}={v}" for k, v in params.as_dict().items()), str(gpu_v), str(HARNESS)],
                d, "verilator")
            (d / "params.json").write_text(json.dumps(params.as_dict(), indent=1) + "\n")
            (obj / "Vgpu").rename(exe)
            shutil.rmtree(obj, ignore_errors=True)
    return exe


def run_verilator(job: Job, timeout_cycles: int) -> RunResult:
    exe = build_verilator(job.params)
    runs = CACHE / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runs) as tmp:
        job_path, result_path = Path(tmp) / "job.txt", Path(tmp) / "result.json"
        mask = (1 << job.params.DATA_MEM_DATA_BITS) - 1
        parts = [job.mem_latency, timeout_cycles, len(job.data), *(int(w) & mask for w in job.data), len(job.launches)]
        for ln in job.launches:
            parts += [ln.threads, len(ln.program), *map(int, ln.program)]
        job_path.write_text(" ".join(map(str, parts)) + "\n")
        r = subprocess.run([str(exe), str(job_path), str(result_path)], capture_output=True, text=True)
        if r.returncode == 3:
            raise TimeoutError(r.stderr.strip())
        if r.returncode or not result_path.exists():
            raise SimulationError(f"verilator harness failed ({r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        res = json.loads(result_path.read_text())
    return RunResult(res["memory"], [LaunchResult(ln["cycles"], ln["counters"]) for ln in res["launches"]], {})
