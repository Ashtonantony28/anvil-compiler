"""RTL backend: sv2v + iverilog build cached per GpuParams, one vvp run per Job (cocotb module anvil.runtime.cocotb_job).

The build is the sim Makefile's `gpu_v` + `compile` recipe, run in .anvil-cache/rtl/<key>/ instead of sim/build.
"""
from __future__ import annotations

import fcntl
import functools
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .api import GpuParams, Job, LaunchResult, RunResult

ROOT = Path(__file__).resolve().parents[2]
SIM = Path(os.environ.get("ANVIL_SIM", ROOT / "sim"))
CACHE = Path(os.environ.get("ANVIL_CACHE", ROOT / ".anvil-cache"))


class SimulationError(RuntimeError):
    pass


def gpu_sources() -> list[Path]:
    """The Makefile's GPU_SRC: src/*.sv without the warp scheduler testbench."""
    return sorted(p for p in (SIM / "src").glob("*.sv") if p.name != "warp_scheduler_tb.sv")


@functools.cache
def _tool_version(tool: str) -> str:
    try:
        r = subprocess.run([tool, "-V" if tool == "iverilog" else "--version"], capture_output=True, text=True)
    except FileNotFoundError:
        raise SimulationError(f"{tool} not found on PATH (see PLAN.md, Environment)") from None
    return (r.stdout or r.stderr).splitlines()[0] if (r.stdout or r.stderr) else "?"


def source_hash() -> str:
    h = hashlib.sha256()
    for p in gpu_sources():
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def build_key(params: GpuParams, tool: str = "iverilog", extra: str = "") -> str:
    h = hashlib.sha256("\n".join([tool, _tool_version("sv2v"), _tool_version(tool), source_hash(),
                                  params.key(), extra]).encode())
    return h.hexdigest()[:16]


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _sh(cmd: list[str], cwd: Path, what: str, **kw) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, **kw)
    if r.returncode:
        raise SimulationError(f"{what} failed ({r.returncode}): {' '.join(map(str, cmd))}\n{r.stdout[-3000:]}"
                              f"\n{r.stderr[-3000:]}")
    return r


def sv2v_output() -> Path:
    """gpu.v (sv2v of GPU_SRC with a leading `timescale), shared by every parameter set of these sources."""
    out = CACHE / "sv2v" / source_hash()[:16] / "gpu.v"
    with _locked(out.parent.with_suffix(".lock")):
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.parent / "sv2v_raw.v"
            _sh(["sv2v", "-w", str(tmp), *map(str, gpu_sources())], out.parent, "sv2v")
            out.write_text("`timescale 1ns/1ns\n" + tmp.read_text())
            tmp.unlink()
    return out


def build(params: GpuParams, gpu_v: Path | None = None) -> Path:
    """sim.vvp for params, built once under .anvil-cache/rtl/<key>/. gpu_v overrides the sv2v output (a rewritten
    source, e.g. to check that the Verilator rewrite keeps iverilog's behaviour)."""
    extra = "" if gpu_v is None else hashlib.sha256(Path(gpu_v).read_bytes()).hexdigest()
    d = CACHE / "rtl" / build_key(params, extra=extra)
    vvp = d / "sim.vvp"
    if vvp.exists():
        return vvp
    gpu_v = gpu_v or sv2v_output()
    with _locked(d.with_suffix(".lock")):
        if not vvp.exists():
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / "sim.vvp.tmp"
            _sh(["iverilog", "-o", str(tmp), "-s", "gpu", "-g2012", *params.plusargs(), str(gpu_v)], d, "iverilog")
            (d / "params.json").write_text(json.dumps(params.as_dict(), indent=1) + "\n")
            tmp.rename(vvp)
    return vvp


@functools.cache
def _cocotb_config(arg: str) -> str:
    return _sh(["cocotb-config", arg], ROOT, "cocotb-config").stdout.strip()


def _check_results():
    """sim/test/helpers/check_results.main, loaded by path (vvp exits 0 when a cocotb test fails)."""
    spec = importlib.util.spec_from_file_location("_anvil_check_results", SIM / "test" / "helpers" / "check_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


def cocotb_env(run_dir: Path, job_path: Path, result_path: Path) -> dict:
    pypath = os.pathsep.join([str(ROOT), str(SIM), *filter(None, [os.environ.get("PYTHONPATH")])])
    return {**os.environ, "PYTHONPATH": pypath, "PYGPI_PYTHON_BIN": sys.executable,
            "LIBPYTHON_LOC": _cocotb_config("--libpython"), "COCOTB_TEST_MODULES": "anvil.runtime.cocotb_job",
            "COCOTB_RESULTS_FILE": str(run_dir / "results.xml"), "ANVIL_JOB": str(job_path),
            "ANVIL_RESULT": str(result_path), "COCOTB_REDUCED_LOG_FMT": "1"}


def run_in_dir(job: Job, timeout_cycles: int, cmd_for, env_extra: dict | None = None) -> RunResult:
    """Write the job, run cmd_for(run_dir) with the cocotb env, gate on results.xml, parse the result JSON."""
    runs = CACHE / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(dir=runs))
    job_path, result_path = run_dir / "job.json", run_dir / "result.json"
    job_path.write_text(json.dumps({**job.to_dict(), "timeout_cycles": int(timeout_cycles)}))
    env = {**cocotb_env(run_dir, job_path, result_path), **(env_extra or {})}
    r = subprocess.run(cmd_for(run_dir), cwd=run_dir, env=env, capture_output=True, text=True)
    log = (r.stdout + r.stderr)[-4000:]
    res = json.loads(result_path.read_text()) if result_path.exists() else None
    if res and "error" in res:
        kind, msg, tb = res["error"]
        if kind == "TimeoutError":
            shutil.rmtree(run_dir, ignore_errors=True)
            raise TimeoutError(msg)
        raise SimulationError(f"{kind}: {msg} (run dir {run_dir})\n{tb}")
    if r.returncode or _check_results()(str(run_dir / "results.xml")) or res is None:
        raise SimulationError(f"simulation failed (exit {r.returncode}, run dir {run_dir}):\n{log}")
    shutil.rmtree(run_dir, ignore_errors=True)
    return RunResult(res["memory"], [LaunchResult(ln["cycles"], ln["counters"]) for ln in res["launches"]], {})


def run_rtl(job: Job, timeout_cycles: int, gpu_v: Path | None = None) -> RunResult:
    vvp = build(job.params, gpu_v)
    return run_in_dir(job, timeout_cycles,
                      lambda d: ["vvp", "-M", _cocotb_config("--lib-dir"), "-m", "libcocotbvpi_icarus", str(vvp)])
