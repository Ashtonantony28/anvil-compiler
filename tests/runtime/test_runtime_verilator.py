"""The Verilator backend against the iverilog RTL (ADR-001): memory, per-launch cycles and every counter must be equal.

Marked rtl: every case also runs iverilog."""
import random

import pytest

from anvil.runtime import GpuParams, Job, Launch, run
from anvil.runtime.api import DEFAULT_TIMEOUT_CYCLES
from anvil.runtime.rtl import run_rtl, sv2v_output
from anvil.runtime.verilator import CSC, verilator_source
from test_runtime_rtl import BENCH_COLS, XF_CONFIGS, _random_program, run_all
from workloads import (KINDS, attention_job, bench_mlp_job, bench_rows, config_params, golden_mismatches,
                       isa_fixtures, mlp_chain_job)

pytestmark = pytest.mark.rtl


def _same(a, b, what):
    assert a.memory == b.memory, f"{what}: memory differs"
    assert [(lr.cycles, lr.counters) for lr in a.launches] == [(lr.cycles, lr.counters) for lr in b.launches], what


def test_bench_mlp_matches_published():
    for cfg in XF_CONFIGS:
        job, names, img, acts, want = bench_mlp_job(config_params(cfg))
        r = run(job, "verilator")
        for lr, row in zip(r.launches, bench_rows(cfg)):
            got = {"cycles": lr.cycles, **lr.counters}
            assert {c: got[c] for c in BENCH_COLS} == {c: row[c] for c in BENCH_COLS}, (cfg, row["kind"], row["case"])
        for kind in KINDS:
            assert golden_mismatches(img, r.memory, want, acts[kind]) == [], (cfg, kind)


def test_isa_fixtures_match_rtl():
    jobs = [Job(fx["data"], [Launch(fx["program"], fx["threads"])]) for fx in isa_fixtures(32).values()]
    for name, job, r in zip(isa_fixtures(32), jobs, run_all(jobs)):
        _same(run(job, "verilator"), r, name)


# The sim Makefile's matrix configurations (CFG_*, ENV_* latency), at the xf widths.
MATRIX = {
    "default": ({}, 0),
    "icache_pipelined": ({"ICACHE_LINES": 16, "PIPELINED": 1}, 0),
    "all": ({"ICACHE_LINES": 16, "PIPELINED": 1, "COALESCE": 1, "DCACHE_LINES": 16, "MEDIC": 1}, 0),
    "cores1": ({"NUM_CORES": 1}, 0),
    "cores4": ({"NUM_CORES": 4}, 0),
    "latency": ({}, 3),
    "warps2": ({"THREADS_PER_BLOCK": 8, "WARPS_PER_CORE": 2}, 0),
    "warps4gto": ({"WARPS_PER_CORE": 4, "WARP_POLICY": 1}, 0),
    "warps4gtlrr": ({"THREADS_PER_BLOCK": 16, "WARPS_PER_CORE": 4, "WARP_POLICY": 2}, 0),
    "simt": ({"SIMT_STACK": 1}, 0),
    "warps2lat": ({"THREADS_PER_BLOCK": 8, "WARPS_PER_CORE": 2, "ICACHE_LINES": 16, "PIPELINED": 1}, 10),
}


@pytest.mark.parametrize("cfg", sorted(MATRIX))
def test_matrix_matches_rtl(cfg):
    """Seeded multi-launch random jobs (several blocks, so cores contend) and fixture kernels at each configuration."""
    p, lat = MATRIX[cfg]
    params = GpuParams(**p)
    rng = random.Random(cfg)
    jobs = [Job([rng.getrandbits(32) for _ in range(rng.randint(0, 400))],
                [Launch(_random_program(rng), rng.randint(1, 24)) for _ in range(rng.randint(2, 4))], params, lat)
            for _ in range(3)]
    fxs = isa_fixtures(32)
    jobs += [Job(fxs[n]["data"], [Launch(fxs[n]["program"], fxs[n]["threads"])] * 2, params, lat)
             for n in ("matmul32", "divergent_relu32", "smem_cross32", "conv1d32")]
    for i, (job, r) in enumerate(zip(jobs, run_all(jobs))):
        _same(run(job, "verilator"), r, f"{cfg} job {i}")


@pytest.mark.parametrize("cfg", ["default", "all", "warps2lat"])
def test_chains_match_rtl(cfg):
    p, lat = MATRIX[cfg]
    params = GpuParams(**p)
    cases = [mlp_chain_job("tiled", params), attention_job("naive", params)]
    jobs = [Job(job.data, job.launches, params, lat) for job, _, _ in cases]
    for (job, img, want), j, r in zip(cases, jobs, run_all(jobs)):
        v = run(j, "verilator")
        _same(v, r, cfg)
        assert golden_mismatches(img, v.memory, want) == []


def test_rewrite_keeps_iverilog_behaviour(tmp_path):
    """iverilog on the Verilator-rewritten source equals iverilog on sv2v's output (counters included)."""
    text = verilator_source(sv2v_output().read_text())
    assert "csc_b[j] = 1'b1;" in text and f"{CSC}[j] = 1'b1;" not in text
    path = tmp_path / "gpu.v"
    path.write_text(text)
    cfg = "COALESCE=1 DCACHE_LINES=16"
    job, *_ = bench_mlp_job(config_params(cfg))
    r = run_rtl(job, DEFAULT_TIMEOUT_CYCLES, path)
    for lr, row in zip(r.launches, bench_rows(cfg)):
        got = {"cycles": lr.cycles, **lr.counters}
        assert {c: got[c] for c in BENCH_COLS} == {c: row[c] for c in BENCH_COLS}


def test_rewrite_rejects_unexpected_source():
    text = sv2v_output().read_text()
    with pytest.raises(ValueError, match="expected 1"):
        verilator_source(text.replace(f"{CSC} <= 0;", f"{CSC} <= 1'b0;"))


def test_verilator_timeout():
    from xf.asm import assemble
    spin = assemble("CMP R0, R0\nloop:\n BRz loop\n RET")
    with pytest.raises(TimeoutError):
        run(Job([], [Launch(spin, 1)]), "verilator", timeout_cycles=300)
