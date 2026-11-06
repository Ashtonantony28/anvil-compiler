"""Contract C2 on the iverilog RTL build (SG0 accept): golden-equal chains, published xf-bench cycles, isasim agreement.

Simulations run at most WORKERS at a time."""
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from anvil.runtime import GpuParams, Job, Launch, run
from anvil.runtime.rtl import build, build_key
from workloads import (KINDS, attention_job, bench_mlp_job, bench_rows, config_params, golden_mismatches,
                       isa_fixtures, mlp_chain_job)
from xf.asm import assemble

pytestmark = pytest.mark.rtl
WORKERS = 4
XF_CONFIGS = ["", "COALESCE=1", "DCACHE_LINES=16", "COALESCE=1 DCACHE_LINES=16"]   # sim/bench/run.py
BENCH_COLS = ["cycles", "data_mem_requests", "data_mem_requests_coalesced", "dcache_hits", "dcache_misses",
              "icache_hits", "icache_misses"]


def run_all(jobs, backend="rtl", **kw):
    with ThreadPoolExecutor(WORKERS) as ex:
        return list(ex.map(lambda j: run(j, backend, **kw), jobs))


def _no_flags(r):
    return not any(r.flags[k] for k in ("divergent", "stores_conflict", "store_races", "race_words"))


def test_chains_golden_on_isasim_and_rtl():
    cases = [(f"mlp_block_0.{k}", *mlp_chain_job(k)) for k in KINDS] + \
            [(f"attention_0.{k}", *attention_job(k)) for k in KINDS]
    rtl = run_all([job for _, job, _, _ in cases])
    for (name, job, img, want), r in zip(cases, rtl):
        model = run(job)
        assert _no_flags(model), (name, model.flags)
        assert golden_mismatches(img, model.memory, want) == [], name
        assert golden_mismatches(img, r.memory, want) == [], name
        assert r.memory == model.memory, name
        assert len(r.launches) == len(job.launches)
        for lr, lm in zip(r.launches, model.launches):
            assert lr.cycles > 0
            assert lr.counters["warp_issues_total"] == lm.counters["warp_issues_total"], name


def test_bench_mlp_cycles_match_published():
    """Per-launch cycles and counters equal sim/bench/xf-mlp.json at each XF_CONFIGS build."""
    built = [bench_mlp_job(config_params(c)) for c in XF_CONFIGS]
    results = run_all([b[0] for b in built])
    for cfg, (job, names, img, acts, want), r in zip(XF_CONFIGS, built, results):
        rows = bench_rows(cfg)
        assert [(row["kind"], row["case"]) for row in rows] == names, cfg
        for (kind, case), lr, row, ln in zip(names, r.launches, rows, job.launches):
            got = {"cycles": lr.cycles, **lr.counters}
            assert {c: got[c] for c in BENCH_COLS} == {c: row[c] for c in BENCH_COLS}, (cfg, kind, case)
            assert row["threads"] == ln.threads
        for kind in KINDS:
            assert golden_mismatches(img, r.memory, want, acts[kind]) == [], (cfg, kind)


def test_isa_fixtures_agree_with_isasim():
    fxs = isa_fixtures(32)
    jobs = {n: Job(fx["data"], [Launch(fx["program"], fx["threads"])]) for n, fx in fxs.items()}
    rtl = dict(zip(jobs, run_all(list(jobs.values()))))
    for n, job in jobs.items():
        model = run(job)
        assert rtl[n].memory == model.memory, n
        assert model.memory[:len(fxs[n]["memory"])] == fxs[n]["memory"], n
        assert rtl[n].launches[0].counters["warp_issues_total"] == model.launches[0].counters["warp_issues_total"], n


OPS = ("ADD", "SUB", "MUL", "DIV", "MAX")


def _random_program(rng):
    """Straight-line ALU/CONST/LDR/STR code; addresses come from registers, so stores may conflict across lanes."""
    reg = lambda: rng.choice([f"R{i}" for i in range(12)] + ["%blockIdx", "%blockDim", "%threadIdx"])
    lines = []
    for _ in range(rng.randint(4, 24)):
        k = rng.random()
        if k < 0.25:
            lines.append(f"CONST R{rng.randint(0, 11)}, #{rng.randint(0, 255)}")
        elif k < 0.7:
            lines.append(f"{rng.choice(OPS)} R{rng.randint(0, 11)}, {reg()}, {reg()}")
        elif k < 0.85:
            lines.append(f"LDR R{rng.randint(0, 11)}, {reg()}")
        else:
            lines.append(f"STR {reg()}, {reg()}")
    return assemble("\n".join(lines) + "\nRET")


def test_random_multi_launch_jobs_agree_with_isasim():
    """Seeded multi-launch jobs of one block each (blocks on different cores would race through memory)."""
    rng = random.Random(1234)
    jobs = [Job([rng.getrandbits(32) for _ in range(rng.randint(0, 300))],
                [Launch(_random_program(rng), rng.randint(1, 4)) for _ in range(rng.randint(2, 5))])
            for _ in range(8)]
    for job, r in zip(jobs, run_all(jobs)):
        model = run(job)
        assert r.memory == model.memory
        assert [lr.counters["warp_issues_total"] for lr in r.launches] == \
               [lm.counters["warp_issues_total"] for lm in model.launches]


def test_params_and_mem_latency():
    """A non-default build and memory latency change cycles, never memory."""
    fx = isa_fixtures(32)["matmul32"]
    base = Job(fx["data"], [Launch(fx["program"], fx["threads"])] * 2)
    jobs = [base, Job(base.data, base.launches, mem_latency=3),
            Job(base.data, base.launches, GpuParams(ICACHE_LINES=16, PIPELINED=1, NUM_CORES=4)),
            Job(base.data, base.launches, GpuParams(THREADS_PER_BLOCK=8, WARPS_PER_CORE=2, WARP_POLICY=1),
                mem_latency=2)]
    res = run_all(jobs)
    model = run(base)
    for r in res:
        assert r.memory == model.memory
    plain, slow, fast = (r.launches[0].cycles for r in res[:3])
    assert slow > plain and fast < plain
    assert res[0].launches[0].cycles == res[0].launches[1].cycles   # the reset makes launches independent
    assert res[2].launches[0].counters["icache_hits"] > 0
    assert run(jobs[3]).memory == model.memory


def test_build_cache_per_params():
    assert build_key(GpuParams()) == build_key(GpuParams())
    assert build_key(GpuParams()) != build_key(GpuParams(COALESCE=1))
    vvp = build(GpuParams())
    assert vvp.exists() and build(GpuParams()) == vvp
    assert "-Pgpu.WORD_BITS=32" in GpuParams().plusargs()


def test_timeout_raises():
    spin = assemble("CMP R0, R0\nloop:\n BRz loop\n RET")   # nzp starts at 000; CMP makes it z
    with pytest.raises(TimeoutError):
        run(Job([], [Launch(spin, 1)]), "rtl", timeout_cycles=300)


def test_rtl_masks_and_pads_memory():
    r = run(Job([-1, 7], [Launch([0xF000], 1)]), "rtl")
    assert len(r.memory) == 4096 and r.memory[:3] == [0xFFFFFFFF, 7, 0]
    assert np.count_nonzero(r.memory) == 2
