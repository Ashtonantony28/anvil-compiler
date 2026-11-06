"""Contract C2 without a Verilog simulator: GpuParams, Job validation and the isasim backend."""
import dataclasses
import json
import re

import pytest
from hypothesis import given, settings, strategies as st

from anvil.runtime import GpuParams, Job, Launch, run
from anvil.runtime.api import GPU_SV_DEFAULTS
from anvil.runtime.isasim_backend import isasim_kwargs
from workloads import KINDS, SIM, attention_job, golden_mismatches, isa_fixtures, mlp_chain_job

from xf.asm import assemble


def _gpu_sv_params():
    src = (SIM / "src" / "gpu.sv").read_text()
    header = src[src.index("module gpu"):src.index("input")]
    return {k: int(v) for k, v in re.findall(r"parameter\s+(\w+)\s*=\s*(\d+)", header)}


def test_defaults_table_matches_gpu_sv():
    sv = _gpu_sv_params()
    assert {k: sv[k] for k in GPU_SV_DEFAULTS} == GPU_SV_DEFAULTS
    assert [f.name for f in dataclasses.fields(GpuParams)] == list(GPU_SV_DEFAULTS)


def test_plusargs_default_is_makefile_xf_params():
    xf = re.search(r"^XF_PARAMS\s*:=\s*(.*)$", (SIM / "Makefile").read_text(), re.M).group(1).split()
    assert sorted(GpuParams().plusargs()) == sorted(xf)


def test_plusargs_lists_every_field_off_the_rtl_default():
    p = GpuParams(COALESCE=1, DCACHE_LINES=16, WARPS_PER_CORE=2, THREADS_PER_BLOCK=8)
    assert p.plusargs() == ["-Pgpu.WORD_BITS=32", "-Pgpu.DATA_MEM_ADDR_BITS=12", "-Pgpu.DATA_MEM_DATA_BITS=32",
                            "-Pgpu.THREADS_PER_BLOCK=8", "-Pgpu.COALESCE=1", "-Pgpu.DCACHE_LINES=16",
                            "-Pgpu.WARPS_PER_CORE=2"]
    assert GpuParams(WORD_BITS=8, DATA_MEM_ADDR_BITS=8, DATA_MEM_DATA_BITS=8).plusargs() == []
    assert all(a.startswith("-Pgpu.") for a in p.plusargs())


def test_params_frozen_hashable_and_checked():
    p = GpuParams()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.NUM_CORES = 4
    assert len({GpuParams(), GpuParams(), GpuParams(COALESCE=1)}) == 2
    for bad in ({"WARP_POLICY": 3}, {"WARPS_PER_CORE": 3}, {"NUM_CORES": -1}, {"WORD_BITS": 33}, {"COALESCE": 1.0}):
        with pytest.raises(ValueError):
            GpuParams(**bad)


def test_isasim_kwargs():
    assert isasim_kwargs(GpuParams()) == {"word_bits": 32, "addr_bits": 12, "threads_per_block": 4,
                                          "warps_per_core": 1, "simt": False, "warp_policy": "lrr"}
    kw = isasim_kwargs(GpuParams(THREADS_PER_BLOCK=8, WARPS_PER_CORE=2, WARP_POLICY=1, SIMT_STACK=1))
    assert (kw["threads_per_block"], kw["warps_per_core"], kw["warp_policy"], kw["simt"]) == (8, 2, "gto", True)
    with pytest.raises(ValueError):
        isasim_kwargs(GpuParams(DATA_MEM_DATA_BITS=16))


RET = [0xF000]


@pytest.mark.parametrize("job,msg", [
    (Job([], [Launch(RET, 0)]), "threads"),
    (Job([], [Launch(RET, 256)]), "threads"),
    (Job([], [Launch([], 4)]), "program"),
    (Job([], [Launch(RET * 257, 4)]), "program"),
    (Job([], [Launch([0x10000], 4)]), "16 bits"),
    (Job([0] * 4097, [Launch(RET, 4)]), "data words"),
    (Job([], []), "no launches"),
    (Job([], [Launch(RET, 4)], mem_latency=-1), "mem_latency"),
])
def test_job_validation(job, msg):
    with pytest.raises(ValueError, match=msg):
        run(job)


def test_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        run(Job([], [Launch(RET, 4)]), "spice")


def test_max_block_is_checked():
    from xf.kernels import Program
    prog = Program(RET)
    prog.max_block = 4
    run(Job([], [Launch(prog, 4)]))
    with pytest.raises(ValueError, match="blockDim"):
        run(Job([], [Launch(prog, 8)], GpuParams(THREADS_PER_BLOCK=8)))


def test_job_dict_round_trip():
    job = Job([1, 2, 3], [Launch([0x9001, 0xF000], 7)], GpuParams(COALESCE=1), mem_latency=3)
    d = json.loads(json.dumps(job.to_dict()))
    assert Job.from_dict(d) == job


@pytest.mark.parametrize("width", [8, 32])
def test_isasim_reproduces_isa_fixtures(width):
    for name, fx in isa_fixtures(width).items():
        p = fx["params"]
        params = GpuParams(WORD_BITS=p["word_bits"], DATA_MEM_DATA_BITS=p["word_bits"], DATA_MEM_ADDR_BITS=p["addr_bits"])
        r = run(Job(fx["data"], [Launch(fx["program"], fx["threads"])], params))
        assert len(r.memory) == 1 << p["addr_bits"]
        assert r.memory[:len(fx["memory"])] == fx["memory"], name
        assert not any(r.memory[len(fx["memory"]):]), name
        assert r.launches[0].cycles is None and r.launches[0].counters["warp_issues_total"] > 0


@pytest.mark.parametrize("kind", KINDS)
def test_isasim_chains_golden(kind):
    for job, img, want in (mlp_chain_job(kind), attention_job(kind)):
        r = run(job)
        assert golden_mismatches(img, r.memory, want) == []
        assert not any(r.flags[k] for k in ("divergent", "stores_conflict", "store_races", "race_words")), r.flags


# Word 0 counts launches: each launch adds 1 to it (one thread), so memory must persist across launches.
INC = assemble("""
    CONST R0, #0
    LDR R1, R0
    CONST R2, #1
    ADD R1, R1, R2
    STR R0, R1
    RET""")


def test_memory_persists_across_launches():
    r = run(Job([5], [Launch(INC, 1)] * 3))
    assert r.memory[0] == 8 and len(r.launches) == 3


def test_data_is_masked_to_word_bits():
    r = run(Job([-1, 1 << 32], [Launch(RET, 1)]))
    assert r.memory[:2] == [0xFFFFFFFF, 0]


DIVERGENT = assemble("""
    CONST R0, #1
    CMP %threadIdx, R0
    BRz done
    CONST R1, #7
    STR %threadIdx, R1
done:
    RET""")
CONFLICT = assemble("""
    CONST R0, #9
    STR R0, %threadIdx
    RET""")


def test_isasim_flags_carry_launch_index():
    r = run(Job([], [Launch(RET, 4), Launch(DIVERGENT, 4), Launch(CONFLICT, 4)]))
    assert r.flags["divergent"] and all(e[0] == 1 for e in r.flags["divergent"])
    assert r.flags["stores_conflict"] == [(2, 0, 1, 9)]
    assert r.memory[9] == 3       # the highest lane wins (ADR-016)


# Straight-line programs (no branches, so every run ends): ALU ops, CONST, LDR and STR at address max(63, x).
OPS = ("ADD", "SUB", "MUL", "DIV", "MAX")
_instr = st.one_of(
    st.tuples(st.just("CONST"), st.integers(0, 12), st.integers(0, 63)).map(lambda t: f"CONST R{t[1]}, #{t[2]}"),
    st.tuples(st.sampled_from(OPS), st.integers(0, 12), st.integers(0, 15), st.integers(0, 15))
      .map(lambda t: f"{t[0]} R{t[1]}, {_reg(t[2])}, {_reg(t[3])}"),
    st.tuples(st.integers(0, 12), st.integers(0, 15)).map(lambda t: f"CONST R12, #63\n MAX R12, R12, {_reg(t[1])}"
                                                        f"\n LDR R{t[0]}, R12"),
    st.tuples(st.integers(0, 15), st.integers(0, 15)).map(lambda t: f"CONST R12, #63\n MAX R12, R12, {_reg(t[0])}"
                                                        f"\n STR R12, {_reg(t[1])}"),
)


def _reg(r):
    return {13: "%blockIdx", 14: "%blockDim", 15: "%threadIdx"}.get(r, f"R{r}")


_program = st.lists(_instr, min_size=1, max_size=20).map(lambda xs: assemble("\n".join(xs) + "\nRET"))


@settings(max_examples=40, deadline=None)
@given(st.lists(st.tuples(_program, st.integers(1, 12)), min_size=2, max_size=4),
       st.lists(st.integers(0, (1 << 32) - 1), max_size=64), st.integers(0, 3))
def test_isasim_job_splits(launches, data, cut):
    """Running launches as one job equals running a prefix, then the rest on the prefix's memory."""
    launches = [Launch(p, t) for p, t in launches]
    cut = 1 + cut % (len(launches) - 1)
    whole = run(Job(data, launches))
    head = run(Job(data, launches[:cut]))
    tail = run(Job(head.memory, launches[cut:]))
    assert whole.memory == tail.memory
    assert [lr.counters for lr in whole.launches] == [lr.counters for lr in head.launches + tail.launches]
