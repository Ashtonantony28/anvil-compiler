"""SG2 accept (rtl): compiled programs run on the iverilog RTL through anvil.runtime with the same memory as isasim:
the recompiled mlp chains, a spill-heavy random program (with loops) and the idiom kernels."""
import random

import numpy as np
import pytest
from hypothesis import HealthCheck, Phase, assume, given, settings, strategies as st

from anvil.backend import ProgramTooLarge, compile_mir
from anvil.backend.dealloc import devirtualize
from anvil.ir import mir
from anvil.runtime import GpuParams, Job, Launch, run
from xf import fixture, mlp

from test_regalloc import IN, SCRATCH, pressure_program
from test_xf_recompile import args_base_of

pytestmark = pytest.mark.rtl


def _same(job):
    model = run(job)
    rtl = run(job, "rtl")
    assert not any(model.flags[k] for k in ("divergent", "stores_conflict", "store_races")), model.flags
    assert rtl.memory == model.memory
    return model, rtl


@pytest.mark.parametrize("kind", ["naive", "tiled"])
def test_recompiled_mlp_chain_rtl_equals_isasim_and_golden(kind):
    d = fixture.load(fixture.FIXTURE_DIR / "mlp_block_0.json")
    img = d["image"]
    img.write("x", d["inputs"]["x"])
    chain = mlp.launches(img, kind)
    launches = []
    for out, (prog, threads) in chain:
        p = compile_mir(devirtualize(mir.decode(prog, out)), args_base=args_base_of(prog))
        launches.append(Launch(p.words, threads))
    _, rtl = _same(Job(list(img.words), launches))
    for k, v in d["golden"].items():
        assert np.array_equal(img.read(k, rtl.memory), v), k


def _spill_job(case, threads, tpb):
    f, data = case
    mem = [0] * IN + data
    mem += [0] * (2048 - len(mem)) + [SCRATCH]
    p = compile_mir(f, args_base=2048, spill_slot=0)
    assert p.stats["spill_loads"] > 0
    return Job(mem, [Launch(p.words, threads)], GpuParams(THREADS_PER_BLOCK=tpb, WARPS_PER_CORE=tpb // 4))


@settings(max_examples=2, deadline=None, database=None, derandomize=True, phases=[Phase.generate],
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
@given(pressure_program(loops=True), st.sampled_from([4, 8]))
def test_spilling_program_rtl_equals_isasim(case, tpb):
    """Two fixed draws (derandomized, generate phase only: a failing case is reported, never shrunk on RTL)."""
    try:
        job = _spill_job(case, 12, tpb)
    except ProgramTooLarge:
        assume(False)                                   # oversize random program: draw another
    model, rtl = _same(job)
    assert len(rtl.launches) == 1 and rtl.launches[0].cycles > 0


def test_idioms_rtl_equals_isasim():
    """One kernel exercising every idiom on 8 threads of random pairs."""
    from anvil.backend.idioms import Builder
    b = Builder()
    g = b.global_tid()
    x = b.ldr(b.mul(g, 2))
    y = b.ldr(b.add(b.mul(g, 2), 1))
    outs = [b.min(x, y), b.clamp_i8(x), b.abs(y), b.lt(x, y), b.ge(x, y), b.ne(x, y), b.select(b.lt(y, x), x, y),
            b.mul_pow2(x, 5), b.requant(x, 77, 9), b.addr(0x12345, (x, 0x10001))]
    base = b.addr(512, (g, len(outs)))
    for k, o in enumerate(outs):
        b.str_(b.add(base, k), o)
    b.emit(mir.RetOp())
    rng = random.Random(7)
    data = [rng.choice([rng.getrandbits(32), rng.randrange(-200, 200) & 0xFFFFFFFF, rng.randrange(-2000, 2000)
                        & 0xFFFFFFFF]) for _ in range(16)]
    p = compile_mir(mir.func("idioms", b.ops), args_base=4000)
    _same(Job(data, [Launch(p.words, 8)]))
