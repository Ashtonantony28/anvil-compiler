"""SG8 (G5) ISA proposals: the extended-ISA fork, idioms and small kernels bit-exact (fast, no RTL)."""
import random

import numpy as np
import pytest

import anvil.codegen as cg
from anvil import semantics as sem
from anvil.backend.idioms import Builder
from anvil.codegen import compile_kernel, kernel_job
from anvil.cost.analytic import predict
from anvil.ir import mir
from anvil.isa_ext import PROPOSALS, enabled, run_job
from anvil.isa_ext import isa as X
from anvil.isa_ext import isasim as fork
from anvil.lower.graph_to_tensor import lower
from anvil.runtime import GpuParams, run
from anvil.schedule import Schedule, Target
from anvil.schedule import templates as T

VARIANTS = [(), *[(m,) for m in PROPOSALS], PROPOSALS]
W32 = 0xFFFFFFFF


def s32(x):
    x &= W32
    return x - (1 << 32) if x >> 31 else x


def test_encoding_extends_base():
    from xf import isa
    for m, (op, _) in isa.OPCODES.items():
        assert X.OPCODES[m] == isa.OPCODES[m]
    for m in PROPOSALS:
        w = X.encode(m, rd=3, rs=4, rt=5)
        assert w >> 12 == X.EXT[m][0] >= 16 and w < 1 << X.WORD_BITS_EXT
        assert X.decode(w) == (m, {"rd": 3, "rs": 4, "rt": 5})
    assert X.encode("ADD", rd=1, rs=2, rt=3) == isa.encode("ADD", rd=1, rs=2, rt=3)


def test_semantics():
    f = X.semantics(32)
    rng = random.Random(0)
    edge = [0, 1, 2, 31, 32, 255, 1 << 31, W32, (1 << 31) - 1, 12345678]
    vals = edge + [rng.getrandbits(32) for _ in range(60)]
    for a in vals:
        for b in vals[:20]:
            assert f["MIN"](0, a, b) == min(s32(a), s32(b)) & W32
            assert f["AND"](0, a, b) == a & b
            assert f["SHL"](0, a, b) == ((a << b) & W32 if b < 32 else 0)
            assert f["SLT"](0, a, b) == int(s32(a) < s32(b))
            assert f["MAD"](a, a, b) == (a + a * b) & W32


def _xf(block):
    from anvil.tune.g3_report import fixture_model
    (m, c), d = fixture_model(block)
    prog = lower(m)
    return prog, prog.run({"x": d["inputs"]["x"]}, c)


def test_fork_equals_xf_isasim_on_base_programs():
    from xf import isasim
    prog, vals = _xf("attention")
    for name, W in (("p", 1), ("scores", 2), ("h", 4)):
        p = GpuParams(WARPS_PER_CORE=W)
        k = prog.kernel(name)
        ck = compile_kernel(T.default_schedule(k, Target.from_params(p, max_threads=4096)).finalize(), p)
        jp = kernel_job(ck, {s: vals[s] for s in ck.buffers if s in vals})
        ln = jp.job.launches[0]
        mem = list(jp.job.data)
        kw = dict(word_bits=32, addr_bits=12, warps_per_core=W)
        a, b = isasim.run(ln.program, mem, ln.threads, **kw), fork.run(ln.program, mem, ln.threads, **kw)
        assert a.memory == b.memory and a.steps == b.steps
        assert (a.divergent, a.stores_conflict, a.store_races) == (b.divergent, b.stores_conflict, b.store_races)
        # tier 1 on the fork's streams == tier 1 on anvil.cost.profile (and Verilator, ADR-001 / G4)
        r = run_job(jp.job)
        assert r.cycles == predict(jp.job).cycles
        ref = run(jp.job, "isasim")
        assert r.memory == ref.memory and r.steps == [lr.counters["warp_issues_total"] for lr in ref.launches]


def _run_mir(build, n_in, props, rng, trials=40):
    """Compile a straight-line Builder program (inputs from args[0..n_in-1], result to args[n_in]) under `props`
    and run it on the fork against `build`'s Python value."""
    with enabled(props):                    # idioms switch at build time, selection + encoding at compile time
        b = Builder()
        ins = [b.ldr(b.add("R12", i)) for i in range(n_in)]
        out = build(b, *ins)
        b.str_(b.add("R12", n_in), out)
        b.emit(mir.RetOp())
        prog = cg.compile_mir(mir.func("t", b.ops), args_base=0)      # the codegen entry (peepholes + encode)
    return prog


IDIOMS = {
    "min": (2, lambda b, x, y: b.min(x, y), lambda x, y: min(x, y)),
    "clamp_i8": (1, lambda b, x: b.clamp_i8(x), lambda x: max(-128, min(127, x))),
    "lt": (2, lambda b, x, y: b.lt(x, y), lambda x, y: int(x < y)),
    "ge": (2, lambda b, x, y: b.ge(x, y), lambda x, y: int(x >= y)),
    "le": (2, lambda b, x, y: b.le(x, y), lambda x, y: int(x <= y)),
    "gt": (2, lambda b, x, y: b.gt(x, y), lambda x, y: int(x > y)),
    "mul_pow2": (1, lambda b, x: b.mul_pow2(x, 13), lambda x: s32(x << 13)),
    "requant": (1, lambda b, x: b.requant(x, 77, 9), lambda x: int(sem.requant(np.int64(x), 77, 9))),
    "mac": (3, lambda b, a, x, y: b.add(a, b.mul(x, y), into=a), lambda a, x, y: s32(a + x * y)),
}


@pytest.mark.parametrize("props", VARIANTS, ids=lambda v: "+".join(v) or "base")
@pytest.mark.parametrize("name", sorted(IDIOMS))
def test_idioms_bit_exact(name, props):
    n, build, ref = IDIOMS[name]
    rng = random.Random(hash((name, props)) & 0xFFFF)
    prog = _run_mir(build, n, props, rng)
    for _ in range(25):
        if name == "requant":
            xs = [rng.randint(-(1 << 20), 1 << 20)]
        elif name in ("mac", "mul_pow2"):
            xs = [rng.randint(-(1 << 14), 1 << 14) for _ in range(n)]
        else:
            xs = [rng.choice([rng.randint(-200, 200), s32(rng.getrandbits(32))]) for _ in range(n)]
        data = [x & W32 for x in xs] + [0]
        r = fork.run(prog.words, data, 1, word_bits=32, addr_bits=8)
        assert s32(r.memory[n]) == ref(*xs), (name, props, xs)
    if props == PROPOSALS and name in ("min", "lt", "mul_pow2", "mac"):
        assert any(w >> 12 >= 16 for w in prog.words)          # the extension was actually used


def test_disabled_is_the_base_compiler():
    prog, _ = _xf("mlp_block")
    p = GpuParams()
    k = prog.kernel("gate")
    s = T.default_schedule(k, Target.from_params(p, max_threads=4096)).finalize()
    base = compile_kernel(s, p).program.words
    with enabled(()):
        assert compile_kernel(s, p).program.words == base
    with enabled(PROPOSALS):
        assert compile_kernel(s, p).program.words != base
    assert compile_kernel(s, p).program.words == base           # patches are undone


@pytest.mark.parametrize("props", VARIANTS, ids=lambda v: "+".join(v) or "base")
def test_small_kernels_bit_exact(props):
    """Every MLP / attention kernel, default and tiled schedules, under each variant: bit-exact vs the tensor
    interpreter on the fork, no flag; the fully extended ISA is never slower on tier 1."""
    for block in ("mlp_block", "attention"):
        prog, vals = _xf(block)
        for W in (1, 2):
            p = GpuParams(WARPS_PER_CORE=W)
            tgt = Target.from_params(p, max_threads=4096)
            for k in prog.kernels:
                for tiled in (False, True):
                    if tiled and k.op not in ("qlinear", "qmatmul"):
                        continue
                    if W == 2 and k.name not in ("h", "gate", "p"):
                        continue
                    s = T.TEMPLATES["matmul_tiled"](Schedule(k, tgt)) if tiled else T.default_schedule(k, tgt)
                    fin = s.finalize()
                    with enabled(props):
                        ck = compile_kernel(fin, p)
                    jp = kernel_job(ck, {s_: vals[s_] for s_ in ck.buffers if s_ in vals})
                    r = run_job(jp.job)
                    assert not r.races and not any(len(v) for v in r.flags.values())
                    for w in k.writes:
                        assert np.array_equal(jp.read(r.memory, w).reshape(np.shape(vals[w])), vals[w]), \
                            (block, k.name, props)
                    if props == PROPOSALS:
                        b = run_job(kernel_job(compile_kernel(fin, p), {s_: vals[s_] for s_ in ck.buffers
                                                                        if s_ in vals}).job)
                        assert r.cycles <= b.cycles
