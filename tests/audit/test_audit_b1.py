"""Audit B1 (correctness and regressions since audit A: SG9's i16 residual stream, SG10's predication / race / stray-
write fixes, SG8's anvil.isa_ext and co-design sweep): strict-xfail repros for every critical/high/medium finding of
AUDIT-B1.md. Each test states the correct behaviour and passes once the finding is fixed (strict xfail then turns it
into a failure, so the marker goes with the fix). No test here needs RTL or torch."""
import importlib

import numpy as np
import pytest

from anvil.codegen import compile_kernel, kernel_job
from anvil.ir import graph as g
from anvil.lower.graph_to_tensor import lower
from anvil.runtime import GpuParams
from anvil.schedule import Target
from anvil.schedule import templates as T

Q = importlib.import_module("anvil.frontend.quantize")      # the module (anvil.frontend.quantize is also a function)


# --------------------------------------------------------------------------------------------- shared builders

def _wide_rmsnorm(rows=5, d=8):
    """(graph module, consts, kernel `h`, finalized default schedule) of one wide (i16) rmsnorm with ragged rows:
    rows % THREADS_PER_BLOCK != 0, so the default rows-per-thread schedule has a lane-varying mask guard."""
    rng = np.random.default_rng(1)
    b = g.GraphBuilder("b1")
    x = b.input("x", [rows, d], "i16", 5e-4)
    lut, s, c, e = Q.rmsnorm_lut_wide(5e-4, 0.01, 0.02, 127, d, 1e-5, 1 << 15)
    consts = {"w": rng.integers(-127, 128, d), "lut": lut}
    h = b.rmsnorm_int("h", x, b.const("w", [d]), b.const("lut", [len(lut)], "i32"), 1, s, 0.02, c, e)
    b.output(h)
    mod = b.module()
    k = lower(mod).kernel("h")
    p = GpuParams()
    fin = T.default_schedule(k, Target.from_params(p, max_threads=4096)).finalize()
    return mod, consts, fin, p


# --------------------------------------------------------------------------------------------- B1-M1

def test_b1_m1_precondition_the_base_compile_spills():
    """The repro kernel is legal and spills on the base ISA (so MAD's documented fallback is the path under test)."""
    _, _, fin, p = _wide_rmsnorm()
    ck = compile_kernel(fin, p)
    assert ck.program.stats["spilled_vregs"] > 0


@pytest.mark.xfail(strict=True, reason="B1-M1: with MAD enabled, a kernel whose MAD accumulator spills raises "
                                       "BackendError ('virtual registers may be read before written') instead of "
                                       "taking the documented recompile-without-MAD fallback")
def test_b1_m1_mad_falls_back_when_its_accumulator_spills():
    from anvil.interp import graph as gi
    from anvil.isa_ext import enabled, run_job
    mod, consts, fin, p = _wide_rmsnorm()
    with enabled(("MAD",)):
        ck = compile_kernel(fin, p)                    # today: BackendError from the register allocator
    assert ck.program.stats["isa_ext"].get("mad_fallback") == 1
    x = np.random.default_rng(2).integers(-(1 << 15), 1 << 15, (5, 8))
    jp = kernel_job(ck, {"x": x, **{s: consts[s] for s in ("w", "lut")}})
    r = run_job(jp.job, timing=False)
    want = gi.run(mod, {"x": x}, consts).values["h"]
    assert np.array_equal(jp.read(r.memory, "h").reshape(want.shape), want)


# --------------------------------------------------------------------------------------------- B1-L (passing tests)
# Coverage gaps found by seeded mutants that survived `make check` (AUDIT-B1, mutant table). The compiler is right
# today; these tests make the mutants die in `make check`.

def test_b1_overflow_proof_uses_the_int16_input_magnitude():
    """Mutant B09 (an i16 input bounded by 2^14): x*Ma + x*Mb with |x| <= 2^15 overflows, with 2^14 it would not."""
    b = g.GraphBuilder("p")
    x = b.input("x", [2, 4], "i16", 1.0)
    b.output(b.add_requant("r", x, x, 1 << 15, 1 << 15, 16, 1.0, dtype="i16"))
    with pytest.raises(OverflowError):
        Q.prove_no_overflow(b.module(), {})


def test_b1_overflow_proof_propagates_the_int16_result_magnitude():
    """Mutant B08 (an i16 add_requant result bounded by 128): its consumer must be bounded with 2^15."""
    b = g.GraphBuilder("p")
    x = b.input("x", [2, 4], "i16", 1.0)
    r = b.add_requant("r", x, x, 1, 1, 1, 1.0, dtype="i16")
    b.output(b.add_requant("r2", r, r, 1 << 15, 1 << 15, 16, 1.0, dtype="i16"))
    with pytest.raises(OverflowError):
        Q.prove_no_overflow(b.module(), {})
    b = g.GraphBuilder("p")                                   # the same chain at int8 magnitudes is provably safe
    x = b.input("x", [2, 4], "i8", 1.0)
    r = b.add_requant("r", x, x, 1, 1, 1, 1.0, dtype="i8")
    b.output(b.add_requant("r2", r, r, 1 << 15, 1 << 15, 16, 1.0, dtype="i16"))
    Q.prove_no_overflow(b.module(), {})


def test_b1_add_requant_with_int16_operands_clamps_to_an_int8_result():
    """Mutant B11 (the wide add_requant path always clamps to int16): i16 operands with an i8 result saturate at
    int8, in the semantics and in the compiled kernel."""
    from anvil import semantics as sem
    a, b_ = np.array([30000, -30000, 200, -129, 5]), np.array([0, 0, 0, 0, 0])
    assert list(sem.add_requant(a, b_, 1, 1, 0, "i8")) == [127, -128, 127, -128, 5]
    import anvil
    gb = g.GraphBuilder("p")
    x = gb.input("x", [1, 5], "i16", 1.0)
    y = gb.input("y", [1, 5], "i8", 1.0)
    gb.output(gb.add_requant("r", x, y, 1, 1, 0, 1.0, dtype="i8"))
    mod = gb.module()
    from anvil.interp import graph as gi
    want = gi.run(mod, {"x": a[None], "y": b_[None]}).values["r"]
    assert list(want.ravel()) == [127, -128, 127, -128, 5]
    cm = anvil.compile(mod)
    assert list(cm.run([a[None], b_[None]], quantized=True).ints["r"].ravel()) == [127, -128, 127, -128, 5]


def test_b1_overflow_proof_rejects_a_wide_rmsnorm_shift_past_31():
    """Mutant B12 (the proof accepts s + kmax = 32): a wide rmsnorm whose shift s + kmax reaches 32 is rejected,
    at 31 it is accepted (a tiny LUT keeps every product inside int32, so only the shift rule decides)."""
    d = 8
    lut, _, c, e = Q.rmsnorm_lut_wide(5e-4, 0.01, 0.02, 127, d, 1e-5, 1 << 15)
    kmax = max(Q.rms_xmax_table(d, len(lut), c, 1 << 15, e))
    consts = {"w": np.full(d, 127), "lut": lut >> 16}
    for shift, ok in ((31 - kmax, True), (32 - kmax, False)):
        b = g.GraphBuilder("p")
        x = b.input("x", [2, d], "i16", 5e-4)
        b.output(b.rmsnorm_int("h", x, b.const("w", [d]), b.const("lut", [len(lut)], "i32"), 1, shift, 0.02, c, e))
        if ok:
            Q.prove_no_overflow(b.module(), consts)
        else:
            with pytest.raises(OverflowError, match="s \\+ kmax"):
                Q.prove_no_overflow(b.module(), consts)


def test_b1_compute_at_past_a_ragged_edge_by_one_row_is_rejected_or_bit_exact():
    """Mutant B25 (compute_at's domain check off by one): 5 rows split by 3 with a masked tail; the last outer
    iteration's region is rows 3..5, one row past the stage's domain. A2-M8's case (4 rows) overshoots by 2."""
    from anvil.interp import graph as gi
    from anvil.schedule import Schedule, ScheduleError
    mod, consts, _, _ = _wide_rmsnorm(rows=5, d=8)
    k = lower(mod).kernel("h")
    x = np.random.default_rng(3).integers(-(1 << 15), 1 << 15, (5, 8))
    want = gi.run(mod, {"x": x}, consts).values["h"]
    s = Schedule(k, Target())
    s.apply("split", "i", 3, "mask")
    try:
        s.apply("compute_at", "stat.i", "i.o")
    except ScheduleError:
        return
    got = s.run({"x": x, "w": consts["w"], "lut": consts["lut"]})["h"]
    assert np.array_equal(np.asarray(got).reshape(want.shape), want)


# isa_ext MIR peepholes (mutants B28, B29, B31) and the SLT idioms on equal operands (B34)

_W = 0xFFFFFFFF


def _eval_lir(code, live_in):
    """Straight-line lir evaluator over 32-bit words (enough for the peephole patterns below)."""
    r = dict(live_in)
    for i in code:
        a, b = r.get(i.a), r.get(i.b)
        if i.op in ("const", "const32"):
            r[i.d] = i.imm & _W
        elif i.op == "add":
            r[i.d] = (a + b) & _W
        elif i.op == "sub":
            r[i.d] = (a - b) & _W
        elif i.op == "mul":
            r[i.d] = (a * b) & _W
        elif i.op == "div":
            r[i.d] = a // b if b else _W
        elif i.op == "and":
            r[i.d] = a & b
        elif i.op == "mad":
            r[i.d] = (r[i.d] + a * b) & _W
        elif i.op == "shl":
            r[i.d] = (a << b) & _W if b < 32 else 0
        elif i.op == "sra":
            sa = a - ((a & (1 << 31)) << 1)
            r[i.d] = (sa >> min(b, 31)) & _W
        else:
            raise AssertionError(i.op)
    return r


def _same(code, rewritten, live_in, outs):
    before, after = _eval_lir(code, live_in), _eval_lir(rewritten, live_in)
    return all(before[o] == after[o] for o in outs)


def test_b1_mad_fusion_keeps_a_product_that_has_another_use():
    """Mutant B28: t = a*b used by the add and by a later instruction must stay computed."""
    from anvil.backend.lir import Ins
    from anvil.isa_ext.backend import fuse_mad
    code = [Ins("mul", "v4", "v1", "v2"), Ins("add", "v3", "v3", "v4"), Ins("add", "v5", "v4", "v1")]
    out, _ = fuse_mad([Ins(i.op, i.d, i.a, i.b, i.imm) for i in code])
    assert _same(code, out, {"v1": 7, "v2": 9, "v3": 100}, ["v3", "v5"])


def test_b1_mad_fusion_does_not_move_a_product_past_a_redefinition():
    """Mutant B29: a = ... between t = a*b and d = d + t; the fused MAD would read the new a."""
    from anvil.backend.lir import Ins
    from anvil.isa_ext.backend import fuse_mad
    code = [Ins("mul", "v4", "v1", "v2"), Ins("add", "v1", "v1", "v2"), Ins("add", "v3", "v3", "v4")]
    out, _ = fuse_mad([Ins(i.op, i.d, i.a, i.b, i.imm) for i in code])
    assert _same(code, out, {"v1": 7, "v2": 9, "v3": 100}, ["v3", "v1"])


def test_b1_and_mod_needs_the_same_dividend():
    """Mutant B31: q = x div 4; t = q*4; r = y - t is not y AND 3 when y != x."""
    from anvil.backend.lir import Fresh, Ins
    from anvil.isa_ext.backend import and_mod
    code = [Ins("const", "v9", imm=4), Ins("div", "v2", "v1", "v9"), Ins("mul", "v3", "v2", "v9"),
            Ins("sub", "v4", "v5", "v3")]
    out, _ = and_mod([Ins(i.op, i.d, i.a, i.b, i.imm) for i in code], Fresh(code))
    assert _same(code, out, {"v1": 13, "v5": 22}, ["v4"])


@pytest.mark.parametrize("name", ["ge", "le", "lt", "gt"])
def test_b1_slt_comparisons_on_equal_operands(name):
    """Mutant B34 (ge as b < a): the SLT idioms are exact when the operands are equal (test_isa_ext draws random
    operands and almost never draws equal ones)."""
    import anvil.codegen as cg
    from anvil.backend.idioms import Builder
    from anvil.ir import mir
    from anvil.isa_ext import enabled
    from anvil.isa_ext import isasim as fork
    ref = {"ge": lambda x, y: x >= y, "le": lambda x, y: x <= y, "lt": lambda x, y: x < y, "gt": lambda x, y: x > y}
    with enabled(("SLT",)):
        b = Builder()
        x, y = b.ldr(b.add("R12", 0)), b.ldr(b.add("R12", 1))
        b.str_(b.add("R12", 2), getattr(b, name)(x, y))
        b.emit(mir.RetOp())
        prog = cg.compile_mir(mir.func("t", b.ops), args_base=0)
    for xv, yv in ((5, 5), (-3, -3), (0, 0), (4, 5), (5, 4), (-(1 << 31), (1 << 31) - 1)):
        r = fork.run(prog.words, [xv & _W, yv & _W, 0], 1, word_bits=32, addr_bits=8)
        assert r.memory[2] == int(ref[name](xv, yv)), (name, xv, yv)


def test_b1_tuner_verdict_rejects_stray_writes():
    """Mutant B26 (measure.verdict ignores stray writes): a candidate whose run changes a word past the job or a
    word of an input buffer is not valid, even when every output buffer equals the reference."""
    from anvil.runtime import run
    from anvil.tune import Task, build, sketches_for
    from anvil.tune.g3_report import fixture_model
    from anvil.tune.measure import verdict
    (module, consts), d = fixture_model("mlp_block")
    prog = lower(module)
    vals = prog.run({"x": d["inputs"]["x"]}, consts)
    t = Task.from_program(prog, "gate", vals, GpuParams())
    b = build(t, sketches_for(t.kernel, t.target, t.params)[0].seeds()[0])
    mem = list(run(b.job, "isasim").memory)
    assert verdict(t, b, mem)[0]
    past = list(mem)
    past[b.plan.end + 3] += 1
    assert not verdict(t, b, past)[0]
    inp = list(mem)
    src = next(s_ for s_ in t.values if s_ in b.plan.buffers and s_ not in t.ref)     # an input buffer
    inp[b.plan.addresses[src]] += 1
    assert not verdict(t, b, inp)[0]


def test_b1_sweep_check_chain_rejects_cross_block_races(monkeypatch):
    """Mutant B42 (the co-design sweep's chain check ignores check_races' races): a race report fails the point."""
    import anvil
    import anvil.runtime as rt
    from anvil.codesign import sweep
    from anvil.codesign.workloads import load
    wl = load("mlp_block")
    cm = anvil.compile(wl.model)
    sweep.check_chain(cm, wl, backend="isasim")
    real = rt.check_races

    def racy(job):
        rr = real(job)
        rr.races = [(0, 0, "read-write", [0, 1])]
        return rr
    monkeypatch.setattr(rt, "check_races", racy)
    with pytest.raises(AssertionError, match="race"):
        sweep.check_chain(cm, wl, backend="isasim")


def _run_builder(props, build, inputs):
    """Compile a straight-line Builder program (inputs from args[0..], result to args[len(inputs)]) under `props`
    and run it on the isa_ext fork; -> the result word."""
    import anvil.codegen as cg
    from anvil.backend.idioms import Builder
    from anvil.ir import mir
    from anvil.isa_ext import enabled
    from anvil.isa_ext import isasim as fork
    with enabled(props):
        b = Builder()
        ins = [b.ldr(b.add("R12", i)) for i in range(len(inputs))]
        b.str_(b.add("R12", len(inputs)), build(b, *ins))
        b.emit(mir.RetOp())
        prog = cg.compile_mir(mir.func("t", b.ops), args_base=0)
    r = fork.run(prog.words, [v & _W for v in inputs] + [0], 1, word_bits=32, addr_bits=8)
    return r.memory[len(inputs)], prog


@pytest.mark.parametrize("f", [0, 1, 2, 3])
def test_b1_slt_mask_predicate_is_f_equal_zero(f):
    """Mutant B35 (the SLT predicate tests f < 2): the codegen predicate for a false indicator f >= 0 with range
    [0, 3] is 1 iff f == 0. In compiled kernels f = 1 at hi >= 2 only reaches lanes that SG10's lane clamp makes
    duplicates of an active lane, so no kernel test sees it; this checks the idiom directly."""
    from anvil.isa_ext.backend import _gen_patches

    class _F:
        def __init__(self, reg):
            self.reg = reg

        def range(self):
            return (0, 3)
    pred_from = _gen_patches({"SLT"})["pred_from"]

    def build(b, x):
        b.lin_reg = lambda fl: fl.reg
        return pred_from(b, _F(x))
    got, _ = _run_builder(("SLT",), build, [f])
    assert got == int(f == 0)


def test_b1_shl_multiply_with_the_power_of_two_first():
    """Mutant B41 (mul(2^k, x) shifts the constant): SHL selection is exact whichever operand is the constant."""
    for k in (8, 10, 13, 20):
        for xv in (3, -7, 12345):
            got, prog = _run_builder(("SHL",), lambda b, x, k=k: b.mul(1 << k, x), [xv])
            assert got == (xv << k) & _W, (k, xv)
            assert any(w >> 12 == 18 for w in prog.words)        # SHL was selected
