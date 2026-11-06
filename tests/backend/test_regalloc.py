"""SG2 accept: hypothesis-generated straight-line and loop programs with register pressure above 12 spill to the
per-thread scratch frame and still equal the MIR interpreter (on isasim, and the allocated MIR on the
interpreter); allocator unit behaviour (args-register reuse, loop-depth spill choice, remat, errors)."""
import random

import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from anvil.backend import BackendError, ProgramTooLarge, compile_mir
from anvil.backend.idioms import Builder
from anvil.ir import mir
from xf import isa

from backend_util import compile_and_check, interp, sim

IN, OUT, SCRATCH = 64, 1024, 2560
ALU = ["add", "sub", "mul", "div", "sra", "max"]


@st.composite
def pressure_program(draw, loops):
    """Load `nin` (13..18) values per thread, mix them with random ALU ops (new values and in-place updates,
    register and wide-constant operands), optionally inside 1-2 nested counted loops, then store all of them:
    every pool value is live at the stores, so more than 12 registers are needed."""
    nin = draw(st.integers(13, 18))
    b = Builder()
    g = b.global_tid()
    src = b.addr(IN, (g, nin))
    pool = [b.ldr(b.add(src, j)) for j in range(nin)]
    pick = st.integers(0, nin - 1)

    def ops(n):
        for _ in range(n):
            kind = draw(st.sampled_from(ALU))
            x = pool[draw(pick)]
            y = draw(st.one_of(st.integers(-(1 << 31), (1 << 32) - 1), st.integers(0, 3))) \
                if draw(st.integers(0, 4)) == 0 else pool[draw(pick)]
            if kind == "sra" and isinstance(y, int):
                y %= 40
            k = draw(pick)
            into = pool[k] if draw(st.booleans()) else None        # in-place (multi-def) or a new value
            pool[k] = getattr(b, kind)(x, y, into=into)

    ops(draw(st.integers(0, 10)))
    if loops:
        depth = draw(st.integers(1, 2))
        counters = []
        for lvl in range(depth):
            c = b.reg()
            b.emit(mir.Const32Op(c, 0))
            # counters are redefined in the loop, so they are not constants; make sure they are multi-def
            counters.append((c, draw(st.integers(1, 3))))
            b.label(f"top{lvl}")
            ops(draw(st.integers(1, 6)))
        for lvl in reversed(range(depth)):
            c, n = counters[lvl]
            b.add(c, 1, into=c)
            b.emit(mir.CmpOp(c, b.const(n)))
            b.emit(mir.BrOp(0b100, f"top{lvl}"))
            ops(draw(st.integers(0, 4)))
    dst = b.addr(OUT, (g, nin))
    for j, v in enumerate(pool):
        b.str_(b.add(dst, j), v)
    b.emit(mir.RetOp())
    rng = random.Random(draw(st.integers(0, 2**32)))
    data = [rng.choice([rng.getrandbits(32), rng.randrange(-300, 300) % (1 << 32)]) for _ in range(18 * 16)]
    return mir.func("pressure", b.ops), data


SETTINGS = settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _run(case, threads, args_base, slot, tpb_warps):
    f, data = case
    mem = [0] * IN + data
    mem += [0] * (args_base + slot - len(mem)) + [SCRATCH]
    tpb, warps = tpb_warps
    try:
        compile_mir(f, args_base=args_base, spill_slot=slot)
    except ProgramTooLarge:
        assume(False)                                   # a legitimate outcome for a random program; rare
    p, _ = compile_and_check(f, mem, threads, args_base=args_base, spill_slot=slot, ignore=[(SCRATCH, 4096)],
                             threads_per_block=tpb, warps_per_core=warps)
    assert p.stats["spilled_vregs"] > 0 and p.stats["frame_words"] > 0, p.stats
    assert p.stats["frame_words"] * threads <= 4096 - SCRATCH
    return p


@SETTINGS
@given(pressure_program(loops=False), st.integers(1, 16), st.sampled_from([1536, 2048]),
       st.sampled_from([0, 1, 7, 300]), st.sampled_from([(4, 1), (8, 2)]))
def test_straight_line_pressure_spills_and_matches(case, threads, args_base, slot, tpb_warps):
    _run(case, threads, args_base, slot, tpb_warps)


@SETTINGS
@given(pressure_program(loops=True), st.integers(1, 16), st.sampled_from([1536, 2048]),
       st.sampled_from([0, 3]), st.sampled_from([(4, 1), (8, 2)]))
def test_loop_pressure_spills_and_matches(case, threads, args_base, slot, tpb_warps):
    _run(case, threads, args_base, slot, tpb_warps)


def _wide(n_live, loop_users=None):
    """n_live loaded values; values in `loop_users` are used in a 3-trip loop, the others only after it."""
    b = Builder()
    g = b.global_tid()
    vals = [b.ldr(b.add(b.mul(g, 32), j)) for j in range(n_live)]
    acc = b.reg()
    b.emit(mir.Const32Op(acc, 0))
    c = b.reg()
    b.emit(mir.Const32Op(c, 0))
    b.label("top")
    for v in (vals if loop_users is None else [vals[i] for i in loop_users]):
        b.add(acc, v, into=acc)
    b.add(c, 1, into=c)
    b.emit(mir.CmpOp(c, b.const(3)))
    b.emit(mir.BrOp(0b100, "top"))
    out = b.addr(1024, (g, 32))
    for j, v in enumerate(vals):
        b.str_(b.add(out, j), v)
    b.str_(b.add(out, 31), acc)
    b.emit(mir.RetOp())
    return b.ops, vals


def test_loop_depth_steers_spills_outside_the_loop():
    ops, vals = _wide(14, loop_users=range(4))
    f = mir.func("depth", ops)
    data = list(range(1, 600)) + [0] * (2000 - 599) + [3000]
    p, _ = compile_and_check(f, data, 8, args_base=1999, spill_slot=1, ignore=[(3000, 4096)])
    names = set(p.stats["spilled_names"])
    assert names and not names & set(vals[:4]), "a value used in the loop was spilled"


def test_no_spill_slot_raises_only_when_spilling():
    ops, _ = _wide(16)
    with pytest.raises(BackendError, match="spill_slot"):
        compile_mir(mir.func("w", ops), args_base=100)
    ops, _ = _wide(6)
    p = compile_mir(mir.func("w", ops), args_base=100)
    assert p.stats["frame_words"] == 0 and p.stats["spill_loads"] == 0


@pytest.mark.parametrize("reuse", [True, False])
def test_args_register_reuse(reuse):
    """R12 is written after the prologue only when reuse_args_reg allows it, and only after its last read."""
    ops, _ = _wide(12)
    f = mir.func("r12", ops)
    data = list(range(1, 600)) + [0] * (2000 - 599) + [3000]
    p = compile_mir(f, args_base=1999, spill_slot=1, reuse_args_reg=reuse)
    writes12 = [k for k, w in enumerate(p.words[6:]) if isa.decode(w)[1].get("rd") == 12
                and isa.decode(w)[0] not in ("STR", "STS", "CMP", "BR")]
    assert bool(writes12) == reuse
    want = interp(f, data, 8, 1999)
    got = sim(p.words, data, 8).memory
    assert want[1024:1024 + 8 * 32] == got[1024:1024 + 8 * 32]


def test_constants_rematerialize_instead_of_spilling():
    """Many cheap constants (products a*b, 3 instructions) live across the program: they are rebuilt at their
    uses, never stored."""
    b = Builder()
    g = b.global_tid()
    cs = [b.const((17 + k) * (100 + 3 * k)) for k in range(16)]
    x = b.ldr(g)
    for c in cs:
        x = b.add(x, c)
    acc = b.reg()
    b.emit(mir.Const32Op(acc, 0))
    for c in cs:                                    # every constant used twice, far apart
        b.add(acc, b.mul(c, x), into=acc)
    b.str_(b.add(g, 100), acc)
    b.emit(mir.RetOp())
    f = mir.func("consts", b.ops)
    p, _ = compile_and_check(f, list(range(64)), 8, args_base=200)
    assert p.stats["remat_vregs"] > 0 and p.stats["spilled_vregs"] == 0


def test_wide_constants_may_live_in_memory():
    """A constant whose materialization is longer than a reload is spilled to its frame slot instead."""
    b = Builder()
    g = b.global_tid()
    cs = [b.const(0x01000000 * (k + 1) + 0x10203 * k + 0x55) for k in range(8)]
    xs = [b.ldr(b.add(g, 8 * k)) for k in range(6)]
    acc = b.reg()
    b.emit(mir.Const32Op(acc, 0))
    for c in cs + cs:
        b.add(acc, b.mul(c, b.add(acc, xs[0])), into=acc)
    for k, x in enumerate(xs):
        b.str_(b.add(g, 100 + 8 * k), b.add(x, acc))
    b.emit(mir.RetOp())
    f = mir.func("wide", b.ops)
    data = list(range(64)) + [0] * (300 - 64) + [3000]
    p, _ = compile_and_check(f, data, 8, args_base=300, spill_slot=0, ignore=[(3000, 4096)])
    assert p.stats["spilled_vregs"] > 0


def test_undefined_virtual_register_is_rejected():
    with pytest.raises(BackendError, match="read before written"):
        compile_mir(mir.func("u", [mir.AddOp("v0", "v1", "R13"), mir.StrOp("v0", "v0"), mir.RetOp()]),
                    args_base=0)
