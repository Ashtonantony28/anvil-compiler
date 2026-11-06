"""SG2 accept: every idiom equals anvil.semantics on random values, on the MIR interpreter (virtual registers)
and compiled on isasim. One thread per input pair: x = MEM[2g], y = MEM[2g+1], out = MEM[512+g]."""
import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st

from anvil import semantics as sem
from anvil.backend import compile_mir
from anvil.backend.idioms import Builder
from anvil.ir import mir

from backend_util import interp, sim

I32 = st.integers(-(1 << 31), (1 << 31) - 1)
EDGE = st.sampled_from([0, 1, -1, 127, 128, -128, -129, (1 << 31) - 1, -(1 << 31), 255, 256])
VAL = st.one_of(I32, EDGE, st.integers(-300, 300))
PAIRS = st.lists(st.tuples(VAL, VAL), min_size=1, max_size=12)
OUT = 512


def run_idiom(build, pairs, extra=()):
    """build(b, x, y) -> result register. Returns signed results from (MIR interp, isasim) after checking they
    agree."""
    b = Builder()
    g = b.global_tid()
    two_g = b.mul(g, 2)
    x = b.ldr(two_g)
    y = b.ldr(b.add(two_g, 1))
    b.str_(b.add(g, OUT), build(b, x, y))
    b.emit(mir.RetOp())
    f = mir.func("idiom", b.ops)
    data = [sem.wrap(v) for p in pairs for v in p] + [0] * (OUT - 2 * len(pairs)) + list(extra)
    n = len(pairs)
    want = interp(f, data, n, 4000)
    p = compile_mir(f, args_base=4000)
    got = sim(p.words, data, n).memory
    assert want[OUT:OUT + n] == got[OUT:OUT + n]
    return [sem.to_signed(v) for v in got[OUT:OUT + n]]


def ref(fn, pairs):
    return [int(fn(a, b)) for a, b in pairs]


S = settings(max_examples=25, deadline=None)

BINARY = {
    "min": (lambda b, x, y: b.min(x, y), lambda a, c: sem.min_i32(a, c)),
    "max": (lambda b, x, y: b.max(x, y), lambda a, c: sem.max_i32(a, c)),
    "lt": (lambda b, x, y: b.lt(x, y), lambda a, c: sem.cmp("lt", a, c)),
    "le": (lambda b, x, y: b.le(x, y), lambda a, c: sem.cmp("le", a, c)),
    "gt": (lambda b, x, y: b.gt(x, y), lambda a, c: sem.cmp("gt", a, c)),
    "ge": (lambda b, x, y: b.ge(x, y), lambda a, c: sem.cmp("ge", a, c)),
    "eq": (lambda b, x, y: b.eq(x, y), lambda a, c: sem.cmp("eq", a, c)),
    "ne": (lambda b, x, y: b.ne(x, y), lambda a, c: sem.cmp("ne", a, c)),
    "clamp_i8": (lambda b, x, y: b.clamp_i8(x), lambda a, c: sem.clamp_i8(np.int64(a))),
    "relu": (lambda b, x, y: b.max(x, 0), lambda a, c: max(a, 0)),
}


@pytest.mark.parametrize("name", sorted(BINARY))
@S
@given(pairs=PAIRS)
def test_binary_idioms_equal_semantics(name, pairs):
    build, fn = BINARY[name]
    assert run_idiom(build, pairs) == ref(fn, pairs)


@S
@given(pairs=PAIRS)
def test_neg_and_abs(pairs):
    assume(all(a != -(1 << 31) for a, _ in pairs))
    assert run_idiom(lambda b, x, y: b.neg(x), pairs) == ref(lambda a, c: sem.sub_i32(0, a), pairs)
    assert run_idiom(lambda b, x, y: b.abs(x), pairs) == [abs(a) for a, _ in pairs]


@S
@given(pairs=PAIRS, bounds=st.tuples(I32, I32))
def test_clamp_constant_bounds(pairs, bounds):
    lo, hi = sorted(bounds)
    got = run_idiom(lambda b, x, y: b.clamp(x, lo, hi), pairs)
    assert got == ref(lambda a, c: sem.min_i32(sem.max_i32(a, lo), hi), pairs)


@S
@given(pairs=PAIRS)
def test_select_and_mask(pairs):
    """select(c, x, y) with c = lt(x, y): the signed minimum again, through the mask arithmetic."""
    got = run_idiom(lambda b, x, y: b.select(b.lt(x, y), x, y), pairs)
    assert got == ref(lambda a, c: sem.min_i32(a, c), pairs)
    got = run_idiom(lambda b, x, y: b.mask(b.ne(x, y)), pairs)
    assert got == [-1 if a != c else 0 for a, c in pairs]


@S
@given(pairs=PAIRS, k=st.integers(0, 31))
def test_mul_pow2(pairs, k):
    pairs = [(a >> k, c) for a, c in pairs]          # keep a * 2^k inside int32
    got = run_idiom(lambda b, x, y: b.mul_pow2(x, k), pairs)
    assert got == ref(lambda a, c: sem.mul_i32(a, 1 << k), pairs)


ACC = st.integers(-(1 << 20), 1 << 20)


@S
@given(accs=st.lists(st.tuples(ACC, st.just(0)), min_size=1, max_size=12), m=st.integers(1, 1 << 10),
       s=st.integers(0, 31))
def test_requant_constant(accs, m, s):
    try:
        want = [int(sem.requant(np.int64(a), m, s)) for a, _ in accs]
    except OverflowError:
        assume(False)
    assert run_idiom(lambda b, x, y: b.requant(x, m, s), accs) == want


@S
@given(accs=st.lists(st.tuples(ACC, st.just(0)), min_size=1, max_size=12), m=st.integers(-(1 << 10), 1 << 10),
       s=st.integers(0, 31))
def test_requant_register_operands(accs, m, s):
    """M, s and the rounding term loaded from memory, as the xf kernels do."""
    def build(b, x, y):
        base = b.const(OUT + 64)
        mm, ss, rr = (b.ldr(b.add(base, k)) for k in range(3))
        return b.requant(x, mm, ss, rnd=rr)
    extra = [0] * 64 + [sem.wrap(m), s, sem.round_term(s)]
    try:
        want = [int(sem.requant(np.int64(a), m, s)) for a, _ in accs]
    except OverflowError:
        assume(False)
    assert run_idiom(build, accs, extra) == want


@S
@given(pairs=st.lists(st.tuples(st.integers(0, 255), st.integers(0, 255)), min_size=1, max_size=12),
       base=st.integers(0, (1 << 32) - 1), s1=st.integers(0, (1 << 32) - 1), s2=st.integers(1, 3))
def test_wide_address_arithmetic(pairs, base, s1, s2):
    got = run_idiom(lambda b, x, y: b.addr(base, (x, s1), (y, s2)), pairs)
    assert got == [sem.to_signed(base + a * s1 + c * s2) for a, c in pairs]
