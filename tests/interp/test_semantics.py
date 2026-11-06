"""C3: anvil.semantics re-exports xf.golden, generalizes it consistently, and matches the ISA word semantics."""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from anvil import semantics as sem
from xf import golden, isa, isasim

i8s = st.integers(-128, 127)
words = st.integers(0, (1 << 32) - 1)


def test_reexports_are_golden():
    for n in ("clamp_i8", "matmul_i8", "requant", "relu", "swiglu_lut", "residual_add", "rsqrt_index",
              "rmsnorm_int", "softmax_lut", "I8", "I32"):
        assert getattr(sem, n) is getattr(golden, n)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.integers(-(1 << 20), 1 << 20), min_size=1, max_size=12), st.integers(1, 1 << 9),
       st.integers(0, 20))
def test_requant_var_uniform_shift_equals_golden(acc, m, s):
    assert np.array_equal(sem.requant_var(acc, m, s), golden.requant(acc, m, s))


@settings(max_examples=40, deadline=None)
@given(st.integers(1, 4), st.integers(1, 8), st.data())
def test_requant_var_per_row_equals_rmsnorm_requant(t, d, data):
    """rmsnorm_int's per-row s + k is requant_var with a broadcast shift column."""
    acc = np.array(data.draw(st.lists(st.integers(-(1 << 22), 1 << 22), min_size=t * d, max_size=t * d))).reshape(t, d)
    k = np.array(data.draw(st.lists(st.integers(0, 4), min_size=t, max_size=t)))
    want = np.stack([golden.requant(acc[i], 3, 10 + int(k[i])) for i in range(t)])
    assert np.array_equal(sem.requant_var(acc, 3, 10 + k[:, None]), want)


@settings(max_examples=40, deadline=None)
@given(st.integers(1, 4), st.integers(1, 6), st.integers(1, 5), st.data())
def test_qmatmul_and_lut_epilogue(m, k, n, data):
    a = np.array(data.draw(st.lists(i8s, min_size=m * k, max_size=m * k))).reshape(m, k)
    b = np.array(data.draw(st.lists(i8s, min_size=k * n, max_size=k * n))).reshape(k, n)
    lut = np.array(data.draw(st.lists(i8s, min_size=256, max_size=256)))
    rq = golden.requant(golden.matmul_i8(a, b), 77, 9)
    assert np.array_equal(sem.qmatmul(a, b, 77, 9), rq)
    assert np.array_equal(sem.qmatmul(a, b, 77, 9, lut), lut[rq + 128])
    assert np.array_equal(sem.lut_i8(rq, lut), lut[rq + 128])


def test_checked_int32_raises():
    with pytest.raises(OverflowError):
        sem.requant([1 << 30], 4, 3)
    with pytest.raises(OverflowError):
        sem.add_i32(golden.I32[1], 1)
    with pytest.raises(OverflowError):
        sem.mul_i32(1 << 16, 1 << 16)
    with pytest.raises(ValueError):
        sem.check_i8([128])
    with pytest.raises(IndexError):
        sem.lut_lookup(np.arange(4), [4])
    with pytest.raises(IndexError):
        sem.lut_lookup(np.arange(4), [-1])        # numpy would wrap silently
    with pytest.raises(ValueError):
        sem.div_floor(-1, 2)
    assert sem.round_term(0) == 0 and sem.round_term(5) == 16


def _isasim_alu(mnem, a, b):
    prog = [isa.encode("CONST", rd=0, imm=0), isa.encode("CONST", rd=1, imm=1),
            isa.encode("LDR", rd=2, rs=0), isa.encode("LDR", rd=3, rs=1)]
    if mnem == "CMP":     # mem[2] = 1 if Z (equal) else 0
        prog += [isa.encode("CONST", rd=4, imm=0), isa.encode("CMP", rs=2, rt=3),
                 isa.encode("BR", nzp=0b100, imm=8), isa.encode("CONST", rd=4, imm=1)]
    else:
        prog += [isa.encode(mnem, rd=4, rs=2, rt=3)]
    prog += [isa.encode("CONST", rd=5, imm=2), isa.encode("STR", rs=5, rt=4), isa.encode("RET")]
    return isasim.run(prog, [a, b], 1, word_bits=32, addr_bits=12).memory[2]


WORD_OPS = {"ADD": sem.w_add, "SUB": sem.w_sub, "MUL": sem.w_mul, "DIV": sem.w_div, "SRA": sem.w_sra, "MAX": sem.w_max}


@settings(max_examples=25, deadline=None)
@given(st.sampled_from(sorted(WORD_OPS) + ["CMP"]), words, st.one_of(words, st.integers(0, 40)))
def test_word_ops_match_isasim(mnem, a, b):
    if mnem == "CMP":
        want = int(sem.w_cmp_nzp(a, b) == sem.NZP_Z)
    else:
        want = WORD_OPS[mnem](a, b)
    assert _isasim_alu(mnem, a, b) == want


def test_word_helpers():
    assert sem.wrap(-1) == (1 << 32) - 1 and sem.to_signed((1 << 32) - 1) == -1
    assert sem.w_sra(sem.wrap(-8), 40) == sem.wrap(-1)       # rt >= WORD_BITS fills with the sign
    assert sem.w_div(5, 0) == 0
    assert sem.w_max(sem.wrap(-1), 0) == 0
    assert sem.w_cmp_nzp(3, 3) == sem.NZP_Z and sem.w_cmp_nzp(2, 3) == sem.NZP_N
