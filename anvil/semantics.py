"""C3: the single source of Anvil's integer arithmetic.

Two families live here and nowhere else:

* **Checked tensor semantics** (graph and tensor IR): int64 numpy arrays whose every intermediate must fit int32,
  raising ``OverflowError`` otherwise, exactly as ``xf.golden``. The xf golden ops are re-exported unchanged; the
  generalized ops (per-element requant shift, LUT lookups, fused qmatmul epilogues, scalar helpers for loop-level
  interpreters) are built on them and tested equal to them where both apply.
* **Machine word semantics** (MIR and the ISA): unsigned ``WORD_BITS``-bit words with wrap-around, the ALU of
  ``docs/isa.md`` (``SRA`` and ``MAX`` signed, ``DIV`` and ``CMP`` unsigned, ``DIV`` by zero = 0), bit for bit with
  ``xf.isasim``.
"""
import numpy as np

from xf import golden
from xf.golden import (  # noqa: F401  (re-exported: C3)
    I8,
    I32,
    clamp_i8,
    matmul_i8,
    relu,
    requant,
    residual_add,
    rmsnorm_int,
    rsqrt_index,
    softmax_lut,
    swiglu_lut,
)

WORD_BITS = 32
SMEM_WORDS = 64

__all__ = [
    "I8", "I32", "WORD_BITS", "SMEM_WORDS",
    "check_i32", "check_i8", "clamp_i8", "matmul_i8", "relu", "requant", "residual_add", "rmsnorm_int",
    "rsqrt_index", "softmax_lut", "swiglu_lut",
    "round_term", "requant_var", "qmatmul", "lut_lookup", "lut_i8", "add_requant",
    "DT_RANGE", "check_dtype", "clamp_dtype", "requant_dtype", "rmsnorm_wide", "sq_shift",
    "add_i32", "sub_i32", "mul_i32", "max_i32", "min_i32", "div_floor", "sra_i32", "cmp",
    "wrap", "to_signed", "w_add", "w_sub", "w_mul", "w_div", "w_sra", "w_max", "w_cmp_nzp",
    "NZP_N", "NZP_Z", "NZP_P",
]


# ---------------------------------------------------------------------------------------------- checked int32

def check_i32(x, what="value"):
    """x as int64 numpy, raising OverflowError if any element leaves int32 (golden._i32)."""
    return golden._i32(x, what)


def check_i8(x, what="value"):
    """x as int64 numpy, raising ValueError if any element leaves int8 (golden._i8)."""
    return golden._i8(x, what)


def round_term(s):
    """The requant rounding addend 2^(s-1), 0 when s == 0 (the kernels' `_rnd`)."""
    s = int(s)
    if not 0 <= s <= 31:
        raise ValueError(f"shift {s} outside [0, 31]")
    return (1 << (s - 1)) if s else 0


def requant_var(acc, m, s):
    """golden.requant with a per-element shift: s broadcasts against acc (rmsnorm_int's s + k)."""
    acc = check_i32(acc, "acc")
    s = np.broadcast_to(np.asarray(s, dtype=np.int64), acc.shape)
    out = np.empty(acc.shape, dtype=np.int64)
    for sv in np.unique(s):
        sel = s == sv
        out[sel] = requant(acc[sel], m, int(sv))
    return out


def lut_lookup(table, idx, offset=0):
    """table[idx + offset] with an explicit bounds check (numpy would wrap negative indices)."""
    table = np.asarray(table, dtype=np.int64)
    i = np.asarray(idx, dtype=np.int64) + int(offset)
    if table.ndim != 1:
        raise ValueError("LUT must be 1-D")
    if i.size and (i.min() < 0 or i.max() >= len(table)):
        raise IndexError(f"LUT index outside [0, {len(table)})")
    return table[i]


def lut_i8(x, table):
    """Unary int8 LUT: table[x + 128], table of 256 int8 entries."""
    x, table = check_i8(x, "x"), check_i8(table, "lut")
    if table.shape != (256,):
        raise ValueError("int8 LUT must have 256 entries")
    return table[x + 128]


def qmatmul(a, b, m, s, lut=None, dtype="i8"):
    """requant(a @ b, m, s), then the optional fused unary LUT epilogue lut[. + 128]; an i16 result (no LUT)
    clamps to int16 (ADR-004)."""
    if dtype != "i8":
        if lut is not None:
            raise ValueError("a LUT epilogue needs an i8 result")
        return requant_dtype(matmul_i8(a, b), m, s, dtype)
    out = requant(matmul_i8(a, b), m, s)
    return out if lut is None else lut_i8(out, lut)


def add_requant(a, b, ma, mb, s, dtype="i8"):
    """Scale-aligned residual clamp((a*Ma + b*Mb + 2^(s-1)) >> s) to `dtype`. All-i8 operands and result: exactly
    golden.residual_add. Otherwise (SG9's wide residual stream, ADR-004) a and b may each be i8 or i16 and the
    result is clamped to `dtype`, with every intermediate checked against int32 in the same order."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    if dtype == "i8" and _is_i8(a) and _is_i8(b):
        return residual_add(a, b, ma, mb, s)
    a, b = check_dtype(a, "i16", "a"), check_dtype(b, "i16", "b")
    acc = check_i32(check_i32(a * int(ma), "a*Ma") + check_i32(b * int(mb), "b*Mb"), "a*Ma + b*Mb")
    return requant_dtype(acc, 1, s, dtype)


# ------------------------------------------------------------------------------------- wide values (ADR-004)
# The residual stream may be i16 (one value per 32-bit word, like every other value). Requant results clamp to
# their dtype's range; rmsnorm accepts an i16 row and sums rounded-shifted squares so the sum stays in int32.

DT_RANGE = {"i8": I8, "i16": (-(1 << 15), (1 << 15) - 1), "i32": I32}


def _is_i8(x):
    return not x.size or (x.min() >= I8[0] and x.max() <= I8[1])


def check_dtype(x, dtype, what="value"):
    """x as int64 numpy, raising ValueError if any element leaves `dtype`'s range (OverflowError for i32)."""
    if dtype == "i32":
        return check_i32(x, what)
    if dtype == "i8":
        return check_i8(x, what)
    lo, hi = DT_RANGE[dtype]
    x = np.asarray(x, dtype=np.int64)
    if x.size and (x.min() < lo or x.max() > hi):
        raise ValueError(f"{what} is not int{dtype[1:]}")
    return x


def clamp_dtype(x, dtype):
    return np.clip(x, *DT_RANGE[dtype])


def requant_dtype(acc, m, s, dtype="i8"):
    """golden.requant clamped to `dtype` instead of int8 (identical for i8)."""
    if dtype == "i8":
        return requant(acc, m, s)
    if not 0 <= s <= 31:
        raise ValueError(f"shift {s} outside [0, 31]")
    acc = check_i32(acc, "acc")
    prod = check_i32(acc * int(m), "acc*M")
    rounded = check_i32(prod + round_term(s), "acc*M + round")
    return clamp_dtype(rounded >> s, dtype)


def sq_shift(x, c):
    """Rounded right shift of the square: (x*x + 2^(c-1)) >> c, the term rmsnorm sums for a wide row."""
    sq = check_i32(x * x, "x*x")
    return check_i32(sq + round_term(c), "x*x + round") >> int(c)


def rmsnorm_wide(x, w, rsqrt_lut, m, s, ss_shift=0, ss_init=0, xdtype="i8"):
    """rmsnorm_int with an i8 or i16 input row: ss = e + sum((x^2 + 2^(c-1)) >> c), c = ss_shift, e = ss_init
    (eps folded into the sum, so the LUT holds an eps-free rsqrt that is exact at every shift k); then exactly
    golden.rmsnorm_int's index, LUT and requant (int8 result). With an i8 input and c = e = 0 it is
    golden.rmsnorm_int."""
    x = np.asarray(x, dtype=np.int64)
    if xdtype == "i8" and int(ss_shift) == 0 and int(ss_init) == 0:
        return rmsnorm_int(x, w, rsqrt_lut, m, s)
    x, w = check_dtype(x, xdtype, "x"), check_i8(w, "w")
    lut = check_i32(rsqrt_lut, "rsqrt_lut")
    if lut.ndim != 1 or lut.min() < 0:
        raise ValueError("rsqrt_lut must be a non-negative 1-D table")
    ss = check_i32(sq_shift(x, ss_shift).sum(axis=-1) + int(ss_init), "sum of squares")
    idx, k = rsqrt_index(ss, len(lut))
    xwr = check_i32(check_i32(x * w, "x*w") * lut[idx][..., None], "x*w*r")
    out = np.empty_like(xwr)
    for kv in np.unique(k):
        out[k == kv] = requant(xwr[k == kv], m, s + int(kv))
    return out


# ---------------------------------------------------------------------------------- checked int32 scalar ALU
# Used element-wise by loop-level interpreters (tensor IR); same int32 discipline as golden.

def add_i32(a, b):
    return check_i32(np.int64(a) + np.int64(b), "add")


def sub_i32(a, b):
    return check_i32(np.int64(a) - np.int64(b), "sub")


def mul_i32(a, b):
    return check_i32(np.int64(a) * np.int64(b), "mul")


def max_i32(a, b):
    return np.maximum(check_i32(a), check_i32(b))


def min_i32(a, b):
    return np.minimum(check_i32(a), check_i32(b))


def div_floor(a, b):
    """Floor division of non-negative values (the only division golden uses: softmax)."""
    a, b = check_i32(a), check_i32(b)
    if (np.asarray(a) < 0).any() or (np.asarray(b) <= 0).any():
        raise ValueError("div_floor needs a >= 0 and b > 0")
    return a // b


def sra_i32(a, b):
    """Arithmetic shift right by 0 <= b <= 31."""
    a, b = check_i32(a), np.asarray(b, dtype=np.int64)
    if (b < 0).any() or (b > 31).any():
        raise ValueError("shift outside [0, 31]")
    return a >> b


_CMP = {"eq": np.equal, "ne": np.not_equal, "lt": np.less, "le": np.less_equal,
        "gt": np.greater, "ge": np.greater_equal}


def cmp(pred, a, b):
    """Signed comparison -> 0/1 int64."""
    return _CMP[pred](check_i32(a), check_i32(b)).astype(np.int64)


# ----------------------------------------------------------------------------------------- machine words
NZP_N, NZP_Z, NZP_P = 0b100, 0b010, 0b001


def wrap(x, bits=WORD_BITS):
    """Unsigned word: x mod 2^bits (python ints and numpy arrays)."""
    if isinstance(x, (int, np.integer)):
        return int(x) & ((1 << bits) - 1)
    return np.asarray(x, dtype=np.int64) & ((1 << bits) - 1)


def to_signed(x, bits=WORD_BITS):
    """Two's-complement reading of an unsigned word."""
    x = wrap(x, bits)
    if isinstance(x, int):
        return x - (1 << bits) if x >> (bits - 1) else x
    return np.where(x >> (bits - 1), x - (1 << bits), x)


def w_add(a, b, bits=WORD_BITS):
    return wrap(int(a) + int(b), bits)


def w_sub(a, b, bits=WORD_BITS):
    return wrap(int(a) - int(b), bits)


def w_mul(a, b, bits=WORD_BITS):
    return wrap(int(a) * int(b), bits)


def w_div(a, b, bits=WORD_BITS):
    """Unsigned division; 0 when b == 0."""
    a, b = wrap(a, bits), wrap(b, bits)
    return a // b if b else 0


def w_sra(a, b, bits=WORD_BITS):
    """Signed a >> unsigned b; b >= bits fills with the sign."""
    return wrap(to_signed(a, bits) >> wrap(b, bits), bits)


def w_max(a, b, bits=WORD_BITS):
    """Signed max of two words."""
    a, b = wrap(a, bits), wrap(b, bits)
    return a if to_signed(a, bits) > to_signed(b, bits) else b


def w_cmp_nzp(a, b, bits=WORD_BITS):
    """CMP: N when a != b (unsigned rs - rt > 0), Z when equal, P never (docs/isa.md)."""
    return NZP_N if wrap(a, bits) != wrap(b, bits) else NZP_Z
