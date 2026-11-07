"""Extended ISA table (model-only study, SG8 / G5): the 16 base opcodes of sim/xf/isa.py plus candidate instructions.

Encoding. The base ISA uses all 16 values of its 4-bit opcode field, so no new instruction fits a 16-bit word. The
study assumes a **5-bit opcode**: a 17-bit instruction word `op[16:12] rd[11:8] rs[7:4] rt[3:0]` (BR/CONST keep
their imm[7:0] / nzp[11:9] fields). Opcodes 0-15 are the base ISA bit for bit (a base program is a valid extended
program, word for word); 16-31 are the extension space. Nothing here changes the RTL: program memory, fetcher and
decoder would all need a wider word (PROGRAM_MEM_DATA_BITS = 17) and new ALU functions, which this study only
models (xf.isasim fork in anvil.isa_ext.isasim; tier-1 timing where every new op costs one ALU instruction).

| op5  | mnemonic | semantics (WORD_BITS-bit words, signed = two's complement)         | replaces (base idiom)                 |
|------|----------|--------------------------------------------------------------------|---------------------------------------|
| 16   | MIN      | rd <- min(rs, rt) signed                                           | a + b - max(a, b) (3 ops)             |
| 17   | AND      | rd <- rs & rt                                                      | x - 2^k * (x div 2^k) (3), masks (MUL)|
| 18   | SHL      | rd <- rs << rt, 0 when rt >= WORD_BITS (rt unsigned)               | MUL by 2^k (const), 0 - SRA(INT_MIN, 31 - S) (3) |
| 19   | SLT      | rd <- 1 if rs < rt (signed) else 0                                 | (max(a, b) - a) div itself (3 ops)    |
| 20   | MAD      | rd <- rd + rs * rt   (rd is read: a tied accumulator)               | MUL t + ADD rd, rd, t (2 ops)         |
"""
from __future__ import annotations

from xf import isa as base

EXT = {
    "MIN": (0b10000, "rd,rs,rt"),
    "AND": (0b10001, "rd,rs,rt"),
    "SHL": (0b10010, "rd,rs,rt"),
    "SLT": (0b10011, "rd,rs,rt"),
    "MAD": (0b10100, "rd,rs,rt"),
}
PROPOSALS = tuple(EXT)                      # every candidate instruction, in table order
OPCODES = {**base.OPCODES, **EXT}
BY_OPCODE = {op: m for m, (op, _) in OPCODES.items()}
FIELDS = {**base.FIELDS, "op": (12, 5)}
WORD_BITS_EXT = 17


def operands(mnemonic):
    fmt = OPCODES[mnemonic][1]
    return fmt.split(",") if fmt else []


def encode(mnemonic, **fields) -> int:
    if mnemonic in base.OPCODES:
        return base.encode(mnemonic, **fields)
    op, _ = EXT[mnemonic]
    word = op << 12
    for name in operands(mnemonic):
        shift, width = FIELDS[name]
        v = fields.get(name, 0)
        if not 0 <= v < (1 << width):
            raise ValueError(f"{mnemonic}: {name}={v} out of range")
        word |= v << shift
    return word


def decode(word):
    mnemonic = BY_OPCODE[word >> 12]
    return mnemonic, {n: (word >> FIELDS[n][0]) & ((1 << FIELDS[n][1]) - 1) for n in operands(mnemonic)}


def semantics(word_bits: int = 32) -> dict:
    """mnemonic -> f(rd_old, rs, rt) -> new rd word, for the extension ops."""
    mask, sign = (1 << word_bits) - 1, 1 << (word_bits - 1)

    def s(x):
        return x - ((x & sign) << 1)
    return {
        "MIN": lambda d, a, b: a if s(a) < s(b) else b,
        "AND": lambda d, a, b: a & b,
        "SHL": lambda d, a, b: (a << b) & mask if b < word_bits else 0,
        "SLT": lambda d, a, b: int(s(a) < s(b)),
        "MAD": lambda d, a, b: (d + a * b) & mask,
    }
