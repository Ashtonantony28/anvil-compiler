"""anvil.mir: machine IR, one op per ISA instruction (sim/xf/isa.py) on named registers, plus pseudo-ops.

A MIR function is a flat instruction list in one block (not SSA: registers are reassigned in loops, as in
any pre-emission machine IR). Register operands are string properties:

    vN          virtual register, unbounded (before register allocation)
    R0 .. R12   physical general-purpose registers
    R13 .. R15  %blockIdx, %blockDim, %threadIdx (read-only)

    nop | bar | ret                                     NOP BAR RET
    br {nzp, target}      sync {target}                 BRnzp (nzp 1..7) / SYNC (BR nzp=000) to a label
    cmp {rs, rt}                                        CMP
    add|sub|mul|div|sra|max {rd, rs, rt}                3-register ALU ops
    ldr|lds {rd, rs}      str|sts {rs, rt}              memory / shared memory
    const {rd, imm}                                     CONST, imm 0..255
    label {sym}                                         pseudo: branch target
    const32 {rd, value}                                 pseudo: any word constant (materialized by the backend)
    sreg {rd, which}                                    pseudo: rd <- %blockIdx | %blockDim | %threadIdx

`encode(func)` turns fully allocated MIR (physical registers, no const32/sreg) into 16-bit words with xf.isa;
`decode(words)` lifts an assembled program back to MIR (labels L<pc> at every branch target).
"""
import re
from typing import ClassVar

from xdsl.dialects.builtin import IntegerAttr, StringAttr, i64
from xdsl.ir import Block, Dialect, Region
from xdsl.irdl import IRDLOperation, irdl_op_definition, prop_def, region_def, traits_def
from xdsl.traits import NoTerminator
from xdsl.utils.exceptions import VerifyException

from xf import isa

SPECIAL = {"blockIdx": 13, "blockDim": 14, "threadIdx": 15}
_REG = re.compile(r"^(v\d+|R(\d|1[0-5]))$")


def is_virtual(r: str) -> bool:
    return r.startswith("v")


def phys_index(r: str) -> int:
    if not r.startswith("R"):
        raise ValueError(f"{r} is not a physical register")
    return int(r[1:])


def _check_reg(op, field, write):
    r = getattr(op, field).data
    if not _REG.match(r):
        raise VerifyException(f"{op.name}: {field}={r!r} is not a register (vN or R0..R15)")
    if write and r in ("R13", "R14", "R15"):
        raise VerifyException(f"{op.name}: {field}={r} is read-only")


class MirOp(IRDLOperation):
    """Base: READS / WRITES name the register properties (for the interpreter and the register allocator)."""
    READS: ClassVar[tuple[str, ...]] = ()
    WRITES: ClassVar[tuple[str, ...]] = ()
    MNEMONIC: ClassVar[str | None] = None     # xf.isa mnemonic, None for pseudo-ops

    def verify_(self):
        for f in self.READS:
            _check_reg(self, f, False)
        for f in self.WRITES:
            _check_reg(self, f, True)

    def regs_read(self) -> list[str]:
        return [getattr(self, f).data for f in self.READS]

    def regs_written(self) -> list[str]:
        return [getattr(self, f).data for f in self.WRITES]


@irdl_op_definition
class FuncOp(IRDLOperation):
    name = "anvil.mir.func"
    sym_name = prop_def(StringAttr)
    body = region_def("single_block")
    traits = traits_def(NoTerminator())

    def __init__(self, sym_name: str, ops=()):
        super().__init__(properties={"sym_name": StringAttr(sym_name)}, regions=[Region(Block(list(ops)))])

    @property
    def instrs(self) -> list[MirOp]:
        return list(self.body.block.ops)


def _nullary(opname, mnem):
    @irdl_op_definition
    class Op(MirOp):
        name = f"anvil.mir.{opname}"
        assembly_format = "attr-dict"
        MNEMONIC: ClassVar[str | None] = mnem

        def __init__(self):
            super().__init__()
    Op.__name__ = Op.__qualname__ = f"{opname.capitalize()}Op"
    return Op


NopOp = _nullary("nop", "NOP")
BarOp = _nullary("bar", "BAR")
RetOp = _nullary("ret", "RET")


@irdl_op_definition
class BrOp(MirOp):
    name = "anvil.mir.br"
    assembly_format = "$nzp `,` $target attr-dict"
    MNEMONIC: ClassVar[str | None] = "BR"
    nzp = prop_def(IntegerAttr)
    target = prop_def(StringAttr)

    def __init__(self, nzp: int, target: str):
        super().__init__(properties={"nzp": IntegerAttr(nzp, i64), "target": StringAttr(target)})

    def verify_(self):
        if not 1 <= self.nzp.value.data <= 7:
            raise VerifyException(f"{self.name}: nzp {self.nzp.value.data} outside 1..7 (SYNC is anvil.mir.sync)")


@irdl_op_definition
class SyncOp(MirOp):
    name = "anvil.mir.sync"
    assembly_format = "$target attr-dict"
    MNEMONIC: ClassVar[str | None] = "BR"
    target = prop_def(StringAttr)

    def __init__(self, target: str):
        super().__init__(properties={"target": StringAttr(target)})


@irdl_op_definition
class CmpOp(MirOp):
    name = "anvil.mir.cmp"
    assembly_format = "$rs `,` $rt attr-dict"
    MNEMONIC: ClassVar[str | None] = "CMP"
    READS: ClassVar[tuple[str, ...]] = ("rs", "rt")
    rs = prop_def(StringAttr)
    rt = prop_def(StringAttr)

    def __init__(self, rs: str, rt: str):
        super().__init__(properties={"rs": StringAttr(rs), "rt": StringAttr(rt)})


def _alu(opname, mnem):
    @irdl_op_definition
    class Op(MirOp):
        name = f"anvil.mir.{opname}"
        assembly_format = "$rd `,` $rs `,` $rt attr-dict"
        MNEMONIC: ClassVar[str | None] = mnem
        READS: ClassVar[tuple[str, ...]] = ("rs", "rt")
        WRITES: ClassVar[tuple[str, ...]] = ("rd",)
        rd = prop_def(StringAttr)
        rs = prop_def(StringAttr)
        rt = prop_def(StringAttr)

        def __init__(self, rd: str, rs: str, rt: str):
            super().__init__(properties={"rd": StringAttr(rd), "rs": StringAttr(rs), "rt": StringAttr(rt)})
    Op.__name__ = Op.__qualname__ = f"{opname.capitalize()}Op"
    return Op


AddOp, SubOp, MulOp, DivOp, SraOp, MaxOp = (_alu(n, n.upper()) for n in ("add", "sub", "mul", "div", "sra", "max"))


def _load(opname, mnem):
    @irdl_op_definition
    class Op(MirOp):
        name = f"anvil.mir.{opname}"
        assembly_format = "$rd `,` $rs attr-dict"
        MNEMONIC: ClassVar[str | None] = mnem
        READS: ClassVar[tuple[str, ...]] = ("rs",)
        WRITES: ClassVar[tuple[str, ...]] = ("rd",)
        rd = prop_def(StringAttr)
        rs = prop_def(StringAttr)

        def __init__(self, rd: str, rs: str):
            super().__init__(properties={"rd": StringAttr(rd), "rs": StringAttr(rs)})
    Op.__name__ = Op.__qualname__ = f"{opname.capitalize()}Op"
    return Op


def _store(opname, mnem):
    @irdl_op_definition
    class Op(MirOp):
        name = f"anvil.mir.{opname}"
        assembly_format = "$rs `,` $rt attr-dict"
        MNEMONIC: ClassVar[str | None] = mnem
        READS: ClassVar[tuple[str, ...]] = ("rs", "rt")
        rs = prop_def(StringAttr)
        rt = prop_def(StringAttr)

        def __init__(self, rs: str, rt: str):
            super().__init__(properties={"rs": StringAttr(rs), "rt": StringAttr(rt)})
    Op.__name__ = Op.__qualname__ = f"{opname.capitalize()}Op"
    return Op


LdrOp, LdsOp = _load("ldr", "LDR"), _load("lds", "LDS")
StrOp, StsOp = _store("str", "STR"), _store("sts", "STS")


@irdl_op_definition
class ConstOp(MirOp):
    name = "anvil.mir.const"
    assembly_format = "$rd `,` $imm attr-dict"
    MNEMONIC: ClassVar[str | None] = "CONST"
    WRITES: ClassVar[tuple[str, ...]] = ("rd",)
    rd = prop_def(StringAttr)
    imm = prop_def(IntegerAttr)

    def __init__(self, rd: str, imm: int):
        super().__init__(properties={"rd": StringAttr(rd), "imm": IntegerAttr(imm, i64)})

    def verify_(self):
        super().verify_()
        if not 0 <= self.imm.value.data <= 255:
            raise VerifyException(f"{self.name}: imm {self.imm.value.data} outside 0..255 (use const32)")


@irdl_op_definition
class LabelOp(MirOp):
    name = "anvil.mir.label"
    assembly_format = "$sym attr-dict"
    sym = prop_def(StringAttr)

    def __init__(self, sym: str):
        super().__init__(properties={"sym": StringAttr(sym)})


@irdl_op_definition
class Const32Op(MirOp):
    name = "anvil.mir.const32"
    assembly_format = "$rd `,` $value attr-dict"
    WRITES: ClassVar[tuple[str, ...]] = ("rd",)
    rd = prop_def(StringAttr)
    value = prop_def(IntegerAttr)

    def __init__(self, rd: str, value: int):
        super().__init__(properties={"rd": StringAttr(rd), "value": IntegerAttr(value, i64)})

    def verify_(self):
        super().verify_()
        if not -(1 << 31) <= self.value.value.data < 1 << 32:
            raise VerifyException(f"{self.name}: {self.value.value.data} is not a 32-bit word")


@irdl_op_definition
class SregOp(MirOp):
    name = "anvil.mir.sreg"
    assembly_format = "$rd `,` $which attr-dict"
    WRITES: ClassVar[tuple[str, ...]] = ("rd",)
    rd = prop_def(StringAttr)
    which = prop_def(StringAttr)

    def __init__(self, rd: str, which: str):
        super().__init__(properties={"rd": StringAttr(rd), "which": StringAttr(which)})

    def verify_(self):
        super().verify_()
        if self.which.data not in SPECIAL:
            raise VerifyException(f"{self.name}: which must be one of {sorted(SPECIAL)}")


ALU = {"ADD": AddOp, "SUB": SubOp, "MUL": MulOp, "DIV": DivOp, "SRA": SraOp, "MAX": MaxOp}
OPS = [FuncOp, NopOp, BarOp, RetOp, BrOp, SyncOp, CmpOp, AddOp, SubOp, MulOp, DivOp, SraOp, MaxOp, LdrOp, LdsOp,
       StrOp, StsOp, ConstOp, LabelOp, Const32Op, SregOp]
Mir = Dialect("anvil.mir", OPS, [])
# every xf.isa opcode has exactly one MIR op (BR has two: br and sync)
assert {c.MNEMONIC for c in OPS if getattr(c, "MNEMONIC", None)} == set(isa.OPCODES)


def func(sym_name, ops):
    f = FuncOp(sym_name, ops)
    f.verify()
    return f


def labels(f: FuncOp) -> dict[str, int]:
    """label -> index of the next real instruction."""
    out, pc = {}, 0
    for op in f.instrs:
        if isinstance(op, LabelOp):
            if op.sym.data in out:
                raise VerifyException(f"duplicate label {op.sym.data}")
            out[op.sym.data] = pc
        else:
            pc += 1
    return out


def encode(f: FuncOp) -> list[int]:
    """Allocated MIR -> 16-bit words via xf.isa.encode. Rejects virtual registers and unlowered pseudo-ops."""
    lab, words = labels(f), []
    for op in f.instrs:
        if isinstance(op, LabelOp):
            continue
        if op.MNEMONIC is None:
            raise ValueError(f"{op.name} must be lowered before encoding")
        fields = {}
        for fld in op.READS + op.WRITES:
            fields[fld] = phys_index(getattr(op, fld).data)
        if isinstance(op, BrOp):
            fields.update(nzp=op.nzp.value.data, imm=lab[op.target.data])
        elif isinstance(op, SyncOp):
            fields.update(nzp=0, imm=lab[op.target.data])
        elif isinstance(op, ConstOp):
            fields["imm"] = op.imm.value.data
        words.append(isa.encode(op.MNEMONIC, **fields))
    if len(words) > 256:
        raise ValueError(f"{len(words)} instructions exceed the 256-word program memory")
    return words


def decode(words, sym_name="decoded") -> FuncOp:
    """Assembled words -> MIR on physical registers, a label L<pc> before every branch target."""
    decoded = [isa.decode(w) for w in words]
    targets = sorted({f["imm"] for m, f in decoded if m == "BR"})
    ops = []
    for pc, (m, f) in enumerate(decoded):
        if pc in targets:
            ops.append(LabelOp(f"L{pc}"))
        R = {k: f"R{v}" for k, v in f.items() if k in ("rd", "rs", "rt")}
        if m == "BR":
            ops.append(SyncOp(f"L{f['imm']}") if f["nzp"] == 0 else BrOp(f["nzp"], f"L{f['imm']}"))
        elif m in ALU:
            ops.append(ALU[m](R["rd"], R["rs"], R["rt"]))
        elif m == "CMP":
            ops.append(CmpOp(R["rs"], R["rt"]))
        elif m in ("LDR", "LDS"):
            ops.append((LdrOp if m == "LDR" else LdsOp)(R["rd"], R["rs"]))
        elif m in ("STR", "STS"):
            ops.append((StrOp if m == "STR" else StsOp)(R["rs"], R["rt"]))
        elif m == "CONST":
            ops.append(ConstOp(R["rd"], f["imm"]))
        else:
            ops.append({"NOP": NopOp, "BAR": BarOp, "RET": RetOp}[m]())
    past = [t for t in targets if t >= len(words)]
    if past:
        raise ValueError(f"branch targets {past} past the end of the program")
    return FuncOp(sym_name, ops)
