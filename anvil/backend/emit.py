"""After allocation: peephole cleanups, the ARGS prologue, label resolution, the 256-word check, assembly text."""
from __future__ import annotations

from .. import semantics as sem
from ..ir import mir
from .lir import ALU, BackendError, Ins, lower

PROGRAM_WORDS = 256
PROLOGUE_WORDS = 6
_FOLD = {"add": sem.w_add, "sub": sem.w_sub, "mul": sem.w_mul, "div": sem.w_div, "sra": sem.w_sra, "max": sem.w_max}


def peephole(code: list[Ins]) -> tuple[list[Ins], int]:
    """Local cleanups on allocated code; returns (code, instructions removed).

    * NOPs go (the ISA has no hazards that need them).
    * A BR whose target is the next instruction goes.
    * Within a straight-line run (state reset at every label), registers holding known words are tracked from
      CONST and folded ALU ops (32-bit words): an instruction that leaves its destination unchanged - a CONST or
      rematerialized constant already in the register, `ADD r,r,0`, `SUB r,r,0`, `MUL r,r,1`, `DIV r,r,1`,
      `SRA r,r,0`, `MAX r,r,r` - goes.

    Nothing is assumed about the registers the ARGS prologue leaves (R0 = lo, R12 = args_base): the program must
    stay correct when words 0 and 4 are patched to another args block.
    """
    out, known, removed = [], {}, 0
    for k, i in enumerate(code):
        if i.op == "label":
            known = {}
            out.append(i)
            continue
        if i.op == "nop":
            removed += 1
            continue
        if i.op == "br":
            j = k + 1
            while j < len(code) and code[j].op == "label" and code[j].sym != i.sym:
                j += 1
            if j < len(code) and code[j].op == "label" and code[j].sym == i.sym:
                removed += 1
                continue
        if i.op == "const":
            if known.get(i.d) == i.imm:
                removed += 1
                continue
            known[i.d] = i.imm
        elif i.op in ALU:
            ka, kb = known.get(i.a), known.get(i.b)
            val = _FOLD[i.op](ka, kb) if ka is not None and kb is not None else None
            same = (val is not None and known.get(i.d) == val) or _identity(i, ka, kb)
            if same:
                removed += 1
                continue
            if val is None:
                known.pop(i.d, None)
            else:
                known[i.d] = val
        elif i.d is not None:
            known.pop(i.d, None)
        out.append(i)
    return out, removed


def _identity(i: Ins, ka, kb) -> bool:
    if i.op == "max":
        return i.d == i.a == i.b
    if i.d == i.a and kb == {"add": 0, "sub": 0, "mul": 1, "div": 1, "sra": 0}.get(i.op, -1):
        return True
    return i.op in ("add", "mul") and i.d == i.b and ka == {"add": 0, "mul": 1}[i.op]


def prologue(args_base: int) -> list[Ins]:
    """The xf ARGS prologue: R12 = hi*256 + lo with hi at word 0 and lo at word 4 (xf.kernels.program patches
    exactly those two immediates)."""
    if not 0 <= args_base < 1 << 16:
        raise BackendError(f"args_base {args_base} outside 0..65535 (the prologue builds hi*256 + lo)")
    return [Ins("const", "R12", imm=args_base >> 8), Ins("const", "R0", imm=16), Ins("mul", "R12", "R12", "R0"),
            Ins("mul", "R12", "R12", "R0"), Ins("const", "R0", imm=args_base & 0xFF), Ins("add", "R12", "R12", "R0")]


def check_size(code: list[Ins], name: str) -> int:
    n = sum(i.op != "label" for i in code)
    if n > PROGRAM_WORDS:
        raise ProgramTooLarge(f"program {name!r} needs {n} words, {n - PROGRAM_WORDS} over the "
                              f"{PROGRAM_WORDS}-word program memory")
    lab, pc = {}, 0
    for i in code:
        if i.op == "label":
            lab[i.sym] = pc
        else:
            pc += 1
    for i in code:
        if i.op in ("br", "sync") and lab[i.sym] >= n:
            raise BackendError(f"program {name!r}: label {i.sym!r} is past the last instruction "
                               "(end the path with RET)")
    return n


class ProgramTooLarge(BackendError):
    pass


_SPECIAL = {"R13": "%blockIdx", "R14": "%blockDim", "R15": "%threadIdx"}


def asm_text(code: list[Ins], name: str, header: str = "") -> str:
    """xf/asm.py syntax; `xf.asm.assemble(asm_text(...))` gives the same words as `encode`."""
    lines = [f"; {name} - compiled by anvil.backend"] + [f"; {h}" for h in header.splitlines() if h]
    r = lambda x: _SPECIAL.get(x, x)  # noqa: E731
    for i in code:
        if i.op == "label":
            lines.append(f"{i.sym}:")
            continue
        if i.op == "br":
            m = "BR" + "".join(c for c, bit in zip("nzp", f"{i.imm:03b}") if bit == "1")
            s = f"{m} {i.sym}"
        elif i.op == "sync":
            s = f"SYNC {i.sym}"
        elif i.op == "const":
            s = f"CONST {r(i.d)}, #{i.imm}"
        elif i.op in ("str", "sts", "cmp"):
            s = f"{i.op.upper()} {r(i.a)}, {r(i.b)}"
        elif i.op in ("ldr", "lds"):
            s = f"{i.op.upper()} {r(i.d)}, {r(i.a)}"
        elif i.op in ALU:
            s = f"{i.op.upper()} {r(i.d)}, {r(i.a)}, {r(i.b)}"
        else:
            s = i.op.upper()
        lines.append(f"    {s}")
    return "\n".join(lines) + "\n"


def encode(code: list[Ins], name: str) -> tuple[list[int], mir.FuncOp]:
    f = lower(code, name)
    return mir.encode(f), f
