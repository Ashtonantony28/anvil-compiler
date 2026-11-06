"""anvil.backend: anvil.mir (virtual registers) -> tiny-gpu-extended program words.

    prog = compile_mir(module, args_base=img.base("c.args"), spill_slot=None)
    prog.words      16-bit words, at most 256, starting with the xf ARGS prologue
    prog.asm_text   the same program in xf/asm.py syntax (assembles to prog.words)
    prog.stats      sizes, spills, rematerializations, allocator rounds (see Program)
    prog.func       the allocated anvil.mir.func (physical registers; runs on anvil.interp.mir)

Input MIR
---------
Virtual registers vN, the pseudo-ops `const32` / `sreg` / `label`, reads of R13-R15 (or `sreg`) and reads of
**R12, the args pointer**: the value the ARGS prologue puts in R12 (args_base). R0-R11 and writes to R12 are
rejected: the backend owns allocation. Every virtual register must be written before it is read on every path.

Calling convention and args-block layout (Anvil ABI v1)
-------------------------------------------------------
* Words 0-5 of every program are the xf ARGS prologue

      CONST R12, #hi ; CONST R0, #16 ; MUL R12, R12, R0 ; MUL R12, R12, R0 ; CONST R0, #lo ; ADD R12, R12, R0

  with args_base = hi*256 + lo, so `xf.kernels.program`-style patching of words 0 and 4 rebinds a compiled
  program to another args block (0 <= args_base < 65536).
* `args[0 .. n-1]` (args[i] = MEM[args_base + i]) are the kernel's own arguments, laid out by whoever builds
  the launch (for the xf kernels: xf/kernels.py and xf/elementwise.py).
* **Extension: the spill-area word.** A program that spills to memory reads `args[spill_slot]`, the absolute
  address of a scratch area; `spill_slot` is chosen by the caller (normally n, the word after the kernel's own
  arguments) and passed to `compile_mir`. Thread g = blockIdx * blockDim + threadIdx owns the words
  `scratch + g * F .. scratch + g * F + F - 1`, where F = `stats["frame_words"]`; the caller reserves
  `threads * F` words (threads = the launch's thread count). Frames are disjoint per thread, so spill stores never
  race (xf.isasim `store_races` stays empty) and no word is read by a core other than the one that wrote it
  (AUDIT-10 M2). A program with `frame_words == 0` never reads `args[spill_slot]`; compiling a program that needs
  spills without a spill_slot raises BackendError.
* R12 keeps the args pointer until its last use (including the spill-frame setup); with `reuse_args_reg=True`
  (default) it is then allocatable. R13-R15 are never written.

Passes (in order)
-----------------
1. lift + validation; `sreg` copy propagation (single-definition sreg results become R13-R15 directly);
2. constant canonicalization: single-definition const/const32 registers -> one register per value, built at
   entry from a minimal CONST/MUL/ADD/SUB plan (CSE + hoisting; `consts`);
3. dead code elimination;
4. register allocation onto R0-R11 (+R12) with rematerialization of constants and memory spills (`regalloc`);
5. peephole cleanups (`emit.peephole`);
6. ARGS prologue, label resolution, 256-word size check (`ProgramTooLarge`, naming the overflow), encode via
   `anvil.ir.mir.encode`.

The idiom library (signed compare, select, min, clamp, requant, ...) is in `anvil.backend.idioms` as MIR
builders; `anvil.backend.dealloc` lifts allocated programs (e.g. the xf kernels) back to virtual registers.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..ir import mir
from . import consts, emit
from .emit import PROGRAM_WORDS, PROLOGUE_WORDS, ProgramTooLarge
from .lir import PURE, BackendError, Fresh, Ins, get_func, is_virtual, lift, liveness
from .regalloc import Allocator

__all__ = ["compile_mir", "Program", "BackendError", "ProgramTooLarge", "PROGRAM_WORDS", "PROLOGUE_WORDS"]


@dataclass
class Program:
    words: list[int]
    asm_text: str
    stats: dict = field(default_factory=dict)
    func: mir.FuncOp | None = None


def _validate(code: list[Ins]) -> None:
    for i in code:
        for r in i.reads():
            if not (is_virtual(r) or r in ("R12", "R13", "R14", "R15")):
                raise BackendError(f"{i}: physical register {r} in backend input (use virtual registers; "
                                   "R12 = args pointer and R13-R15 may be read)")
        for r in i.writes():
            if not is_virtual(r):
                raise BackendError(f"{i}: writes physical register {r}; the backend input writes only vN")


def _sregs(code: list[Ins], fresh: Fresh) -> list[Ins]:
    """sreg rd, which: a single-definition rd is replaced by R13/R14/R15 at its uses; otherwise rd = Rk + 0."""
    defs = Counter(r for i in code for r in i.writes())
    sub, out = {}, []
    for i in code:
        if i.op == "sreg":
            src = f"R{mir.SPECIAL[i.sym]}"
            if defs[i.d] == 1:
                sub[i.d] = src
                continue
            z = fresh()
            out += [Ins("const", z, imm=0), Ins("add", i.d, src, z)]
            continue
        out.append(i)
    for i in out:
        if i.a in sub:
            i.a = sub[i.a]
        if i.b in sub:
            i.b = sub[i.b]
    return out


def dce(code: list[Ins]) -> tuple[list[Ins], int]:
    removed = 0
    while True:
        _, lout = liveness(code, is_virtual)
        keep = [i for k, i in enumerate(code) if not (i.op in PURE and is_virtual(i.d) and i.d not in lout[k])]
        if len(keep) == len(code):
            return code, removed
        removed += len(code) - len(keep)
        code = keep


def compile_mir(module, *, args_base: int, spill_slot: int | None = None, reuse_args_reg: bool = True,
                peephole: bool = True, name: str | None = None) -> Program:
    """Compile one anvil.mir.func (or a module holding one) to a program bound to the args block at args_base."""
    f = get_func(module)
    name = name or f.sym_name.data
    code = lift(f)
    _validate(code)
    fresh = Fresh(code)
    args = fresh()
    for i in code:
        if i.a == "R12":
            i.a = args
        if i.b == "R12":
            i.b = args
    code = _sregs(code, fresh)
    code, const_vals = consts.canonicalize(code, fresh)
    code, dead = dce(code)
    const_vals = {r: v for r, v in const_vals.items() if any(i.d == r for i in code)}
    alloc = Allocator(code, args=args, const_vals=const_vals, fresh=fresh, spill_slot=spill_slot,
                      reuse_args_reg=reuse_args_reg)
    code = alloc.run()
    removed = 0
    if peephole:
        code, removed = emit.peephole(code)
    body = emit.check_size(emit.prologue(args_base) + code, name) - PROLOGUE_WORDS
    full = emit.prologue(args_base) + code
    words, func = emit.encode(full, name)
    s = alloc.stats
    stats = {
        "words": len(words), "prologue_words": PROLOGUE_WORDS, "body_words": body,
        "input_instrs": sum(i.op != "label" for i in lift(f)),
        "const_values": len(const_vals), "dead_removed": dead, "peephole_removed": removed,
        "alloc_rounds": s.rounds, "spilled_vregs": len(s.spilled), "remat_vregs": len(s.remat),
        "spill_loads": s.spill_loads, "spill_stores": s.spill_stores, "remat_instrs": s.remat_instrs,
        "frame_words": s.frame_words, "regs_used": s.regs_used,
    }
    header = (f"args_base={args_base} spill_slot={spill_slot} " +
              " ".join(f"{k}={v}" for k, v in stats.items()))
    stats["spilled_names"] = list(s.spilled)
    return Program(words, emit.asm_text(full, name, header), stats, func)
