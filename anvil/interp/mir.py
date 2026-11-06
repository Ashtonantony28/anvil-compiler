"""Interpreter for anvil.mir: per-lane registers (virtual or physical), lockstep warps, global memory and per-block SMEM.

Machine semantics are anvil.semantics' word ops (WORD_BITS wrap, docs/isa.md), so on allocated programs this agrees
with xf.isasim. Differences, all stricter than the hardware:

* a branch on which the warp's active lanes disagree raises DivergenceError (MIR assumes uniform control flow);
* reading a virtual register before any write raises (physical R0..R12 start at 0 as on the GPU);
* running past the last instruction raises (the GPU would run into zero words, i.e. NOPs).

Blocks run one after another in blockIdx order on one memory; within a block, warps of consecutive lanes issue one
instruction at a time round-robin (xf.isasim's "lrr"), BAR holds a warp until every unfinished warp of the block
has reached one, RET retires the warp.
"""
from dataclasses import dataclass

from .. import semantics as sem
from ..ir import mir


class MirInterpError(RuntimeError):
    pass


class DivergenceError(MirInterpError):
    pass


@dataclass
class MirResult:
    memory: list
    steps: int = 0


_ALU = {"ADD": sem.w_add, "SUB": sem.w_sub, "MUL": sem.w_mul, "DIV": sem.w_div, "SRA": sem.w_sra, "MAX": sem.w_max}


def _lower(f: mir.FuncOp):
    """Flatten to (kind, *fields) tuples with branch targets resolved to instruction indices."""
    lab = mir.labels(f)
    prog = []
    for op in f.instrs:
        if isinstance(op, mir.LabelOp):
            continue
        r = {k: getattr(op, k).data for k in op.READS + op.WRITES}
        if isinstance(op, mir.BrOp):
            prog.append(("BR", op.nzp.value.data, lab[op.target.data]))
        elif isinstance(op, mir.SyncOp):
            lab[op.target.data]          # the reconvergence label must exist; SYNC is a no-op without divergence
            prog.append(("NOP",))
        elif op.MNEMONIC in _ALU:
            prog.append(("ALU", _ALU[op.MNEMONIC], r["rd"], r["rs"], r["rt"]))
        elif isinstance(op, mir.ConstOp):
            prog.append(("SET", r["rd"], op.imm.value.data))
        elif isinstance(op, mir.Const32Op):
            prog.append(("SET", r["rd"], op.value.value.data))
        elif isinstance(op, mir.SregOp):
            prog.append(("MOV", r["rd"], f"R{mir.SPECIAL[op.which.data]}"))
        elif isinstance(op, mir.CmpOp):
            prog.append(("CMP", r["rs"], r["rt"]))
        elif isinstance(op, (mir.LdrOp, mir.LdsOp)):
            prog.append((op.MNEMONIC, r["rd"], r["rs"]))
        elif isinstance(op, (mir.StrOp, mir.StsOp)):
            prog.append((op.MNEMONIC, r["rs"], r["rt"]))
        else:
            prog.append((op.MNEMONIC,))     # NOP BAR RET
    return prog


def run(f, data, threads, *, word_bits=sem.WORD_BITS, addr_bits=12, threads_per_block=4, warps_per_core=1,
        smem_size=sem.SMEM_WORDS, max_steps=2_000_000) -> MirResult:
    """Run MIR function `f` (or a module holding one) over `threads` threads on a copy of `data`."""
    if not isinstance(f, mir.FuncOp):
        fs = [op for op in f.ops if isinstance(op, mir.FuncOp)]
        if len(fs) != 1:
            raise ValueError(f"expected one anvil.mir.func, found {len(fs)}")
        f = fs[0]
    if threads_per_block % warps_per_core:
        raise ValueError(f"{warps_per_core} warps do not divide {threads_per_block} threads")
    if len(data) > 1 << addr_bits:
        raise ValueError("data larger than memory")
    prog = _lower(f)
    wb = word_bits
    amask, smask = (1 << addr_bits) - 1, smem_size - 1
    mem = [sem.wrap(int(v), wb) for v in data] + [0] * ((1 << addr_bits) - len(data))
    res = MirResult(mem)
    tpb, lpw = threads_per_block, threads_per_block // warps_per_core

    for b in range(-(-threads // tpb)):
        n = min(tpb, threads - b * tpb)
        regs = []
        for t in range(n):
            r = {f"R{i}": 0 for i in range(13)}
            r.update(R13=sem.wrap(b, wb), R14=sem.wrap(tpb, wb), R15=sem.wrap(t, wb))
            regs.append(r)
        nzp, smem = [0] * n, [0] * smem_size
        warps = [{"pc": 0, "lanes": list(range(w * lpw, min((w + 1) * lpw, n))), "bar": False}
                 for w in range(warps_per_core)]
        warps = [w for w in warps if w["lanes"]]
        cur = 0

        def rd(t, name):
            try:
                return regs[t][name]
            except KeyError:
                raise MirInterpError(f"block {b} lane {t}: {name} read before written") from None

        while warps:
            if all(w["bar"] for w in warps):
                for w in warps:
                    w["bar"] = False
            ready = [i for i, w in enumerate(warps) if not w["bar"]]
            wi = next(i for i in [*range(cur, len(warps)), *range(cur)] if i in ready)
            cur = (wi + 1) % len(warps)
            warp = warps[wi]
            res.steps += 1
            if res.steps > max_steps:
                raise MirInterpError(f"more than {max_steps} steps")
            pc = warp["pc"]
            if pc >= len(prog):
                raise MirInterpError(f"block {b}: pc {pc} ran past the last instruction")
            ins, lanes = prog[pc], warp["lanes"]
            warp["pc"] = pc + 1
            k = ins[0]
            if k == "ALU":
                fn, d, s, tt = ins[1:]
                vals = [fn(rd(t, s), rd(t, tt), wb) for t in lanes]
                for t, v in zip(lanes, vals):
                    regs[t][d] = v
            elif k == "SET":
                for t in lanes:
                    regs[t][ins[1]] = sem.wrap(ins[2], wb)
            elif k == "MOV":
                for t in lanes:
                    regs[t][ins[1]] = regs[t][ins[2]]
            elif k == "LDR":
                vals = [mem[rd(t, ins[2]) & amask] for t in lanes]
                for t, v in zip(lanes, vals):
                    regs[t][ins[1]] = v
            elif k == "LDS":
                vals = [smem[rd(t, ins[2]) & 0xFF & smask] for t in lanes]
                for t, v in zip(lanes, vals):
                    regs[t][ins[1]] = v
            elif k == "STR":       # lanes in order: the highest lane wins a shared word (ADR-016)
                for a, v in [(rd(t, ins[1]) & amask, rd(t, ins[2])) for t in lanes]:
                    mem[a] = v
            elif k == "STS":
                for a, v in [(rd(t, ins[1]) & 0xFF & smask, rd(t, ins[2])) for t in lanes]:
                    smem[a] = v
            elif k == "CMP":
                for t in lanes:
                    nzp[t] = sem.w_cmp_nzp(rd(t, ins[1]), rd(t, ins[2]), wb)
            elif k == "BR":
                taken = {bool(nzp[t] & ins[1]) for t in lanes}
                if len(taken) > 1:
                    raise DivergenceError(f"block {b}: divergent branch at instruction {pc}")
                if taken.pop():
                    warp["pc"] = ins[2]
            elif k == "BAR":
                warp["bar"] = True
            elif k == "RET":
                warps.pop(wi)
                cur = wi % len(warps) if warps else 0
            # NOP / SYNC: nothing
    return res
