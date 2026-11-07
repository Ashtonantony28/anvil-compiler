"""Dynamic instruction profile of a Job: per launch, per block, per warp, the instruction stream the GPU executes
(pc, opcode, global addresses per active lane). Input of the tier-1 timing model (`anvil.cost.analytic`) and of the
tier-2 features (`anvil.cost.features`).

    prof = profile(job)                     # Profile: launches -> blocks -> warps -> [Instr]
    prof.memory                             # final data memory (equals xf.isasim's for race-free programs)

The executor is an independent functional model of the 16-opcode ISA (docs/isa.md) at SIMT_STACK=0: a warp follows
its lowest enabled lane on branches, R13/R14/R15 hold %blockIdx/%blockDim/%threadIdx. Warps of a block run in
round-robin order between BARs; for programs without races (every candidate the tuner measures is checked for
races on isasim) the values, hence the addresses and trip counts, do not depend on the issue order, so one
functional run serves every warp policy. Tests compare its memory with xf.isasim on every kernel it profiles.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# opcodes
NOP, BR, CMP, ADD, SUB, MUL, DIV, LDR, STR, CONST, LDS, STS, BAR, SRA, MAX, RET = range(16)
ALU_OPS = (ADD, SUB, MUL, DIV, SRA, MAX)
GLOBAL_OPS = (LDR, STR)
SMEM_OPS = (LDS, STS)
OP_NAMES = ("NOP", "BR", "CMP", "ADD", "SUB", "MUL", "DIV", "LDR", "STR", "CONST", "LDS", "STS", "BAR", "SRA",
            "MAX", "RET")


class ProfileError(RuntimeError):
    pass


@dataclass
class Block:
    block_id: int
    threads: int                    # live threads of this block (short for the last block)
    warps: list                     # per warp: list of (pc, op, addrs) where addrs = ((lane, addr), ...) for LDR/STR


@dataclass
class LaunchProfile:
    program: list
    threads: int
    blocks: list = field(default_factory=list)

    @property
    def issues(self) -> int:
        return sum(len(w) for b in self.blocks for w in b.warps)


@dataclass
class Profile:
    launches: list
    memory: list
    tpb: int
    warps_per_core: int


def _run_launch(program, mem, threads, *, word_bits, addr_bits, tpb, W, smem_size=64, max_steps=2_000_000):
    wmask, amask, sign = (1 << word_bits) - 1, (1 << addr_bits) - 1, 1 << (word_bits - 1)
    smask = 0xFF & (smem_size - 1)
    prog = [int(w) for w in program] + [0] * (256 - len(program))
    dec = [(w >> 12, (w >> 8) & 15, (w >> 4) & 15, w & 15, w & 0xFF, (w >> 9) & 7) for w in prog]
    lpw = tpb // W
    out = LaunchProfile(list(program), threads)
    steps = 0

    def signed(x):
        return x - ((x & sign) << 1)

    for b in range(-(-threads // tpb)):
        n = min(tpb, threads - b * tpb)
        regs = [[0] * 13 + [b & wmask, tpb & wmask, t & wmask] for t in range(tpb)]
        nzp = [0] * tpb
        smem = [0] * smem_size
        streams = [[] for _ in range(W)]
        lanes_of = [[t for t in range(w * lpw, min((w + 1) * lpw, n))] for w in range(W)]
        pcs = [0] * W
        done = [not lanes_of[w] for w in range(W)]
        at_bar = [False] * W
        while not all(done):
            progressed = False
            for w in range(W):
                if done[w] or at_bar[w]:
                    continue
                lanes = lanes_of[w]
                rec = streams[w]
                pc = pcs[w]
                while True:              # run warp w until BAR or RET
                    steps += 1
                    if steps > max_steps:
                        raise ProfileError(f"more than {max_steps} warp instructions")
                    op, rd, rs, rt, imm, cond = dec[pc]
                    npc = (pc + 1) & 0xFF
                    addrs = None
                    if op == ADD:
                        if rd < 13:
                            for t in lanes:
                                r = regs[t]
                                r[rd] = (r[rs] + r[rt]) & wmask
                    elif op == SUB:
                        if rd < 13:
                            for t in lanes:
                                r = regs[t]
                                r[rd] = (r[rs] - r[rt]) & wmask
                    elif op == MUL:
                        if rd < 13:
                            for t in lanes:
                                r = regs[t]
                                r[rd] = (r[rs] * r[rt]) & wmask
                    elif op == DIV:
                        if rd < 13:
                            for t in lanes:
                                r = regs[t]
                                r[rd] = r[rs] // r[rt] if r[rt] else 0
                    elif op == SRA:
                        if rd < 13:
                            for t in lanes:
                                r = regs[t]
                                r[rd] = (signed(r[rs]) >> r[rt]) & wmask
                    elif op == MAX:
                        if rd < 13:
                            for t in lanes:
                                r = regs[t]
                                r[rd] = r[rs] if signed(r[rs]) > signed(r[rt]) else r[rt]
                    elif op == CONST:
                        if rd < 13:
                            for t in lanes:
                                regs[t][rd] = imm
                    elif op == LDR:
                        addrs = tuple((t, regs[t][rs] & amask) for t in lanes)
                        if rd < 13:
                            for t, a in addrs:
                                regs[t][rd] = mem[a]
                    elif op == STR:
                        addrs = tuple((t, regs[t][rs] & amask) for t in lanes)
                        for t, a in addrs:
                            mem[a] = regs[t][rt]
                    elif op == LDS:
                        if rd < 13:
                            vals = [smem[regs[t][rs] & smask] for t in lanes]
                            for t, v in zip(lanes, vals):
                                regs[t][rd] = v
                    elif op == STS:
                        for t in lanes:
                            smem[regs[t][rs] & smask] = regs[t][rt]
                    elif op == CMP:
                        for t in lanes:
                            nzp[t] = 0b100 if regs[t][rs] != regs[t][rt] else 0b010
                    elif op == BR:
                        if cond and nzp[lanes[0]] & cond:
                            npc = imm
                    rec.append((pc, op, addrs))
                    pc = npc
                    if op == RET:
                        done[w] = True
                        break
                    if op == BAR and W > 1:
                        at_bar[w] = True
                        break
                pcs[w] = pc
                progressed = True
            live = [w for w in range(W) if not done[w]]
            if live and all(at_bar[w] for w in live):
                for w in live:
                    at_bar[w] = False
            elif not progressed and live:
                raise ProfileError("deadlock at BAR")
        out.blocks.append(Block(b, n, streams))
    return out


def profile(job, *, max_steps: int = 2_000_000) -> Profile:
    """Profile every launch of `job` (anvil.runtime.Job) from its initial memory."""
    p = job.params
    if p.SIMT_STACK:
        raise ProfileError("SIMT_STACK=1 is not modelled (the tier-1 model assumes lane-0 branch semantics)")
    mask = (1 << p.WORD_BITS) - 1
    mem = [int(w) & mask for w in job.data] + [0] * ((1 << p.DATA_MEM_ADDR_BITS) - len(job.data))
    launches = []
    for ln in job.launches:
        launches.append(_run_launch(ln.program, mem, ln.threads, word_bits=p.WORD_BITS,
                                    addr_bits=p.DATA_MEM_ADDR_BITS, tpb=p.THREADS_PER_BLOCK, W=p.WARPS_PER_CORE,
                                    max_steps=max_steps))
    return Profile(launches, mem[:len(job.data)], p.THREADS_PER_BLOCK, p.WARPS_PER_CORE)
