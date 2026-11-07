"""Fork of the instruction-level reference model sim/xf/isasim.py (pinned at 096c6ef) for the extended ISA of
anvil.isa_ext.isa (5-bit opcodes, model only). sim/ is never edited: this is a wrapped copy of `xf.isasim.run` with

  * the opcode taken from bits [16:12] (base words are unchanged, so base programs behave exactly as on xf.isasim);
  * the candidate ops MIN, AND, SHL, SLT, MAD executed like the base ALU ops (all lanes read before any writes;
    MAD also reads rd), carried through the race taint like ALU ops;
  * optional recording of each warp's instruction stream (pc, op, global addresses per lane), the input of the
    tier-1 timing model (anvil.cost.analytic), so one run gives both the functional result and the profile.

SIMT_STACK and multi-warp semantics are xf.isasim's, line for line. The race taint and flags are xf.isasim's.
"""
from __future__ import annotations

from xf.isasim import IsaSimError, Result, _Warp

from . import isa as X

PC_MASK = 0xFF
_ALU_BASE = (0b0011, 0b0100, 0b0101, 0b0110, 0b1101, 0b1110)
_EXT_OPS = {X.EXT[m][0]: m for m in X.EXT}
_MAD = X.EXT["MAD"][0]
_WRITES_RD = _ALU_BASE + (0b0111, 0b1001, 0b1010) + tuple(_EXT_OPS)


def _taint(op, rd, rs, rt, lanes, regs, reg_t, smem_t, mem_t, amask, smask):
    for t in lanes:
        ts, tt = reg_t[t] >> rs & 1, reg_t[t] >> rt & 1
        if op == 0b1000:
            (mem_t.add if ts or tt else mem_t.discard)(regs[t][rs] & amask)
        elif op == 0b1011:
            (smem_t.add if ts or tt else smem_t.discard)(regs[t][rs] & smask)
        elif rd < 13 and op in _WRITES_RD:
            if op == 0b0111:
                tt = regs[t][rs] & amask in mem_t
            elif op == 0b1010:
                tt = regs[t][rs] & smask in smem_t
            elif op == 0b1001:
                ts = tt = 0
            elif op == _MAD:
                ts = ts or reg_t[t] >> rd & 1
            reg_t[t] = reg_t[t] & ~(1 << rd) | bool(ts or tt) << rd


def run(program, data, threads, *, word_bits=8, addr_bits=8, threads_per_block=4, warps_per_core=1,
        smem_size=64, simt=False, stack_depth=8, max_steps=2_000_000, warp_policy="lrr", record=False):
    """xf.isasim.run for extended programs. With record=True the Result has `streams`: per block, per warp, the list
    of (pc, op, addrs) with addrs = ((lane, address), ...) for LDR/STR, else None (anvil.cost.profile's format)."""
    wmask, amask, sign = (1 << word_bits) - 1, (1 << addr_bits) - 1, 1 << (word_bits - 1)
    if smem_size < 1 or smem_size & (smem_size - 1):
        raise ValueError(f"smem_size {smem_size} is not a power of two")
    if threads_per_block % warps_per_core:
        raise ValueError(f"{warps_per_core} warps do not divide {threads_per_block} threads")
    if len(data) > 1 << addr_bits or len(program) > 1 << 8:
        raise ValueError("data or program larger than its memory")
    if not callable(warp_policy) and warp_policy not in ("lrr", "gto"):
        raise ValueError(f"unknown warp policy {warp_policy!r}")

    def signed(x):
        return x - ((x & sign) << 1)

    alu = {
        0b0011: lambda a, b: (a + b) & wmask,
        0b0100: lambda a, b: (a - b) & wmask,
        0b0101: lambda a, b: (a * b) & wmask,
        0b0110: lambda a, b: a // b if b else 0,
        0b1101: lambda a, b: (signed(a) >> b) & wmask,
        0b1110: lambda a, b: a if signed(a) > signed(b) else b,
    }
    sem = X.semantics(word_bits)
    ext = {op: sem[m] for op, m in _EXT_OPS.items()}
    prog = [int(w) for w in program] + [0] * ((1 << 8) - len(program))
    for w in prog:
        if not 0 <= w < 1 << X.WORD_BITS_EXT or (w >> 12) not in X.BY_OPCODE:
            raise ValueError(f"word {w:#x} is not an extended-ISA instruction")
    res = Result([int(v) & wmask for v in data] + [0] * ((1 << addr_bits) - len(data)))
    res.streams = [] if record else None
    mem, tpb, lanes_per_warp = res.memory, threads_per_block, threads_per_block // warps_per_core

    for b in range(-(-threads // tpb)):
        n = min(tpb, threads - b * tpb)
        regs = [[0] * 13 + [b & wmask, tpb & wmask, t & wmask] for t in range(tpb)]
        nzp, smem = [0] * tpb, [0] * smem_size
        reg_t, smem_t, stored = [0] * tpb, set(), {}
        warps = [_Warp(sum(1 << t for t in range(w * lanes_per_warp, min((w + 1) * lanes_per_warp, n))))
                 for w in range(warps_per_core)]
        streams = [[] for _ in range(warps_per_core)]
        if record:
            res.streams.append((b, n, streams))
        cur = 0
        while True:
            live = [w for w in warps if not w.done]
            if not live:
                break
            if all(w.bar for w in live):
                stored.clear()
                for w in live:
                    w.bar = False
            if callable(warp_policy):
                cur = warp_policy(b, [i for i, w in enumerate(warps) if not w.done and not w.bar])
            else:
                order = range(cur, cur + warps_per_core) if warp_policy == "lrr" else [cur, *range(warps_per_core)]
                cur = next(i % warps_per_core for i in order if not warps[i % warps_per_core].done
                           and not warps[i % warps_per_core].bar)
            wid = cur
            warp = warps[wid]
            if warp_policy == "lrr":
                cur = (cur + 1) % warps_per_core
            res.steps += 1
            if res.steps > max_steps:
                raise IsaSimError(f"more than {max_steps} steps")

            stack = warp.stack
            top = stack[-1]
            pc, mask = top[0], top[2]
            lanes = [t for t in range(tpb) if mask >> t & 1]
            word = prog[pc]
            op, rd, rs, rt, imm = word >> 12, (word >> 8) & 15, (word >> 4) & 15, word & 15, word & 0xFF
            top[0] = (pc + 1) & PC_MASK
            if res.store_races:
                _taint(op, rd, rs, rt, lanes, regs, reg_t, smem_t, res.race_words, amask, 0xFF & (smem_size - 1))
            addrs = None
            if op in alu:
                f = alu[op]
                vals = [f(regs[t][rs], regs[t][rt]) for t in lanes]
                if rd < 13:
                    for t, v in zip(lanes, vals):
                        regs[t][rd] = v
            elif op in ext:
                f = ext[op]
                vals = [f(regs[t][rd], regs[t][rs], regs[t][rt]) for t in lanes]
                if rd < 13:
                    for t, v in zip(lanes, vals):
                        regs[t][rd] = v
            elif op in (0b0111, 0b1001, 0b1010):   # LDR, CONST, LDS
                if op == 0b0111:
                    addrs = tuple((t, regs[t][rs] & amask) for t in lanes)
                    vals = [mem[a] for _, a in addrs]
                elif op == 0b1001:
                    vals = [imm] * len(lanes)
                else:
                    vals = [smem[regs[t][rs] & 0xFF & (smem_size - 1)] for t in lanes]
                if rd < 13:
                    for t, v in zip(lanes, vals):
                        regs[t][rd] = v
            elif op == 0b1000:                     # STR
                addrs = tuple((t, regs[t][rs] & amask) for t in lanes)
                seen, clash = {}, set()
                for a, v in [(regs[t][rs] & amask, regs[t][rt]) for t in lanes]:
                    if seen.get(a, v) != v and a not in clash:
                        clash.add(a)
                        res.stores_conflict.append((b, pc, a))
                    seen[a] = mem[a] = v
                    ws, vs = stored.setdefault(a, (set(), set()))
                    raced = len(ws) > 1 and len(vs) > 1
                    ws.add(wid)
                    vs.add(v)
                    if len(ws) > 1 and len(vs) > 1:
                        if not raced:
                            res.store_races.append((b, pc, a))
                        res.race_words.add(a)
            elif op == 0b1011:                     # STS
                for a, v in [(regs[t][rs] & 0xFF & (smem_size - 1), regs[t][rt]) for t in lanes]:
                    smem[a] = v
            elif op == 0b0010:                     # CMP
                for t in lanes:
                    nzp[t] = 0b100 if regs[t][rs] != regs[t][rt] else 0b010
            elif op == 0b1100:                     # BAR
                warp.bar = True
            elif op == 0b1111:                     # RET
                if record:
                    streams[wid].append((pc, op, None))
                for e in stack:
                    e[2] &= ~mask
                stack[:] = [e for e in stack if e[2]]
                if not stack:
                    warp.done, warp.bar = True, False
                    continue
            elif op == 0b0001:
                cond = (word >> 9) & 7
                if cond == 0:                      # SYNC
                    if simt:
                        warp.sync = imm
                else:
                    taken = sum(1 << t for t in lanes if nzp[t] & cond)
                    if not simt:
                        if taken and taken != mask:
                            res.divergent.append((b, pc))
                        if taken & (mask & -mask):
                            top[0] = imm
                    elif taken == mask:
                        top[0] = imm
                    elif taken:
                        if warp.sync is None:
                            raise IsaSimError(f"block {b}: divergent branch at pc {pc} without SYNC")
                        r, warp.sync = warp.sync, None
                        top[0] = r
                        stack += [[(pc + 1) & PC_MASK, r, mask & ~taken], [imm, r, taken]]
                        if len(stack) - 1 > stack_depth:
                            raise IsaSimError(f"block {b}: SIMT stack deeper than {stack_depth} at pc {pc}")
            if record and op != 0b1111:
                streams[wid].append((pc, op, addrs))
            while len(stack) > 1 and stack[-1][0] == stack[-1][1]:
                stack.pop()
    return res
