"""Register allocation: iterated Chaitin-Briggs graph coloring with optimistic spilling, rematerialization of
constants and memory spills to a per-thread scratch frame in global memory.

Registers. R0-R11 are allocatable. R12 holds the args pointer (set by the ARGS prologue): the virtual register
`args` standing for it is precolored R12 and lives until its last use; with `reuse_args_reg` (default) R12 is
allocatable to registers that do not interfere with `args`, so a kernel whose args are read early gets 13
registers afterwards. R13-R15 are the read-only special registers and never allocated.

Liveness is computed over the instruction-level CFG (labels fall through, every BR has both edges, RET ends a
path; SYNC is a plain fall-through for dataflow: under the SIMT stack each lane still follows one CFG path).
Interference is the classic def-vs-live-out rule.

Spill choice minimizes cost / degree, where cost counts every def and use weighted by 10^loop-depth (loops are
spans [label, backward BR]) times the price of the spill code: a memory spill costs MEM_WEIGHT per occurrence
(three instructions and a memory round trip), a constant costs its materialization length per use (no memory
traffic). Registers created by spill code are unspillable, so the iteration converges.

Spill frame (see anvil.backend for the args-block convention): on the first memory spill the allocator emits,
at entry,

    fp = LDR(args + spill_slot) + (blockIdx * blockDim + threadIdx) * FRAME

and slot k of the thread lives at fp + k. FRAME is the final number of slots, patched once allocation converges.
A reload into t is `CONST t,#k; ADD t,fp,t; LDR t,t` (just `LDR t,fp` for slot 0), a store of t is
`CONST a,#k; ADD a,fp,a; STR a,t` (`STR fp,t` for slot 0). Slots are numbered most-used first.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from . import consts
from .lir import BackendError, Fresh, Ins, is_virtual, liveness, loop_depth

GPRS = [f"R{k}" for k in range(12)]
ARGS_REG = "R12"
MEM_WEIGHT = 6
MAX_ROUNDS = 32


@dataclass
class AllocStats:
    rounds: int = 0
    spilled: list = field(default_factory=list)       # memory-spilled virtual registers (in order)
    remat: list = field(default_factory=list)         # rematerialized constant registers
    spill_loads: int = 0
    spill_stores: int = 0
    remat_instrs: int = 0
    frame_words: int = 0
    regs_used: int = 0


class Allocator:
    def __init__(self, code: list[Ins], *, args: str | None, const_vals: dict[str, int], fresh: Fresh,
                 spill_slot: int | None, reuse_args_reg: bool = True):
        self.code, self.args, self.const_vals, self.fresh = code, args, dict(const_vals), fresh
        self.spill_slot, self.reuse = spill_slot, reuse_args_reg
        self.unspillable: set[str] = set()
        self.fp: str | None = None
        self.frame_ins: Ins | None = None
        self.slot: dict[str, int] = {}
        self.remat_ok: set[str] = set()
        self.stats = AllocStats()

    # ------------------------------------------------------------------------------------------ analysis
    def _graph(self):
        lin, lout = liveness(self.code, is_virtual)
        undefined = lin[0] - {self.args} if self.code else set()
        if undefined:
            raise BackendError(f"virtual registers may be read before written: {sorted(undefined)}")
        nodes = {r for i in self.code for r in (i.d, i.a, i.b) if is_virtual(r)}
        adj = defaultdict(set)
        for k, i in enumerate(self.code):
            for d in i.writes():
                if not is_virtual(d):
                    continue
                for live in lout[k]:
                    if live != d:
                        adj[d].add(live)
                        adj[live].add(d)
        return nodes, adj

    def _costs(self, nodes):
        """Spill cost per register; for constants the cheaper of rematerialization and a memory slot (recorded in
        self.remat_ok)."""
        depth = loop_depth(self.code)
        mem = dict.fromkeys(nodes, 0.0)
        remat = dict.fromkeys(self.const_vals, 0.0)
        for k, i in enumerate(self.code):
            w = 10.0 ** min(depth[k], 8)
            for r in i.reads():
                if r in mem:
                    mem[r] += w * MEM_WEIGHT
                if r in remat:
                    remat[r] += w * consts.cost(self.const_vals[r])
            for r in i.writes():
                if r in mem:
                    mem[r] += w * MEM_WEIGHT
        self.remat_ok = {r for r in remat if r in mem and remat[r] <= mem[r]}
        cost = {r: remat[r] if r in self.remat_ok else mem[r] for r in nodes}
        for r in self.unspillable | {self.args}:
            if r in cost:
                cost[r] = math.inf
        return cost

    def _allowed(self, r):
        if r == self.args:
            return [ARGS_REG]
        return GPRS + [ARGS_REG] if self.reuse else GPRS

    # ------------------------------------------------------------------------------------------ coloring
    def _color(self, nodes, adj, cost):
        remaining = set(nodes) - {self.args}
        stack = []
        deg = {n: len(adj[n]) for n in nodes}
        while remaining:
            pick = next((n for n in sorted(remaining, key=lambda n: (deg[n], n)) if deg[n] < len(self._allowed(n))),
                        None)
            if pick is None:       # potential spill: cheapest per interference, pushed optimistically (Briggs)
                pick = min(sorted(remaining), key=lambda n: (cost[n] / max(deg[n], 1), -deg[n]))
            stack.append(pick)
            remaining.discard(pick)
            for m in adj[pick]:
                deg[m] -= 1
        color = {self.args: ARGS_REG} if self.args in nodes else {}
        spilled = []
        while stack:
            n = stack.pop()
            used = {color[m] for m in adj[n] if m in color}
            free = [c for c in self._allowed(n) if c not in used]
            if free:
                color[n] = free[0]
            else:
                spilled.append(n)
        return color, spilled

    # ------------------------------------------------------------------------------------------ spill code
    def _ensure_fp(self):
        if self.fp is not None:
            return
        if self.spill_slot is None:
            raise BackendError("register pressure needs memory spills, but no spill_slot was given: pass "
                               "compile_mir(..., spill_slot=k) with args[k] = the scratch area's address")
        f = self.fresh
        if self.args is None:
            self.args = f()
        pre = []
        if self.spill_slot:
            tree, k, _ = consts.emit_tree(self.spill_slot, f)
            addr = f()
            pre += tree + [Ins("add", addr, self.args, k)]
        else:
            addr = self.args
        base, g1, g2, frame, g3, fp = (f() for _ in range(6))
        self.frame_ins = Ins("const", frame, imm=0, sym="FRAME")
        pre += [Ins("ldr", base, addr), Ins("mul", g1, "R13", "R14"), Ins("add", g2, g1, "R15"), self.frame_ins,
                Ins("mul", g3, g2, frame), Ins("add", fp, base, g3)]
        self.unspillable |= {r for i in pre for r in i.writes()}
        self.fp = fp
        self.code = pre + self.code

    def _slot_addr(self, k, into):
        """Code leaving fp + k in `into` (no code for slot 0: fp itself)."""
        if k == 0:
            return [], self.fp
        return [Ins("const", into, imm=k), Ins("add", into, self.fp, into)], into

    def _rewrite(self, spilled, cost):
        rematted = {r for r in spilled if r in self.remat_ok}
        mem_sp = sorted((r for r in spilled if r not in rematted), key=lambda r: (-cost[r], r))
        if mem_sp:
            self._ensure_fp()
            for r in mem_sp:
                self.slot[r] = len(self.slot)
                self.stats.spilled.append(r)
                self.const_vals.pop(r, None)          # a constant kept in memory: built once, stored, reloaded
        self.stats.remat += sorted(rematted)
        drop = set(spilled)
        out = []
        for i in self.code:
            if i.d in rematted:
                continue                                        # entry definition of a rematerialized constant
            for r in sorted({x for x in i.reads() if x in drop}):
                if r in rematted:
                    tree, t, regs = consts.emit_tree(self.const_vals[r], self.fresh)
                    self.unspillable.update(regs)
                    self.stats.remat_instrs += len(tree)
                    out += tree
                else:
                    t = self.fresh()
                    self.unspillable.add(t)
                    pre, a = self._slot_addr(self.slot[r], t)
                    out += pre + [Ins("ldr", t, a)]
                    self.stats.spill_loads += 1
                if i.a == r:
                    i.a = t
                if i.b == r:
                    i.b = t
            post = []
            if i.d in drop:
                t = self.fresh()
                self.unspillable.add(t)
                r, i.d = i.d, t
                a = self.fresh()
                self.unspillable.add(a)
                pre, a = self._slot_addr(self.slot[r], a)
                post = pre + [Ins("str", None, a, t)]
                self.stats.spill_stores += 1
            out.append(i)
            out += post
        self.code = out

    # ------------------------------------------------------------------------------------------ driver
    def run(self) -> list[Ins]:
        for rnd in range(1, MAX_ROUNDS + 1):
            self.stats.rounds = rnd
            nodes, adj = self._graph()
            cost = self._costs(nodes)
            color, spilled = self._color(nodes, adj, cost)
            if not spilled:
                break
            bad = [r for r in spilled if math.isinf(cost[r])]
            if bad:
                raise BackendError(f"cannot allocate: unspillable registers {bad} do not fit")
            self._rewrite(spilled, cost)
        else:
            raise BackendError(f"register allocation did not converge in {MAX_ROUNDS} rounds")
        if self.frame_ins is not None:
            if len(self.slot) > 255:
                raise BackendError(f"spill frame of {len(self.slot)} words exceeds 255")
            self.frame_ins.imm = len(self.slot)
            self.frame_ins.sym = None
        self.stats.frame_words = len(self.slot)
        for i in self.code:
            i.rename(color)
        self.stats.regs_used = len({r for i in self.code for r in (i.d, i.a, i.b)
                                    if r is not None and r.startswith("R") and int(r[1:]) < 13})
        return self.code
