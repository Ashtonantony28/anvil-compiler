"""Tier 1: an analytic timing model of tiny-gpu-extended, built from the RTL's FSM state durations (no training).

    t = predict(job)                              # Timing: cycles, launch_cycles, breakdown
    t = predict_profile(prof, params, mem_latency)

Every delay is a closed-form state duration read off the RTL (sim/src), evaluated over the dynamic instruction
stream of each block (`anvil.cost.profile`). Edges are numbered from the launch's `start`; S is the edge at which a
core's scheduler enters FETCH, R the edge at which it enters REQUEST, X the edge at which it enters EXECUTE.

  scheduler (scheduler.sv)   FETCH -> DECODE (skipped when PIPELINED) -> REQUEST -> WAIT -> EXECUTE -> UPDATE, one
                             instruction in flight per core; WARPS_PER_CORE > 1 adds the IDLE issue slot after
                             UPDATE and before each FETCH (warp_scheduler.sv: LRR, GTO, GTLRR).
  fetch (fetcher.sv)         I-cache hit: FETCH lasts 2 cycles. Miss / no I-cache: the fetcher raises valid at S+1,
                             the program controller (one channel shared by every core, lowest core first) grabs it
                             at G >= S+2; the fetcher has the word at G+3+L and the scheduler leaves FETCH at G+4+L
                             (uncontended: 6+L cycles). The channel can grab again at G+5+L (READ_WAITING,
                             READ_RELAYING until the consumer drops valid, IDLE).
  WAIT (scheduler, lsu.sv)   ALU/CONST/CMP/BR/BAR/RET: 1 cycle. LDS/STS: 3 cycles (REQUESTING, WAITING, DONE).
                             LDR/STR: each active lane's LSU raises valid at R+2; without L1-D the data controller
                             (4 channels; per edge channel 0 first, each grabs the lowest pending lane of any core;
                             COALESCE merges pending reads of the same address into the grab) grabs at G >= R+3, the
                             LSU is DONE at G+3+L and the scheduler enters EXECUTE one edge after the last lane
                             (uncontended: 6+L cycles). With L1-D (l1d_cache.sv, shared, direct-mapped, one word per
                             line): lookup at R+3; a read hit is DONE at R+4; a miss or a (write-through) store goes
                             downstream at R+4 and is DONE at G+4+L, installing the line at G+3+L.
  warps (scheduler.sv)       at WARPS_PER_CORE > 1 a global LDR/STR yields at R+2 (the warp is pending) and is issued
                             again, FETCH included, once its LSUs are DONE; SMEM ops wait in WAIT. BAR holds a warp
                             until every live warp is at a BAR.
  dispatch (dispatch.sv)     blocks go to cores in index order; the first FETCH is at edge 2; a core that finishes a
                             block at E (done) starts its next block's FETCH at E+3; the launch takes E_last+3 cycles.
  memory (harness)           L = mem_latency waiting edges per transaction, program memory included.

A launch is evaluated as discrete events at edge resolution (shared program/data channels and the L1-D are the
only coupling between cores). The model is exact wherever these rules are the whole story (tests/cost compare it
with Verilator on straight-line programs and single blocks); known simplifications: MEDIC's priority pass is
modelled per lane window but not validated, SIMT_STACK=1 is not modelled.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from .profile import BAR, GLOBAL_OPS, LDR, RET, SMEM_OPS, STR, profile

NUM_CHANNELS = 4          # gpu.sv DATA_MEM_NUM_CHANNELS
FETCH_HIT = 2             # FETCH cycles on an I-cache hit
FETCH_GRAB = 2            # S -> earliest program-channel grab
FETCH_DONE = 4            # grab -> scheduler leaves FETCH (+L)
CHAN_BUSY = 5             # grab -> earliest next grab on the channel (+L)
LSU_GRAB = 3              # R -> earliest data-channel grab / L1-D lookup
LSU_DONE = 3              # grab -> LSU DONE (+L), no L1-D
L1D_HIT_DONE = 1          # lookup -> LSU DONE on a hit
L1D_MISS_DONE = 4         # downstream grab -> LSU DONE (+L)
L1D_FILL = 3              # downstream grab -> line installed (+L)
SMEM_WAIT = 3
FIRST_FETCH = 2           # launch start -> first FETCH edge
BLOCK_GAP = 3             # block done (E) -> next block's FETCH on that core
LAUNCH_TAIL = 3           # last block done -> launch cycles


@dataclass
class Timing:
    cycles: int
    launch_cycles: list
    breakdown: dict = field(default_factory=dict)     # summed over launches


class _Chan:
    """Program or data controller: `n` channels, pending requests, grabs at edge resolution."""

    def __init__(self, sim, n, busy):
        self.sim, self.free, self.busy = sim, [0] * n, busy
        self.pending = []            # [consumer, ready, is_read, addr, cb, prio]
        self.scheduled = set()

    def request(self, consumer, ready, cb, is_read=True, addr=None):
        self.pending.append([consumer, ready, is_read, addr, cb])
        self._schedule(max(ready, min(self.free)))

    def _schedule(self, t):
        if t not in self.scheduled:
            self.scheduled.add(t)
            self.sim.at(t, 1, self.arbitrate, t)

    def arbitrate(self, t):
        self.scheduled.discard(t)
        sim = self.sim
        for ch in range(len(self.free)):
            if self.free[ch] > t:
                continue
            ready = [p for p in self.pending if p[1] <= t]
            if not ready:
                break
            if sim.medic and self.coalesce is not None:
                hi = [p for p in ready if p[2] and sim.high_util(p[0])]
                pick = min(hi or ready, key=lambda p: p[0])
            else:
                pick = min(ready, key=lambda p: p[0])
            group = [pick]
            if self.coalesce and pick[2]:
                group += [p for p in ready if p is not pick and p[2] and p[3] == pick[3]]
            for p in group:
                self.pending.remove(p)
            self.free[ch] = t + self.busy
            sim.stats["grabs"] += 1
            sim.stats["grab_wait"] += sum(t - p[1] for p in group)
            for p in group:
                p[4](t)
        if self.pending:
            nxt = max(t + 1, min(min(p[1] for p in self.pending), min(self.free)))
            self._schedule(nxt)


class _Core:
    def __init__(self, sim, idx):
        self.sim, self.idx = sim, idx
        self.icache = {}
        self.block = None

    # ------------------------------------------------------------------ block lifecycle
    def start(self, block, S):
        sim = self.sim
        self.block = block
        W = sim.W
        self.streams = block.warps
        self.pos = [0] * W
        self.done = [not s for s in self.streams]
        self.bar = [False] * W
        self.pending = [False] * W
        self.lane_done = [None] * W          # per warp: {lane: done edge or None}
        self.age = [0] * W
        self.cur = 0
        self.lrr = 0
        self.gto = 0
        self.sleeping = False
        sim.at(S, 0, self.issue if W > 1 else self.fetch, S)

    def finish(self, E):
        self.block = None
        self.sim.block_done(self, E)

    # ------------------------------------------------------------------ issue (WARPS_PER_CORE > 1)
    def _busy(self, w, i):
        ld = self.lane_done[w]
        if not ld:
            return False
        return any(d is None or d >= i for d in ld.values())

    def issue(self, i):
        """IDLE issue attempt at edge i (state is IDLE before the edge)."""
        sim, W = self.sim, self.sim.W
        self.sleeping = False
        live = [w for w in range(W) if not self.done[w]]
        if not live:
            self.finish(i)
            return
        release = all(self.bar[w] for w in live)
        if release:
            for w in live:
                self.bar[w] = False
        ready = [w for w in range(W) if not self.done[w] and not self.bar[w]
                 and not (self.pending[w] and self._busy(w, i))]
        if not ready:
            sim.stats["idle_stall"] += 1
            nxt = []
            for w in live:
                if self.pending[w]:
                    ld = self.lane_done[w].values()
                    if all(d is not None for d in ld):
                        nxt.append(max(ld) + 1)
            if nxt:
                sim.at(max(i + 1, min(nxt)), 0, self.issue, max(i + 1, min(nxt)))
            else:
                self.sleeping = True
                self.sleep_since = i
            return
        rs = set(ready)
        pol = sim.policy
        sel = None
        if pol == 1:                                   # GTO
            if self.gto in rs:
                sel = self.gto
            else:
                sel = min(ready, key=lambda w: (self.age[w], w))
                if self.age[sel] >= 0xFF:
                    sel = None
        elif pol == 2 and self.cur in rs:              # GTLRR: the progressing (last issued) warp first
            sel = self.cur
        if sel is None:                                # LRR (and the GTLRR fallback)
            for k in range(W):
                w = (self.lrr + k) % W
                if w in rs:
                    sel = w
                    break
        self.lrr = (sel + 1) % W
        self.gto = sel
        self.cur = sel
        if not self.pending[sel]:
            if self.age[sel] != 0xFE:
                self.age[sel] += 1
            sim.stats["issues"] += 1
        sim.stats["idle_cycles"] += 1
        self.fetch(i, sel)

    def poke(self):
        """A lane's grant became known; wake an IDLE core that had nothing to wait for."""
        if self.sleeping:
            self.sleeping = False
            nxt = []
            for w in range(self.sim.W):
                if self.pending[w] and not self.done[w]:
                    ld = self.lane_done[w].values()
                    if all(d is not None for d in ld):
                        nxt.append(max(ld) + 1)
            if nxt:
                t = max(self.sleep_since + 1, min(nxt))
                self.sim.at(t, 0, self.issue, t)
            else:
                self.sleeping = True

    # ------------------------------------------------------------------ one instruction
    def fetch(self, S, w=0):
        sim = self.sim
        if sim.W == 1:
            sim.stats["issues"] += 1
        pc, op, addrs = self.streams[w][self.pos[w]]
        lines = sim.p.ICACHE_LINES
        if lines:
            idx, tag = pc % lines, pc // lines
            if self.icache.get(idx) == tag:
                sim.stats["icache_hits"] += 1
                sim.stats["fetch_cycles"] += FETCH_HIT
                sim.at(S + FETCH_HIT, 0, self.decode, S + FETCH_HIT, w)
                return
        sim.stats["icache_misses"] += 1

        def granted(G):
            if lines:
                sim.at(G + 3 + sim.L, 2, self.icache.__setitem__, pc % lines, pc // lines)
            F = G + FETCH_DONE + sim.L
            sim.stats["fetch_cycles"] += F - S
            sim.at(F, 0, self.decode, F, w)
        sim.prog.request(self.idx, S + FETCH_GRAB, granted)

    def decode(self, F, w):
        R = F if self.sim.p.PIPELINED else F + 1
        self.request(R, w)

    def request(self, R, w):
        sim = self.sim
        pc, op, addrs = self.streams[w][self.pos[w]]
        if op in GLOBAL_OPS:
            if self.pending[w]:                          # re-issue after a yield: the LSUs are DONE
                self.pending[w] = False
                self.lane_done[w] = None
                sim.at(R + 2, 0, self.execute, R + 2, w)
                return
            self.lane_done[w] = {lane: None for lane, _ in addrs}
            base = self.idx * sim.p.THREADS_PER_BLOCK
            for lane, a in addrs:
                sim.lsu(self, w, base + lane, lane, op == LDR, a, R + LSU_GRAB)
            if sim.W > 1:
                self.pending[w] = True
                sim.stats["yields"] += 1
                sim.at(R + 3, 0, self.issue, R + 3)     # IDLE at R+2, issue attempts from R+3
            else:
                self.wait_start = R
            return
        if op in SMEM_OPS:
            sim.stats["smem_wait"] += SMEM_WAIT
            sim.at(R + 1 + SMEM_WAIT, 0, self.execute, R + 1 + SMEM_WAIT, w)
        else:
            sim.at(R + 2, 0, self.execute, R + 2, w)

    def lane_finished(self, w, lane, D):
        self.lane_done[w][lane] = D
        if self.sim.W == 1:
            ld = self.lane_done[w]
            if all(d is not None for d in ld.values()):
                X = max(ld.values()) + 1
                self.sim.stats["mem_wait"] += X - self.wait_start - 1
                self.lane_done[w] = None
                self.sim.at(X, 0, self.execute, X, w)
        else:
            self.poke()

    def execute(self, X, w):
        sim = self.sim
        pc, op, addrs = self.streams[w][self.pos[w]]
        self.pos[w] += 1
        nxt = X + 2                                     # EXECUTE, UPDATE
        if sim.W == 1:
            if op == RET:
                self.finish(nxt)
            else:
                sim.at(nxt, 0, self.fetch, nxt)
            return
        if op == RET:
            self.done[w] = True
        elif op == BAR:
            self.bar[w] = True
        sim.at(nxt + 1, 0, self.issue, nxt + 1)


class _Sim:
    def __init__(self, params, mem_latency):
        self.p, self.L = params, int(mem_latency)
        self.W = params.WARPS_PER_CORE
        self.policy = params.WARP_POLICY
        self.medic = bool(params.MEDIC and params.DCACHE_LINES)

    def at(self, t, cls, fn, *args):
        self.seq += 1
        heapq.heappush(self.q, (t, cls, self.seq, fn, args))

    def high_util(self, consumer):
        return sum(self.window.get(consumer, (1, 1, 1, 1))) >= 2

    # ------------------------------------------------------------------ memory side
    def lsu(self, core, w, consumer, lane, is_read, addr, Q):
        p = self.p
        L = self.L
        self.stats["mem_lane_requests"] += 1
        if not p.DCACHE_LINES:
            def granted(G):
                core.lane_finished(w, lane, G + LSU_DONE + L)
            self.data.request(consumer, Q, granted, is_read, addr)
            return

        def lookup(t):
            lines = p.DCACHE_LINES
            idx, tag = addr % lines, addr // lines
            if is_read and self.dcache.get(idx) == tag:
                self.stats["dcache_hits"] += 1
                if self.medic:
                    self._win(consumer, 1)
                core.lane_finished(w, lane, t + L1D_HIT_DONE)
                return
            if is_read:
                self.stats["dcache_misses"] += 1
                if self.medic:
                    self._win(consumer, 0)
            install = is_read and not (self.medic and not self.high_util(consumer))

            def granted(G):
                if install:
                    self.at(G + L1D_FILL + L, 2, self.dcache.__setitem__, idx, tag)
                core.lane_finished(w, lane, G + L1D_MISS_DONE + L)
            self.data.request(consumer, t + 1, granted, is_read, addr)
        self.at(Q, 0, lookup, Q)

    def _win(self, consumer, hit):
        win = self.window.get(consumer, (1, 1, 1, 1))
        self.window[consumer] = win[1:] + (hit,)

    # ------------------------------------------------------------------ dispatch
    def block_done(self, core, E):
        self.end = max(self.end, E)
        if self.next_block < len(self.blocks):
            b = self.blocks[self.next_block]
            self.next_block += 1
            core.start(b, E + BLOCK_GAP)

    def launch(self, lp) -> int:
        self.q, self.seq = [], 0
        self.prog = _Chan(self, 1, CHAN_BUSY + self.L)
        self.prog.coalesce = None
        self.data = _Chan(self, NUM_CHANNELS, CHAN_BUSY + self.L)
        self.data.coalesce = bool(self.p.COALESCE)
        self.dcache, self.window = {}, {}
        self.blocks, self.next_block, self.end = lp.blocks, 0, 0
        cores = [_Core(self, c) for c in range(self.p.NUM_CORES)]
        for c in cores:
            if self.next_block < len(self.blocks):
                c.start(self.blocks[self.next_block], FIRST_FETCH)
                self.next_block += 1
        while self.q:
            t, cls, _, fn, args = heapq.heappop(self.q)
            fn(*args)
        return self.end + LAUNCH_TAIL


def _new_stats():
    keys = ("issues", "icache_hits", "icache_misses", "fetch_cycles", "mem_wait", "smem_wait", "idle_cycles",
            "idle_stall", "yields", "grabs", "grab_wait", "mem_lane_requests", "dcache_hits", "dcache_misses")
    return dict.fromkeys(keys, 0)


def predict_profile(prof, params, mem_latency: int = 0) -> Timing:
    sim = _Sim(params, mem_latency)
    sim.stats = _new_stats()
    lc = [sim.launch(lp) for lp in prof.launches]
    return Timing(sum(lc), lc, dict(sim.stats))


def predict(job, prof=None) -> Timing:
    """Tier-1 cycles of an anvil.runtime.Job (all launches)."""
    prof = prof if prof is not None else profile(job)
    return predict_profile(prof, job.params, job.mem_latency)


# --------------------------------------------------------------------------------------- closed form (tier 1a)

def closed_form(prof, params, mem_latency: int = 0) -> float:
    """A contention-free closed form of the same state durations: per block, the sum of its instructions'
    uncontended durations (I-cache and L1-D hits from a per-core / per-launch direct-mapped replay in program
    order; at WARPS_PER_CORE > 1 the IDLE slot and the re-issue of yielding loads, with no overlap credited);
    blocks are list-scheduled over the cores (FIRST_FETCH, BLOCK_GAP, LAUNCH_TAIL); the result is the larger of
    that makespan and the program/data channel occupancy (one program channel, NUM_CHANNELS data channels,
    CHAN_BUSY+L per transaction). Cheap and interpretable, but blind to arbitration and warp overlap."""
    p, L = params, int(mem_latency)
    W = p.WARPS_PER_CORE
    dec = 0 if p.PIPELINED else 1
    total = 0.0
    for lp in prof.launches:
        icaches = [dict() for _ in range(p.NUM_CORES)]
        dcache = {}
        free = [FIRST_FETCH] * p.NUM_CORES
        prog_busy = data_busy = 0
        for b in lp.blocks:
            c = min(range(p.NUM_CORES), key=lambda i: (free[i], i))
            ic = icaches[c]
            t = 0
            for stream in b.warps:
                for pc, op, addrs in stream:
                    fetches = 2 if (W > 1 and op in GLOBAL_OPS) else 1
                    for _ in range(fetches):
                        if p.ICACHE_LINES and ic.get(pc % p.ICACHE_LINES) == pc // p.ICACHE_LINES:
                            t += FETCH_HIT
                        else:
                            t += FETCH_GRAB + FETCH_DONE + L
                            prog_busy += CHAN_BUSY + L
                            if p.ICACHE_LINES:
                                ic[pc % p.ICACHE_LINES] = pc // p.ICACHE_LINES
                    t += dec + 1 + 2 + (1 if W > 1 else 0)
                    if op in SMEM_OPS:
                        t += SMEM_WAIT
                    elif op in GLOBAL_OPS:
                        lanes = [a for _, a in addrs]
                        trans = len(set(lanes)) if (p.COALESCE and op == LDR) else len(lanes)
                        if p.DCACHE_LINES:
                            n = p.DCACHE_LINES
                            hit = op == LDR and all(dcache.get(a % n) == a // n for a in lanes)
                            if op == LDR:
                                for a in lanes:
                                    dcache[a % n] = a // n
                            if hit:
                                t += 4
                                trans = 0
                            else:
                                t += 1 + LSU_GRAB + L1D_MISS_DONE + L + (2 if W > 1 else 0)
                        else:
                            t += LSU_GRAB + LSU_DONE + L + (1 if W > 1 else 0)
                        data_busy += trans * (CHAN_BUSY + L)
                    else:
                        t += 1
            free[c] += t + BLOCK_GAP
        makespan = max(free) - BLOCK_GAP + LAUNCH_TAIL
        total += max(makespan, prog_busy, data_busy / NUM_CHANNELS)
    return total
