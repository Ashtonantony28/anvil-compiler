"""Hardware cost proxy: declared state bits (flip-flops / RAM bits) of `module gpu` at a GpuParams, read off the RTL
(sim/src at 096c6ef). No synthesis: combinational logic (ALU datapaths, decoders, arbiters, `always @(*)` temporaries)
and the 32-bit telemetry counters (icache_hits/misses, warp_issues, bar_stall_cycles, mem_requests, hits_total, ...)
are left out. The formula, with WB = WORD_BITS, A = DATA_MEM_ADDR_BITS, D = DATA_MEM_DATA_BITS, T =
THREADS_PER_BLOCK, W = WARPS_PER_CORE, LANES = T / W (core.sv:51), C = NUM_CORES, clog2 as in Verilog:

per core (x C)
  registers      T * 16 * WB                       registers.sv:30  reg [WORD_BITS-1:0] registers [15:0], one per thread
  nzp            T * 3                             pc.sv:25         reg [2:0] nzp, one per thread
  lsu            T * (3*WB + 2*A + 22)             lsu.sv:24-42     read/write valid+address, write data, smem enables+
                                                                    8-bit addresses+data, lsu_state, lsu_out
  issued_op      T * 4 if W > 1                    core.sv:239      per-thread latched memory op (ADR-014)
  alu            LANES * WB                        alu.sv:23        alu_out_reg, one ALU per lane of a warp
  decoder        38                                decoder.sv:15-31 decoded fields
  fetcher        30                                fetcher.sv:19-35 valid, address(8), state(3), instruction(16), 2 flags
  I-cache        L_I * (1 + (8 - clog2(L_I)) + 16) fetcher.sv:84-86  valid, tag, 16-bit word per line (L_I = ICACHE_LINES)
  scheduler      8 + 3 + 1 + clog2(W+1)            scheduler.sv:33-39 current_pc, core_state, done, current_warp
  per-warp state W * (8 + 8 + 3) if W > 1          scheduler.sv:62-64 warp_pc, warp_age, done/bar/pending
  warp scheduler 2 * clog2(W+1) if W > 1           warp_scheduler.sv:37-38 lrr_ptr, gto_current
  SIMT stacks    W * (11*(16 + LANES) + 4 + 1 + 8) simt_stack.sv:23-31 (DEPTH 8 -> N = 11 entries of pc, rpc, mask; sp,
                 if SIMT_STACK                                       sync_valid, sync_pc; n_* are combinational)
  shared memory  64 * WB                           shared_memory.sv:25 (SMEM_SIZE = 64, core.sv:17)
global
  dispatch       17                                dispatch.sv:26-28
  controller(N consumers, K channels, data bits B, address bits a):    controller.sv:22-58
                 N*(B + 2) + K*(3 + clog2(N+1) + 2*a + B + 2) + N + (K*N if COALESCE)
                 data controller: N = C*T lanes, K = 4 (gpu.sv:12), B = D, a = A;
                 program controller: N = C fetchers, K = 1 (gpu.sv:15), B = 16, a = 8, never coalescing
  L1-D (shared)  L_D * (1 + (A - clog2(L_D)) + D)  l1d_cache.sv:57-63 valid, tag, data per line (L_D = DCACHE_LINES)
                 + N*(D + 2) + N*(2*A + D + 2)     l1d_cache.sv:30-43 consumer- and downstream-side output registers
                 + N*2 (+ N*8 with MEDIC)          l1d_cache.sv:67-72 per-consumer state (MeDiC windows)
PIPELINED and WARP_POLICY add no state (scheduler.sv:160 skips DECODE; the policy is combinational).
"""
from __future__ import annotations

import math

SMEM_SIZE = 64
DATA_CHANNELS = 4
PROGRAM_CHANNELS = 1
SIMT_DEPTH = 8


def clog2(n: int) -> int:
    return max(0, math.ceil(math.log2(n))) if n > 1 else 0


def _controller(n, k, data_bits, addr_bits, coalesce=False):
    return (n * (data_bits + 2) + k * (3 + clog2(n + 1) + 2 * addr_bits + data_bits + 2) + n
            + (k * n if coalesce else 0))


def breakdown(p) -> dict:
    """State bits per component (summed over cores) for GpuParams p."""
    WB, A, D, T, W, C = p.WORD_BITS, p.DATA_MEM_ADDR_BITS, p.DATA_MEM_DATA_BITS, p.THREADS_PER_BLOCK, \
        p.WARPS_PER_CORE, p.NUM_CORES
    lanes = T // W
    out = {}
    out["registers"] = C * T * 16 * WB
    out["nzp"] = C * T * 3
    out["lsu"] = C * T * (3 * WB + 2 * A + 22) + (C * T * 4 if W > 1 else 0)
    out["alu"] = C * lanes * WB
    out["decode+fetch"] = C * (38 + 30)
    Li = p.ICACHE_LINES
    out["icache"] = C * Li * (1 + max(1, 8 - max(1, clog2(Li))) + 16) if Li else 0
    sched = 8 + 3 + 1 + clog2(W + 1)
    if W > 1:
        sched += W * 19 + 2 * clog2(W + 1)
    out["scheduler"] = C * sched
    out["simt"] = C * W * (11 * (16 + lanes) + clog2(SIMT_DEPTH + 3) + 1 + 8) if p.SIMT_STACK else 0
    out["smem"] = C * SMEM_SIZE * WB
    out["dispatch"] = 17
    out["data_ctrl"] = _controller(C * T, DATA_CHANNELS, D, A, bool(p.COALESCE))
    out["prog_ctrl"] = _controller(C, PROGRAM_CHANNELS, 16, 8)
    Ld = p.DCACHE_LINES
    if Ld:
        n = C * T
        out["dcache"] = (Ld * (1 + max(1, A - max(1, clog2(Ld))) + D) + n * (D + 2) + n * (2 * A + D + 2) + n * 2
                         + (n * 8 if p.MEDIC else 0))
    else:
        out["dcache"] = 0
    return out


def state_bits(p) -> int:
    return sum(breakdown(p).values())


def kbits(p) -> float:
    return state_bits(p) / 1000.0
