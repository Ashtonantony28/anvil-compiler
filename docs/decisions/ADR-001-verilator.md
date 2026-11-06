# ADR-001: A Verilator backend for `anvil.runtime.run`

**Status:** accepted (SG0, 2026-11-06). The design doc's M0 spike: build the simulator's sv2v output with Verilator,
adopt it only if it reproduces the iverilog RTL's per-launch cycles exactly, and record the speedup either way.

## Decision

Adopt it. `run(job, "verilator")` builds the sv2v output with Verilator 5.020 and a C++ harness, and matches the
iverilog + cocotb RTL on final memory, per-launch cycles and every counter on every case we tried. On the xf-bench
MLP job it runs about **300x faster** than iverilog (about 2 M cycles/s against 6 k). `"rtl"` (iverilog + cocotb, the
simulator's own helpers) stays the reference. The Verilator suite in `tests/runtime/test_runtime_verilator.py`
compares the two, and has to keep passing for `"verilator"` to stay trusted.

## What worked and what failed

1. **cocotb on Verilator: does not build.** cocotb 2.0.1 needs Verilator 5.036 or newer: its Makefile refuses
   5.020, and its `verilator.cpp` calls `VerilatedVpi::doInertialPuts`, `evalNeeded` and `clearEvalNeeded`, which
   5.020 lacks. It would also not have been much faster, because the per-cycle Python memory model dominates the
   iverilog run time (see the tiny-job numbers below). We dropped this path.
2. **C++ harness (`anvil/runtime/verilator_harness.cpp`): works.** About 250 lines. It ports
   `sim/test/helpers/memory.py` (per-channel latency countdown, the "served" flag that stops a stale `valid` from
   being serviced twice, writes applied from channel 0 up so the highest channel wins) and the launch sequence of
   `setup.launch` + `runner._run`. The timing follows cocotb on iverilog. At each rising edge the testbench reads the
   DUT's pre-edge outputs. Its writes are queued to the ReadWrite phase, so the DUT first samples them at the next
   edge. The harness does the same: it samples, clocks, then applies inputs. It also checks
   `data_mem_requests == serviced reads + writes` per launch, as `runner.check_counters` does.
3. **Verilator rejects `controller.sv` as written.**
   - `BLKANDNBLK`: `channel_serving_consumer` gets blocking writes (`= 1`, claiming a consumer; later loop iterations
     must see the claim) and non-blocking writes (`<= 0`, releasing it) in one `always` block. With the error waived,
     cycles still matched on the four bench configs. But `data_mem_requests` was wrong: 704 against the published 248
     at `COALESCE=1` for the naive `h` launch, and off by one to five at `DCACHE_LINES=16`. Matching cycles alone does
     not prove a Verilator build correct; the tests compare every counter.
   - **Fix, applied in the build cache only (the simulator stays read-only):** `verilator_source()` rewrites the sv2v
     output's `controller` module with blocking shadows. `csc_b` holds the variable as the block sees it.
     `csc_nm`/`csc_nv` collect the non-blocking writes. One non-blocking write at the end applies them over `csc_b`.
     This is the Verilog semantics of the original. The rewrite checks that every pattern it expects is present and
     that no use of the variable is left over, so a change to `controller.sv` makes the build fail instead of
     producing a wrong one. `test_rewrite_keeps_iverilog_behaviour` builds iverilog on the rewritten source and gets
     the published `COALESCE=1 DCACHE_LINES=16` cycles and counters.
   - `BLKLOOPINIT` at `COALESCE=1` ("delayed assignment to array inside for loops"): the coalescing loop makes the
     controller's loop nest exceed Verilator's default unroll limits. `--unroll-count 1024 --unroll-stmts 1000000`
     fixes it.
4. **X handling.** iverilog starts registers at X, and the cocotb helpers read X as 0. Verilator starts them at 0
   (`--x-initial fast`). We found no case where this matters: every register the results depend on is reset.

## Evidence (all exact: memory, cycles, all 8 counters)

- `sim/bench/xf-mlp.json`: all 12 launches (naive and tiled MLP chains) at all four `XF_CONFIGS`, cycles and the
  published request, coalescing, L1-D and I-cache counters.
- All 37 32-bit ISA fixtures at the default build.
- An 11-configuration matrix modelled on the simulator's `make matrix`: default, I-cache + pipelined, "all"
  (I-cache, pipelined, coalesce, L1-D, MeDiC), 1 and 4 cores, memory latency 3, 2 warps, 4 warps GTO, 16-thread
  blocks with 4 warps GTLRR, SIMT stack, and 2 warps with I-cache, pipelining and latency 10. Each runs seeded random
  multi-launch jobs (up to 24 threads, so several blocks contend for the memory controller; including cross-warp STR
  races, which are unspecified but must still be identical between two builds of the same RTL) plus 4 fixture
  kernels launched twice.
- The MLP (tiled) and attention (naive) chains at the default, "all" and "warps2lat" configurations.

## Numbers (4-core container, warm build cache, one process per job)

| Job | isasim | iverilog + cocotb | Verilator | Speedup vs iverilog |
|---|---|---|---|---|
| xf-bench MLP, 12 launches, 87 939 cycles, default | 0.025 s | 12.9 s (6.8 k cycles/s) | 0.037 s (2.4 M cycles/s) | 345x |
| same at `COALESCE=1 DCACHE_LINES=16`, 89 592 cycles | 0.025 s | 14.4 s (6.2 k cycles/s) | 0.055 s (1.6 M cycles/s) | 260x |
| one-instruction kernel (start-up cost) | — | 0.27 s | 0.005 s | 54x |

Build per `GpuParams`: sv2v 0.25 s (shared by every configuration), iverilog 0.04 s, Verilator 3.4 s (5.0 s at
`COALESCE=1 DCACHE_LINES=16`), using `-j` over the 4 cores. Builds are cached under `.anvil-cache/verilator/<key>/`.
The key covers the sources, tool versions, harness, flags and the rewritten text.

## Consequences

- SG6/SG7 measure on `"verilator"`. An MLP-sized measurement takes tens of milliseconds instead of about 13 s, so
  the measurement-cost risk R1 is retired. `"rtl"` stays the reference for audits and for any new configuration
  before we trust Verilator on it.
- The harness reimplements the cocotb testbench. If `sim/test/helpers/memory.py` or `setup.py` change on a submodule
  bump, the Verilator suite will fail until the harness follows. That is intended.
- Not modelled: `run_kernel`'s optional warp-issue trace. `check_counters` checks only
  `data_mem_requests == reads + writes`, not the coalesced-read and I-cache identities, which need internal signals.
