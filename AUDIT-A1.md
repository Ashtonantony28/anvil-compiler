# AUDIT-A1 — compiler correctness and backend/ISA consistency

Audit round A, lens 1. Independent auditor (wrote none of the audited code). Tree audited: `main` at `28e7b44`
(SG0–SG6 and SG5b done, SG7 in progress; `anvil/cost` and `tests/cost` were being edited by another worker and were not
audited). Nothing outside this file and `tests/audit/test_audit_a1.py` was changed; `make check` stays green
(611 passed, 11 xfailed).

## Summary

| ID | Severity | Finding |
|---|---|---|
| A1-M1 | Medium | Masked-off lanes load out-of-bounds words that other cores write in the same launch (cross-core read/write race). This is the root cause of SG6's "isasim and RTL disagree on scratch words". The bug is in the compiler, not the simulator. |
| A1-M2 | Medium | `GpuParams(WORD_BITS < 32)` is accepted by `compile_kernel` / `anvil.compile` and silently miscompiled (wrong outputs, no error). |
| A1-M3 | Medium | The C4 level-4 oracles (codegen `check_kernel`, driver `ints`, tuner `verdict`) read back only the kernel's named buffers, so writes anywhere else in memory go unnoticed. Seeded mutant M02 survived 13 split-launch configurations because of this. |
| A1-L1 | Low | C4 is enforced unevenly. Level 3 (MIR interpreter) runs only at `SIMT_STACK=0` and only per kernel, never on chains, the driver or G2. Level 4's flags cannot see smem races, cross-block write-write races or cross-core read/write races, and the cross-core checker promised in design §3 does not exist. |
| A1-L2 | Low | Three seeded mutants survived `make check` through coverage gaps: M15 (register-shift requant at S = 31), M20 (warp-private smem with more than one warp) and M30 (unpredicated register stores under a lane-varying guard; ragged row kernels). Passing tests that kill all three are added in `tests/audit/`. |
| A1-L3 | Low | Three surviving mutants are equivalent under today's schedule language. M04 and M22 (load CSE across barriers and loop iterations) are safe only because of an undocumented invariant. M18 (rsqrt trip count) is unreachable for int8 rows. |
| A1-L4 | Low | A graph with no kernels (its output is an input) compiles, then `run` raises the opaque `ValueError: Job has no launches`. |
| A1-L5 | Low | Test readability: `flags_clean()` returns the *bad* flags, so `assert not flags_clean(r, simt)` (tests/codegen) reads as if it asserted the opposite. The assertion itself is correct. |

There are no critical or high findings. Across about 1,280 random graphs × schedules × GpuParams on isasim and 212 on
Verilator (including `DCACHE_LINES=8` and `COALESCE=1`), no live output differed from the graph interpreter. No
`store_races`, `stores_conflict` or divergence flag fired, and no crash or internal error occurred.

Mutation testing: 32 seeded compiler mutants. `make check` killed **26/32 (81 %)**. With the audit's three new passing
tests, 29/32 (91 %) are killed. The remaining 3 are argued equivalent (A1-L3).

---

## Method

1. **Differential fuzzer**
   (`/tmp/.../scratchpad/fz/fuzz.py`; the logic is reproduced in this report, and the script is not committed). Each
   case is a random graph of 1–3 ops drawn from qlinear (optionally with a LUT epilogue), qmatmul, lut, lut_mul,
   add_requant, transpose, rmsnorm_int and softmax_lut (causal when square).
   - Shapes: R ∈ 1..9, C ∈ 1..12 and N ∈ 1..12. Big mode uses R up to 47, C/N up to 39 and 16-bit addresses; it
     exercises launch splitting (22 of 45 sampled kernels split) and spills.
   - Quantization: requant (M, s) with s up to 31 and M up to the int32 limit. Extreme mode uses M = the largest legal
     value, |Ma| = |Mb| at their limits, and inputs from {−128, −1, 0, 127}.
   - Schedules: the `default` and `tiled` presets, or random primitive sequences over a random template prefix (the
     tests/schedule strategy, re-implemented with numpy RNG).
   - Predication: `mask` forced in 30 % of cases.
   - GpuParams: THREADS_PER_BLOCK 4/8, WARPS_PER_CORE 1/2/4, SIMT_STACK 0/1 and NUM_CORES 1/2/4. Verilator mode adds
     DCACHE_LINES 0/8 and COALESCE 0/1.
   - Oracle: graph interpreter = isasim on every readable segment (`keep="all"`) with clean flags. For Verilator,
     memory equal to isasim below the scratch area, with scratch differences counted separately.

   Results:

   | Batch | Cases | Live mismatches | Scratch-only differences |
   |---|---|---|---|
   | isasim, normal (seeds 0–40, 100–700, 6000–6400) | 829 | 0 | — |
   | isasim, extreme (4000–4300) | 237 | 0 | — |
   | isasim, big (5000–5120) | 91 | 0 | — |
   | isasim, big + extreme (7000–7150) | 121 | 0 | — |
   | Verilator, hardware knobs (seeds 1000–1010, 2000–2150, 3000–3100 extreme) | 212 | 0 | 9 (A1-M1) |

   Skipped cases: about 15 % where the graph interpreter raises (int32 overflow or a LUT index out of range, which is
   legitimate), and about 3 % `ProgramTooLarge`. No CodegenError, ScheduleError or BackendError reached the fuzzer
   other than these.
2. **Mutation testing.** Thirty-two mutants (table below) were applied one at a time to a scratch copy of the repo,
   never to this tree. Each copy ran `pytest -x -m "not rtl and not torch"` (= `make check`) and was then reverted.
   Survivors were also run against `tests/codegen/test_codegen_rtl.py` and `tests/backend/test_backend_rtl.py`, and
   against 200 fuzzer seeds. The clean copy was diffed against `anvil/` after every batch.
3. **SG6 open issue.** Reproduced from the measurement cache (`.anvil-cache/measure`: 7 records with
   `scratch_differs: true`, all 2-active-lane matmul candidates at the xf-bench params) and traced to the root cause
   (A1-M1).
4. **C4 review.** Every test that claims a C4 level was read. Hypothesis vacuity was measured with a pytest plugin that
   counts examples reaching the oracle (`test_random_schedules_compile_bit_exact`: 119/120 examples finalize, 112 are
   checked bit-exact and 7 hit `ProgramTooLarge`; not vacuous). `skip`/`assume` were grepped: the codegen grid skips
   nothing at the current fixtures, the `assume(False)` calls in backend tests discard only oversize or invalid random
   programs, and the torch skips are importorskip only.

---

## Findings

### A1-M1 (Medium) — masked-off lanes race other cores through out-of-bounds loads

**What happens.** In `mask` predication (`SIMT_STACK=0`, or `predication="mask"`), `anvil/codegen/lower.py` lets
every lane run the body. It redirects only *stores*, to the sink and smem sink words; its docstring says "Loads, gathers
and arithmetic need no masking (no faults; addresses wrap to the memory size)". Lanes outside the bound lane extent
(`l >= Lx`), and lanes under a false lane-varying guard, therefore compute out-of-range indices and **load** from them.
Those addresses land in whatever buffer follows: the kernel's own output buffer, other kernels' buffers, the args
blocks, or the shared sink words that other threads are writing.

**SG6 repro.** The 1×8×8 qlinear `y` candidate (`ti=1, tj=2, lanes=2`, at the xf-bench params) has two blocks of four
threads with lanes 2–3 inactive. Block 1's inactive lanes compute columns j = 8..11 and read `w[7, 8..11]`, which is
address 72..75 = `y[0..3]`. That is exactly the output block 0 is storing **concurrently on core 0**, plus
`y[4..7]` and args/sink words at 80+.
- On isasim, blocks run one after another, so the inactive lane sees block 0's final values.
- On RTL/Verilator, it sees a timing-dependent mix.

The loaded value only reaches the lane's sink word, so the sink words of threads 6–7 differ between the two simulators.
With the tuner's inputs these are words 90–91 (−17 and −11 on isasim, 20 and 32 on Verilator); with the fixture input
in the test, word 91 differs. All 7 cached records with `scratch_differs` have this shape (2 active lanes of 4), and the
Verilator fuzzer found 9 more scratch-only differences.

**Correction to PLAN.md SG6.** The open issue says "simulator-side, uninvestigated". It is compiler-side: both simulators
are right, and the program has a cross-core read-during-write race.

**Can it affect a live output?** Not in any configuration the audit could reach. Masked-off lanes' values flow only
into masked stores (sink), the smem sink word, and their own unpredicated registers (inactive lanes), which feed only
masked stores. Cross-lane reductions read partials only from active threads. Verilator fuzzing with
`DCACHE_LINES=8`/`COALESCE=1` found no live mismatch.

The hazard is still real:
- It contradicts canonical rule 2 ("inactive threads must not load or store") and the design's own mitigation for the
  shared L1-D stale fill (design §3 and R6: "the compiler never reads a word in the same launch another core writes …
  and a checker asserts it"). AUDIT-10 M2 means that with `DCACHE_LINES > 0`, a masked lane's miss fill racing another
  core's write-through can install a stale line. Any *live* read of that word later in the same launch (for example a
  `distribute` + BAR exchange through a global temp) would then be wrong.
- isasim cannot detect it, because it runs blocks sequentially and has no read/write race checker.
- `anvil.tune.measure.execute` exempts the scratch area from the Verilator comparison precisely to hide the symptom.

**Fix direction.** Guard masked loads like stores: `addr = select(p, addr, own_sink_word)` for global LDR and
gathers in mask regions, including the lane-activity region. Alternatively, clamp inactive lanes' indices into the
buffer. Then drop the scratch exemption in `measure.execute`. A cheap static checker (the address-trace helper in the
test) can assert rule 2 in CI.

**Repro.**
```
python3 -m pytest -q --runxfail tests/audit/test_audit_a1.py -k a1_m1     # make check part + Verilator part
```
Tests:
- `test_a1_m1_inactive_lane_loads_stay_inside_read_buffers` (strict xfail, runs in `make check`). It traces the
  allocated program per thread and asserts that lanes 2–3 load only `x`/`w` words. Today they also load y = 72..79 and
  args/sink words at 80+.
- `test_a1_m1_masked_off_lanes_do_not_race_other_cores` (strict xfail, `rtl`/Verilator). It asserts full-memory
  equality isasim == Verilator including scratch. Today word 91 differs.

### A1-M2 (Medium) — WORD_BITS below 32 is silently miscompiled

`GpuParams` (C2) accepts `WORD_BITS` and `DATA_MEM_DATA_BITS` from 1 to 32, and nothing in `anvil.codegen`,
`anvil.backend` or `anvil.driver` checks them. The backend assumes 32-bit words throughout:
- `const32` plans built from 2^32 wraps,
- `peephole` folding with `sem.w_*` at 32 bits,
- requant rounding `0 − SRA(INT_MIN, 31−S)`,
- the int32 sign tricks in `idioms`.

At `WORD_BITS=16`, the `gate` / `y` / `act` fixture kernels compile without error and return 62 / 11 / 54 wrong
elements on isasim. Expected behaviour: a `CodegenError` naming the unsupported word size, or a real narrow-word
backend.

**Repro.** `python3 -m pytest -q --runxfail tests/audit/test_audit_a1.py -k a1_m2`. The test
`test_a1_m2_word_bits_below_32_rejected[16|24]` is a strict xfail.

### A1-M3 (Medium) — level-4 oracles ignore memory outside the named buffers

`tests/codegen/codegen_util.check_kernel` compares `plan.read(r.memory, s)` for `s in ck.buffers` only. The same holds
for:
- the driver tests (`cm.run(...).ints`),
- G2 (`g2_report.check`),
- the tuner's `measure.verdict` (only `task.ref`, the buffers the kernel writes).

A compiler bug that writes outside its output buffer is therefore invisible unless it happens to clobber an input or a
later kernel's operand. Such bugs include an extra block, an off-by-one tail, a wrong stride, or a store into the args
block.

**Evidence.**
- Mutant M02 (one extra block in the last split launch, writing past `c`) passed 13 `test_launch_splitting`
  configurations. It was only caught at `matmul_tiled (8,4,1)`, where the stray block happened to corrupt something
  read back.
- The tuner can rank such a kernel as valid.
- Isasim-vs-RTL tests compare `memory[:plan.end]`, but they compare two runs of the same wrong program.

Expected: the oracle checks that every word below the scratch area (and every word past `plan.end`) is either an
expected output value or unchanged from `job.data`.

**Repro.** `python3 -m pytest -q --runxfail tests/audit/test_audit_a1.py -k a1_m3`. The test
`test_a1_m3_check_kernel_catches_stray_writes` (strict xfail) flips one word past the job's data and expects
`check_kernel` to fail.

### A1-L1 (Low) — C4 is enforced unevenly

| Level | Where it is enforced | Gap |
|---|---|---|
| 1 graph = xf.golden | `tests/interp`, `tests/frontend` (fixtures) | — |
| 2 tensor after each transform | `tests/schedule/test_random_schedules.py` checks after **every** accepted primitive, 20 examples × 10 ops | Fixture shapes only (d=8, 4×16, and one 3×5 lut). The audit fuzzer covered ragged and size-1 shapes at levels 2–4 with no failure. |
| 3 MIR = tensor, before/after regalloc | `check_kernel(mir_level=…)` in the codegen grid and random tests | Only at `SIMT_STACK=0` (the MIR interpreter rejects divergence), only per kernel, never for driver chains or G2. It compares memory only below `plan.scratch`. |
| 4 isasim, flags empty | codegen, driver, G2 and tuner | isasim's `store_races` covers only cross-warp STR within one block. Smem races (STS) are not checked; mutant M20's warp-shared smem only failed through wrong values. Cross-block write-write and cross-core read/write races (A1-M1) are not checked, because blocks run sequentially. The tensor-interpreter race checker covers level 2 only. |
| 5 RTL = isasim | `-m rtl` samples (codegen 7 kernels + barriers + split, backend chains, G1, G2) | Full-memory compares, so A1-M1 would be caught if a sample had the shape. None does. |
| 6 float vs torch | torch-marked tests | — |

Recommendation: add an Anvil-side race checker over an address trace (global and smem; cross-block and cross-core
included), and a level-3 run for the driver chains at `SIMT_STACK=0`.

### A1-L2 (Low) — coverage gaps exposed by surviving mutants (now covered)

- **M15**: requant with a register shift computes `rnd = SRA(2^31, 1)` instead of `2^31 div 2`. It is wrong only at
  S = s + k = 31, which no test reaches. Killed by `test_a1_rmsnorm_register_shift_31_rounds_like_golden`.
- **M20**: warp-private smem buffers at offset 0 for every warp. No `make check` test binds W > 1 warps around a
  `cache_read(scope="warp")` buffer, and the random-schedule strategy rarely reaches one (0 of 200 fuzzer seeds).
  Killed by `test_a1_warp_scope_cache_read_with_two_warps[0|1]`.
- **M30**: register stores under a lane-varying `mask` guard not predicated. Every `make check` row kernel has rows that
  are a multiple of the lane count. Killed by `test_a1_row_per_block_ragged_rows` (rmsnorm and softmax, 5-wide rows,
  three configurations), and by 11 of 200 fuzzer seeds.

These are passing tests (the compiler is right), added so the mutants die in `make check`.

### A1-L3 (Low) — equivalent mutants rely on an undocumented invariant

- **M04** (barrier does not invalidate the load cache) and **M22** (load cache crosses loop scopes) survive `make
  check`, the RTL suites and 200 fuzzer seeds each. They are equivalent today only because codegen-time CSE is safe
  whenever every barrier is preceded, in the same thread's program text, by a store to the same space, which calls
  `forget_loads`. Every schedule primitive that inserts a BAR (`cache_read`, `distribute`, `cross_lane_reduce`)
  currently produces that shape. A future primitive that reads data another thread produced, without storing itself,
  would silently read stale registers. The invariant should be documented next to `load_mem`, or `forget_loads()` kept
  as the only mechanism and tested directly.
- **M18** (`_rsqrt_trips` one short when an intermediate equals n): unreachable. It needs a row sum of squares near 2^31,
  that is d ≥ 131,072 int8 values. The loop runs the int32 worst case (SG3b open issue); bounding it by d·128² would
  make the trip count exact and save cycles.

### A1-L4 (Low) — a kernel-less graph fails late with an opaque error

```
python3 -c "import sys;sys.path[:0]=['sim','.'];import numpy as np,anvil;from anvil.ir.graph import GraphBuilder as G
g=G('id');x=g.input('x',[2,2]);g.output(x);anvil.compile(g.module()).run(np.zeros((2,2),int),quantized=True)"
# ValueError: Job has no launches
```
`anvil.compile` should either return a model whose `run` copies inputs to outputs, or reject the graph at compile time
with a clear message.

### A1-L5 (Low) — inverted helper name

`tests/codegen/codegen_util.flags_clean` returns the dict of *offending* flags. Call sites read
`assert not flags_clean(r, simt)` (test_codegen_predication, test_codegen_rtl), which is correct but reads as "assert
the flags are dirty". Rename it to `bad_flags`.

---

## Mutant table

Killed by = the first failing test under `make check` (`pytest -x`). For survivors, the RTL suites and the 200-seed
fuzzer were also run.

| Mutant | Seeded bug | Killed by (make check) | Result |
|---|---|---|---|
| M01 | split launch: `block_offset = i·(bpl−1)` | test_codegen_predication::test_launch_splitting[naive-(4,1,0)] | killed |
| M02 | split launch: one extra block in the last launch | test_launch_splitting[tiled-(8,4,1)] (13 earlier configs passed; A1-M3) | killed |
| M03 | BAR never emitted | test_codegen_chains::test_chain_golden_on_isasim[attention-(8,2,1)-tiled] | killed |
| M04 | barrier does not invalidate load CSE | — | **survived** (equivalent today, A1-L3) |
| M05 | regalloc: dead defs do not interfere | test_regalloc::test_straight_line_pressure_spills_and_matches | killed |
| M06 | regalloc: no interference with constant registers | test_consts::test_const32_compiled_on_isasim | killed |
| M07 | const32: negative constant built with ADD | test_consts::test_plan_builds_the_word | killed |
| M08 | const32: `(hi+1)·256 − (255−lo)` | test_consts::test_plan_builds_the_word | killed |
| M09 | mask sink: lanes of a block share one sink word | test_codegen_predication::test_lane_varying_guards_mask_and_sync[0] | killed |
| M10 | masked-store address select inverted | test_codegen_ops::test_op_template_grid[lut-y-elementwise-4-1-0] | killed |
| M11 | lane-activity guard admits one inactive lane | test_inactive_lanes_and_warps_with_barriers[1-0] | killed |
| M12 | args block: first two buffer slots swapped | test_chain_golden_on_isasim[mlp_block-(4,1,0)-default] | killed |
| M13 | memplan: lifetimes touching at one kernel may share words | test_driver_isasim::test_fixture_cases_golden_on_isasim[mlp_block-default] | killed |
| M14 | requant rounding `2^(s−1) − 1` | test_idioms::test_requant_constant | killed |
| M15 | register-shift requant rounding via SRA (wrong at S = 31) | — | **survived**; now killed by tests/audit (A1-L2) |
| M16 | clamp_i8 lower bound −127 | test_idioms::test_binary_idioms_equal_semantics[clamp_i8] | killed |
| M17 | signed `lt` computes `gt` | test_idioms::test_binary_idioms_equal_semantics[ge] | killed |
| M18 | rsqrt_index trip count one short | — | **survived** (unreachable, A1-L3) |
| M19 | guard predicate admits x == ub | test_op_template_grid[lut-y-elementwise-4-1-0] | killed |
| M20 | warp-private smem shared by all warps | — | **survived**; now killed by tests/audit (A1-L2) |
| M21 | masked smem sink overlaps the last smem word | test_inactive_lanes_and_warps_with_barriers[1-0] | killed |
| M22 | load CSE across loop iterations | — | **survived** (equivalent today, A1-L3) |
| M23 | idle warps not retired when W = WARPS_PER_CORE − 1 | test_inactive_lanes_and_warps_with_barriers[2-0] | killed |
| M24 | induction register re-initialized every iteration | test_chain_golden_on_isasim[mlp_block-(4,1,0)-default] | killed |
| M25 | spill frame index ignores threadIdx | test_regalloc::test_straight_line_pressure_spills_and_matches | killed |
| M26 | memplan reserves no scratch words | test_driver_isasim::test_other_configurations[WARPS_PER_CORE=2] | killed |
| M27 | peephole: known constants survive labels | test_emit::test_peephole_rules | killed |
| M28 | counted loop one trip short (trips > 2) | test_chain_golden_on_isasim[mlp_block-(4,1,0)-default] | killed |
| M29 | uniform guard branch condition inverted | test_op_template_grid[lut-y-elementwise-4-4-0] | killed |
| M30 | mask predication: register stores not predicated | — | **survived**; now killed by tests/audit (A1-L2) |
| M31 | memplan: buffer dies one kernel before its last read | test_fixture_cases_golden_on_isasim[mlp_block-default] | killed |
| M32 | spill area overlaps the sink area | test_op_template_grid[mlp_block-h-none-4-1-0] | killed |

Kill rate: `make check` 26/32 = 81 %. With tests/audit: 29/32 = 91 %. Remaining survivors: M04, M22 and M18, all
equivalent under the current schedule language and value ranges.

RTL suites on survivors (`tests/codegen/test_codegen_rtl.py`, `tests/backend/test_backend_rtl.py`, `-m rtl`):
M04, M18, M20, M22 and M30 survive (16 passed each). M15 "fails"
`test_matmul_templates_on_verilator_vs_hand_kernels[naive]`, but only because that test compares cycle counts against
the published `docs/codegen-bench.md` table. The mutant changes the program, not the checked values, so this is not a
correctness kill. The RTL samples are a subset of the isasim grid's shapes, so they cannot see these bugs either. The
200-seed fuzzer killed M30 (11 failures) and none of the others.

## Files

- `tests/audit/test_audit_a1.py`: 4 strict xfails in `make check` (A1-M1 trace, A1-M2 ×2, A1-M3), 1 strict xfail under
  `-m rtl` (A1-M1 Verilator), and 9 passing mutant-killer tests (A1-L2).
- This report.
