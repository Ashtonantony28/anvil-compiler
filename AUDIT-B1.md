# AUDIT-B1: correctness and regressions since audit A

Audit round B, lens 1. I am an independent auditor and wrote none of the audited code. Tree: `main` at `6461764`
(SG9, SG10 and SG8 done). Scope: code added or changed after audit A.

- **SG9:** the i16 residual stream. This covers the `anvil.semantics` wide ops, `rmsnorm_wide` with
  `ss_shift`/`ss_init`, `quantize(residual="i16")`, the wide branch of `prove_no_overflow`, and the clamps in
  `lower/graph_to_tensor`.
- **SG10:** masked-lane load redirection and the inactive-lane clamp in `codegen/lower.py`,
  `anvil/runtime/races.py`, `JobPlan.stray_writes` and its callers, and the `compute_at` domain check.
- **SG8:** `anvil/isa_ext` (the ISA fork, the MIR peepholes, the idiom switches and the `enabled()` patch scope), and
  the correctness of the `anvil/codesign` sweep.

I changed only this file and `tests/audit/test_audit_b1.py`. `make check` stays green: 753 passed, 3 xfailed (B1-M1 plus AUDIT-B2's two), in 2 min 44 s. I did not touch AUDIT-B2 or its tests, PLAN.md, `sim/` or any code.

## Summary

| ID | Severity | Finding |
|---|---|---|
| B1-M1 | Medium | When MAD's accumulator is the register that spills, compiling under `isa_ext.enabled({"MAD"})` raises `BackendError: virtual registers may be read before written`. The documented fallback (recompile without MAD) never runs, because it checks the stats of a compile that already raised. The tuner turns the error into an `Invalid` candidate. So in the G5 study's MAD re-tune, 8 MLP and 17 attention candidates that compile on the base ISA were dropped without any message: the MAD variant searched a different space from the base variant. |
| B1-L1 | Low | Coverage gaps. 16 of the 42 seeded mutants survived `make check`, and 13 of those are real gaps (the other 3 are B1-L5): an unsound i16 magnitude in the overflow proof (B08, B09), the wide `add_requant` clamp (B11), the `s + kmax <= 31` rule (B12), an off-by-one in `compute_at`'s domain check (B25), the tuner verdict's stray-write check (B26), three `isa_ext` peephole guards (B28, B29, B31), the SLT `ge`/`le` idioms on equal operands (B34), the SLT mask predicate (B35), SHL with the constant first (B41), and the sweep's race check (B42). I added passing tests that kill all 13. |
| B1-L2 | Low | SG10's inactive-lane clamp hides wrong lane-activity predicates. A lane with `l >= Lx` recomputes lane `Lx - 1`'s address and value, so if its predicate is wrongly 1 it stores the same value to the same word. No oracle can see that (`stores_conflict` needs different values). Mutant B35 (SLT predicate `f < 2`) survived 117 SLT + mask fuzz cases for this reason. |
| B1-L3 | Low | `check_races` misses a read-write race when the reader also writes the word with the same value as the other writer (readers ⊆ writers, one value). A 2-block, 8-instruction repro is below. No compiled kernel has this shape today. Mutant B17 shows the shape needs a kernel that loads a global buffer it also writes, and none does. |
| B1-L4 | Low | Nested `isa_ext.enabled()` blocks compose correctly: the code equals the flat union. Two pieces of bookkeeping are wrong. `active()` returns `frozenset()` after an inner block exits, although the outer patches are still in force. `program.stats["isa_ext"]["props"]` names only the innermost set. No code reads either value today. |
| B1-L5 | Low | Three of the 42 mutants are equivalent under today's schedule language: B17 (safe word for a written global buffer), B27 (smem safe word 1 instead of 0) and B40 (no MAD fallback). B40 also shows the MAD fallback is too broad. A MAD program whose spills do not touch an accumulator is correct, yet any spill disables MAD for the whole kernel. |

There are no critical or high findings. The SG9, SG10 and SG8 code is right on everything the fuzzer and the
mutants reach. Across 1,167 compiled random cases with no live-output mismatch:
- graphs with i16 inputs and i16 `qlinear`/`add_requant` results, wide rmsnorm (also on i8 rows), extreme values,
  random schedules and random `GpuParams`;
- isasim, the `isa_ext` fork and Verilator.

No race, divergence, store-conflict or stray-write flag fired. Verilator equals isasim on the whole job memory,
**scratch included**, in all 71 cases (DCACHE_LINES 0/8, COALESCE 0/1, every warp policy). This confirms SG10's claim
that masked-off lanes now load nothing racy, so the sink words are deterministic.

The overflow proof was sound on every case I tried:
- 156 graphs it accepted, each run on adversarial inputs (the proof rejected 423 others);
- 397,681 adversarial rows against the tight wide-RMSNorm bound, over 400 random (d, scale, eps, wmax) configurations.

Mutation testing: 42 seeded mutants in the SG9, SG10 and SG8 code.
- `make check` killed **26/42 (62 %)**.
- With the passing tests added here, it kills **39/42 (93 %)**.
- The remaining 3 are argued equivalent (B1-L5).
- The fuzzers alone (differential plus proof-soundness) also killed B08 and B09.

The new checks do fire on injected bugs:
- B16 (no masked-load redirection) is caught by `check_races` in `tests/codegen/test_codegen_races.py`.
- B19, B20 and B21 (blinding the race checker) are caught by that file's checker unit tests.
- B22 and B23 (blinding `stray_writes`) are caught by A1-M3's test and by `test_stray_writes_catch_...`.
- B14 and B15 (lane clamp off by one, or removed) are caught by A1-M1's address-trace test.

---

## Method

1. **Differential fuzzer.** I wrote it for this audit (`scratchpad/fz/gen.py`, not committed). The logic is described
   here.
   - **Graph.** Random, with 1–3 ops drawn from qlinear (i8 or i16 result; an optional LUT epilogue on i8),
     qmatmul (i8 or i16), add_requant (any mix of i8/i16 operands and result), rmsnorm (wide form via
     `rmsnorm_lut_wide` on i16 rows, and on 30 % of i8 rows; xf form otherwise), lut, lut_mul, softmax_lut and
     transpose.
   - **Input.** 60 % i16. Scales are drawn from {1e-4, 5e-4, 0.01, 0.03}. Requant constants come from `fixed_point`.
   - **Extreme mode.** Ratios are drawn from {1e-6, 0.5, 3, 40}. Weights come from {±wmax, 0, −1}. Inputs come from
     {−32768, 32767, 0, ±1} and constant rows at both limits.
   - **Shapes.** R ≤ 9 and C ≤ 12; big mode uses ≤ 39 and 16-bit addresses.
   - **Schedules.** `default`, `tiled`, or random primitive sequences (tests/schedule's strategy on a numpy RNG).
     `predication="mask"` is forced in 30 % of cases, or in all of them with `FZ_MASK`.
   - **GpuParams.** THREADS_PER_BLOCK 4/8, WARPS_PER_CORE 1/2/4, SIMT_STACK 0/1 and NUM_CORES 1/2/4. Verilator mode
     adds DCACHE_LINES 0/8, COALESCE 0/1 and WARP_POLICY 0–2.
   - **Oracle.** Every readable value from `cm.job(..., keep="all")` must equal the graph interpreter. The C4 flags
     must be empty (divergence only at SIMT_STACK=0). `JobPlan.stray_writes` must be empty, with every const
     unchanged. `check_races` must be empty. Verilator must equal isasim on the whole job memory, scratch included.
     In `isa_ext` mode, a random non-empty subset of proposals is compiled under `enabled()` and run on the fork,
     with the same checks.

   | Batch | Seeds | Compiled cases | Live mismatches | Errors |
   |---|---|---|---|---|
   | isasim | 0–250 | 203 | 0 | 0 |
   | isasim, extreme | 1000–1250 | 194 | 0 | 0 |
   | isasim, big (16-bit addresses) | 2000–2080 | 60 | 0 | 0 |
   | isasim, mask forced | 7000–7200 | 165 | 0 | 0 |
   | isa_ext, random proposals | 3000–3250 | 204 | 0 | 1 (B1-M1) |
   | isa_ext, random proposals, extreme | 3500–3700 | 157 | 0 | 2 (B1-M1) |
   | isa_ext, all five proposals, mask forced | 7500–7650 | 113 | 0 | 0 |
   | Verilator | 4000–4060 | 41 | 0 (scratch included) | 0 |
   | Verilator, extreme | 4500–4540 | 30 | 0 (scratch included) | 0 |

   Skipped cases (≈ 20 %) are graphs the proof rejects, `ProgramTooLarge`, and schedules the scheduler or codegen
   rejects (smem). Inputs on which the graph interpreter raises are also skipped. These are legitimate.

   The fuzzer is not vacuous. Of 97 compiled cases (seeds 0–119):
   - 53 have an i16 input;
   - 45 have a mask region and 29 a SIMT `sync` region;
   - 16 redirect masked loads (`stats["mask_loads"]`);
   - 55 use random schedules.
2. **Overflow-proof soundness** (`scratchpad/fz/proof.py`).
   - **Graphs.** Random chains of add_requant, wide or i8 rmsnorm and qlinear, with constants that are *not*
     designed safe: M log-uniform up to 2^31, s up to 31. Some rmsnorm LUTs are inflated ×1.5–16, `ss_shift` is
     lowered by 1–2 and `s` is raised by 1–2.
   - **Check.** For every graph `prove_no_overflow` accepts, the graph interpreter (which checks every int32 term)
     runs on these inputs: both dtype limits, random extremes, and rows with one loud entry from 1 to 2^15 plus
     filler.
   - **Result.** 600 seeds: 156 accepted, 423 rejected, **0 unsound**. On mutants B08 and B09 it finds 3 and 33
     unsound graphs.
   - **Wide-RMSNorm bound.** A separate test hits the tight bound directly: 400 random configurations (d ∈ 1..128,
     sx ∈ [1e-6, 0.1], eps ∈ [1e-8, 1], wmax, i8 and i16 magnitudes) and 397,681 adversarial rows. Every row stays
     within `rms_xmax_table` for its (idx, k). No `rmsnorm_wide` raises.
3. **Mutation testing.**
   - **Procedure.** 42 mutants (table below) were applied one at a time to scratch copies of the repo, never to this
     tree. Each copy ran `make check`'s selection with `-x`. Survivors were then fuzzed (60 normal + 40 extreme seeds,
     plus 60 `isa_ext` seeds for SG8 mutants), and the SG9 survivors also by the proof fuzzer. Each mutant was
     reverted by restoring the original file. The scratch trees were reset with git after each manual check.
   - **Kill table.** "make check" is the stock suite. "+B1 tests" adds this audit's passing tests.

   | ID | Area | Mutant | make check | +B1 tests | Killed by |
   |---|---|---|---|---|---|
   | B01 | SG9 | `sq_shift` truncates (no rounding term) | killed | killed | test_wide_residual (tight bound) |
   | B02 | SG9 | lowered square rounds with 2^c, not 2^(c−1) | killed | killed | test_wide_residual (lowering) |
   | B03 | SG9 | lowered rmsnorm drops `ss_init` | killed | killed | test_wide_residual (lowering) |
   | B04 | SG9 | wide requant clamp hi = 32768 | killed | killed | test_wide_residual (lowering) |
   | B05 | SG9 | wide requant clamp lo = −32767 | killed | killed | test_wide_residual (lowering) |
   | B06 | SG9 | wide requant lowering drops the rounding term | killed | killed | test_wide_residual (lowering) |
   | B07 | SG9 | tight bound uses the middle of each (i, k) bucket (unsound) | killed | killed | test_wide_residual (tight bound) |
   | B08 | SG9 | proof: an i16 `add_requant` result bounded by 128 (unsound) | **survived** | killed | B1 test; proof fuzz |
   | B09 | SG9 | proof: an i16 input bounded by 2^14 (unsound) | **survived** | killed | B1 test; proof fuzz |
   | B10 | SG9 | `ss_shift_for` one too small | killed | killed | test_wide_residual (tight bound) |
   | B11 | SG9 | wide `add_requant` always clamps to i16 | **survived** | killed | B1 test |
   | B12 | SG9 | proof accepts `s + kmax = 32` | **survived** | killed | B1 test |
   | B13 | SG9 | tight bound zeroes reachable indices n/4..n/2 at k ≥ 1 | killed | killed | test_wide_residual (tight bound) |
   | B14 | SG10 | inactive-lane clamp to `Lx` (one past) | killed | killed | A1-M1 address trace |
   | B15 | SG10 | no inactive-lane clamp (A1-M1 regression) | killed | killed | A1-M1 address trace |
   | B16 | SG10 | no safe-address redirection for masked loads | killed | killed | test_codegen_races (`check_races`) |
   | B17 | SG10 | safe word of a written global buffer = its word 0 | survived | survived | equivalent (B1-L5) |
   | B18 | SG10 | safe-address term hoisted to the outermost scope | killed | killed | A1-M3 / backend |
   | B19 | SG10 | race checker ignores read-write races | killed | killed | test_codegen_races |
   | B20 | SG10 | race checker attributes every access to block 0 | killed | killed | test_codegen_races |
   | B21 | SG10 | write-write needs 3 distinct values | killed | killed | test_codegen_races |
   | B22 | SG10 | `stray_writes` exempts everything past scratch | killed | killed | A1-M3 test |
   | B23 | SG10 | `stray_writes` ignores `unchanged` | killed | killed | test_driver_isasim |
   | B24 | SG10 | `compute_at` domain check removed (A2-M8 regression) | killed | killed | A2-M8 test |
   | B25 | SG10 | `compute_at` domain check off by one | **survived** | killed | B1 test |
   | B26 | SG10 | tuner `verdict` ignores stray writes | **survived** | killed | B1 test |
   | B27 | SG10 | smem safe word is word 1 | survived | survived | equivalent (B1-L5) |
   | B28 | SG8 | MAD fusion ignores other uses of the product | **survived** | killed | B1 test |
   | B29 | SG8 | MAD fusion ignores redefinitions between mul and add | **survived** | killed | B1 test |
   | B30 | SG8 | AND-mod mask 2^k + 1 | killed | killed | test_isa_ext |
   | B31 | SG8 | AND-mod drops the same-dividend / single-def checks | **survived** | killed | B1 test |
   | B32 | SG8 | SHL rounding term 2^S instead of 2^(S−1) | killed | killed | test_isa_ext |
   | B33 | SG8 | SHL for `mul` by 2^k shifts k−1 | killed | killed | test_isa_ext |
   | B34 | SG8 | SLT `ge(a, b)` as `b < a` (wrong on equality) | **survived** | killed | B1 test |
   | B35 | SG8 | SLT mask predicate `f < 2` | **survived** | killed | B1 test (see B1-L2) |
   | B36 | SG8 | MIN semantics unsigned | killed | killed | test_isa_ext |
   | B37 | SG8 | regalloc does not see MAD's tied read | killed | killed | test_isa_ext |
   | B38 | SG8 | `enabled()` leaks the idiom patches after exit | killed | killed | test_isa_ext |
   | B39 | SG8 | fork: MAD does not read rd | killed | killed | test_isa_ext |
   | B40 | SG8 | no MAD fallback on spills | survived | survived | equivalent (B1-L5) |
   | B41 | SG8 | SHL `mul(2^k, x)` shifts the constant | **survived** | killed | B1 test |
   | B42 | codesign | sweep `check_chain` ignores `check_races` races | **survived** | killed | B1 test |

4. **Targeted reviews.** These parts were read line by line:
   - the SG9/SG10/SG8 diffs (`git show 11f554b 9191a0e`);
   - the race checker's warp-policy replay against `xf.isasim`'s LRR/GTO loop;
   - the `enabled()` patch list against every by-name import of the patched names: `anvil.backend.PURE`,
     `emit.ALU`, `emit.encode`/`asm_text`, `lir.Ins.reads`, `cg.compile_mir`;
   - the fork against `xf.isasim` (diff);
   - all 104 sweep records against their config IDs (params, latency, `state_bits`): 0 inconsistencies.

   Re-running the G5 MAD and base re-tunes (`study.retuned`, scratch caches) reproduces the committed cycles exactly:
   MLP 22,644 / 24,844 and attention 20,577 / 22,425. Running an extended-ISA `CompiledModel` on a base backend is
   rejected by `runtime.validate` ("program words outside 16 bits"), not silently mis-executed.

---

## Findings

### B1-M1 (Medium): MAD's documented spill fallback does not run when an accumulator spills

**What happens.** `anvil/isa_ext/backend.py` documents the fallback in `enabled()`: "a program whose allocation spills
while it contains a MAD is recompiled without MAD fusion (the spill rewriter does not handle a tied spilled
accumulator)". The code for it is:

```python
prog = base_compile(g, **kw)
if st.get("mad") and prog.stats["spilled_vregs"]:      # only reached when base_compile returned
    g, st = select(f, props - {"MAD"}) ...
```

When the register that spills is a MAD accumulator, the allocator's spill rewriter renames the MAD's destination to
a fresh register. It stores that register to the slot, but inserts no reload for MAD's tied read of it:
`mad v238, v8, v86` with `v238` defined nowhere. The next allocation round raises
`BackendError: virtual registers may be read before written: ['v238']` inside `base_compile`, so the fallback is
never reached.

When only other registers spill, MAD with spills is correct. Mutant B40 removes the fallback, and a 14-input
synthetic program then still computes the right value on the fork.

**Where it bites.**
- `anvil.compile` or `compile_kernel` under `enabled({"MAD"})` raises on legal kernels that spill on the
  base ISA. The smallest repro I found is a wide (i16) rmsnorm with ragged rows (5 × 8) under its default schedule
  at `GpuParams()`, which spills 6 registers on the base ISA. The fuzzer hit it 3 times in 450 `isa_ext` seeds
  (3167, 3549, 3635). All three were an rmsnorm kernel with ragged rows, and one had MAD alone enabled.
- `anvil.tune.measure.build` maps `BackendError` to `Invalid(stage="backend")`, so the tuner drops these candidates
  without any message.

**Effect on G5.** I re-ran `study.retuned` with a wrapper on `build`, using scratch caches. The re-tune numbers
reproduce the committed records exactly.

| Variant | MLP candidates dropped by this error | Attention candidates dropped |
|---|---|---|
| MAD | 8 (up 3, down 5) | 17 (q 1, k 6, v 2, scores 1, o 3, y 4) |
| base | 0 | 0 |

- The MAD variant was searched over a different candidate set than the base variant. That breaks the study's
  stated "identical pipeline, so the deltas compare like with like".
- The dropped candidates spill on the base ISA and are rarely competitive, so the effect on the headline (MAD ≈ 10 %)
  is probably small. It is not measured.
- In the fixed-schedule half of the study, a spilling best schedule would have crashed the study instead. None
  does.

**Expected.** Either the spill rewriter reloads a spilled tied accumulator before a MAD, or the fallback catches the
allocation error (or pre-checks for spills) and recompiles without MAD. Narrowing the fallback to "an accumulator
spilled" would also stop it from discarding correct MAD programs (B1-L5).

**Repro.**
```
python3 -m pytest -q --runxfail tests/audit/test_audit_b1.py -k b1_m1
```
- `test_b1_m1_precondition_the_base_compile_spills` passes. It shows the kernel is legal and spills on the base ISA.
- `test_b1_m1_mad_falls_back_when_its_accumulator_spills` is a strict xfail. It compiles under MAD, expects
  `mad_fallback == 1`, and checks the result on the fork against the graph interpreter.

### B1-L1 (Low): coverage gaps exposed by surviving mutants (now covered)

The compiler is right in every case below. Each test is a passing test in `tests/audit/test_audit_b1.py`, written
so the mutant dies in `make check`. I checked each test against its mutant in a scratch tree.

- **B08, B09: the i16 magnitudes of the overflow proof.** Inputs are 2^15, and i16 `add_requant` and
  `qlinear`/`qmatmul` results are 2^15. No test asserts that the proof *rejects* a graph that is unsafe only
  because of an i16 magnitude. The differential fuzzer cannot see these mutants, because a graph built with
  `fixed_point`/`rmsnorm_lut_wide` is safe whatever the proof says. The proof fuzzer does see them.
  - Tests: `test_b1_overflow_proof_uses_the_int16_input_magnitude` and `..._propagates_the_int16_result_magnitude`.
- **B11: i16 operands with an i8 `add_requant` result.** Such an add saturates at int8. `test_wide_semantics_...`
  only uses int8-valued operands, which take the golden path, because `sem.add_requant` picks its path from the
  operand *values*.
  - Test: `test_b1_add_requant_with_int16_operands_clamps_to_an_int8_result` (semantics and the compiled kernel).
- **B12: `s + kmax > 31` for a wide rmsnorm.** No test reaches this rule. `rmsnorm_lut_wide` stops well below it
  (s + kmax = 24 at d = 8).
  - Test: `test_b1_overflow_proof_rejects_a_wide_rmsnorm_shift_past_31`.
- **B25: `compute_at` one row past a ragged edge.** A2-M8's case overshoots the domain by 2 rows, so an off-by-one in
  `_check_box_in_domain` passes.
  - Test: `test_b1_compute_at_past_a_ragged_edge_by_one_row_is_rejected_or_bit_exact` (5 rows split by 3).
- **B26: the tuner verdict's stray-write rule.** tests/tune never runs `verdict` on a memory with a stray write.
  - Test: `test_b1_tuner_verdict_rejects_stray_writes` (a word past the job, and a word of an input buffer).
- **B28, B29, B31: the guards of the MIR peepholes.** These are "product used once", "operands not redefined before
  the add" and "same dividend, single def". The codegen never emits the patterns the guards exclude, so kernel tests
  cannot see them.
  - Tests: `test_b1_mad_fusion_*` and `test_b1_and_mod_needs_the_same_dividend`. They are unit tests on `fuse_mad` /
    `and_mod` with a small lir evaluator.
- **B34: SLT `ge`/`le` on equal operands.** `test_isa_ext`'s idiom test draws random operands and almost never draws
  equal ones.
  - Test: `test_b1_slt_comparisons_on_equal_operands`.
- **B35: SLT mask predicate.** See B1-L2.
  - Test: `test_b1_slt_mask_predicate_is_f_equal_zero` (the idiom over f ∈ 0..3).
- **B41: SHL `mul(2^k, x)` with the constant first.** The codegen always puts the register first.
  - Test: `test_b1_shl_multiply_with_the_power_of_two_first`.
- **B42: the sweep's `check_chain` race rule.** This is only exercised by the RTL replay test, and that test never
  sees a race.
  - Test: `test_b1_sweep_check_chain_rejects_cross_block_races` (`check_races` patched to report one race,
    `backend="isasim"`).

### B1-L2 (Low): the inactive-lane clamp makes wrong lane-activity predicates invisible

Under mask predication, SG10 runs inactive lanes (`l >= Lx`) with `l` clamped to `Lx − 1`. Each such lane then
computes exactly lane `Lx − 1`'s addresses and values. If its store predicate is wrongly 1, it stores the same value
to the same word as lane `Lx − 1`:
- no memory word changes;
- `stores_conflict` stays empty (it needs two different values);
- the race checker, `stray_writes` and Verilator all agree.

Mutant B35 shows this. It makes the SLT predicate test `f < 2` instead of `f == 0`. With 1 ≤ Lx < lanes the
lane-activity indicator is `f = l` (hi ≥ 2), so lane 1 gets predicate 1. The mutant survived `make check` and 117
fuzz cases with SLT and mask predication forced.

I found no other kind of lane that gets `f = 1` with hi ≥ 2. A single split by a factor ≤ the extent gives
`floordiv(x, n) ≤ 1`, so hi ≤ 1 and the base path is used. A two-dimensional ragged tile bound to lanes failed to
schedule (`reorder` of a masked split is not a perfect nest). The 117 fuzz cases with SLT and mask forced never
reached such a lane either. This is evidence, not proof.

So the bug class is unobservable today, but would bite as soon as either clamp or predicate changes. The added
idiom test pins the predicate directly. Suggestion: a debug option that turns off the clamp (or sets inactive lanes
to a poison lane index) in the fuzz and the codegen tests, so predicate bugs show again.

### B1-L3 (Low): `check_races` misses a read of a word that the reader also writes with the other writer's value

`races.py` reports a read-write race only for readers that do not write the word (`reads - writes`). It reports a
write-write race only when the values differ. Now take two blocks that each read word w and then write the same
constant to it. Block 1's read can see the initial value or block 0's write, depending on timing, and neither rule
fires:

```python
# 2 blocks x 4 threads: r1 = mem[10]; mem[10] = 7; mem[20 + blockIdx] = r1
prog = [CONST r0,#10; LDR r1,r0; CONST r2,#7; STR r0,r2; CONST r3,#20; ADD r3,r3,%blockIdx; STR r3,r1; RET]
check_races(Job([0]*32, [Launch(prog, 8)])).races   # [] ; isasim: mem[20] = 0, mem[21] = 7 (block 1 saw block 0's write)
```

The compiler never emits this shape. Kernels never load a global buffer they write: mutant B17 (B1-L5) reaches no
such load in 300 extra fuzz cases. The checker is the promised "no read in a launch of a word another core writes"
guard, and this is a hole in it. The fix: report a read of word w by block b whenever some *other* block writes w,
unless b's read follows its own write of w.

### B1-L4 (Low): `enabled()` bookkeeping under nesting

`with enabled({"MIN"}): with enabled({"SLT"}): ...` behaves as follows:
- The code is the same as `enabled({"MIN", "SLT"})`: the inner `compile_mir` wraps the outer one, and the patches are
  restored in order.
- `program.stats["isa_ext"]["props"]` reads `["SLT"]`.
- After the inner block exits, `active()` returns `frozenset()` while MIN is still in force. The `finally` resets
  `_ACTIVE` to empty instead of to the outer value.

Nothing reads `active()` today, and the study never nests. Low.

### B1-L5 (Low): equivalent mutants and an over-broad MAD fallback

- **B17.** For a buffer the kernel also writes, the masked-load safe word is the buffer's own word 0 instead of
  the args block. This is equivalent because no finalizable kernel loads a global buffer it writes. `bind` needs a
  single top-level nest, so multi-stage kernels must `compute_at`, which turns the temps into registers. No template
  kernel has such a load, and 300 random-schedule fuzz cases on the mutant found no race.
- **B27.** The smem safe word is word 1 instead of word 0. Masked lanes' smem values are discarded, smem addresses
  wrap (`& (SMEM_SIZE − 1)`), and smem is not race-traced, so nothing can observe it. The docstring's claim ("a word
  an active lane may read in this phase") is itself not checked by any tool (A1-L1).
- **B40.** Without the MAD fallback, every program that compiles is still correct: spills that do not hit an
  accumulator are handled, and those that do hit one raise (B1-M1). The fallback is therefore too broad as well as
  too late. Any spill disables MAD for the whole kernel. The MAD study's `mad_fallback` kernels thus under-count
  what MAD could fuse.

---

## Notes (not findings)

- Default schedules of a wide rmsnorm with `rows % THREADS_PER_BLOCK != 0` spill 6 registers at `GpuParams()`,
  against 0 when the rows divide evenly. The cause is the masked register stores plus the wide square's
  round-and-shift. The SG5 model (T = 16) is not affected. This is performance only.
- A1-L1's remaining gaps are unchanged. Shared memory is not race-traced. Within-block cross-warp read/write
  ordering is checked only for STR/STR (`store_races`).
- Every new check runs where SG10 says it does:
  - `check_races` runs in the tuner (`execute` → `block_races` flag), the sweep and the ISA study;
  - `stray_writes` runs in `check_kernel`, the driver tests, G1/G2 and the tuner verdict.

## Repro index

```bash
python3 -m pytest -q tests/audit/test_audit_b1.py                    # in make check: 20 passed, 1 xfailed, ~1 s
python3 -m pytest -q --runxfail tests/audit/test_audit_b1.py -k b1_m1 # B1-M1 fails today
```
