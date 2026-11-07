# AUDIT-B2: honesty of the G1–G5 claims, numerics of model contract v2

Independent final audit (round B, lens 2) of Anvil at `6461764` (main), 2026-11-07. I wrote none of this code.
Scope: regenerate every report and diff it, re-measure a sample of headline numbers on Verilator and iverilog,
check that each headline points at a committed or cached measurement, check that comparisons are like for like,
and re-check the numerics of contract v2 (SG9, ADR-004). Compiler correctness since audit A is lens 1
(AUDIT-B1) and is not repeated here.

Repros: each medium finding has a strict-xfail test in `tests/audit/test_audit_b2.py`. There are no critical or
high findings. Two of the tests run in `make check` and one is marked `torch`:

```bash
python3 -m pytest -q tests/audit/test_audit_b2.py -m "not torch and not rtl"                 # 2 xfailed, < 1 s
NANOGPT_DIR=/home/claude/nanogpt python3 -m pytest -q -m torch tests/audit/test_audit_b2.py   # 1 xfailed, ~2 min
```

I changed no code, docs or data. The only files I added are this report and the test file. Scratch scripts used
below (cross-replay, v1/v2 numerics, iverilog re-runs) are described inline, so they can be rebuilt from the text.

## Summary

| id | severity | finding |
|---|---|---|
| B2-M1 | medium | Contract v2's TOL (5 OUT_UNIT = 0.79 absolute) is 1.9× looser than v1's (6 LSB = 0.41). The unfixed v1 model passes it everywhere (σ = 0.02: 4.01; domain: 3.40 OUT_UNIT), so the now-passing A2-H1 test cannot tell the defect from its fix. The fix is real, but the test does not show it. |
| B2-M2 | medium | PLAN says that after audit A "G3 ratios are reported against both hand chains". `g3-report.md` still has only "tuned / hand tiled", and mm1x8x8 still reads 0.30 (0.87 vs hand naive; A2-L5). |
| B2-L1 | low | In A2-H1's own metric (matched calibration, the model's own output LSB, limit 6), σ = 0.02 still gives 7.64 (v1 10.86). σ = 0.05 gives 5.98 (v1 7.92), which passes with almost no margin. ADR-004 and the contract say this. |
| B2-L2 | low | The G5 caveat blames C4 losing to C2 (no I-cache) on search noise. It is a hardware effect. Every one of the sweep's schedules is slower at C4 than the best at C2, for both blocks. |
| B2-L3 | low | Which parts make up the G5 knee depends on the proxy. PIPELINED is free and COALESCE costs 32 bits. "2 cores + an I-cache" is robust. "+ CO + PI, 32 lines" is not (it moves to I16, no-CO or no-PI under plausible weightings). |
| B2-L4 | low | The model's knee comes from a 9-point front whose only combinations are the blocks' knee and fastest point (`model_design()`), so "knee for every workload" is partly built in. The reduced coverage is disclosed. |
| B2-L5 | low | The G4 0.24× cost compares in-process Python with a Verilator subprocess per run (≈ 15 ms fixed overhead). Minus all fixed overhead it is ≈ 0.34×. LightGBM inference and feature extraction are not counted. The conclusion (cheaper than measuring) holds. |
| B2-L6 | low | The G5 ISA timing (a new op costs one ALU op) is stated, but its hardware implications are not. MAD reads three registers (a third register-file read port), and every extension needs a 17-bit program word and I-cache line. The cost proxy and the report leave both out. |
| B2-L7 | low | Stale or conflicting headline text. PLAN SG7 "G4 met, with a caveat" still quotes tier 1, τ = 1.000, "80 vs 1,322", 43,830 rows. The README lists reports g1–g4 only (no g5). The `docs/design.md` header still says "plan only … no implementation code has been written". |
| B2-L8 | low | The G5 "every number points at a cached measurement" check (`--verify`) reads `.anvil-cache/codesign/measure`, which is gitignored. From a clean checkout the committed records can be replayed (`tests/codesign`, 5 points), not re-verified. |
| B2-L9 | low | Held-out hygiene for v2 is clean in data, but the integer 20261107 also seeds the cost dataset's synthetic inputs (SG7, before SG9) and the G5 random config sample. The codesign sweep checks chains on `heldout_domain(1)[0]`, though `model.py` says HELDOUT_SEED is used "for nothing but the final check". |
| B2-L10 | low | The G5 baselines are slightly under-tuned compared with schedules found at other points (C2: MLP 24,844 vs 24,720; attention 22,425 vs 22,333; model 3,549,798 vs 3,530,874), so ratios against the baseline are flattered by ≤ 0.6 %. |

There are no critical or high findings. Every regenerated report matches the committed one, apart from wall
times and the `--online` section (details below). Every headline number I re-measured reproduces exactly, on
Verilator and on iverilog.

## 1. Report regeneration

The fresh tree came from `git archive HEAD` plus `sim/` in a scratch directory. The output was compared with
`diff` against the committed docs. `.anvil-cache` was shared (only Verilator builds and measurement caches are
read from it).

| artifact | command | result |
|---|---|---|
| `docs/g1-report.md` | `anvil.g1_report.main(path=…)` (NANOGPT_DIR) | **identical** |
| `docs/codegen-bench.md` | `anvil.codegen.bench.main(path=…)` | **identical** |
| `docs/g2-report.md` | `anvil.g2_report.main(path=…, rtl=False)` | identical apart from the wall-time table (machine under load: 1.4/3.67/4.74 vs 0.8/1.94/2.17 s) |
| `docs/g4-report.md` | `python3 -m anvil.cost.g4_report --no-train` | identical apart from the "Online" section, which needs `--online` (live tuning, wall times), and the size of `g4-metrics.json` in the file list |
| `docs/g5-report.md`, `docs/g5-pareto.svg` | `python3 -m anvil.codesign.report --verify --out … --svg …` | **identical** (md and svg byte for byte); `--verify` 1,081 / 1,081 kernel measurements equal the records (local cache) |
| `docs/g3-report.md` | not re-run (12.7 min tuning); A2 re-ran it on the default config. Its chain numbers are re-measured below | — |

Only g3 (`--out`) and g5 (`--out`, `--svg`) have output options. None of the scripts has a `--check` mode, and
g4 writes `docs/` and `data/cost/g4-metrics.json` unconditionally. This is A2's note and unchanged.

## 2. Independent re-measurement

I compiled from graph IR (`from_image`) with the presets and the committed tuned store, and replayed the G5
records' schedules. Hand chains came from `g1_report.hand_chain`. Every run was golden-checked or reference-checked.

| claim | source | Verilator | iverilog (RTL) |
|---|---|---:|---:|
| G1 MLP tiled | g1-report | 35,084 | **35,084** |
| G1 hand tiled MLP | g1-report | 40,329 | **40,329** |
| G1 attention tiled / hand tiled | g1-report | 31,668 / 39,498 | — |
| G3 MLP tuned | g3-report | 24,720 | **24,720** |
| G3 attention tuned | g3-report | 22,549 | **22,549** |
| G2 default / tiled total | g2-report (regenerated, 32 inputs) | 6,732,468 / 5,545,262, bit-exact, max 0.871 OUT_UNIT | (not re-run: ~16 min per preset) |
| G5 MLP knee C2.I32.CO.PI | `sweep/mlp_block/C2.I32.CO.PI.json` | 18,203 | **18,203** |
| G5 MLP baseline C2 | `sweep/mlp_block/C2.json` | 24,844 | **24,844** |
| G5 attention fastest C4.I32.CO.PI | `sweep/attention/C4.I32.CO.PI.json` | 15,553 | **15,553** |
| G5 model C2 | `sweep/model/C2.json` | 3,549,798 | — |

The comparisons are like for like. G1 and G3 run hand and compiled chains through the same `run()` at the same
GpuParams. The G5 points of one workload use the same budget, seed and ranker, and their tuned chains are
re-checked end to end. The G5 ISA table compares the `base` row through the identical fork + tier-1 pipeline,
and tier 1 equals Verilator on all three base chains (24,844 / 22,425 / 3,549,798).

## 3. Numerics of contract v2 (item 1)

All numbers below are mine. I used held-out inputs `heldout_inputs(10000) · σ` or `heldout_domain(10000)` and
`model.measure`. "v1" is contract v1's recipe (`residual="i8"`, N(0, 1) calibration, ADR-004 option A), and
"v2" is the contract model. OUT_UNIT = 0.15846, v1's LSB = 0.0686 (my v1 build: 0.0693), and v1's TOL is
6 × 0.0686 = 0.41 absolute.

| model | σ | max error, absolute | OUT_UNIT | v1 LSB | own output LSB |
|---|---|---:|---:|---:|---:|
| v1 | 0.02 | 0.636 | 4.01 | 9.27 | 9.17 |
| v2 | 0.02 | **0.235** | 1.48 | 3.42 | 1.48 |
| v1 | 0.05 | 0.467 | 2.95 | 6.81 | 6.74 |
| v2 | 0.05 | **0.259** | 1.64 | 3.78 | 1.63 |
| v1 | 1 | 0.202 | 1.28 | 2.94 | 2.91 |
| v2 | 1 | 0.098 | 0.62 | 1.43 | 0.62 |
| v1 | domain [0.02, 3] | 0.539 | 3.40 | 7.85 | 7.77 |
| v2 | domain [0.02, 3] | **0.237** | 1.50 | 3.46 | 1.50 |
| v1, matched calibration | 0.02 | 0.391 | 2.47 | 5.70 | **10.86** |
| v2, matched calibration | 0.02 | 0.275 | 1.74 | 4.01 | **7.64** |
| v1, matched calibration | 0.05 (A2-H1's setting) | 0.286 | 1.80 | 4.16 | **7.92** |
| v2, matched calibration | 0.05 | 0.216 | 1.36 | 3.14 | **5.98** |

**Is the A2-H1 fix real?** Yes, in absolute terms, for the contract model. The error at σ = 0.02 falls 2.7×
(0.636 → 0.235) and at σ = 0.05 1.8×. Over the domain it falls 2.3×. v2's sampled maximum over the whole domain
(0.237) is inside v1's own absolute TOL (0.41), and v1 is not (0.539). These numbers agree with ADR-004's option
table and the contract's held-out table to within sampling.

**Is it partly an artefact of the unit change?** The test's pass is, the improvement is not (B2-M1).

### B2-M1 (medium): the v2 TOL does not discriminate the A2-H1 defect

TOL went from 6 LSB of v1's output (0.41 absolute) to 5 OUT_UNIT (0.79 absolute), 1.9× looser. The derivation
rule (1.5 × the climbed max) is the same, and the looser bound follows from the hill climb finding 3.03 OUT_UNIT
on v2. The contract states both numbers ("v1's TOL of 6 LSB is 0.41 absolute = 2.6 OUT_UNIT"; "TOL = 5.0
OUT_UNIT (0.79 absolute)"). Nowhere does it say what follows from them:

* The unfixed v1 model passes the v2 contract at every σ I measured. It gives 4.01 OUT_UNIT at σ = 0.02 and 3.40
  over the domain, against TOL 5, with 0 of 10k inputs over TOL.
* The A2-H1 test (`test_h1_tolerance_holds_for_small_inputs_with_matched_calibration`) now measures in OUT_UNIT.
  On v1's recipe it would read 1.80 at σ = 0.05, below 5. So it passes with or without the SG9 fix and no longer
  guards the regression it was written for. PLAN's SG9 note concedes "passes partly because the unit is
  absolute". The measurement shows that the pass depends on the unit and TOL change alone.

The improvement is real (table above). But the acceptance evidence that SG9 cites (an A2-H1 test that passes,
and "TOL holds over [0.02, 3]") would also hold for the code A2 flagged. A regression back to the int8 residual
would not be caught by TOL. It would be caught only by the per-segment bounds re-derived on v2 (L0.r1 RMS 0.046 vs
v1 0.33–0.64), which I did not re-check against v1.

Repro: `test_m1_v2_tol_is_not_looser_than_v1_in_absolute_terms` (fast) and
`test_m1_v2_tol_rejects_the_v1_model_at_small_sigma` (torch: v1 at σ = 0.02 on 4k held-out inputs must exceed
TOL). Fix options: a v1-discriminating check (for example, a sampled-max bound of 2.6 OUT_UNIT, which v2 meets at
1.68 on 200k held-out inputs), or state TOL as a regression bound and add a v1-recipe negative test.

### B2-L1 (low): A2-H1's original metric still fails at σ = 0.02

With calibration matched to the input scale and errors in the model's own output LSB, the exact A2-H1 measurement
gives 10.86 → 7.64 at σ = 0.02 and 7.92 → 5.98 at σ = 0.05 (the contract's 20k held-out run: 7.64 / 6.06). This
is disclosed in the contract and ADR-004 ("in C2's own LSB, sigma = 0.05 is 5.6, below v1's 6 but not by much").
It is low because a user who calibrates to tiny inputs gets a smaller output range, so a finer unit, and the
absolute error still falls.

### Held-out seed

20261107 is drawn from only by `heldout_inputs`/`heldout_domain` (stream 20261107 and `(20261107, 1)`),
`nongaussian_sets` (+1), `g2_report.inputs`, tests and the codesign reference input. `model_derive._rng` asserts it
never derives. The sets are disjoint from calibration (0) and derivation (2, `(2, k)`). Calibration and derivation
never see it, so the held-out set is clean in data. From the repo I cannot prove that it was first drawn after TOL
was frozen: SG9 is a single commit, `11f554b`. Its derivation maximum (1.779) is consistent with the held-out
maximum (1.675). B2-L9 covers reuse of the same integer elsewhere.

## 4. G4 (item 2)

* **Is the partial miss clear?** In the report, yes. The headline line reads "Within-kernel tau … h3 0.727
  (target >= 0.8: MISSED at h3)", next to "pooled 0.956 - 0.999 (met)". PLAN's SG10 note says the same. PLAN's
  SG7 "Done — G4 met, with a caveat" paragraph was not updated. It still headlines tier 1's τ = 1.000 and the
  pre-dedupe numbers (B2-L7), and the status table says only "DONE".
* **Cost (0.24×).** Recomputed from `data/cost/rows.jsonl.gz`: over the ok rows t_model 505 s, t_sim 2,127 s,
  t_tier1 2,495 s. All rows give 505.8 / 2,128.9 / 2,497.8. The per-row profile cost is charged to every
  (program, config) row. That is the right per-prediction cost for tuning at one configuration, if anything
  conservative. Unfair in the model's favour (B2-L5): `run(job, "verilator")` spawns the harness executable and
  writes and reads JSON for each run. A least-squares fit gives t_sim ≈ 14.9 ms + cycles / 1.92 M/s, so ≈ 640 s of
  the 2,127 s is fixed per-run overhead. Without it, the ratio is ≈ 505 / 1,490 = 0.34×. LightGBM inference and
  `program_features` are not in t_model. The conclusion, that tier2_static is several times cheaper than a
  Verilator run, stands. The ratio is specific to this harness.
* **Guided-search ratio.** It is mostly a property of the pools. A perfect model needs 68 measurements (one per
  pool), tier2_static needs 69, and random needs 1,137. The report says so in its notes.

## 5. G5 (item 3)

**Search noise.** To test whether Pareto points are artefacts of search noise, I cross-replayed. For each target
point I recompiled the workload with every other sweep point's per-kernel best schedules (from
`data/codesign/best.json`, no search) and ran it on Verilator with the reference check:

| workload | target | record | best over all sources (source) |
|---|---|---:|---|
| mlp_block | C2 (baseline) | 24,844 | 24,720 (C4's), 0.995 |
| mlp_block | C4 | 25,696 | 25,696 (own) |
| mlp_block | C2.I32.CO.PI (knee) | 18,203 | 18,203 (own) |
| mlp_block | C4.I32.CO.PI (fastest) | 13,985 | 13,985 (own) |
| mlp_block | C1 / C2.L3 / C2.I32.CO.PI.L3 | 43,856 / 34,732 / 23,274 | equal |
| attention | C2 | 22,425 | 22,333 (C2.D8's), 0.996 |
| attention | C4 | 22,673 | 22,587, 0.996 |
| attention | C2.I32.CO.PI / C4.I32.CO.PI / C1 / C2.L3 / C2.I32.CO.PI.L3 | 17,152 / 15,553 / 37,517 / 31,248 / 22,308 | equal, except C1 37,345 (0.995) |
| model | C2 | 3,549,798 | 3,530,874 (C2.PI's), 0.995 |
| model | C4 | 3,012,887 | 3,012,887 (own) |

(I stopped the model's knee and fastest-point targets for time. For the blocks they are locally best.)

So the front and knee points are locally best among everything the sweep found. The baselines are
under-tuned by ≤ 0.6 % (B2-L10), so "0.73× / 0.76× / 0.64× of baseline" are flattered by at most that much.
The search-noise caveat names the wrong case. C4 is slower than C2 with no I-cache under every schedule the sweep
found: MLP best 25,696 vs 24,720, attention 22,587 vs 22,333. That is a real effect of four fetchers sharing one
program channel (consistent with the simulator's AUDIT-10 M1), not noise (B2-L2).

**Cost proxy robustness (B2-L3).** I recomputed the fronts and knees from the committed records under other
weightings, using `report.pareto`/`report.knee`:

| weighting | MLP L0 | MLP L3 | attention L0 | attention L3 | model |
|---|---|---|---|---|---|
| state bits (report) | C2.I32.CO.PI | C2.I32.CO.PI | C2.I32.CO.PI | C2.I32.CO.PI | C2.I32.CO.PI |
| + 3k or 10k bits per core (ALU/MUL/DIV logic) | same | same | same | same | same |
| I-cache bits × 0.5 | same | same | same | same | same |
| I-cache bits × 3 | C2.PI | same | C2.I16.CO.PI | same | C2.PI |
| PIPELINED + 500 bits per core | same | C2.I32 | same | same | C2.I32 |
| COALESCE + 2,000 bits | C2.I32 | C2.I32 | C2.PI | same | C2.I32 |

"Two cores" is robust in every case. The I-cache is in the knee unless I-cache bits are weighted 3×.
CO and PI are in the knee mainly because the proxy prices them at 32 and 0 bits. CO alone is worth 0.5 %
(C2.CO 0.995). The report states that "PIPELINED and WARP_POLICY are free". It does not say that this decides the
knee's composition.

**Coverage (B2-L4).** It is disclosed in the Method section and in PLAN: the model gets 10 points at budget 12.
The model's front has 5 points, and the only combinations evaluated are C2.I32.CO.PI and C4.I32.CO.PI, chosen
from the blocks' results. So "knee for every workload" is a confirmation for the model, not an independent
finding.

**ISA study.** Tier 1 is exact on the base ISA (cross-check row: 24,844 / 22,425 / 3,549,798 equal Verilator).
New ops are assumed to cost one ALU instruction, and the report says so. RTL is unchanged. "Fixed" uses base-ISA
schedules, which is conservative for the variants. "Retuned" runs the base row through the identical pipeline. The
MAD-best ranking is the same under fixed and retuned and does not involve the cost proxy. Not stated (B2-L6): MAD
reads `rd` as well, so it needs a third register-file read port, and all proposals need 17-bit program memory and
I-cache lines. Neither is costed or mentioned in the report. The table lives only in `isa_ext/isa.py`'s docstring.

## 6. README and design vs reality (item 4)

* README: install, usage and test commands are accurate (A2-L9 fixed). The report list stops at g4 and omits
  `docs/g5-report.md`.
* `docs/design.md`: G1–G5 are met as worded in PLAN, with three exceptions. G4's within-kernel τ at h3 is 0.727.
  G5's ISA proposals were evaluated "in the ISA model" with a fork plus tier 1, which is what the design asked
  for. G6 is skipped (D3). The header still says the document is an unimplemented plan (B2-L7).
* PLAN's G3 line "G3 ratios are reported against both hand chains" is not true of `g3-report.md` (B2-M2).

### B2-M2 (medium): G3's mm1x8x8 ratio

`docs/g3-report.md`'s chain table has one ratio column, "tuned / hand tiled". For mm1x8x8 it shows 0.30
(1,148 / 3,848), but the relevant hand kernel is naive (1,324): 0.87×. For attention, best-of-hand per op is 0.60×,
not 0.57× (A2-L5). After audit A, PLAN states "G3 ratios are reported against both hand chains", which the
committed report does not do. The headline MLP number (0.61×) is unaffected, because best-of-hand equals hand tiled
there. Repro: `test_m2_g3_report_states_ratio_against_hand_naive` (fast).

## Defensible headline claims

* **G1.** From the nanoGPT Block with no hand-written assembly, the MLP block and the attention head run
  golden-equal on iverilog RTL. They take 35,084 and 31,668 cycles, 0.87× and 0.80× of the hand tiled chains at
  identical GpuParams. I re-measured both on Verilator and the MLP chain on iverilog, and the report regenerates
  byte for byte.
* **G2.** The 2-layer d=32 two-head model (40 kernels, 120/84 launches, ~30k words) is bit-exact against the
  graph interpreter on isasim and Verilator (32 held-out inputs) and on iverilog (1 input). It takes
  6,732,468 / 5,545,262 cycles (default/tiled). Max error vs PyTorch is 0.87 OUT_UNIT. Over σ ∈ [0.02, 3] the
  held-out sampled max is 0.27 absolute (1.68 OUT_UNIT), 2.3–2.7× lower than contract v1's error on the same
  inputs. TOL (5 OUT_UNIT = 0.79 absolute) is looser than v1's and would also pass v1's model.
* **G3.** The tuned MLP chain takes 24,720 cycles, 0.61× hand tiled and 0.52× hand naive. Attention takes 22,549,
  0.57× hand tiled and 0.60× best-of-hand. Both were re-measured on Verilator and iverilog. For mm1x8x8 the tuned
  kernel is 0.87× hand naive.
* **G4.** Partly met. tier2_static (no event model) reaches pooled Kendall τ 0.956–0.999 on held-out kernels ×
  held-out configs, MAPE 2.6 %, and guided search to within 5 % with 6.1 % of random search's measurements. Its
  inputs cost ≈ 0.24–0.34× a Verilator run in this harness. Within-kernel τ misses 0.8 at h3 (0.727). Tier 1's
  τ = 1.000 is an RTL replay that is no cheaper than Verilator.
* **G5.** Across 104 separately tuned, bit-exact points, two cores with an I-cache is the knee on every workload
  under the state-bit proxy and under reweighted proxies. With the report's proxy that knee is
  C2.I32.CO.PI: 0.73× / 0.76× / 0.64× of baseline cycles for 1.13× the state bits. The points are locally best
  against cross-replayed schedules (baselines within 0.6 %). Four cores without an I-cache is slower than two (a
  hardware effect). In an ISA-model fork with tier-1 timing that is exact on the base ISA, and assuming one-ALU-op
  cost and no RTL change, MAD saves the most (≈ 10 % mean, 8–13 %) and all five ops together save ≈ 15–17 %.
