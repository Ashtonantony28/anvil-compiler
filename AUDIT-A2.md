# AUDIT-A2: numerics, performance-claim honesty, API and docs

Independent audit of Anvil at `5eaf3d6` (main), 2026-11-07. Lenses: (A) numerics, (B) honesty of the
performance claims (G1, G3, G4), (C) API, docs and reproducibility. Compiler correctness by fuzzing and mutation is
covered by audit A1 and is not repeated here. One correctness bug turned up anyway while checking that `make check`
works from a clean checkout (A2-M8).

Repros: every critical, high and medium finding has a strict-xfail test in `tests/audit/test_audit_a2.py`. Each test
passes once its finding is fixed. Seven of them run in `make check` and two are marked `torch`:

```bash
python3 -m pytest -q tests/audit/test_audit_a2.py -m "not torch and not rtl"          # 7 xfailed, ~2 s
NANOGPT_DIR=/home/claude/nanogpt python3 -m pytest -q -m torch tests/audit/test_audit_a2.py   # 2 xfailed, ~2 min
python3 -m pytest -q --runxfail tests/audit/test_audit_a2.py ...                          # shows each failure
```

No code, docs or data were changed. The only files added are this report and the test file.

## Summary

| id | severity | finding |
|---|---|---|
| A2-H1 | high | Below σ = 0.5 the error exceeds TOL even with matched calibration (7.8–8.4 LSB at σ = 0.05). nanoGPT's own embedding init is σ = 0.02. |
| A2-M1 | medium | TOL = 6 LSB cannot see a 3 % output gain error (3.7 LSB). The compiled-model acceptance (G1/G2) has no per-segment gain check. |
| A2-M2 | medium | "G4 met" rests on tier 1, an RTL replay that costs 1.22× Verilator's wall time in total. The defensible G4 result is tier2_static. |
| A2-M3 | medium | Held-out G4 kernels duplicate training kernels: 121 of 2,221 held-out programs have a feature-identical training twin (`xf_attention.h` = `xf_mlp.h`, 64 of 127). |
| A2-M4 | medium | `anvil.compile` on graph IR skips `prove_no_overflow`. An overflowing graph compiles and silently wraps. |
| A2-M5 | medium | `run(x, quantized=True)` accepts out-of-int8 and float inputs and returns wrong outputs with no error. |
| A2-M6 | medium | `anvil.compile(graph_module)` without `consts` compiles and runs on unset weight memory. |
| A2-M7 | medium | Unknown keywords are swallowed on the graph path, so `shedules="tiled"` silently compiles the defaults. |
| A2-M8 | medium | `make check` fails from a clean checkout: hypothesis finds `split(i,3,mask)` then `compute_at(stat.i, i.o)` reading out of bounds. |
| A2-L1 | low | An off-by-one rsqrt LUT index in all four RMSNorms passes every check (gain −0.0012, `scores` −0.0024 inside ±0.0025). |
| A2-L2 | low | The TOL/SEGMENTS derivation (DERIVE_SEED sweep, hill climb, seed-99/seed-7 families) has no code. `derive_inputs` is never called. |
| A2-L3 | low | The overflow proof is graph-level only. Codegen idioms are wrap-safe by construction, but no proof or test states this. |
| A2-L4 | low | Under DCACHE_LINES=16, cycles depend on the input (≤ 44 cycles, ≤ 0.1 %). The reports quote fixture input 0 only. |
| A2-L5 | low | The G3 ratios use the hand *tiled* chain: mm1x8x8 shows 0.30 (0.87 vs hand naive); attention is 0.60 vs the best hand kernel per op. |
| A2-L6 | low | The G4 held-out configs mostly recombine knob values already in training; only DCACHE_LINES=8 is a new value. Within-kernel τ of tier2_static at h3 is 0.732 (< 0.8). |
| A2-L7 | low | codegen-bench's hand cycles at DCACHE_LINES=16 (from `sim/bench/xf-mlp.json`) differ from G1's hand chain on the same config (48,797 vs 48,825). |
| A2-L8 | low | Misuse gives unclear or no errors: a transposed same-size input is reshaped silently, NaN becomes 0, a dict with an unknown kernel is ignored, a bad template name raises JSONDecodeError, `schedules=3` and `params={...}` raise AttributeError. |
| A2-L9 | low | Docs: `python3 -m anvil.model` (model-contract "Reproduce with") needs `PYTHONPATH=sim`. README has no install or usage. |

There are no critical findings.

## Defensible headline claims

* **G1.** From the nanoGPT Block, with no hand-written assembly, both xf workloads run golden-equal on iverilog RTL.
  The compiled chains are 0.87× (MLP, tiled) and 0.80× (attention, tiled) of the hand chains' cycles at identical
  GpuParams. I re-measured these on Verilator and iverilog: 35,084 / 40,329 exactly. The cycles do not depend on the
  input at the xf default and COALESCE=1. Read the ratio as "compiler vs hand kernels with different algorithms per
  op": rmsnorm uses 4 threads compiled vs 32 hand, and the requant constants are immediates in compiled code vs
  memory loads in the hand kernels.
* **G2.** The 2-layer d=32 model is bit-exact on isasim, Verilator and iverilog vs the graph interpreter. Its error
  vs PyTorch is ≤ 2.3 LSB on 32 held-out inputs, and the contract numbers reproduce exactly (200k held-out max 3.455).
  The accuracy claim holds only for σ ∈ [0.5, 3] (A2-H1), and the TOL check alone does not bound gain errors (A2-M1).
* **G3.** The tuned MLP chain takes 24,720 cycles, 0.61× the hand tiled chain and 0.52× hand naive. I re-measured it
  on Verilator and iverilog and it is input-independent at the default config. Attention takes 22,549 cycles, 0.57×
  hand tiled and 0.60× the best hand kernel per op. For mm1x8x8 the honest ratio is 0.87× hand naive, not 0.30×.
* **G4.** The learned model without the event model (tier2_static) has pooled Kendall τ of 0.956–0.999 on held-out
  kernels × held-out configs, MAPE 2.6 %. It guides the tuner to within 5 % with 6.2 % of random search's
  measurements. Its within-kernel τ reaches 0.732 at h3. Tier 1's τ = 1.000 is an RTL replay that is no cheaper than
  Verilator, so it should not headline G4 (A2-M2). The held-out kernels are partly duplicates of training kernels
  (A2-M3).

## A. Numerics

### A2-H1 (high): the contract fails for small inputs, even with matched calibration

The contract's domain is σ ∈ [0.5, 3] and the open issue (PLAN SG5a) attributes the failure below 0.5 to the
int8 residual stream. I checked whether it is only a calibration mismatch, since the model is calibrated at σ = 1.
It is not:

| calibration σ | input σ | n | max error (LSB) |
|---|---|---|---|
| 1 | 0.5 / 0.3 / 0.2 / 0.05 | 10k | 3.73 / 4.48 / 4.70 / 6.87 |
| 0.2 | 0.2 | 10k | **6.14** |
| 0.05 | 0.05 | 10k / 4k | **8.39 / 7.81** |

RMSNorm is scale-invariant. At small σ the residual stream is dominated by the attention/MLP branch, and the int8
rounding of the small skip path is re-amplified by the next RMSNorm. Recalibrating does not help. This matters
because nanoGPT initialises `wte` with std 0.02, so a nanoGPT model's first Block sees inputs of roughly σ = 0.02,
well outside the domain where the end-to-end accuracy claim holds. The limit is documented, so this is not hidden.
But the contract's claim is narrower than "the model runs within TOL": it holds only for a input scale that the
target model itself does not produce at init. I rate it high: it bounds where the compiler's numerics can be used
at all, and the PLAN's own fix (an int16 residual or per-token scales) changes contracts 3 and 8.

Repro: `test_h1_tolerance_holds_for_small_inputs_with_matched_calibration` (torch, ~110 s), or
`M.measure(M.quantize_model(m, calib=M.calib_inputs()*0.05), M.heldout_inputs(4000)*0.05, segments=False)`.

### A2-M1 (medium): TOL alone does not bound gain errors, and compiled acceptance has no gain check

I injected bugs into the quantized target model and ran all three checks. "Held-out" means 4k inputs; "G2 32"
means the same check G2 runs; "segments" is `segment_errors` on 4k held-out inputs.

| injected bug | held-out max | G2 32 max | segment violations |
|---|---|---|---|
| none | 3.079 | 2.300 | — |
| `L1.out` Ma, Mb × 1.01 (output gain +1 %) | 3.079 | 2.737 | L1.out |
| `L1.out` Ma, Mb × 1.03 (output gain +3 %) | 3.718 | 4.077 | L1.out |
| `L0.out` Ma × 1.01 (skip path) | 2.829 | 2.514 | L0.out, L1.r1, L1.out |
| `L0.r1` Ma × 1.01 (skip path) | 3.035 | 2.492 | L0.r1, L0.out, L1.r1, L1.out |
| `L0.r1` Mb × 1.01 (attention branch) | 3.122 | 2.356 | none (L0.r1 gain moves < 3e-4) |
| `L1.down` M × 1.01 | 3.079 | 2.300 | L1.down |
| SiLU LUT × 1.05 (`L1`) / × 1.03 (`L0`) | 3.079 / 2.937 | 2.688 / 2.344 | act, down |
| SiLU LUT × 1.01 (both layers) | 3.079 | 2.300 | none: int8 rounding absorbs it, act gain moves < 3e-4 |
| rsqrt LUT read one index late (all four norms) | 3.079 | 2.356 | none (see A2-L1) |

The per-segment gain/RMS bounds work. They catch every ≥ 1 % gain error that actually reaches a segment, and the
existing `test_segment_check_catches_a_gain_error` is right, if synthetic. But the end-to-end TOL catches none of
these bugs: a 3 % gain error on the output stays at 3.7 LSB, below TOL = 6. The segment checks run only in
`tests/model` on the graph interpreter. The compiled-model acceptance (G1 `compiled_chain`, G2 `g2_report.check` /
`tests/g2`) checks bit-exactness against the graph interpreter plus TOL, so the composition holds only while
`tests/model` and `tests/g2` quantize identically. Any path that builds the `Quantized` differently, such as a
user's own calibration, `anvil.compile(module, …, **quant_kw)` or a future `schedules="tuned"` rewrite of
consts, has no gain check at all. Recommendation: run the per-segment gain rule in `g2_report.check` (or in a
`CompiledModel.verify`) on the readable segments.

Repro: `test_m1_compiled_acceptance_catches_a_3pct_output_gain_error` (torch, ~20 s).

### Held-out seed hygiene: OK

`HELDOUT_SEED` (20261106) is used only in `model.heldout_inputs`, `nongaussian_sets` (as seed + 1),
`g2_report.inputs` and checks. It is not used by calibration (`CALIB_SEED = 0`), tuning (the tuner verifies on
seeded uniform int8 inputs) or the cost dataset. `python3 -m anvil.model` reproduces every held-out number in
`docs/model-contract.md` exactly: 3.455 / 0 over TOL / peak 104.8, and 3.666 / 2.894 / 3.518 / 3.061 / 3.357.
I cannot verify from the repo that TOL was frozen before the held-out set was first measured (A2-L2).

### A2-L1 (low): an off-by-one rsqrt index passes every check

Reading every RMSNorm's rsqrt LUT one entry late (`lut[i+1]`) shifts each RMSNorm-fed segment's gain by about
−0.0012, and `scores` (q·k) by −0.0024. That stays inside GAIN_TOL = ±0.0025, so no check fires. The impact is
below quantization noise (≈ 0.1 %), so the rating is low. It shows the detection floor: a systematic bias of
≤ 0.25 % per segment is invisible by design. The bit-exact fixture tests catch such bugs only in code shared with
xf (d = 8, one head).

### A2-L2 (low): the tolerance derivation is not reproducible from the repo

`docs/model-contract.md` derives TOL = 6 and SEGMENTS from a 200k DERIVE_SEED sweep, a hill climb from the 4 worst
inputs (60 × 128 candidates), σ = 0.5/2/3 at seed 99 and the Laplace/t5/channel families at seed 7. None of this
has code. `anvil.model.derive_inputs` is defined and never called, and `main()` only runs the held-out check.
`SEGMENTS` is hard-coded. A reader cannot regenerate the numbers the contract is built on, nor confirm that the
held-out set played no part in them.

Repro: `grep -rn "derive_inputs\|climb" anvil tests` (the only hit is the definition).

### A2-L3 (low): the overflow proof covers graph-level intermediates only

`prove_no_overflow` bounds every graph-level int32 intermediate, including the rounding term
(`acc*M + 2^(s-1)`, `x*w*r*M + 2^(s+kmax-1)`, `e*one + S/2`). Generated code has more intermediates: `min(a,b) =
a+b−max`, `select = y + c·(x−y)`, `lt = nz(max(a,b) − a)`, `nz(d) = d div d`, address terms, and the
register-shift rounding term `(0 − SRA(INT_MIN, 31−S)) div 2`. I checked each by reading and found them correct
under wrap-around words and unsigned DIV, including S = 31: 0 − INT_MIN = 2^31, unsigned div 2 = 2^30. The only
sign-sensitive steps (SRA, signed MAX in clamp) see exactly the proven values. So nothing overflows harmfully, but
this argument is not written down or tested. A future idiom using signed compare on a sum would break silently.
The proof is also not run on the graph-IR compile path (A2-M4).

## B. Performance-claim honesty

### Independent re-measurement (G1 / G3)

I ran `anvil.compile` on the `from_image` graphs and `g1_report.hand_chain` through `anvil.runtime.run` on
Verilator. There is no result cache: `.anvil-cache/runs` holds only temp dirs.

| workload | config | hand naive | hand tiled | default | tiled | tuned (replayed from `anvil/tune/best`) |
|---|---|---:|---:|---:|---:|---:|
| mlp_block | default | 47,610 | 40,329 | 44,422 | 35,084 | 24,720 |
| attention | default | 44,163 | 39,498 | 38,974 | 31,668 | 22,549 |
| mlp_block | COALESCE=1 | 47,610 | 40,329 | 44,422 | 35,084 | 24,720 |
| attention | COALESCE=1 | 44,163 | 39,498 | 38,974 | 31,668 | 22,333 |
| mlp_block | DCACHE_LINES=16 | 48,825 | 40,846 | 45,644 | 35,616 | 25,377 |
| attention | DCACHE_LINES=16 | 45,272 | 39,901 | 40,066 | 32,066 | 22,897 |

Every number equals `docs/g1-report.md` / `docs/g3-report.md`. On iverilog RTL, the MLP tuned chain gives 24,720,
tiled 35,084 and hand tiled 40,329, all golden-equal. Chain totals count every launch: the job's launch count equals
the sum of `CompiledModel.kernels[*].launches`, and the sum of all launch cycles equals `total_cycles`. The hand and
compiled chains run through the same `run()` with the same GpuParams, each as one Job, reset between launches.
The comparisons are apples to apples at the chain level. The memory layout differs (planner vs xf image), which is
inherent.

### A2-L4 (low): data dependence

I ran each preset on fixtures 0–3, two uniform-random int8 inputs and zeros. Cycles are identical for every input at
the default and COALESCE=1 configs, including the tuned schedules, so the tuned schedules are not overfit to fixture
data. At DCACHE_LINES=16 they vary by up to 44 cycles (≤ 0.1 %), for the hand chains as well (48,825–48,844). Tuned
MLP is invariant there and tuned attention varies by ≤ 8. The reports quote input 0 without saying so.

### A2-L5 (low): which hand chain the ratios use

The G3 table's "tuned / hand tiled" column gives mm1x8x8 = 0.30. The hand tiled kernel (3,848) is the wrong
baseline for a 1×8×8 matmul: against hand naive (1,324) the tuned kernel is 0.87×. For attention, the hand tiled
chain uses tiled `scores` (4,601), worse than naive (2,969). The best hand kernel per op totals 37,866, so tuned
attention is 0.60× of that rather than 0.57×. For MLP, best-of-hand equals hand tiled (40,329), so the headline
0.61× stands.

### A2-M2 (medium): what "G4 met" rests on

Tier 1 is an event-level replay of the RTL state machines and is cycle-exact on 43,818 of 43,826 rows. Its
τ = 1.000 and 1-measurement-per-pool tuning are what PLAN's "G4 met" headline quotes. In `data/cost/rows.jsonl.gz`
the tier-1 prediction (`t_model`) totals 2,873 s against 2,345 s of Verilator (`t_sim`), a 1.22× aggregate with
median per row 0.71×. That excludes the functional profile that tier 1 needs, which `dataset._work` does not time.
So tier 1 is not cheaper than measuring, and the cost model saves nothing whenever Verilator is available. My spot
timings on the whole MLP chain: profile 4–6 ms, closed form 1–2 ms, tier-1 event model 22–32 ms, Verilator
16–71 ms, isasim 10–12 ms.

The PLAN caveat ("about as much wall time") understates it, but it is disclosed. The defensible G4 result is
tier2_static. It uses profile and closed-form features, roughly 2–9× cheaper than a Verilator run here. Its pooled
τ is 0.956–0.999 on the four held-out configs, its within-kernel mean is 0.871–0.998 (min 0.732 at h3, below the
0.8 target), MAPE is 2.6 %, and its guided-search ratio is 0.062. That meets G4's pooled-τ and 10 % targets.
Recommendation: headline tier2_static and report the within-kernel minimum as a miss at h3.

Repro: `test_m2_tier1_prediction_is_cheaper_than_the_measurement_it_replaces`, which asserts tier-1 time ≤ 0.5×
Verilator time over the dataset.

### A2-M3 (medium): held-out kernels duplicate training kernels

I matched every held-out-kernel program's full profile-feature vector, warp count and program size against the
training kernels:

| held-out kernel | programs with an identical training twin | twin kernel |
|---|---|---|
| xf_attention.h | 64 / 127 | xf_mlp.h (same rmsnorm 4×8) |
| xf_attention.scores | 14 / 128 | syn.qlinear_4x4x8 |
| xf_attention.kT | 12 / 128 | syn.transpose_8x4 |
| xf_attention.o | 11 / 128 | syn.qlinear_4x8x4 |
| model.L0.scores | 11 / 72 | model.L0.o (same 2×16×16×16 qmatmul) |
| model.L0.r1 | 9 / 72 | model.L0.out (same 16×32 add_requant) |
| total | 121 / 2,221 | |

The headline slice (held-out kernels × held-out configs) is protected by the config split, since no training row
is at h0–h3. The `heldout_kernels` slice (train configs) in the report contains exact duplicates of training rows.
`xf_attention.h` is not a held-out kernel in any meaningful sense, and `model.L0.r1`/`L0.out` and
`L0.scores`/`L0.o` are the same op at the same shape. Recommendation: split by (op, shape) equivalence class, not
by kernel name.

Repro: `test_m3_heldout_kernels_have_no_feature_identical_twin_in_train`.

### A2-L6 (low): held-out configs and within-kernel τ

h0 (COALESCE + DCACHE16 + ICACHE16 + 4 cores, lat 3), h2 and h3 recombine knob values all present in training
(c09, c10, c11, c13, c14). h1 introduces the only unseen value (DCACHE_LINES=8). This is a fair interpolation test
but a weak test of generalization to new hardware. tier2_static's within-kernel τ, which is the ranking a tuner
consumes, is 0.732 for the worst h3 kernel. The guided-search pools are small (N = 8–80), and a perfect model needs
1 measurement per pool, so "80 vs 1,322" is the floor, not a model property (the report notes this).

### A2-L7 (low): two different "hand" numbers for the same chain

`docs/codegen-bench.md` takes hand cycles from `sim/bench/xf-mlp.json`: at DCACHE_LINES=16 the naive total is 48,797
and tiled 40,795. `docs/g1-report.md` measures the same hand chains through `anvil.runtime` and gets 48,825 and
40,846. The difference is the args/image layout under the L1-D. The two docs disagree without saying why.

## C. API, docs and reproducibility

### Report regeneration

| artifact | command | result |
|---|---|---|
| `docs/codegen-bench.md` | `anvil.codegen.bench.main(path=…)` | identical |
| `docs/g1-report.md` | `anvil.g1_report.main(path=…)` | identical |
| `docs/model-contract.md` held-out numbers | `PYTHONPATH=sim python3 -m anvil.model` | identical (needs PYTHONPATH, A2-L9) |
| `docs/g3-report.md` | `anvil.tune.g3_report --configs default --fresh-cache --out … --best …` | see below |
| `docs/g4-report.md` | `anvil.cost.g4_report --no-train` (paths redirected) | see below |

Note: `g4_report.svg(curves, path=SVG)` binds its default path at definition time. Patching `G.SVG` does not
redirect it, and a dry run of the script always rewrites `docs/g4-curves.svg`. The report scripts have no
`--check`/`--out` mode except g3's.

G3 with the default config only, a fresh measurement cache, seed 0 and budget 200: the per-kernel tables and the
chosen schedules are identical, so same seed gives the same choice. Only the tuning wall-time column differs (41 s
became 111 s, under load). G4 with `--no-train` against the committed models: identical apart from the "Online"
section, which needs `--online` and contains wall times, so it cannot be reproduced byte for byte.
`docs/g4-curves.svg` and the repo were left untouched (checked with `git status` and md5).

### A2-M4 (medium): no overflow proof on the graph-IR path

`prove_no_overflow` runs only inside `frontend.quantize`. `anvil.compile((module, consts))` and
`anvil.compile(Quantized)` never call it. Multiplying the MLP fixture's `gate` M by 4096 gives a graph the proof
rejects (`acc*M + round reaches 7.29e12`) and the graph interpreter rejects (`OverflowError`). `anvil.compile`
accepts it and `run` returns wrapped garbage with no error.

Repro: `test_m4_compile_of_graph_ir_proves_no_overflow`.

### A2-M5 (medium): `quantized=True` inputs are not validated

`cm.run(x * 50, quantized=True)` writes values outside int8 into memory and returns outputs. The graph interpreter
raises `ValueError: … is not int8`. Float arrays with `quantized=True` are silently truncated by
`np.asarray(v, dtype=np.int64)` (x + 0.7 gives wrong outputs). Every overflow guarantee assumes int8 inputs.

Repro: `test_m5_run_rejects_out_of_range_integer_inputs`.

### A2-M6 (medium): compiling a module without its consts

`anvil.compile(module)` for a module with non-inline `graph.const`s, without `consts=`, compiles and runs. The
weights and LUTs are whatever the job leaves in memory. The fixture MLP returns `[6 15 6 -24]` vs the golden
`[6 16 7 -25]`, near enough to pass a casual look.

Repro: `test_m6_compile_without_consts_is_an_error`.

### A2-M7 (medium): typos in keywords are swallowed

`compile(model, …, **quant_kw)` forwards extra keywords to `quantize` only for torch modules. On the graph path they
are dropped, so `anvil.compile((mod, consts), shedules="tiled")` compiles the default schedules with no warning. On
the torch path the same typo reaches `quantize` (TypeError there), so the behaviour also differs by input type.

Repro: `test_m7_unknown_keyword_is_an_error`.

### A2-M8 (medium): `make check` fails from a clean checkout

I exported `git archive HEAD` plus `sim/` into an empty directory and ran `make check`: 601 passed, **1 failed**,
in 4 m 39 s. The failure is
`tests/schedule/test_random_schedules.py::test_random_primitive_sequences_bit_exact[mlp_block-h]`. Hypothesis
found `Schedule(h, Target()).apply("split", "i", 3, "mask")` followed by `apply("compute_at", "stat.i", "i.o")`.
The schedule accepts the pair as legal, then the tensor interpreter reads `x[4, 0]` of a `[4, 8]` buffer: the
masked tail is not propagated into the computed-at stage. In the working tree the same test passes because the
local `.hypothesis` database and luck steer the search, so `make check` is effectively nondeterministic. This is a
compiler-correctness bug (A1's lens); it is recorded here because it breaks the PLAN's Verify commands from a
clean checkout. It is a deterministic repro and needs no hypothesis.

Repro: `test_m8_masked_split_then_compute_at_is_rejected_or_bit_exact`.

### A2-L8 (low): unclear or missing errors on misuse

| call | result |
|---|---|
| `cm.run(x.T-shaped, quantized=True)` (same size, wrong shape) | silently reshaped, wrong output |
| `cm.run(np.full(shape, nan))` | RuntimeWarning only; NaN becomes 0 |
| `schedules={"nosuch": "matmul_tiled"}` | unknown kernel ignored |
| `schedules={"gate": "no_template"}` | `JSONDecodeError` (the name is parsed as a JSON trace) |
| `schedules=3` | `AttributeError: 'int' object has no attribute 'get'` |
| `params={"NUM_CORES": 2}` | `AttributeError` |
| unknown backend, bad preset, bad `keep`, wrong input count, address space too small, WORD_BITS ≠ data bits | clear errors (good) |

### A2-L9 (low): docs

`docs/model-contract.md` says "Reproduce with `NANOGPT_DIR=... python3 -m anvil.model`", which fails with
`ModuleNotFoundError: No module named 'xf'` unless `PYTHONPATH=sim` is set. The other report docstrings include
it. `README.md` is three lines, with no install, environment, `anvil.compile` usage or test commands. The usage
lives in docstrings and `PLAN.md` only. The PLAN Verify commands themselves are correct as written, given the
Environment section, apart from A2-M8.
