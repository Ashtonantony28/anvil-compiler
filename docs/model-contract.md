# Target model contract (v2: SG9, ADR-004)

The numbers SG5b's compiled model must meet. The code form is `anvil/model.py` (`OUT_UNIT`, `TOL_LSB`,
`TOL_SIGMA`, `SEGMENTS`, `INPUT_SEGMENTS`, `GAIN_TOL`, `RMS_TOL`, `PEAK_LSB`, `FOOTPRINT_WORDS`, `ADDR_BITS`).

* Derive with `NANOGPT_DIR=... PYTHONPATH=sim python3 -m anvil.model_derive` (about 17 min). Add `--options` for
  ADR-004's option table.
* Re-check on held-out data with `python3 -m anvil.model`.
* Test with `NANOGPT_DIR=... python3 -m pytest -q -m torch tests/model`.

v1 (SG5a) had an int8 residual stream, N(0, 1) calibration and held for sigma in [0.5, 3] only (AUDIT-A2 H1).
v2 changes the residual stream, the calibration, the error unit, the domain and the seeds. ADR-004 has the why.

## Model

Two stacked nanoGPT Extended Blocks (03dbab8, `models.gpt.Block`, `x -> x + attn(ln_1(x)) -> + ff(ln_2(x))`):
`n_embd=32, n_head=2, block_size=T=16, bias=False, pos_embedding='none', norm_type='rmsnorm', activation='swiglu'`,
float64, eval. Block `i` is built after `torch.manual_seed(0 + i)`. As `xf/export.py::build_block` does, there are
two exporter overrides:

* **SwiGLU hidden = 64 (2·d).** nanoGPT computes `int(8·d/3) = 85` and rounds up to a multiple of 256, i.e. 256 at
  d=32. That quadruples the MLP weights (98k words for two layers) and, with activations, overflows 16-bit
  addresses.
* **RMSNorm weights ~ U(0.5, 1.5).** nanoGPT initialises them to ones, which would leave the norm-weight path
  untested.

**Quantization:** `anvil.frontend.quantize(..., residual="i16")`. This is contract 8's recipe (symmetric
per-tensor, `HEADROOM = 1.5`, (M, s) from the worst case over all inputs, SiLU/exp LUTs as xf builds them), except:

* **i16 residual stream.** The input `x`, the residual adds `L<i>.r1` / `L<i>.out` and the branch outputs that
  feed them (`L<i>.y`, `L<i>.down`) are i16 at scale `1.5 · max-abs / 32767`, one value per 32-bit word.
  Everything else is int8.
* **Wide rmsnorm.** RMSNorm of an i16 row sums `(x² + 2^(c-1)) >> c` (c = 5). The sum starts at
  `ss_init = round(eps·d / (sx²·2^c))`. The rsqrt LUT is eps-free and its shift is chosen under the tight
  per-(index, k) bound.

Multi-head is head-batched (ADR-003). IR syms are `L<i>.<seg>` with segments
`h1 q k v qh kh vh kT scores p o om y r1 h2 gate up act down out`. The input is `x` and the output is `L1.out`.

**Calibration:** `contract_calib()`, 2^16 N(0, 1) inputs from `CALIB_SEED`. Each is scaled by its own sigma,
log-uniform over [0.02, 3], drawn from a separate stream of the same seed.

## Data (disjoint seeds)

| Use | Seed | Size |
|---|---|---|
| Calibration (scales) | `CALIB_SEED = 0`, N(0, 1) · sigma, log-uniform sigma over [0.02, 3] | 2^16 |
| Derivation (TOL, SEGMENTS, INPUT_SEGMENTS) | `DERIVE_SEED = 2` (v1 used 1) and streams `(2, k)` | 200k domain + sets below |
| Held-out (check only) | `HELDOUT_SEED = 20261107`, `+ 1` for the other families | 200k (CLI), 20k (pytest) |

The held-out seed is new for v2. v1's 20261106 was retired because it was looked at while choosing the SG9 fix.
`anvil/model_derive.py` asserts that it never draws from `HELDOUT_SEED`.

Errors are input-quantization-free: the float model runs on `dequant(x_q)`. The i16 input has a scale of about
4e-4, so a sigma = 0.02 input is about 50 LSB RMS. v1's int8 input put it at about 0.5 LSB, mostly zeros.

## Unit

Errors are in **OUT_UNIT = 0.15846** (float). That is 1/127 of the v2 model's calibrated `L1.out` range: the LSB
an int8 output over that range would have.

* `L1.out` is i16, so its own LSB is 258× smaller and would make TOL meaningless.
* The unit is fixed. `model.measure` uses it for every model, including ones calibrated differently (the
  A2-H1 test's matched calibration). So TOL states one absolute accuracy.
* For comparison, v1's unit was 0.0686, and v1's TOL of 6 LSB is 0.41 absolute = 2.6 OUT_UNIT.

## Bounds

**End to end: TOL = 5.0 OUT_UNIT** (0.79 absolute). It is a per-input max |error| of `L1.out` on inputs
`x = σ · z` with `σ ∈ [0.02, 3]`. `z` is per element N(0, 1), Laplace or Student-t(5) at unit variance, or N(0, 1)
with any one channel ×4.

Rule (`model_derive`): TOL = max(1.5 × climbed max, 1.25 × worst sampled max), rounded up to 0.5. The derivation
measurements (OUT_UNIT):

| Set (DERIVE_SEED) | n | max error |
|---|---|---|
| Domain: N(0, 1) · σ, σ log-uniform over [0.02, 3] | 200k | **1.779** (99.99 % quantile 1.253), peak 97.1 |
| σ = 0.02 / 0.05 / 0.2 / 1 / 3 | 40k each | 1.624 / 1.699 / 1.163 / 0.725 / 0.643 |
| Laplace / Student-t(5) / channel ×4, unit variance | 20k each | 0.669 / 0.658 / 0.686 |
| Laplace / Student-t(5) / channel ×4, σ = 0.02 | 20k each | 1.566 / 1.562 / 1.499 |
| Matched calibration (re-calibrated on `calib_inputs() · σ`), σ = 0.02 / 0.05 / 0.2 / 1 | 20k each | 1.645 / 1.590 / 1.132 / 0.697 |
| Hill climb from the 4 worst domain inputs (integer inputs, 60 × 128 candidates each) | — | **3.031** |

TOL = max(1.5 × 3.031, 1.25 × 1.779) = 4.55, rounded up to 5.0.

* The climb finds rows whose residual is small against its calibrated range (an early token at σ ≈ 0.04). There,
  the int8 attention internals (`o`, `om`) limit `y`'s absolute precision, and the next RMSNorm amplifies it.
  That is the remaining error mode of the i16 residual, and the reason TOL is not tighter.
* In the matched-calibration models' own (smaller) output LSB, the matched sets above are 7.24 / 6.99 / 5.03 /
  1.60.

Held-out (`python3 -m anvil.model`; HELDOUT_SEED, never used above):

| Set | n | max error |
|---|---|---|
| **Domain, σ log-uniform over [0.02, 3]** | **200k** | **1.675**, 0 over TOL, peak 106.9, no segment violation |
| N(0, 1) · σ, σ = 0.02 / 0.05 / 0.2 / 1 / 3 | 20k each | 1.480 / 1.635 / 1.167 / 0.689 / 0.626 |
| Laplace / Student-t(5) / channel ×4 (unit variance) | 20k each | 0.670 / 0.650 / 0.653 |
| Laplace / Student-t(5) / channel ×4 at σ = 0.02 | 20k each | 1.472 / 1.527 / 1.715 |
| Matched calibration σ = 0.02 / 0.05 (the A2-H1 setting) | 20k each | 1.735 / 1.379 (own LSB 7.64 / 6.06) |

For comparison on the same footing, v1 measured 3.60 OUT_UNIT at σ = 0.02, 1.33 at σ = 1 and 1.17 at σ = 3
(ADR-004, option A). v2's sampled maxima are about half of v1's everywhere.

**Per segment, pooled** (contract 8's rule): on held-out inputs over the domain, every segment's least-squares
gain `Σ(dequant·ref) / Σ(ref²)` stays within **±0.0025** of `SEGMENTS[k][0]`, and its RMS error within **1.1×**
`SEGMENTS[k][1]`. RMS is in the segment's own int8-equivalent LSB: the LSB for i8, scale · 32767/127 for i16.
`scores` is compared on causal entries only, and `p` against the float softmax. Values on the 200k derivation
inputs (gain / RMS):

| seg | L0 | L1 |
|---|---|---|
| h1 | 1.0008 / 0.294 | 0.9989 / 0.844 |
| q (= qh) | 1.0009 / 0.466 | 0.9989 / 1.037 |
| k (= kh = kT) | 1.0012 / 0.447 | 0.9989 / 1.114 |
| v (= vh) | 1.0005 / 0.465 | 0.9988 / 1.099 |
| scores | 1.0017 / 0.543 | 0.9976 / 1.025 |
| p | 1.0001 / 0.262 | 1.0000 / 0.366 |
| o (= om) | 0.9983 / 0.391 | 0.9992 / 0.631 |
| y (i16) | 0.9983 / 0.320 | 0.9994 / 0.510 |
| r1 (i16) | 1.0000 / 0.046 | 0.9999 / 0.111 |
| h2 | 0.9998 / 0.532 | 0.9989 / 0.720 |
| gate | 0.9998 / 0.733 | 0.9986 / 0.887 |
| up | 0.9999 / 0.771 | 0.9991 / 0.906 |
| act | 0.9980 / 0.382 | 0.9990 / 0.428 |
| down (i16) | 0.9979 / 1.236 | 0.9988 / 1.275 |
| out (i16) | 1.0000 / 0.081 | 0.9999 / 0.137 |

* The residual stream's RMS errors are now 0.05-0.14 int8-equivalent LSB (v1: 0.33-0.64).
* L1's int8 segments carry more error than v1's (h1 0.84 vs 0.51). These are pooled over the whole domain, and
  small-σ inputs make the norms work on branch-dominated rows (see the climb above).

**Per segment, per input** (A2-M1: what a single compiled run is checked with). `INPUT_SEGMENTS[k]` =
(bound on the gain projection `(gain_i − SEGMENTS gain) · ‖ref_i‖`, bound on the RMS error), both in LSB. They
are 1.5× and 1.25× the extremes over the 200k derivation inputs.

* The projection is the error along the input's own reference. Rounding noise barely projects there, and a gain
  error grows it with the signal.
* `g2_report.check` applies it to every readable segment of every compiled run (with `keep="all"`, all 40). So
  does the G2 acceptance in `tests/g2`.
* It flags a 3 % output gain error on every unit-variance input (`tests/model`) and in the compiled G2 path
  (AUDIT-A2 `test_m1_...`).
* On inputs with a small output norm (σ ≲ 0.3), a 3 % gain error is below the per-input floor. The pooled rule
  above still sees ≥ 1 % on the graph interpreter.

**No saturation:** every non-softmax activation's float peak stays ≤ 127.5 of its int8-equivalent LSB on the
held-out inputs (derivation peak 97.1, held-out 106.9).

**Overflow:** `prove_no_overflow` proves every int32 intermediate of all 30 ops with one (28 requant ops,
2 softmaxes) over all inputs of their dtypes: int8, and the whole int16 range for the residual values. The worst
is 2,077,547,776 (< 2^31 − 1).

* The wide rmsnorm is bounded per reachable (LUT index, shift k) pair: `|x| ≤ sqrt(((i + 1)·4^k − e)·2^c)`
  (`rms_xmax_table`).
* `tests/lower/test_wide_residual.py` checks the bound against adversarial rows and full-range int16 inputs.

## q/k precision (AUDIT-6)

int8 `q`/`k` hold with two heads: `scores` gain is within 0.0025 of 1 in both layers. Since L1's attention now sees
small-σ rows too, its RMS is 1.03 LSB pooled (v1 0.80). The error that matters end to end is the attention output
`y`'s absolute precision on rows whose residual is small, not `scores` (above).

## Footprint

One element per 32-bit word (i16 values included), every graph value and kernel temp in its own buffer, no reuse:
weights 20,608 + LUTs 5,120 = 25,728 const words, 24,064 activation words, 256 temp words: **50,048 words**, i.e.
`DATA_MEM_ADDR_BITS = 16`. SG4b's liveness-based planner places it in about 30k words (`docs/g2-report.md`).
