# Target model contract (SG5a → SG5b)

The numbers SG5b's compiled model must meet. The code form is `anvil/model.py` (`TOL_LSB`, `TOL_SIGMA`, `SEGMENTS`,
`GAIN_TOL`, `RMS_TOL`, `PEAK_LSB`, `FOOTPRINT_WORDS`, `ADDR_BITS`). Reproduce with
`NANOGPT_DIR=... python3 -m anvil.model` and check with `NANOGPT_DIR=... python3 -m pytest -q -m torch tests/model`.

## Model

Two stacked nanoGPT Extended Blocks (03dbab8, `models.gpt.Block`, `x -> x + attn(ln_1(x)) -> + ff(ln_2(x))`):
`n_embd=32, n_head=2, block_size=T=16, bias=False, pos_embedding='none', norm_type='rmsnorm', activation='swiglu'`,
float64, eval. Block `i` is built after `torch.manual_seed(0 + i)`. As `xf/export.py::build_block` does, two
exporter overrides:

* **SwiGLU hidden = 64 (2·d).** nanoGPT computes `int(8·d/3) = 85` and rounds up to a multiple of 256, i.e. 256 at
  d=32. That quadruples the MLP weights (98k words for two layers) and, with activations, overflows 16-bit
  addresses. 64 is the design's assumption ("≈ 20k words of weights", D4) and xf's ratio (16 at d=8).
* **RMSNorm weights ~ U(0.5, 1.5)** (nanoGPT initialises them to ones, which would leave the norm-weight path
  untested).

Quantization: `anvil.frontend.quantize` with contract 8's recipe (symmetric per-tensor int8, `HEADROOM = 1.5`,
(M, s) from the worst case over all int8 inputs, rsqrt/SiLU/exp LUTs as xf builds them). Multi-head is head-batched
(ADR-003): one scale per tensor across both heads, int8 `q` and `k`. IR syms are `L<i>.<seg>` with segments
`h1 q k v qh kh vh kT scores p o om y r1 h2 gate up act down out` (`qh/kh/vh` the split heads, `om` the merged
heads, `r1` the attention residual); the input is `x`, the output `L1.out`.

## Data (three disjoint seeds)

| Use | Seed | Size |
|---|---|---|
| Calibration (scales) | `CALIB_SEED = 0`, N(0, 1) | 2^16 inputs of (16, 32) |
| Derivation (TOL, SEGMENTS) | `DERIVE_SEED = 1`, N(0, 1) | 200k |
| Held-out (check only) | `HELDOUT_SEED = 20261106`, N(0, 1); `HELDOUT_SEED + 1` for the other families | 200k (CLI), 20k (pytest) |

The held-out seeds were not used to choose the headroom, the hidden width, the tolerance or the segment bounds.
Errors are input-quantization-free: the float model runs on `dequant(x_q)`.

## Bounds

**End to end: TOL = 6 output LSB** (`6 · scale(L1.out)`), per input max |error| of `L1.out`, on inputs
`x = σ · z` with `σ ∈ [0.5, 3]` and `z` per element N(0, 1), Laplace or Student-t(5) at unit variance, or N(0, 1)
with any one channel ×4.

Derived from measurements (LSB of `L1.out`):

| Set | n | max error |
|---|---|---|
| Derivation N(0, 1) | 200k | 3.490 (99.99 % quantile 2.961) |
| Hill climb from the 4 worst derivation inputs (int8, unsaturated, 60 × 128 candidates each) | — | 3.763 |
| σ = 0.5 / 2 / 3 (seed 99) | 100k each | 4.365 / 2.961 / 3.066 |
| Laplace / Student-t(5) / channel ×4 (seed 7) | 50k / 50k / 128k | 3.324 / 3.198 / 3.268 |
| **Held-out N(0, 1)** | **200k** | **3.455**, 0 over TOL, peak activation 104.8 LSB |
| Held-out σ = 0.5 / σ = 3 / Laplace / Student-t(5) / channel ×4 | 20k each | 3.666 / 2.894 / 3.518 / 3.061 / 3.357 |

6 LSB is 1.6× the climbed N(0, 1) maximum and 1.37× the worst sampled point of the domain (σ = 0.5), the same
value as contract 8's attention TOL, which was also set from a search, not a Gaussian sample (AUDIT-6).

**Outside the domain** (not covered by TOL): small inputs. The int8 residual stream (`r1`, `out`) then uses a
fraction of its calibrated range (peak 29 LSB at σ = 0.05) and the next RMSNorm re-amplifies its rounding error
(`L0.h2` RMS 2.1 LSB at σ = 0.05 vs 0.40 at σ = 1). Measured: σ = 0.3 4.54 (100k; a climb reaches 6.12),
σ = 0.2 6.07, σ = 0.1 6.51, σ = 0.05 8.65 LSB. Fixing it needs a wider residual stream (int16 in 32-bit words) or
per-token scales, i.e. a change to contracts 3/8; recorded as an open issue, not part of SG5a.

**Per segment** (contract 8's rule, SG16): on the held-out inputs every segment's least-squares gain
`Σ(dequant·ref) / Σ(ref²)` stays within **±0.0025** of `SEGMENTS[k][0]` and its RMS error within **1.1×**
`SEGMENTS[k][1]` LSB (`scores` compared on causal entries only; `p` against the float softmax). Values on the 200k
derivation inputs (gain / RMS LSB):

| seg | L0 | L1 |
|---|---|---|
| h1 | 1.0009 / 0.294 | 1.0000 / 0.508 |
| q (= qh) | 1.0009 / 0.465 | 0.9999 / 0.664 |
| k (= kh = kT) | 1.0013 / 0.446 | 1.0002 / 0.681 |
| v (= vh) | 1.0005 / 0.465 | 0.9999 / 0.686 |
| scores | 1.0019 / 0.539 | 1.0001 / 0.802 |
| p | 1.0001 / 0.262 | 1.0000 / 0.315 |
| o (= om) | 0.9980 / 0.390 | 0.9999 / 0.496 |
| y | 0.9981 / 0.430 | 1.0001 / 0.503 |
| r1 | 1.0001 / 0.326 | 1.0002 / 0.545 |
| h2 | 1.0003 / 0.396 | 0.9993 / 0.599 |
| gate | 1.0002 / 0.581 | 0.9992 / 0.741 |
| up | 1.0004 / 0.589 | 0.9995 / 0.751 |
| act | 1.0005 / 0.358 | 0.9965 / 0.400 |
| down | 1.0004 / 1.118 | 0.9964 / 1.157 |
| out | 1.0003 / 0.464 | 1.0002 / 0.640 |

**No saturation:** every non-softmax activation's float peak stays ≤ 127.5 of its LSB on the held-out inputs
(derivation peak 102.3, held-out 104.8).

**Overflow:** `prove_no_overflow` proves every int32 intermediate of all 30 ops with one (28 requant ops, 2 softmaxes) over all
int8 inputs (worst 2,086,369,408 at `L0.up`, < 2^31 − 1).

## q/k precision (AUDIT-6)

int8 `q`/`k` hold at d=32 with two heads: `scores` RMS is 0.54 / 0.80 LSB (xf's d=8 single head: 0.542), its gain
within 0.002 of 1, and the end-to-end error stays within TOL on every family above, so no wider `q`/`k`, per-head
scales or headroom change were needed. The error grows layer to layer (L1 segment RMS ≈ 1.3–1.5× L0's) from the
residual stream, not from `scores`.

## Footprint

One element per 32-bit word, every graph value and kernel temp in its own buffer (no reuse):
weights 20,608 + LUTs 5,120 = 25,728 const words, 24,064 activation words, 256 temp words: **50,048 words**,
i.e. `DATA_MEM_ADDR_BITS = 16` (> 4,096, ≤ 65,536). SG4b's liveness-based planner places it in 30,072 words.
