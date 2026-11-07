# ADR-004: An i16 residual stream for small inputs

Status: accepted (SG9, 2026-11-07). Fixes AUDIT-A2 H1; changes contract 3 (integer semantics: new dtype and two
rmsnorm properties) and the target-model contract (v2, `docs/model-contract.md`). The xf fixtures and their
contract 8 are unchanged: every new path is opt-in (`quantize(..., residual="i16")`) and the int8 paths are the
golden ops, bit for bit.

## Context

Contract v1 held TOL = 6 output LSB only for inputs `x = sigma * z` with sigma in [0.5, 3]. nanoGPT's embeddings
are sigma = 0.02. AUDIT-A2 H1 measured 7.8-8.4 LSB at sigma = 0.05 even with calibration matched to the input
scale. Three causes, found by per-segment gain/RMS at small sigma:

1. **int8 residual stream.** `r1 = x + y` and `out = r1 + down` are int8 at a scale set by their largest
   calibrated value. RMSNorm is scale-invariant, so the next norm divides by the row RMS. When the row RMS is a
   small fraction of the calibrated range, the int8 rounding is amplified. At sigma = 0.02 with v1's
   calibration, `L0.r1` peaks at 24 LSB and `L0.h2`'s RMS error grows from 0.40 to 2.3 LSB, `L1.h1`'s to 3.6.
2. **int8 branch outputs.** At small sigma the stream is mostly the attention and MLP outputs (`y`, `down`), so
   their own int8 rounding is most of the stream's error, and the next norm amplifies it in the same way.
3. **eps folded into the rsqrt LUT (xf's form) is exact only at shift k = 0.** `rmsnorm_int` reads
   `lut[ss >> 2k] / 2^k`, so a LUT entry `rsqrt(ms(i) + eps)` stands for `rsqrt(4^k ms(i) + 4^k eps)`: eps is
   inflated 4^k times. With xf's int8 scales this is invisible. With a finer input scale (i16, or a small
   calibration) k reaches 11 and the norm's gain collapses. Measured: `L0.h1` gain 0.19 at sigma = 0.05 with an
   i16 input before the fix. Contract v1's own small-sigma numbers carry a little of it too.

## Options (numbers)

Max error over n = 4,000 inputs per sigma, `DERIVE_SEED`-derived (never the held-out seed), in the v2 contract's
fixed unit `OUT_UNIT` (absolute: 1/127 of the v2 model's calibrated output range, 0.1585). In brackets: the same
error in the option's own output LSB (its int8-equivalent unit). Reproduce with
`NANOGPT_DIR=... PYTHONPATH=sim python3 -m anvil.model_derive --options`.

| option | sigma 0.02 | sigma 0.05 | sigma 0.2 | sigma 1 | sigma 3 |
|---|---|---|---|---|---|
| A: v1 (i8 residual, N(0,1) calibration) | 3.60 (8.24) | 2.91 (6.64) | 2.02 (4.61) | 1.33 (3.04) | 1.17 (2.68) |
| B: i8 residual, sigma-range calibration | 9.99 (9.99) | 5.58 (5.59) | 3.84 (3.85) | 2.61 (2.61) | 2.20 (2.20) |
| C: i8 residual, matched calibration (one model per sigma) | 1.89 (8.31) | 1.96 (8.61) | 1.27 (5.63) | 1.33 (3.04) | 2.83 (2.35) |
| D: i16 residual adds, N(0,1) calibration | 1.59 (3.64) | 1.61 (3.69) | 1.07 (2.45) | 0.70 (1.60) | 0.66 (1.51) |
| E: i16 residual adds, sigma-range calibration | 2.40 (2.40) | 1.50 (1.50) | 1.12 (1.12) | 0.70 (0.70) | 0.70 (0.70) |
| **F: i16 residual adds + branch outputs, sigma-range calibration (chosen)** | **1.43** | **1.35** | **1.38** | **0.87** | **0.60** |
| G: F with N(0,1) calibration | 1.51 (3.47) | 1.17 (2.68) | 1.10 (2.51) | 0.66 (1.50) | 0.63 (1.43) |
| C2: F with matched calibration (the A2-H1 test's setting) | 1.24 (5.45) | 1.27 (5.58) | 1.01 (4.50) | 0.66 (1.50) | 0.59 (0.49) |

All i16 rows include the eps fix (cause 3). For scale, v1's TOL of 6 LSB is 2.6 `OUT_UNIT`.

* **Calibration alone (B, C)** does not help. B spends int8 resolution on sigma = 3 and is worse everywhere.
  C fixes only the scales, not the amplification: 8.3-8.6 own LSB at small sigma, as A2-H1 found.
* **i16 residual (D-G)** removes cause 1. **D** and **G** keep the N(0,1) calibration, so the input and the
  stream clip for sigma > ~1.2. About 8 % of the elements of a sigma = 3 input clip. That passes only because the
  contract measures against `dequant(x_q)`. So the calibration has to cover the domain: **E** or **F**.
  **E** leaves cause 2 (2.40 at sigma = 0.02). **F** also widens the two branch linears that feed the stream
  (`y`, `down`: a qlinear with an i16 result). This flattens the error over the domain (worst 1.43).
* **Per-token scales** (a runtime scale per row of the residual) were not prototyped. They need a per-row
  max-abs reduction and a reciprocal at run time. Every requant that reads the stream would get a runtime
  multiplier, so `(M, s)` stops being a compile-time constant and the overflow proof becomes data-dependent. That
  adds new graph, tensor and codegen ops. They also do nothing for cause 2, since the branch outputs are not
  rescaled. F gets the benefit with one new dtype and no new op kinds.

## Decision

Option F, with the eps fix.

* **IR (contract 3).** `anvil.graph` gains dtype `i16` (one value per 32-bit word, like every value).
  * `add_requant` takes i8 or i16 operands and clamps to its result dtype.
  * `qmatmul` / `qlinear` may have an i16 result when there is no LUT epilogue.
  * `rmsnorm_int` takes an i16 `x` and two optional properties. `ss_shift` = c: the row sum adds
    `(x^2 + 2^(c-1)) >> c` (c = 5 at d = 32 keeps the sum in int32). `ss_init` = e: the sum starts at
    `e = round(eps d / (sx^2 2^c))`, which folds eps into the sum. So the LUT holds an eps-free rsqrt that is
    exact at every k.
  * With both properties absent and int8 operands, every op is `xf.golden`'s, unchanged. Semantics:
    `anvil.semantics.{requant_dtype, add_requant(dtype), qmatmul(dtype), rmsnorm_wide, sq_shift}`.
* **Overflow proof.** For a wide rmsnorm, `prove_no_overflow` bounds `x*w*r` per reachable (index, k) pair. A row
  whose sum maps to (i, k) has `ss < (i + 1) 4^k`, so `|x| <= sqrt(((i + 1) 4^k - e) 2^c)`. That is the tight bound.
  The int8 bound `128 * wmax * max(lut)` would cost 2^15 / 2^7 more of the LUT's precision. `rmsnorm_lut_wide`
  picks the largest LUT shift that this bound admits: 9-10 significant bits at the loud end, a relative step
  below 0.2 %. The int8 bound is still used for int8 inputs, so the xf proof is unchanged.
* **Lowering.** Wide results are clamped with plain tensor ops (`mul`, `add`, `sra`, `max`, `min`) instead of the
  int8 `requant` op. So no new tensor-IR op, no codegen change, and the schedules and templates apply unchanged.
  The rmsnorm reduction starts at `ss_init` and shifts each square.
* **Frontend.** `quantize(..., residual="i16")` widens `residual_nodes(fg)`: the graph inputs and adds that feed
  only RMSNorm/add (or are graph outputs), plus the linears that feed only such adds. Their scale is
  `headroom * max-abs / 32767`. The default stays `"i8"` (contract 8's recipe).
* **Contract model (v2).** `anvil.model.quantize_model` uses `residual="i16"` and calibrates on
  `contract_calib()`: 2^16 N(0,1) inputs from `CALIB_SEED`, each scaled by a sigma log-uniform over [0.02, 3].
  The domain is sigma in [0.02, 3]. TOL and the segment bounds are re-derived by `anvil/model_derive.py` on
  `DERIVE_SEED = 2`. The held-out seed is the fresh 20261107. v1's 20261106 was retired because it was looked at
  while choosing this fix.
* **Error unit.** An i16 output's LSB is 258x smaller than an int8 one's, so "output LSB" no longer means what TOL
  meant. Errors are stated in `OUT_UNIT`, a fixed float: 1/127 of the v2 model's calibrated output range, i.e.
  the LSB an int8 output over that range would have. `model.measure` uses it for every model, including
  differently calibrated ones (the A2-H1 test's). So TOL is one absolute accuracy statement. It does not tighten
  when a calibration shrinks the output range: in C2's own LSB, sigma = 0.05 is 5.6, below v1's 6 but not by
  much. Per-segment RMS stays in each segment's own int8-equivalent LSB.

## Consequences

* A2-H1: TOL holds over sigma in [0.02, 3] on the held-out set, including matched calibration at sigma = 0.05.
  Numbers are in `docs/model-contract.md`.
* Cost (`docs/g2-report.md`, Verilator, against the v1 report): total cycles 6,732,468 (default) and
  5,545,262 (tiled), up from 6,721,506 and 5,534,652 (+0.16 % / +0.19 %, which also includes SG10's concurrent
  codegen fixes).
  * Per wide kernel: rmsnorm 33,063 vs 31,655 cycles; add_requant 39,531-40,571 vs 38,816; `y` 269,176 vs
    268,461. `down`'s default schedule is unchanged at 495,896.
  * Programs grow by 1-2 words. The footprint grows 3 words (30,075 / 29,811); one value per word, as before.
  * `ss_init` is added after the 0-initialised reduction, so the row templates (`row_per_block`'s cross-lane
    reduction) still apply.
* Data dependence: a value's dtype is now a quantization decision. `CompiledModel.quantize_input` clamps to the
  input's dtype.
* Detection floor (A2-L1) and the per-segment rule are unchanged in kind. Compiled runs are now checked per input
  against `INPUT_SEGMENTS` (A2-M1).
