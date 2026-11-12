# Vortex-V backend: the same graphs on a RISC-V core

Anvil's graph IR now has a second target besides tiny-gpu. That target is
[Vortex-V](https://github.com/Ashtonantony28/vortex-v), an RV64IM core (in-order, plus out-of-order variants)
with the Xmlacc int8 extension. `anvil.vortexv` lowers an `anvil.graph` module to one RV64IM + Xmlacc program.
It covers every graph op, and its results match the golden interpreter bit for bit.

```bash
git submodule update --init vortexv                         # Vortex-V at d04a6d7 (read-only)
python3 -m pytest -q tests/vortexv/test_vortexv.py          # ISS: fixtures, random graphs, 2-layer model (~5 s)
python3 -m pytest -q -m "rtl and not slow" tests/vortexv    # RTL lockstep on all 6 cores (~40 s)
python3 -m pytest -q -m "rtl and slow" tests/vortexv        # the 2-layer model on all 6 cores (~20 min)
```

```python
from anvil import vortexv
prog = vortexv.compile(module, consts)
res = prog.run({"x": x})            # Vortex-V's golden ISS: res.outputs, res.values[sym], res.instret
stub, data = prog.image({"x": x})   # for Vortex-V's cocotb harness (RTL)
```

## Why the numbers line up exactly

Anvil's requant is `clamp((acc*M + 2^(s-1)) >> s, -128, 127)`, with every intermediate checked against int32
(`xf.golden.requant`). Vortex-V's `mlacc.rqs` is `clamp((x + 2^(s-1)) >> s, -128, 127)` on a 64-bit `x`. So
`requant(acc, M, s) == rqs(acc*M, s)` whenever the golden model accepts its inputs, because the product is exact
in 64 bits. `test_rqs_is_anvil_requant` checks this on 3,000 Hypothesis cases against Vortex-V's own numpy model
of `rqs`. It is the same rounding NeuroFlight uses (see Vortex-V's `verif/workloads/neuroflight`). Earlier
cross-project notes listed Anvil's rule as a different one; it is not.

## How each op lowers

| Graph op | Vortex-V code |
|---|---|
| `qmatmul`, `qlinear` | `ld` + `ld` + `mlacc.macc` per 8 int8 lanes, unrolled up to 64 lanes. Then `mul` by M, `mlacc.rqs`, and an optional LUT epilogue with `lb`. An i16 result uses round, `sra` and a branch clamp. |
| `rmsnorm_int` | Sum of squares on `macc` (an int8 row with itself), or a scalar loop for i16 rows with `ss_shift`/`ss_init`. Then the `rsqrt_index` loop, an `lw` from the LUT, and `mul` + `rqs` with shift `s + k`. |
| `softmax_lut` | Per row: max over the kept columns, `lw` from the exp LUT, sum, and `(e*one + S/2) divu S`. Causal rows keep `min(i+1, T2)` columns. Masked outputs stay at the buffer's zero. |
| `lut`, `lut_mul`, `add_requant` | Elementwise loops: `lb` from the LUT, `mul`, `rqs`, or the wide clamp. |
| `transpose` | A strided copy for any rank and perm. A last-two-axes transpose that is only read as a matmul's right operand is never materialised: the matmul reads its source in place. |
| `reshape` | A view when the row length is unchanged, otherwise an element-order copy. |

Layout rules:
- Every value has its own buffer.
- int8 rows are padded to 8 bytes with zeros, so `macc` can always read 8 lanes.
- Matmuls read their right operand transposed. Constant weights are transposed when the image is built.
- The program sits at `0x8001_0000`, above Vortex-V's tohost word, and a 3-instruction stub at the reset PC jumps to it.

## Verification

| Check | What | Result |
|---|---|---|
| Fixture blocks | `mlp_block_0..3`, `attention_0..3` (xf fixtures), every value | bit-exact on the ISS |
| Random graphs | 24 seeded multi-head transformer-style graphs (see below), every value | bit-exact on the ISS |
| 2-layer model | SG5a model (2 Blocks, d=32, 2 heads, T=16, i16 residual stream), all 75 values | bit-exact on the ISS |
| Mutant: floor | `rqs` replaced by a plain `sra` | caught on both blocks |
| Mutant: scalar | `macc` replaced by RV64IM byte arithmetic | same answers, 3.3x the instructions |
| RTL | `mlp_block_0`, `attention_0`, one random graph on all 6 Vortex-V core configurations | lockstep with the ISS on every retired instruction; data region byte-identical; values equal the golden model |
| RTL, model | the 2-layer model on all 6 core configurations | same lockstep and memory checks pass on all 6 (19 min in total under Icarus) |

The random graphs cover these features:
- Batched multi-head matmuls with both kinds of right operand: transposed in place, or plain (runtime transpose).
- 3-D transposes with arbitrary perms.
- Reshapes that change the row length.
- int8 rows that are not a multiple of 8.
- LUT epilogues.
- i16 qmatmul and add_requant feeding an i16 rmsnorm with `ss_shift`/`ss_init`.
- Causal and non-causal softmax with T1 != T2, and an i32 softmax output.

Constants are scaled so values spread out instead of saturating.

## Instructions and cycles

Vortex-V counts are retired instructions on the ISS and clock cycles on the RTL (cocotb + Icarus, single-cycle
memory model). The tiny-gpu figures come from this repo's own reports (`docs/g1-report.md`, `docs/closeout.md`) on
iverilog RTL: 4 threads per block, 32-bit words.

| Graph | Vortex-V instructions | without Xmlacc | in-order | OoO | OoO 2-wide | + speculation | + LSQ | + pipelined Xmlacc | tiny-gpu (tuned) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mlp_block_0` | 4,028 | 13,240 | 4,869 | 5,061 | 3,172 | 3,117 | **3,078** | 4,015 | 24,720 |
| `attention_0` | 3,596 | 12,056 | 4,765 | 5,011 | 3,352 | 3,145 | **3,079** | 3,864 | 22,549 |
| 2-layer model | 462,483 | 2,580,115 | 550,804 | 601,155 | 436,736 | 394,228 | **394,153** | 459,097 | 5,545,262 (tiled) |

These are two very different machines, so read the last column as context, not a race. tiny-gpu is a didactic
GPU whose memory system and thread model dominate these tiny blocks. Vortex-V here has an ideal single-cycle
memory.

What the table does show:
- Xmlacc cuts instructions 3.3x on the blocks and 5.6x on the model, where matmuls dominate.
- The OoO core with the LSQ is the fastest configuration.
- The pipelined Xmlacc unit (`XM_LATENCY=3`, ADR 0010) is slower on this code. The matmul inner loop is a single
  chain of dependent `macc`s, so a 3-cycle unit stalls it. The fix is in the compiler: interleave 2 to 3
  accumulators per output (or per output tile). The compiler could do that now that it targets the machine.

## Limits

- The ISS and RTL runs use Vortex-V's single-cycle memory model. Cycles on a real memory system would be higher.
- No register blocking or multi-accumulator scheduling yet: one output element at a time.
- Buffers are never reused (92.8 KB of data for the 2-layer model). Anvil's `memplan` reuse is not wired to this target.
- Graph outputs that are an elided transpose are rejected (`NotImplementedError`). No current graph has one.
