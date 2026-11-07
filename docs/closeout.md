# Anvil close-out (2026-11-07)

Anvil compiles nanoGPT Extended transformer blocks from PyTorch (`torch.export`) to the tiny-gpu-extended ISA through
three xDSL IRs (graph → scheduled loop nests → machine IR), with register allocation for 13 registers, an autotuner,
learned and analytic cost models, and a hardware/software co-design study. Every result below is bit-exact against an
integer golden model and was re-measured by the final independent audit (AUDIT-B2).

## Headline claims (as worded by the final audit)

- **G1 met.** From the nanoGPT Block, with no hand-written assembly, the MLP block and attention head run golden-equal on
  iverilog RTL in 35,084 and 31,668 cycles: 0.87× and 0.80× of the hand tiled chains at identical GPU parameters.
- **G2 met.** The 2-layer, d=32, two-head model is bit-exact against the graph interpreter on isasim and Verilator
  (32 held-out inputs) and iverilog (1 input): 6,732,468 / 5,545,262 cycles (default / tiled presets). Over input scales
  σ ∈ [0.02, 3] the held-out max error is 0.27 absolute, 2.3–2.7× lower than the first contract on the same inputs.
  The tolerance (0.79 absolute) is looser than it needs to be (AUDIT-B2 M1).
- **G3 met, stretch met.** Tuned MLP chain 24,720 cycles (0.61× hand tiled, 0.52× hand naive); attention 22,549
  (0.57× hand tiled). The tuned 1×8×8 matmul is 0.87× hand naive.
- **G4 partly met.** The learned model (tier2_static) reaches pooled Kendall τ 0.956–0.999 on held-out kernels and
  configs, 2.6 % mean error, and guides the tuner to within 5 % of best with 6.1 % of random search's measurements, at
  about 0.24–0.34× the cost of a Verilator run. Within-kernel τ misses the 0.8 target on one held-out config (0.727).
  The analytic tier-1 model is cycle-exact but is a replay of the RTL that costs as much as simulating.
- **G5 met.** Across 104 separately tuned, bit-exact hardware configurations, 2 cores + an instruction cache is the
  knee on every workload; with the state-bit cost proxy the knee is 2 cores + 32-line I-cache + coalescing +
  pipelined decode at 0.64–0.76× of baseline cycles for 1.13× the state bits. In an ISA-model study (RTL unchanged,
  each new op costed as one ALU op), a fused multiply-add saves the most (≈ 10 %); all five candidates ≈ 15–17 %.
- **G6 (FPGA)** skipped by decision D3.

## Engineering facts
- Verilator backend matches iverilog exactly and runs 260–345× faster; it made autotuning and the co-design sweep
  practical.
- 753 fast tests plus RTL/torch suites; two audit rounds with differential fuzzing (~2,650 cases, no wrong output) and
  74 seeded compiler mutants.

## Open items
Three mediums with strict-xfail repros (MAD spill fallback, loose v2 tolerance, G3 report lacks the naive ratio) and the
lows listed in AUDIT-A1, AUDIT-A2, AUDIT-B1 and AUDIT-B2. See PLAN.md "Close-out".
