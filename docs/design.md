# Anvil: a tensor compiler for tiny-gpu-extended (plan for review)

**Status:** plan only, waiting on Ashton's review. No implementation code has been written.
**Written:** 2026-11-06, from `project-analysis/tiny-gpu-extended.md` and the simulator repo as it stands at the close-out
(`Ashtonantony28/Didactic_GPU_Simulator`, branch `claude/complete-markdown-tasks-IG7yf`, commit `096c6ef`, AUDIT-21).
**Working name:** Anvil (rename freely).

---

## 1. The idea in one paragraph

Today every program that runs on tiny-gpu-extended is hand-written assembly: `matmul_tiled.asm` (147 words),
`rmsnorm.asm`, `softmax.asm`, and a hand-built launch chain in `xf/mlp.py`. Anvil replaces that hand work with a
compiler in the Triton/TVM tradition. You give it a PyTorch module (nanoGPT Extended's `Block`). It captures the graph
with `torch.export`, quantizes it to the same int8/fixed-point semantics the xf track proved out, fuses and plans
memory, lowers each op through a schedulable loop-nest IR, picks a schedule with an autotuner guided by a learned cost
model, does register allocation for 13 registers and a 256-instruction program limit, and emits assembly plus a launch
chain that runs on `xf.isasim`, on the RTL under cocotb, and (optionally) on an FPGA. Every layer is checked bit-exact
against an integer golden model, the same way the xf track was.

What makes it interesting rather than a toy port of TVM: the target is hostile in ways real GPUs are not. There is no
signed compare, no AND/OR/shift-left, no MIN, 8-bit immediates, one instruction in flight per warp, 64 words of shared
memory, at most 255 threads per launch and 256 instructions per program. Each of those becomes a compiler problem with
a measurable answer, and the cost model has to learn the machine's real quirks (the channel claim order, I-cache
reuse only across blocks, MeDiC classification) rather than a textbook GPU.

## 2. Goals and success criteria

| # | Goal | Measured by |
|---|---|---|
| G1 | **Compile, don't hand-write.** The MLP block and the attention head that xf hand-built are produced from `torch.export` with no hand-written assembly. | Compiled output on the existing xf fixtures (`mlp_block_0`, `attention_0`) is bit-exact with `xf.golden` and within the frozen contract-8 tolerances against PyTorch. |
| G2 | **Go where xf never got.** A full transformer block (ln_1 → attention → residual → ln_2 → MLP → residual), multi-head, at a larger size (default d=32, 2 heads, T=16, 2 layers). This closes AUDIT-1 M1 (halves never composed). | End-to-end bit-exact vs the graph interpreter; within a tolerance derived and stated by the compiler, checked on held-out inputs that were never used for tuning. |
| G3 | **Beat the hand-written kernels.** | Same GPU params as `docs/xf-bench.md`: MLP block total < 40,329 cycles (hand tiled) with a stretch target of 25 % fewer; tuner picks naive for 1×8×8 on its own (1,324 vs 3,848 cycles). |
| G4 | **A cost model that earns its keep.** | On held-out kernels and held-out hardware configs: Kendall τ ≥ 0.8 between predicted and measured cycles; reaches within 5 % of the best measured schedule with ≤ 10 % of the measurements random search needs. |
| G5 | **Hardware/software co-design.** | A Pareto report of cycles vs hardware cost over `ICACHE_LINES`, `DCACHE_LINES`, `COALESCE`, `WARPS_PER_CORE`, `WARP_POLICY`, `NUM_CORES`, `SIMT_STACK`, each point a separately tuned compile; plus compiler-driven ISA proposals ("adding MIN saves X % on this model") evaluated in the ISA model. |
| G6 | **(Optional) silicon.** | The compiled block runs on an FPGA and its memory image matches the RTL bit-exactly; cycle counts match the RTL simulation exactly. Depends on decision D3. |

Non-goals: training, floating point on the GPU, dynamic shapes, general PyTorch coverage beyond the ops a nanoGPT
block needs (linear, matmul, RMSNorm, SiLU/SwiGLU, softmax with causal mask, add, reshape/transpose/view).

## 3. What the target gives and takes away

Everything below was checked in the repo at `096c6ef`. Each row is a compiler requirement.

| Fact about the machine | Consequence for the compiler |
|---|---|
| 16 opcodes, all used. `CMP` is unsigned and only distinguishes equal/not-equal (`BRp` never fires). No AND/OR/XOR/SHL/MIN. | Instruction selection needs an idiom library: `MIN(x,c) = -MAX(-x,-c)`, clamps via `MAX`, `x*2^k` via `MUL`, bounds as uniform trip counts, predication as arithmetic masks. Loops are counted equality loops. |
| `CONST` takes an 8-bit immediate. | Constant materialization (`hi*256+lo` sequences) with rematerialization and a per-kernel constant pool in the args block. |
| Registers R0–R12 general, R13–R15 are `%blockIdx/%blockDim/%threadIdx`. The ARGS prologue uses R12 as the args pointer. | Register allocation with spilling (to SMEM or a per-thread global scratch area), under a cost model that knows spills cost memory round trips. |
| 8-bit PC: ≤ 256 instructions per program. | Unroll factors and fused epilogues are bounded by code size; schedules that do not fit are pruned before measurement; very large fused kernels are split. |
| 8-bit DCR: ≤ 255 threads per launch. | Launch partitioning: one op can become several launches with offset args blocks. |
| `DATA_MEM_ADDR_BITS` is a parameter (xf uses 12 = 4,096 words; AUDIT-12 exercised 16-bit addresses). | Static memory planning with liveness-based reuse; the d=32 two-layer model (≈ 20k words of weights) needs 16-bit addresses. |
| Shared memory: 64 words per core, zeroed per block, split across warps of a block. | Tiling must fit SMEM; `cache_read` to SMEM is a schedule choice, not a given. |
| One instruction in flight per warp; latency hiding comes only from other warps (`WARPS_PER_CORE` > 1) and other cores. | Occupancy vs SMEM-per-warp is a real trade-off; instruction count matters more than instruction order. |
| Branches: with `SIMT_STACK=0` the warp follows its lowest lane; with `SIMT_STACK=1` divergence works under `SYNC`, 8-entry stack, `$fatal` on overflow. | Divergence strategy is a lowering choice per region: uniform control flow, arithmetic masking, or `SYNC`-delimited divergence. The compiler proves stack depth ≤ 8. |
| `BAR` is real at `WARPS_PER_CORE > 1`. Cross-warp `STR`s to one word without `BAR` are an unspecified race (ADR-016). | Barrier insertion plus a disjointness check; `xf.isasim`'s `store_races` is run on every compiled kernel in CI and must be empty. |
| Coalescing merges only identical addresses; L1-D is one word per line, write-through. | Layouts that make lanes load the same word (broadcast) are cheap; unit stride is not. The cost model must learn this, not assume line coalescing. |
| Known open mediums: fixed claim order starves cores 2–3 without the I-cache (AUDIT-10 M1); the shared L1-D can install a stale value when a fill races a write-through (AUDIT-10 M2). | The cost model has to capture M1. For M2, the compiler never reads a word in the same launch another core writes (true of dataflow kernels anyway), and a checker asserts it. |
| Simulation speed: ~3–4k simulated cycles/s on iverilog (SG17 baseline); the hand MLP block is ~40k cycles. | Measurement is the bottleneck for autotuning. See M6 and risk R1. |
| `xf.isasim` is bit-exact with the RTL on memory but counts steps, not cycles. | Functional checks are fast; timing needs the RTL, a faster RTL simulator, or a calibrated model. |

## 4. Architecture

```
torch.nn.Module (nanoGPT Extended Block, pinned commit)
   │  torch.export → ATen core graph
   ▼
[1] Frontend ─ op coverage check, shape propagation, decomposition to a small op set
   │
   ▼
[2] Graph IR  (anvil.graph)          interpreter: float (torch semantics) and integer (golden semantics)
   │  passes: calibration + quantization (int8 sym per-tensor, (M, s) requant, LUTs),
   │          overflow proof (generalized check_bounds), fusion (matmul+requant+LUT+residual epilogues),
   │          layout choice, static memory planning, launch partitioning
   ▼
[3] Tensor IR (anvil.tensor)         loop nests over buffers in global / smem / reg; interpreter (numpy int)
   │  schedule primitives: split, reorder, fuse, tile, bind(block|warp|lane), cache_read(smem),
   │  compute_at, unroll, pad-vs-mask, reduce_across_lanes(smem+BAR), divergence(uniform|mask|sync)
   │  ◄──── [6] Autotuner: search space per op, evolutionary search, measurement cache
   │                 ▲
   │            [7] Cost model: analytic (FSM timing) + learned (GBDT on features; stretch: sequence model)
   ▼
[4] Machine IR (anvil.mir)           virtual-register ops 1:1 with the 16 opcodes + pseudo-ops
   │  passes: instruction selection with idioms, constant materialization, strength reduction,
   │          register allocation (13 regs, spills), SYNC/BAR placement + stack-depth proof,
   │          code-size check (≤ 256), ARGS prologue (calling convention from xf/kernels.py)
   ▼
[5] Emit: .asm (xf/asm.py syntax) + Image (xf/image.py) + launch chain [(out segment, (program, threads))]
   │
   ├── xf.isasim       functional, bit-exact, race and divergence flags      (seconds)
   ├── RTL via cocotb  cycle-exact, counters (icache/dcache/requests/warps)  (minutes)
   ├── Verilator build (if the M0 spike pays off) cycle-exact, faster        (target 50×+)
   └── FPGA            via a host bridge (optional, D3)
```

### 4.1 Verification spine (the part that keeps it honest)

The xf track's strongest idea was two-level checking: kernels bit-exact against an integer golden, golden within a
tolerance of PyTorch, so a kernel bug can never hide inside a tolerance. Anvil extends that to every IR boundary:

1. Graph IR integer interpreter = generalized `xf.golden` (and on the xf fixtures, equal to it bit for bit).
2. Tensor IR interpreter after every schedule transform = graph interpreter (translation validation per pass).
3. Machine IR interpreter = Tensor IR interpreter, before and after register allocation.
4. `xf.isasim` on the emitted program = Machine IR interpreter; `divergent`, `store_races` and `stores_conflict` empty.
5. RTL (sampled in CI, always for the headline benchmarks) = `xf.isasim`, through the existing differential harness.
6. Float tolerance against PyTorch only at the graph level, with per-segment gain/RMS checks (the SG16 lesson).

Every schedule the autotuner times must first pass level 4; a fast wrong kernel never enters the results table.
A compiler fuzzer (random small graphs × random legal schedules × random hardware params) runs levels 2–4 and
mutation testing (seeded compiler bugs must be caught) is part of each audit, as AUDIT-3/-7/-12 did for the RTL.

### 4.2 Search and cost model

- **Search space** per op from templates (Ansor-style sketches): tile sizes, block/warp/lane binding, loop order,
  SMEM staging on/off, unroll factor, epilogue fusion depth, pad-vs-mask for edges, divergence strategy. Plus the
  hardware parameters for the co-design sweep.
- **Search:** random sampling to seed, then evolutionary search ranked by the cost model, measuring the top-k each
  round; results cached by (program hash, GPU params).
- **Tier 1 analytic model:** derived from the scheduler FSM (FETCH/REQUEST/WAIT/EXECUTE/UPDATE per instruction,
  I-cache hit path, LSU handshake, channels, coalescing, dispatch of blocks to cores). Interpretable, no training.
- **Tier 2 learned model:** gradient-boosted trees (XGBoost or LightGBM) on features from the Machine IR and the run
  config: instruction mix, memory ops per lane, distinct addresses per warp-instruction (coalescing potential),
  SMEM ops, BARs, trip counts, blocks per core, warps, code size vs `ICACHE_LINES`, and the Tier 1 prediction as a
  feature (residual learning). Trained on RTL measurements; evaluated on held-out ops and held-out hardware configs.
- **Stretch tier 3:** a small transformer over instruction-token sequences, trained with your nanoGPT Extended code,
  compared against tier 2. Nice symmetry: your own model predicting the cost of your own model's kernels.

### 4.3 IR framework

Recommended (decision D1): **xDSL**, a pure-Python framework that implements MLIR's data structures and textual
format. It gives real MLIR concepts (dialects, ops, regions, rewrite patterns, pass pipelines, interop with
`mlir-opt` text) without a C++/LLVM build in a cloud container, and it sits next to the existing Python toolchain
(`xf/asm.py`, `xf/isasim.py`, `xf/image.py`). Three dialects: `anvil.graph`, `anvil.tensor`, `anvil.mir`. If the M1
spike shows xDSL getting in the way, fall back to a plain Python IR with an MLIR-like text format and the same passes.

## 5. Milestones

Each milestone becomes one or more sub-goals in a `PLAN.md` with frozen contracts, one PR per sub-goal and a Verify
command, the same process the simulator used. MVP is M0–M4. At most two build threads at once.

| M | Name | Delivers | Accept / Verify |
|---|---|---|---|
| M0 | Bootstrap and measurement harness | Repo, simulator pinned at `096c6ef` as a submodule, env setup (iverilog 12, sv2v 0.0.13, cocotb 2.0.1, torch, nanoGPT Extended pin). One Python API: `run(program, image, launches, params, backend=isasim\|rtl\|verilator)` → memory, cycles, counters. **Verilator spike:** build the sv2v output with Verilator and measure the speedup. | All existing xf fixtures run through the API on isasim and RTL with the published cycle counts from `docs/xf-bench.md`; Verilator either reproduces those cycle counts exactly (adopt) or is dropped with a note. |
| M1 | IR foundations and contracts | Three dialects, textual round-trip, interpreters at each level, the integer semantics contract (requant, LUT, rsqrt, softmax copied from `xf/golden.py` and frozen). | Interpreters reproduce `xf.golden` on both fixtures bit-exactly; round-trip property tests. |
| M2 | Backend: Machine IR → assembly | Instruction selection + idioms, constant materialization, register allocator with spilling, SYNC/BAR placement, size check, ARGS prologue, emitter. | Hand-translated Machine IR versions of `matmul_naive/tiled`, `rmsnorm`, `softmax` assemble, run bit-exact on isasim and RTL; differential fuzz of random MIR programs vs isasim; register-pressure tests that force spills. |
| M3 | Tensor IR, lowering and schedules | Lowering of matmul, elementwise, LUT, row reductions (RMSNorm, softmax with causal mask), transpose; schedule primitives incl. cross-lane reductions via SMEM+BAR and the three divergence strategies. | Every op × a grid of schedules × `WARPS_PER_CORE` 1/2/4 × `SIMT_STACK` 0/1 bit-exact on isasim; no races or divergence flags. |
| M4 | Frontend and the xf workloads from PyTorch | `torch.export` capture, quantization pass (generalizing `export.py`: calibration with headroom, `fixed_point`, overflow proof), fusion, memory planning, launch partitioning. Default schedules (no tuning yet). | **G1:** compiled MLP block and attention head on the xf fixtures bit-exact with `xf.golden`, within contract-8 tolerances, and run on RTL. |
| M5 | Full transformer block and scale-up | Composition with residuals, multi-head attention, 2 layers at d=32/T=16, 16-bit addresses, multi-launch ops beyond 255 threads, a derived end-to-end tolerance with per-segment checks. | **G2** on held-out inputs; RTL run of the full model recorded. |
| — | **Audit round A** | Independent sessions, one lens each: compiler correctness (fuzz + seeded compiler mutants), numerics, ISA/backend consistency, API/docs. Highs become sub-goals. | — |
| M6 | Autotuner and measurement at scale | Search spaces, evolutionary search, measurement cache, parallel measurement pool, code-size and SMEM pruning. | **G3** at the xf-bench params; tuned schedules reproducible (two runs, same choice). |
| M7 | Cost models | Tier 1 analytic, tier 2 GBDT with active learning from tuner runs, transfer to unseen hardware configs; tier 3 as stretch. | **G4** metrics on a frozen held-out split, reported with seeds. |
| M8 | Co-design sweep and ISA proposals | Joint search over GPU params, Pareto report; each candidate instruction (e.g. MIN, AND, SHL, signed CMP) modelled in a fork of `xf.isasim`, re-tuned, savings reported. | **G5** report where every number points at a cached measurement. |
| M9 | FPGA (optional) | Synthesis flow (sv2v → Yosys → nextpnr, or Vivado), BRAM-backed program/data memories replacing the cocotb memory model, a UART host bridge to load images, start launches, read memory and counters. | **G6.** |
| — | **Audit round B (final)** | Same lenses plus performance-model honesty (are reported speedups real on RTL?). Ends the project: highs fixed, mediums and lows recorded with strict-xfail repros, then close-out. | — |

Stated end point, per your 2026-10-27 instruction: two audit rounds, A after M5 and B after M8 (or M9), then close-out.
No open-ended audit loop.

## 6. Risks

| # | Risk | Likelihood / impact | Mitigation |
|---|---|---|---|
| R1 | Simulation is too slow to autotune: ~4k cycles/s means ~10 s per MLP-sized measurement and hours per op search. | High / high | Verilator spike in M0; cost model prunes before measuring; isasim filters wrong kernels for free; measurement cache; small shapes for training data; parallel workers. |
| R2 | 256-instruction and 13-register limits make good schedules not fit. | High / medium | Size and pressure are first-class schedule constraints; spill cost in the model; kernel splitting; report which schedules were pruned and why. |
| R3 | Missing ops (no AND/SHL/signed compare) make masks and bounds expensive, so compiled code loses to hand tricks like `matmul_tiled`'s `-11` walk. | Medium / high for G3 | Idiom library seeded from the hand kernels; pad-to-tile as an alternative to masking; G3 compares at equal params and reports honestly if a case loses. |
| R4 | Learned model has too little data or does not transfer across hardware configs. | Medium / medium | Residual learning on top of the analytic model; active learning; report τ per config, including failures. |
| R5 | Numerics at d=32, multi-head and 2 layers break the int8 tolerances that held at d=8 (AUDIT-6 showed q/k rounding is fragile). | Medium / high for G2 | Compiler-derived bounds and per-segment gain checks from day one; option of int16 activations for q/k (still 32-bit words); held-out set never used for tuning (AUDIT-6 L1). |
| R6 | Cross-warp races, L1-D stale-fill (AUDIT-10 M2) or core starvation (M1) produce flaky or misleading results. | Medium / medium | `store_races` must be empty in CI; no cross-core read-after-write within a launch; the cost model sees core-count effects explicitly. |
| R7 | The RTL is simulation-only in places (Python memory model, `$fatal`, wide combinational SMEM), so FPGA needs real work, and timing closure is unknown. | High / medium (optional milestone) | M9 is optional and last; BRAM memory wrapper and host bridge live in the compiler repo, simulator RTL unchanged unless you approve. |
| R8 | xDSL API churn or performance. | Low / medium | Pin the version; M1 spike with a fallback to a plain Python IR. |
| R9 | Scope: nine milestones is a lot. | Medium / medium | M0–M4 is a complete, demoable result on its own; each later milestone stands alone. |
| R10 | torch.export output varies across torch versions. | Low / low | Pin torch; decompose to a small fixed op set and test it. |

## 7. Decisions for you (defaults in bold; build starts on the defaults unless you change them)

- **D1 IR framework:** **xDSL (MLIR-compatible, Python)** / plain Python IR / real MLIR in C++.
- **D2 Where the code lives:** **a new repo you create (for example `anvil-compiler`), with the simulator as a pinned
  submodule** / a `compiler/` directory inside `Didactic_GPU_Simulator`. I cannot create GitHub repos from here, so
  the default needs you to make an empty repo and tell me its name.
- **D3 FPGA:** which board do you have, if any? **No board: M9 is skipped** / ECP5 (e.g. ULX3S) with the open
  Yosys/nextpnr flow / Xilinx Artix-7 (e.g. Arty A7) with Vivado.
- **D4 Model size for G2:** **d=32, 2 heads, T=16, 2 layers** / stay at d=8 single head / larger (d=64, needs more
  launches and more memory).
- **D5 Simulator changes:** **none; the compiler treats the simulator as read-only, ISA proposals are evaluated only in
  a forked ISA model** / allow small tooling-only PRs (e.g. a Verilator target) / allow RTL changes for winning ISA
  proposals (would reopen the simulator's verification loop).

## 8. Why this fits your portfolio

It joins three things you already built: nanoGPT Extended is the source model, tiny-gpu-extended is the target, and the
xf track's quantization and golden models become the compiler's semantics. It adds the skills those projects lack:
compiler IR design (MLIR concepts), scheduling and autotuning (TVM/Ansor/Triton ideas), learned performance models,
and hardware/software co-design with measured numbers. Every claim in its final report would point at a measurement,
the same standard the simulator's audits held.
