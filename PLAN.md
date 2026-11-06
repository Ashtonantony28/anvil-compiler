# PLAN — Anvil, a tensor compiler for tiny-gpu-extended

Design and rationale: `docs/design.md` (the approved plan, 2026-11-06). This file is the tracker: sub-goals,
frozen contracts, Verify commands. Every sub-goal is one PR (or, until the GitHub repo exists, one commit on `main`).

Approved by Ashton on 2026-11-06 with the default decisions: D1 xDSL 0.75.0, D2 new repo with the simulator as a pinned
submodule, D3 no FPGA (M9 skipped), D4 d=32 / 2 heads / T=16 / 2 layers, D5 the simulator is read-only.

## Status

| ID | Sub-goal | Milestone | Depends on | Status |
|---|---|---|---|---|
| SG0 | Runtime: one `run()` over isasim, RTL and Verilator; Verilator spike | M0 | — | DONE |
| SG1 | IR contracts: semantics, three dialects, interpreters | M1 | — | DONE |
| SG2 | Backend: MIR → assembly (isel, constants, regalloc, emit) | M2 | SG1 | DONE |
| SG3a | Graph → tensor lowering and schedule primitives | M3 | SG1 | DONE |
| SG3b | Tensor → MIR lowering (codegen), end to end on isasim/RTL | M3 | SG2, SG3a | DONE |
| SG4a | Frontend: torch.export capture and quantization to graph IR | M4 | SG1 | DONE |
| SG4b | Memory plan, launch chain, compile driver; xf workloads from PyTorch on RTL (G1) | M4 | SG3b, SG4a | IN PROGRESS |
| SG5a | Multi-head + full Block at graph/tensor level, derived tolerance | M5 | SG4a, SG3a | IN PROGRESS |
| SG5b | Full 2-layer d=32 model compiled and run on RTL (G2) | M5 | SG4b, SG5a | TODO |
| AUD-A | Audit round A (4 lenses) | — | SG5 | TODO |
| SG6 | Autotuner and measurement cache (G3) | M6 | SG4 | TODO |
| SG7 | Cost models: analytic + GBDT (G4) | M7 | SG6 | TODO |
| SG8 | Co-design sweep and ISA proposals (G5) | M8 | SG7 | TODO |
| AUD-B | Final audit round, then close-out | — | SG8 | TODO |

End point: audit rounds A and B only. Highs from a round become sub-goals; mediums and lows are recorded with
strict-xfail repros. No third round.

## Environment

```bash
apt-get update && apt-get install -y iverilog verilator          # iverilog 12.0, verilator 5.020
curl -sSL -o /tmp/sv2v.zip https://github.com/zachjs/sv2v/releases/download/v0.0.13/sv2v-Linux.zip
python3 -c "import zipfile;zipfile.ZipFile('/tmp/sv2v.zip').extractall('/tmp/sv2v')" && install /tmp/sv2v/*/sv2v /usr/local/bin/
pip install --break-system-packages cocotb==2.0.1 numpy pytest hypothesis xdsl==0.75.0 torch
git submodule update --init                                       # sim/ at 096c6ef
git clone https://github.com/Ashtonantony28/nanoGPT_LLMArchitecturalExtension $NANOGPT_DIR && git -C $NANOGPT_DIR checkout 03dbab8
```

## Frozen contracts

**C1 Layout.** `anvil/semantics.py` (integer semantics), `anvil/ir/{graph,tensor,mir}.py` (xDSL dialects),
`anvil/interp/` (one interpreter per IR), `anvil/backend/`, `anvil/frontend/`, `anvil/runtime/`, `anvil/tune/`,
`anvil/cost/`. Tests in `tests/`, marked `rtl` (runs a Verilog simulator) or `torch` (needs torch + `NANOGPT_DIR`).
`tests/conftest.py` puts `sim/` on `sys.path`; Anvil imports `xf.*` from the pinned submodule and never copies it.

**C2 Runtime.** `anvil.runtime`:
```python
@dataclass(frozen=True)
class GpuParams:                       # names and defaults = sim/src/gpu.sv parameters
    WORD_BITS: int = 32; DATA_MEM_ADDR_BITS: int = 12; DATA_MEM_DATA_BITS: int = 32
    NUM_CORES: int = 2; THREADS_PER_BLOCK: int = 4; ICACHE_LINES: int = 0; COALESCE: int = 0
    PIPELINED: int = 0; DCACHE_LINES: int = 0; MEDIC: int = 0; WARPS_PER_CORE: int = 1
    WARP_POLICY: int = 0; SIMT_STACK: int = 0
    def plusargs(self) -> list[str]     # "-Pgpu.NAME=V" for every field that differs from gpu.sv's default,
                                        # so GpuParams() alone builds the xf GPU (= the Makefile's XF_PARAMS)

@dataclass
class Launch:  program: list[int]; threads: int          # 16-bit words, 1..255 threads
@dataclass
class Job:     data: list[int]; launches: list[Launch]; params: GpuParams = GpuParams(); mem_latency: int = 0

@dataclass
class LaunchResult: cycles: int | None; counters: dict[str, int]      # cycles None on isasim
@dataclass
class RunResult:
    memory: list[int]                  # final data memory, unsigned words
    launches: list[LaunchResult]
    flags: dict                        # isasim only: divergent, stores_conflict, store_races, race_words

def run(job: Job, backend: str = "isasim", *, timeout_cycles: int = 2_000_000) -> RunResult
                                       # "isasim" | "rtl" | "verilator"; isasim maps GTLRR to gto
```
Memory persists across launches, the GPU resets between them (as `sim/test/helpers/runner.py::run_launches`).
RTL builds are cached per `GpuParams` under `.anvil-cache/`. The RTL backend reuses the simulator's own cocotb helpers.

**C3 Integer semantics.** `anvil.semantics` re-exports and generalizes `xf.golden` (requant, LUT ops, rsqrt index,
RMSNorm, softmax, saturating int8) with int32 overflow raising `OverflowError`. Every interpreter uses it; nothing
else defines arithmetic.

**C4 Verification levels.** (1) graph integer interpreter = `xf.golden` on the xf fixtures; (2) tensor interpreter after
each schedule transform = graph interpreter; (3) MIR interpreter = tensor interpreter, before and after regalloc;
(4) `xf.isasim` = MIR interpreter, with `divergent`, `store_races`, `stores_conflict` empty; (5) RTL = isasim through
C2; (6) float tolerance vs PyTorch only at the graph level, with per-segment gain/RMS checks. Every new pass ships with
a test at the level it touches; `make check` (no RTL, no torch) stays under 5 minutes.

## Sub-goals

### SG0 — Runtime and Verilator spike
Done: Verilator adopted (ADR-001). It matches iverilog on memory, per-launch cycles and every counter across the
xf-bench configs, all 37 32-bit ISA fixtures and 11 matrix-style configurations, at 260–345× the speed (≈2.4M
simulated cycles/s). Verilator is the default timing backend for SG6+; iverilog stays the reference.

C2 as code. `isasim` backend wraps `xf.isasim.run` per launch on one memory. `rtl` backend: a generic cocotb test
module that loads a Job from a JSON file named by an env var, runs every launch with the simulator's helpers and
writes a result JSON; `run()` builds (sv2v → iverilog) once per params and invokes vvp. `verilator` backend: build the
same sv2v output with Verilator plus a C++ or cocotb harness; adopt only if it reproduces RTL cycles exactly; record
the speedup in `docs/decisions/ADR-001-verilator.md` either way.
Accept: all launches of `mlp_block_0` (tiled and naive chains from `xf.mlp.launches`) and `attention_0` produce
golden-equal memory on isasim and RTL; RTL per-launch cycles equal `sim/bench/xf-mlp.json` at the four xf-bench
configs; isasim and RTL agree on random seeded programs from the simulator's ISA fixtures.
Verify: `make check && python3 -m pytest -q -m rtl tests/runtime`

### SG1 — IR contracts
C3 as code. Three xDSL dialects with textual round-trip: `anvil.graph` (quantized tensor ops: qlinear/qmatmul with
(M, s) requant, lut, rmsnorm_int, softmax_lut with causal mask, add_requant, transpose/reshape), `anvil.tensor`
(loop nests over buffers in `global`/`smem`/`reg` with affine index expressions), `anvil.mir` (virtual-register ops 1:1
with the 16 opcodes plus pseudo-ops for labels, loops and constants). A numpy integer interpreter for each.
Accept: graph IR for the MLP block and the attention head, built by hand from the fixture images, interprets
bit-exactly to `xf.golden` (`export.mlp_block_int`, `attention.attention_int`); round-trip property tests per dialect;
tensor and MIR interpreters pass hand-written unit programs.
Verify: `make check`

### SG2 — Backend: MIR → assembly
Passes over `anvil.mir`: (a) idiom library as MIR builders, each checked against `anvil.semantics` on random values:
neg, min, clamp-to-int8, abs, signed less-than and select via arithmetic masks (no signed CMP, no logic ops),
requant `(acc*M + 2^(s-1)) >>> s` with clamp, `x * 2^k`, wide address arithmetic; (b) `const32` materialization
(`CONST hi`, `MUL 256`, `ADD lo` chains, sign handling at 32 bits) with rematerialization and CSE of constants;
(c) liveness over the CFG (labels, `br`, `sync`), register allocation onto R0–R11 (R12 is the args pointer set by
the ARGS prologue, R13–R15 read-only) with spilling to a per-thread scratch area in global memory whose base comes from
the args block, and spill-cost heuristics (loop depth); (d) peephole cleanups; (e) label resolution, the 256-word
program limit (clear error naming the overflow), the ARGS prologue compatible with `xf.kernels.program` patching,
and `encode()` to words. Entry point `anvil.backend.compile_mir(module, *, args_base) -> Program(words, asm_text,
stats)`.
Accept: every xf kernel decoded to MIR, de-allocated to virtual registers and re-compiled, runs the full mlp and
attention launch chains on isasim with golden-equal memory; hypothesis-generated straight-line and loop programs with
register pressure above 12 spill and still equal the MIR interpreter on isasim; idioms equal `anvil.semantics`;
oversize programs fail with the size error; a small set of RTL runs (marked `rtl`) confirms isasim == RTL.
Verify: `make check && python3 -m pytest -q -m rtl tests/backend`

Done. Deviation accepted: R12 is allocatable after the args pointer's last read (`reuse_args_reg=True`), otherwise
`matmul_tiled` needs 13 registers. Anvil ABI v1 (spill frames per thread from `args[spill_slot]`) is documented in
`anvil/backend/__init__.py`. Recompiled xf kernels are within 6 % of hand-written size (tiled matmul 215 vs 203 words,
constants rebuilt at each use; follow-up for SG6). No xf kernel spills.

### SG3a — Graph → tensor lowering and schedule primitives
`anvil.lower.graph_to_tensor`: each graph op → a default loop nest in `anvil.tensor` (matmul/qlinear with fused
requant and LUT epilogue, lut, lut_mul, rmsnorm_int, softmax_lut with causal mask, add_requant, transpose, reshape as
a view). `anvil.schedule`: primitives as tensor-IR rewrites with legality checks: split, reorder, fuse, tile, bind
(block | warp | lane, respecting `THREADS_PER_BLOCK`, `WARPS_PER_CORE` and the 255-thread launch limit), cache_read
to smem (with barrier insertion and a 64-word/`WARPS_PER_CORE` budget check), compute_at, unroll, pad-vs-mask for
ragged edges, cross-lane reductions through smem + barrier. A `Schedule` object records the primitive trace so a
tuner can replay it. Extend the tensor interpreter for nested warp × lane barriers (SG1 open issue 1).
Accept: every graph op on both fixtures lowers and interprets bit-exactly to the graph interpreter; hypothesis tests
apply random legal primitive sequences to every op and stay bit-exact (C4 level 2); illegal schedules (smem over
budget, bad binding, > 255 threads per launch) are rejected with clear errors; hand schedules that mirror
`matmul_tiled.asm` (4×4 tile staged in smem) and `matmul_naive.asm` exist as named templates.
Verify: `make check`

Done. Canonical form for codegen: `anvil/schedule/__init__.py`. Open: ragged M in `matmul_tiled`, conservative `pad`
legality, affine maps unsimplified, launch splitting not yet (255-thread limit enforced at bind).

### SG3b — Tensor → MIR lowering
Codegen from bound tensor loop nests to `anvil.mir` through SG2's idiom library: index arithmetic, block-uniform
counted loops (equality exits only), predication strategy per region (uniform, arithmetic mask, or `SYNC`
divergence when `SIMT_STACK=1`, with a stack-depth ≤ 8 proof), barriers, args-block layout per launch, launch
splitting above 255 threads. Then `compile_mir` and run.
Accept: every op × a grid of schedules × `WARPS_PER_CORE` 1/2/4 × `SIMT_STACK` 0/1 is bit-exact on isasim with no
race/divergence flags (C4 level 3–4); the two matmul templates run on Verilator and their cycles are reported next
to the hand kernels'; a sample is checked on iverilog RTL.
Verify: `make check && python3 -m pytest -q -m rtl tests/codegen`

Done. Untuned default/template schedules already beat the hand chain: tiled MLP chain 34,046 vs 40,329 cycles (0.84×),
naive 44,422 vs 47,610 (`docs/codegen-bench.md`). Open: programs near 256 words at wide configs; masked smem stores
need one spare smem word at SIMT_STACK=0; rsqrt loop runs worst-case trips; elementwise ~6 % slower (constant remat).

### SG4a — Frontend: capture and quantization
`anvil.frontend.capture(module, example_inputs)` via `torch.export` (pinned torch) → decomposition to a small op set
(linear, matmul, RMSNorm, SiLU/SwiGLU, softmax with causal mask, add, reshape/view/transpose) with a clear error for
anything else → `anvil.frontend.quantize(graph, calib_inputs, headroom=1.5)` producing `anvil.graph` IR: symmetric
per-tensor int8 scales, (M, s) via a generalized `fixed_point`, LUTs (SiLU, exp, rsqrt) built as `xf/export.py`
builds them, and an overflow proof (generalized `check_bounds`) that raises if any int32 intermediate can overflow.
Accept: on the nanoGPT Extended `Block` built exactly as `xf/export.py::build_block` builds it, the quantized graph
IR reproduces the `mlp_block_0` and `attention_0` images (weights, LUTs, M, s) and, through the graph interpreter, the
fixture goldens bit-exactly; float tolerance vs PyTorch with per-segment gain/RMS checks (C4 level 6); unsupported
ops rejected by name; tests marked `torch`.
Verify: `make check && NANOGPT_DIR=... python3 -m pytest -q -m torch tests/frontend`

Done. From torch, both fixtures' graph IR equals the images (weights, LUTs, M, s, scales) and the goldens bit-exactly.
Open for SG5: multi-head needs head-batched or per-head graph ops (capture works, quantize raises); the full-Block
tolerance is still to be derived (measured ≈ 2.4 LSB on 1–2 stacked Blocks).

### SG5a — Multi-head and the full Block above codegen
Extend `anvil.graph` (head-batched `qmatmul`/softmax, or per-head ops with merge) so `from_torch` quantizes nanoGPT
Extended Blocks with `n_head > 1`; lower the new ops to tensor IR with default schedules. Build the target model
(d=32, 2 heads, T=16, 2 stacked Blocks with residuals, `pos_embedding='none'`, RMSNorm, SwiGLU) and derive its
end-to-end tolerance from calibration plus held-out data that tuning never sees (fresh seed), with per-segment gain/RMS
bounds recorded as a contract for SG5b. Check the overflow proof and the 16-bit address footprint of the image.
Accept: graph interpreter = tensor interpreter (default schedules) bit-exactly on the target model; derived tolerance
holds on ≥ 20k held-out inputs; n_head ∈ {1, 2, 4} captured and quantized; tests marked `torch` where needed.
Verify: `make check && NANOGPT_DIR=... python3 -m pytest -q -m torch tests/frontend tests/model`

### SG4b — Memory plan, compile driver, G1 on RTL
`anvil.compile(module_or_graph, example_inputs, calib_inputs, params, schedules=None) -> CompiledModel` with
`run(x, backend)`: liveness-based static memory planner (activations reuse dead buffers, weights/LUTs packed, args
blocks + scratch/sink/spill after; checks the address-space size and reports the footprint), launch chain assembly,
input/output (de)quantization helpers. Replaces `bump_allocate` in the main path.
Accept (G1): nanoGPT `Block` → `anvil.compile` → RTL (iverilog) for the MLP block and the attention head: memory
golden-equal to the xf fixtures and within contract-8 tolerances vs PyTorch, with no hand-written assembly; planned
footprint ≤ the xf images' sizes; Verilator cycle table for the compiled chains next to the hand chains.
Verify: `make check && NANOGPT_DIR=... python3 -m pytest -q -m "rtl or torch" tests/driver`
