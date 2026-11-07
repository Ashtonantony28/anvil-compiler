"""G1 report (SG4b): the xf MLP block and attention head compiled from PyTorch by `anvil.compile`, next to the
hand-written xf chains, on Verilator (cycle-exact with the iverilog RTL, ADR-001).

    NANOGPT_DIR=... PYTHONPATH=sim python3 -m anvil.g1_report        # rewrites docs/g1-report.md

Compiled: nanoGPT Extended `Block` built as xf/export.py builds it, captured and quantized by anvil.frontend on xf's
calibration sets, scheduled with the driver's "default" and "tiled" presets, memory-planned by anvil.memplan.
Hand: `xf.mlp.launches` / `xf.attention.launches` on the fixture image with the naive and tiled matmul kernels.
Every run is checked golden-equal (the hand chains against xf.golden, the compiled ones against the fixture).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["", "COALESCE=1", "DCACHE_LINES=16", "COALESCE=1 DCACHE_LINES=16"]
BLOCKS = ("mlp_block", "attention")
OUT = {"mlp_block": "out", "attention": "y"}
SCHEDULES = {"default": "naive", "tiled": "tiled"}          # compiled preset -> the hand chain it is compared with
NAMES = {      # FGraph node names -> xf image segment names (= tests/frontend/conftest.py)
    "mlp_block": {"b.ln_2": "h", "b.ff.gate_proj": "gate", "b.ff.up_proj": "up", "mul_2": "act",
                  "b.ff.down_proj": "down", "add_1": "out"},
    "attention": {"b.ln_1": "h", "b.attn.c_attn.0": "q", "b.attn.c_attn.1": "k", "b.attn.c_attn.2": "v",
                  "scaled_dot_product_attention.kT": "kT", "scaled_dot_product_attention.scores": "scores",
                  "scaled_dot_product_attention.p": "p", "scaled_dot_product_attention": "o",
                  "b.attn.c_proj": "y"},
}
IMAGE_WORDS = {"mlp_block": 2005, "attention": 1847}


def config_params(cfg: str):
    from .runtime import GpuParams
    return GpuParams(**{k: int(v) for k, v in (kv.split("=") for kv in cfg.split())})


def torch_modules(nanogpt_dir=None):
    """{block: torch module} for the xf Block (xf/export.py::build_block), as tests/frontend wraps it."""
    import torch
    from xf import export as xe

    b = xe.build_block(nanogpt_dir or os.environ["NANOGPT_DIR"])

    class MlpBlock(torch.nn.Module):
        def __init__(self, b):
            super().__init__()
            self.b = b

        def forward(self, x):
            return x + self.b.ff(self.b.ln_2(x))

    class AttentionHead(torch.nn.Module):
        def __init__(self, b):
            super().__init__()
            self.b = b

        def forward(self, x):
            return self.b.attn(self.b.ln_1(x))

    return {"mlp_block": MlpBlock(b), "attention": AttentionHead(b)}


def calib(block):
    from xf import export as xe
    n = xe.N_CALIB if block == "mlp_block" else xe.ATTN_N_CALIB
    return np.random.default_rng(xe.SEED).standard_normal((n, xe.T, xe.N_EMBD))


def compile_block(module, block, params=None, schedules=None):
    """anvil.compile of one xf block module on xf's calibration set."""
    import torch
    from xf import export as xe

    from .driver import compile
    ex = torch.zeros(1, xe.T, xe.N_EMBD, dtype=torch.float64)
    return compile(module, (ex,), calib(block), params, schedules, headroom=xe.HEADROOM, names=NAMES[block],
                   sym_name=block)


def hand_chain(block, kind, params, backend="verilator", index=0):
    """{segment: cycles} of the hand-written xf chain (golden-checked)."""
    from xf import attention, export, fixture, mlp

    from .runtime import Job, Launch, run
    fx = fixture.load(fixture.FIXTURE_DIR / f"{block}_{index}.json")
    img, x = fx["image"], fx["inputs"]["x"]
    img.write("x", x)
    mod, golden = (mlp, export.mlp_block_int) if block == "mlp_block" else (attention, attention.attention_int)
    ls = mod.launches(img, kind)
    r = run(Job(list(img.words), [Launch(p, t) for _, (p, t) in ls], params), backend)
    for seg, want in golden(img, x).items():
        if not np.array_equal(img.read(seg, r.memory), want):
            raise AssertionError(f"hand {block} {kind}: {seg} differs from the golden")
    return {seg: lr.cycles for (seg, _), lr in zip(ls, r.launches)}


def compiled_chain(cm, block, backend="verilator", index=0):
    """{kernel: cycles} of a compiled chain (golden-checked on every fixture segment)."""
    from xf import fixture
    fx = fixture.load(fixture.FIXTURE_DIR / f"{block}_{index}.json")
    r = cm.run(fx["inputs"]["x"], backend, quantized=True, keep="all")
    for seg, want in fx["golden"].items():
        if not np.array_equal(r.ints[seg], want):
            raise AssertionError(f"compiled {block}: {seg} differs from the golden")
    r2 = cm.run(fx["inputs"]["x"], backend, quantized=True)          # the reuse plan: same cycles, output only
    if not np.array_equal(r2.ints[OUT[block]], fx["golden"][OUT[block]]):
        raise AssertionError(f"compiled {block} (reuse plan): output differs from the golden")
    for x in (r, r2):
        if x.stray_writes():                                         # whole memory (AUDIT-A1 M3)
            raise AssertionError(f"compiled {block}: words outside the plan's buffers or in consts changed: "
                                 f"{x.stray_writes()[:4]}")
    if r2.cycles != r.cycles:
        raise AssertionError(f"compiled {block}: cycles depend on the memory plan: {r.cycles} vs {r2.cycles}")
    return r.cycles


def collect(configs=CONFIGS, backend="verilator", modules=None):
    modules = modules or torch_modules()
    rows, plans = [], {}
    for cfg in configs:
        p = config_params(cfg)
        for block in BLOCKS:
            for preset, kind in SCHEDULES.items():
                cm = compile_block(modules[block], block, p, preset)
                c = compiled_chain(cm, block, backend)
                h = hand_chain(block, kind, p, backend)
                words = {k.name: cm.kernels[k.name].program.stats["words"] for k in cm.program.kernels}
                rows.append((cfg, block, preset, kind, h, c, words))
                plans[(cfg, block, preset)] = cm.plan
    return rows, plans


def cycle_table(rows) -> str:
    out = ["| block | config | compiled schedule | hand chain | kernel | hand cycles | compiled cycles | ratio | "
           "compiled words |", "|---|---|---|---|---|---:|---:|---:|---:|"]
    for cfg, block, preset, kind, h, c, words in rows:
        for k in c:
            out.append(f"| {block} | {cfg or 'default'} | {preset} | {kind} | {k} | {h[k]} | {c[k]} | "
                       f"{c[k] / h[k]:.2f} | {words[k]} |")
        th, tc = sum(h.values()), sum(c.values())
        out.append(f"| {block} | {cfg or 'default'} | {preset} | {kind} | **total** | **{th}** | **{tc}** | "
                   f"**{tc / th:.2f}** | |")
    return "\n".join(out)


def footprint_table(plans) -> str:
    out = ["| block | schedule | consts | activations (reuse) | activations (no reuse) | args | scratch | "
           "footprint | xf image |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for (cfg, block, preset), mp in plans.items():
        if cfg:
            continue
        r = mp.regions
        out.append(f"| {block} | {preset} | {r['consts'][1]} | {r['activations'][1]} | {mp.unshared} | "
                   f"{r['args'][1]} | {r['scratch'][1]} | {mp.footprint} | {IMAGE_WORDS[block]} |")
    return "\n".join(out)


def render(rows, plans) -> str:
    return f"""# G1 report: the xf workloads compiled from PyTorch

Generated by `NANOGPT_DIR=... PYTHONPATH=sim python3 -m anvil.g1_report` (Verilator, cycle-exact with the iverilog
RTL per ADR-001); do not edit. `tests/driver/test_g1_rtl.py` checks the default-configuration rows are current.

The nanoGPT Extended `Block` (built as `xf/export.py::build_block`) is captured and quantized by `anvil.frontend`,
then `anvil.compile` lowers, schedules and compiles every op and lays the chain out with `anvil.memplan`; no
hand-written assembly is involved. Compiled schedule presets: `default` (matmuls one thread per output,
rmsnorm/softmax one thread per row, elementwise one thread per element) is compared with the hand chain built on
`matmul_naive.asm`; `tiled` (matmuls `matmul_tiled`, rmsnorm/softmax `row_per_block`) with the hand chain on
`matmul_tiled.asm`. Hand chains: `xf.mlp.launches` / `xf.attention.launches` on the fixture image. Every run is
golden-equal to the fixture (`mlp_block_0`, `attention_0`) on every segment, and the compiled chain's cycles are
the same under the reuse plan and the keep-all plan. `THREADS_PER_BLOCK=4`, `WARPS_PER_CORE=1`, `WORD_BITS=32`.

## Cycles

{cycle_table(rows)}

## Memory footprint (words, default configuration)

`anvil.memplan`: consts packed first, activations at liveness-assigned offsets (dead buffers reused), then one
args block per launch and the shared sink/spill scratch. The xf images also hold the requant (M, s) words, which
compiled programs carry as immediates, and lay every activation out separately; they do not count the args blocks
`xf.kernels.program` appends.

{footprint_table(plans)}
"""


def main(path=ROOT / "docs" / "g1-report.md", configs=CONFIGS):
    rows, plans = collect(configs)
    text = render(rows, plans)
    Path(path).write_text(text)
    return rows, plans, text


if __name__ == "__main__":
    print(main()[2])
