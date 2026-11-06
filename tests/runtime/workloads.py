"""Jobs built from the simulator's xf fixtures, mirroring sim/test/test_xf_bench.py and test_xf_{mlp_block,attention}."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anvil.runtime import GpuParams, Job, Launch

SIM = Path(__file__).resolve().parents[2] / "sim"
BENCH_MLP = SIM / "bench" / "xf-mlp.json"
KINDS = ("naive", "tiled")


def _launch(prog_threads):
    prog, threads = prog_threads
    return Launch(prog, threads)


def bench_mlp_job(params: GpuParams = GpuParams()):
    """sim/test/test_xf_bench.py::bench_mlp: the naive then the tiled chain on mlp_block_0, one memory.
    Returns (job, [(kind, case)] per launch, image, {kind: acts}, golden)."""
    from xf import export, fixture, mlp
    fx = fixture.load(fixture.FIXTURE_DIR / "mlp_block_0.json")
    img, x = fx["image"], fx["inputs"]["x"]
    launches, names, acts = [], [], {}
    for kind in KINDS:
        acts[kind] = mlp.alloc_acts(img, f"{kind}.")
        img.write(acts[kind]["x"], x)
        for out, pt in mlp.launches(img, kind, acts[kind]):
            launches.append(_launch(pt))
            names.append(tuple(out.split(".")))
    return Job(list(img.words), launches, params), names, img, acts, export.mlp_block_int(img, x)


def mlp_chain_job(kind: str, params: GpuParams = GpuParams(), index: int = 0):
    """One MLP block chain (xf.mlp.launches) on mlp_block_<index>. Returns (job, image, golden by segment)."""
    from xf import export, fixture, mlp
    fx = fixture.load(fixture.FIXTURE_DIR / f"mlp_block_{index}.json")
    img, x = fx["image"], fx["inputs"]["x"]
    img.write("x", x)
    launches = [_launch(pt) for _, pt in mlp.launches(img, kind)]   # allocates kernel argument segments
    job = Job(list(img.words), launches, params)
    want = export.mlp_block_int(img, x)
    for n, g in fx["golden"].items():
        assert np.array_equal(want[n], g), f"golden {n} is stale"
    return job, img, want


def attention_job(kind: str = "tiled", params: GpuParams = GpuParams(), index: int = 0):
    """The attention head chain (xf.attention.launches) on attention_<index>. Returns (job, image, golden)."""
    from xf import attention, fixture
    fx = fixture.load(fixture.FIXTURE_DIR / f"attention_{index}.json")
    img, x = fx["image"], fx["inputs"]["x"]
    img.write("x", x)
    launches = [_launch(pt) for _, pt in attention.launches(img, kind)]   # allocates kernel argument segments
    job = Job(list(img.words), launches, params)
    return job, img, attention.attention_int(img, x)


def golden_mismatches(img, memory, want, acts=None):
    """Segments whose read-back differs from the golden chain."""
    acts = acts or {n: n for n in want}
    return [n for n in want if n in acts and not np.array_equal(img.read(acts[n], memory), want[n])]


def bench_rows(config: str):
    return [r for r in json.loads(BENCH_MLP.read_text())["rows"] if r["params"] == config]


def config_params(config: str) -> GpuParams:
    """A bench/run.py XF_CONFIGS entry ("COALESCE=1 DCACHE_LINES=16") as GpuParams at the xf widths."""
    return GpuParams(**{k: int(v) for k, v in (kv.split("=") for kv in config.split())})


def isa_fixtures(word_bits: int = 32):
    from xf.isa_fixtures import FIXTURE_DIR
    out = {}
    for p in sorted(FIXTURE_DIR.glob("*.json")):
        fx = json.loads(p.read_text())
        if fx["params"]["word_bits"] == word_bits:
            out[p.stem] = fx
    return out
