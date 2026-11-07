"""SG4b Accept (G1, isasim part): nanoGPT's Block -> anvil.compile -> the MLP block and the attention head, every
fixture case golden-equal on isasim (every segment), within contract-8 tolerance vs PyTorch, footprint within the
xf images, and no hand-written assembly anywhere on the path."""
import numpy as np
import pytest

from anvil import g1_report
from xf import export as xe

from g1_util import BLOCKS, OUT, assert_golden, compiled, fx, modules

pytestmark = pytest.mark.torch


def test_names_match_the_frontend_tests():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("frontend_conftest", Path(sys.path[0]) / "conftest.py")
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)
    sys.path.pop(0)
    assert conf.NAMES == g1_report.NAMES


@pytest.mark.parametrize("preset", ["default", "tiled"])
@pytest.mark.parametrize("block", BLOCKS)
def test_fixture_cases_golden_on_isasim(block, preset):
    cm = compiled(block, preset)
    for i in range(4):
        assert_golden(cm, block, i)


@pytest.mark.parametrize("block", BLOCKS)
def test_within_contract8_tolerance(block):
    """|dequant(compiled) - torch| <= the fixture tol: on the fixture references (cases 0-3) and on xf's seeded
    check inputs (export.N_CHECK for the MLP, the first 500 of ATTN_N_CHECK for attention), the float reference
    run on dequant(x_q) as contract 8 states."""
    import torch

    cm, out = compiled(block, "tiled"), OUT[block]
    tol = fx(block)["tol"]
    for i in range(4):
        d = fx(block, i)
        r = cm.run(d["inputs"]["x"], quantized=True)
        assert np.abs(r.output - d["reference"][out]).max() <= tol, i
    n = xe.N_CHECK if block == "mlp_block" else 500
    sx = cm.scale("x")
    xq = xe.quantize(np.random.default_rng(xe.SEED + 1).standard_normal((n, xe.T, xe.N_EMBD)), sx)
    with torch.no_grad():
        ref = modules()[block](torch.as_tensor(xq * sx)).numpy()
    err = max(np.abs(cm.run(x, quantized=True).output - rf).max() for x, rf in zip(xq, ref))
    assert err <= tol, f"{block}: {err / cm.scale(out):.3f} LSB > {tol / cm.scale(out):.3f}"


@pytest.mark.parametrize("block", BLOCKS)
def test_float_input_path(block):
    """run(float x) quantizes with the calibrated input scale: equal to the integer path on x_q * scale."""
    cm = compiled(block)
    d = fx(block, 2)
    x = np.asarray(d["inputs"]["x"]) * cm.scale("x")
    r = cm.run(x[None])                                   # a leading batch dim of 1 is accepted
    assert np.array_equal(r.ints[OUT[block]], d["golden"][OUT[block]])


@pytest.mark.parametrize("block", BLOCKS)
def test_footprint_within_xf_image(block):
    for preset in ("default", "tiled"):
        mp = compiled(block, preset).plan
        assert mp.footprint <= g1_report.IMAGE_WORDS[block], mp.report()
        assert mp.footprint <= len(fx(block)["image"].words)


def test_no_hand_written_assembly(monkeypatch):
    """Compile and run both blocks with every hand-kernel entry point of the simulator disabled; no launched
    program equals a hand kernel (ARGS prologue words masked)."""
    from pathlib import Path

    from xf import asm, attention, elementwise, kernels, mlp

    hand = set()
    for f in sorted((Path(kernels.__file__).resolve().parents[1] / "kernels").glob("*.asm")):
        w = asm.assemble(f.read_text())
        hand.add(tuple(x for i, x in enumerate(w) if i not in (0, 4)))
    assert hand

    def boom(*a, **k):
        raise AssertionError("hand-written kernel used")
    for mod, names in ((asm, ["assemble"]), (kernels, ["program", "matmul"]), (mlp, ["launches"]),
                       (attention, ["launches"])):
        for n in names:
            monkeypatch.setattr(mod, n, boom)
    for n in dir(elementwise):
        if not n.startswith("_") and callable(getattr(elementwise, n)):
            monkeypatch.setattr(elementwise, n, boom)
    from anvil import g1_report as gr
    for block in BLOCKS:
        for preset in ("default", "tiled"):
            cm = gr.compile_block(modules()[block], block, None, preset)
            full, _ = assert_golden(cm, block, 0)
            for launch in full.job.job.launches:
                assert tuple(x for i, x in enumerate(launch.program) if i not in (0, 4)) not in hand
