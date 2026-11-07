"""SG3b Accept: the full mlp and attention fixture chains, built as graph IR (anvil.frontend.from_image), lowered,
scheduled, compiled and run as one Job on isasim, produce memory golden-equal to xf.golden (the fixture goldens)
for every intermediate value."""
import numpy as np
import pytest

from anvil.codegen import compile_kernel, program_job
from anvil.frontend import from_image
from anvil.lower.graph_to_tensor import lower
from anvil.runtime import run
from anvil.schedule import Schedule, ScheduleError, templates as T
from codegen_util import bad_flags, params, target
from xf import fixture

PLANS = {
    "default": lambda op: None,
    "tiled": lambda op: {"qmatmul": "matmul_tiled", "qlinear": "matmul_tiled",
                         "rmsnorm_int": "row_per_block", "softmax_lut": "row_per_block"}.get(op),
}


def compile_chain(prog, p, plan):
    tgt = target(p)
    out = {}
    for k in prog.kernels:
        tpl = PLANS[plan](k.op)
        s = T.default_schedule(k, tgt)
        if tpl is not None:
            try:
                s = T.TEMPLATES[tpl](Schedule(k, tgt))
            except ScheduleError:
                pass
        out[k.name] = compile_kernel(s.finalize(), p)
    return out


@pytest.mark.parametrize("plan", sorted(PLANS))
@pytest.mark.parametrize("cfg", [(4, 1, 0), (4, 2, 1), (4, 4, 0), (8, 2, 1)], ids=str)
@pytest.mark.parametrize("block", ["mlp_block", "attention"])
def test_chain_golden_on_isasim(block, cfg, plan):
    d = fixture.load(fixture.FIXTURE_DIR / f"{block}_0.json")
    module, consts = from_image.BLOCKS[block](d["image"])
    prog = lower(module)
    p = params(*cfg)
    cks = compile_chain(prog, p, plan)
    jp = program_job(prog, cks, {**consts, "x": d["inputs"]["x"]}, p)
    r = run(jp.job)
    assert not bad_flags(r, p.SIMT_STACK)
    assert not jp.stray_writes(r.memory, unchanged=list(consts))      # whole memory (A1-M3)
    for name, want in d["golden"].items():
        if name in jp.buffers:
            assert np.array_equal(jp.read(r.memory, name), want), name
    assert all(n in jp.buffers for n in prog.outputs)
