"""Random legal schedules (SG3a's primitive-sequence strategy, tests/schedule) compile and run bit-exactly on
isasim for every op, at WARPS_PER_CORE 1/2/4 x SIMT_STACK 0/1 (C4 level 4). A schedule the code generator cannot
fit (program over 256 words, no spare smem word for a masked store at SIMT_STACK=0) must fail with a clear error."""
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from anvil.backend import ProgramTooLarge
from anvil.codegen import CodegenError
from anvil.schedule import Schedule, ScheduleError, templates as T
from codegen_util import CONFIGS, OPS, case, check_kernel, params, target
from test_random_schedules import _draw_step, _template_names

CFGS = CONFIGS + [(8, 2, 0), (8, 2, 1), (8, 4, 1)]


@pytest.mark.parametrize("block,name", OPS)
@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_random_schedules_compile_bit_exact(block, name, data):
    k, bufs, _ = case(block, name)
    tpb, wpc, simt = data.draw(st.sampled_from(CFGS))
    p = params(tpb, wpc, simt)
    tgt = target(p)
    s = Schedule(k, tgt)
    if data.draw(st.booleans()):
        tpl = data.draw(st.sampled_from(_template_names(k.op)))
        full = T.TEMPLATES[tpl](Schedule(k, tgt)).trace
        for prim, a, kw in full[:data.draw(st.integers(0, len(full)))]:
            s.apply(prim, *a, **kw)
    for _ in range(data.draw(st.integers(0, 6))):
        prim, a, kw = _draw_step(data.draw, s)
        try:
            s.apply(prim, *a, **kw)
        except ScheduleError:
            continue
    try:
        fin = s.finalize()
    except ScheduleError:
        return
    try:
        check_kernel(fin, bufs, p, mir_level=not simt and data.draw(st.booleans()))
    except ProgramTooLarge:
        pass
    except CodegenError as e:
        assert "smem" in str(e) and not simt, e
