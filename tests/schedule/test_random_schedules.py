"""SG3a Accept: random primitive sequences on every op stay bit-exact with the graph interpreter (C4 level 2).

Each example optionally starts from a random prefix of a template's trace (so deep states, with smem, barriers
and cross-lane reductions, are reached), then draws primitives with arguments picked from the current loops.
Illegal draws must raise ScheduleError and leave the schedule untouched; every accepted state is interpreted
(with the race checker on) and compared, and the trace replays to the same IR.
"""
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from anvil.ir import print_ir
from anvil.schedule import Schedule, ScheduleError, Target, templates as T
from sched_cases import OPS, case

TARGETS = [Target(), Target(4, 2), Target(8, 2)]
KINDS = ("block", "warp", "lane")


def _template_names(op):
    if op in ("qmatmul", "qlinear"):
        return ["matmul_naive", "matmul_tiled"]
    if op in ("rmsnorm_int", "softmax_lut"):
        return ["rows_per_thread", "row_per_block"]
    return ["elementwise"]


def _divisor_or_any(draw, n):
    divs = [d for d in range(1, n + 1) if n % d == 0]
    return draw(st.one_of(st.sampled_from(divs), st.integers(1, max(1, n))))


def _draw_step(draw, s):
    loops = s.loops()
    tags = [x[0] for x in loops]
    ext = {x[0]: x[1] for x in loops}
    tag = draw(st.sampled_from(tags))
    tag2 = draw(st.sampled_from(tags))
    tail = draw(st.sampled_from(["mask", "pad"]))
    prim = draw(st.sampled_from(["split", "split", "reorder", "fuse", "tile", "unroll", "interchange", "bind", "bind",
                                 "replicate", "distribute", "cross_lane_reduce", "cache_read", "compute_at"]))
    if prim == "split":
        return prim, (tag, _divisor_or_any(draw, ext[tag]), tail), {}
    seq = lambda kind: [x[0] for x in loops if x[2] == "seq" and x[3] == kind] or tags  # noqa: E731
    if prim == "compute_at":
        stages = [o.tag.data for o in s.kernel.func.body.block.ops if hasattr(o, "tag") and o.tag is not None]
        return prim, (draw(st.sampled_from(stages or tags)), tag2), {}
    if prim == "interchange":
        return prim, (draw(st.sampled_from(seq("spatial"))), draw(st.sampled_from(seq("reduce")))), {}
    if prim in ("reorder", "fuse"):
        return prim, (tag, tag2), {}
    if prim == "tile":
        return prim, (tag, tag2, _divisor_or_any(draw, ext[tag]), _divisor_or_any(draw, ext[tag2])), {"tail": tail}
    if prim == "unroll":
        return prim, (tag, _divisor_or_any(draw, ext[tag])), {}
    if prim == "bind":
        return prim, (tag, draw(st.sampled_from(KINDS))), {}
    if prim == "replicate":
        return prim, (draw(st.sampled_from(["warp", "lane"])), draw(st.integers(1, 4))), {}
    if prim in ("distribute", "cross_lane_reduce"):
        return prim, (draw(st.sampled_from(seq("spatial" if prim == "distribute" else "reduce"))),), {"tail": tail}
    sym = draw(st.sampled_from(s.kernel.reads))
    return prim, (sym, tag), {"scope": draw(st.sampled_from(["block", "warp"]))}


def _check(s, bufs, ref, name):
    assert np.array_equal(s.run(bufs)[name], ref), s.trace
    try:
        fin = s.finalize()
    except ScheduleError:
        return False
    assert np.array_equal(fin.run(bufs)[name], ref), s.trace
    return True


@pytest.mark.parametrize("block,name", OPS)
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_random_primitive_sequences_bit_exact(block, name, data):
    k, bufs, ref = case(block, name)
    tgt = data.draw(st.sampled_from(TARGETS))
    s = Schedule(k, tgt)
    if data.draw(st.booleans()):                       # start inside a template's trace
        tpl = data.draw(st.sampled_from(_template_names(k.op)))
        full = T.TEMPLATES[tpl](Schedule(k, tgt)).trace
        for prim, a, kw in full[:data.draw(st.integers(0, len(full)))]:
            s.apply(prim, *a, **kw)
    for _ in range(data.draw(st.integers(1, 8))):
        prim, a, kw = _draw_step(data.draw, s)
        before, n = print_ir(s.kernel.module), len(s.trace)
        try:
            s.apply(prim, *a, **kw)
        except ScheduleError:
            assert len(s.trace) == n and print_ir(s.kernel.module) == before
            continue
        assert np.array_equal(s.run(bufs)[name], ref), s.trace
    _check(s, bufs, ref, name)
    again = Schedule.replay(k, s.to_json(), tgt)
    assert print_ir(again.kernel.module) == print_ir(s.kernel.module)
