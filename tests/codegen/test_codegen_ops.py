"""SG3b Accept: every op x the schedule templates (and the unscheduled canonical form) x WARPS_PER_CORE 1/2/4 x
SIMT_STACK 0/1 compiles and runs bit-exactly on isasim against the tensor interpreter, with no divergent /
store_races / stores_conflict flags (C4 levels 3-4); at SIMT_STACK=0 the virtual and the allocated MIR also run on
the MIR interpreter with the same memory (C4 level 3)."""
import pytest

from anvil.schedule import ScheduleError
from codegen_util import CONFIGS, OPS, case, check_kernel, params, schedule, target, templates_for

GRID = [(blk, name, tpl) for blk, name in OPS for tpl in templates_for(case(blk, name)[0].op)]


@pytest.mark.parametrize("tpb,wpc,simt", CONFIGS, ids=lambda x: str(x))
@pytest.mark.parametrize("blk,name,tpl", GRID, ids=lambda x: str(x))
def test_op_template_grid(blk, name, tpl, tpb, wpc, simt):
    k, bufs, ref = case(blk, name)
    p = params(tpb, wpc, simt)
    try:
        fin = schedule(k, tpl, target(p)).finalize()
    except ScheduleError as e:
        pytest.skip(f"template not applicable: {e}")
    ck, r = check_kernel(fin, bufs, p, mir_level=not simt)
    assert ck.program.stats["words"] <= 256
    if simt and ck.code.stats["regions"]["sync"]:
        assert ck.code.stats["simt_stack_entries"] <= 8


@pytest.mark.parametrize("blk,name,tpl", [g for g in GRID if g[2] != "none"], ids=lambda x: str(x))
def test_op_templates_8_threads(blk, name, tpl):
    """THREADS_PER_BLOCK=8 with 2 warps of 4 lanes, both predication flavours."""
    k, bufs, _ = case(blk, name)
    for simt in (0, 1):
        p = params(8, 2, simt)
        try:
            fin = schedule(k, tpl, target(p)).finalize()
        except ScheduleError as e:
            pytest.skip(f"template not applicable: {e}")
        check_kernel(fin, bufs, p)
