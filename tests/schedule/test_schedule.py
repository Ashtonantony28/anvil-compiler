"""SG3a schedule primitives and templates: bit-exact (C4 level 2), canonical, replayable; illegal schedules rejected
with clear errors and no change."""
import numpy as np
import pytest

from anvil.ir import parse_ir, print_ir
from anvil.ir import tensor as t
from anvil.schedule import Schedule, ScheduleError, Target, canonicalize, templates as T, validate
from anvil.schedule import ir_util as u

from sched_cases import OPS, case

TARGETS = [Target(), Target(4, 2), Target(8, 2), Target(8, 4)]


def _check(s, bufs, ref, name):
    assert np.array_equal(s.run(bufs)[name], ref)
    fin = s.finalize()
    validate(fin, s.target)
    assert np.array_equal(fin.run(bufs)[name], ref)
    text = print_ir(fin.module)
    assert print_ir(parse_ir(text)) == text
    return fin


def _templates(op):
    if op in ("qmatmul", "qlinear"):
        return ["matmul_naive", "matmul_tiled"]
    if op in ("rmsnorm_int", "softmax_lut"):
        return ["rows_per_thread", "row_per_block"]
    return ["elementwise"]


@pytest.mark.parametrize("tgt", TARGETS, ids=lambda x: f"tpb{x.threads_per_block}w{x.warps_per_core}")
@pytest.mark.parametrize("block,name", OPS)
def test_templates_bit_exact_and_canonical(block, name, tgt):
    k, bufs, ref = case(block, name)
    for tpl in _templates(k.op):
        s = T.TEMPLATES[tpl](Schedule(k, tgt))
        _check(s, bufs, ref, name)
        again = Schedule.replay(k, s.to_json(), tgt)
        assert print_ir(again.kernel.module) == print_ir(s.kernel.module)
    _check(T.default_schedule(k, tgt), bufs, ref, name)


def _count(fin, cls, space=None):
    return [o for o in fin.func.walk() if isinstance(o, cls)
            and (space is None or t.space_of(o.result.type) == space)]


def test_matmul_tiled_mirrors_the_hand_kernel():
    """matmul_tiled.asm: ceil(M/4) * N threads, a 4 x 4 A tile in SMEM (4 * blockDim words), BAR after staging and
    after each K chunk, four register accumulators per thread."""
    k, bufs, ref = case("mlp_block", "gate")                      # 4x8 @ 8x16
    fin = _check(T.matmul_tiled(Schedule(k)), bufs, ref, "gate")
    bnd = u.bound(fin.func)
    assert u.extent(bnd["block"]) * 4 == 1 * 16                    # ceil(4/4) * N threads
    assert u.extent(bnd["lane"]) == 4
    (sm,) = _count(fin, t.AllocOp, "smem")
    assert sm.result.type.get_shape() == (4, 4)
    (acc,) = _count(fin, t.AllocOp, "reg")
    assert acc.result.type.get_shape() == (4, 1)
    assert len(_count(fin, t.BarrierOp)) == 2
    assert u.extent(u.find(fin.func, "k.o")) == 2                  # K / 4 chunks


def test_matmul_naive_mirrors_the_hand_kernel():
    k, bufs, ref = case("attention", "scores")                    # 4x8 @ 8x4: M*N = 16 threads
    fin = _check(T.matmul_naive(Schedule(k)), bufs, ref, "scores")
    bnd = u.bound(fin.func)
    assert u.extent(bnd["block"]) * u.extent(bnd["lane"]) == 16
    assert not _count(fin, t.AllocOp, "smem") and not _count(fin, t.BarrierOp)
    assert [x for x in Schedule(fin).loops() if x[0] == "k"][0][1] == 8


def test_row_per_block_uses_cross_lane_reductions():
    k, bufs, ref = case("attention", "p")
    fin = _check(T.row_per_block(Schedule(k, Target(4, 2))), bufs, ref, "p")
    assert len(_count(fin, t.AllocOp, "smem")) == 2                # max and sum partials
    assert len(_count(fin, t.BarrierOp)) == 4
    assert not fin.temps                                           # compute_at made the row temps registers


# ------------------------------------------------------------------------------------- illegal schedules

def _rejects(s, match, prim, *a, **kw):
    before, trace = print_ir(s.kernel.module), list(s.trace)
    with pytest.raises(ScheduleError, match=match):
        s.apply(prim, *a, **kw)
    assert print_ir(s.kernel.module) == before and s.trace == trace       # atomic


def test_bad_bindings_are_rejected():
    k, _, _ = case("mlp_block", "gate")
    s = Schedule(k)
    _rejects(s, "only statement", "bind", "j", "block")
    _rejects(s, "reduction loop", "bind", "k", "lane")
    s.bind("i", "block")
    _rejects(s, "already bound", "bind", "j", "block")
    _rejects(s, "16 lanes > 4 lanes per warp", "bind", "j", "lane")
    _rejects(s, "2 warps > WARPS_PER_CORE=1", "replicate", "warp", 2)
    s2 = Schedule(k).fuse("i", "j").split("i_j", 4)
    s2.bind("i_j.o", "block").bind("i_j.i", "lane")
    _rejects(s2, "bind block, then warp, then lane", "replicate", "warp", 1)


def test_too_many_threads_per_launch():
    k, _, _ = case("mlp_block", "gate")                           # 64 outputs
    s = Schedule(k, Target(threads_per_block=4)).fuse("i", "j")
    _rejects(s, r"64 blocks x 4 threads = 256 > 255 threads per launch", "bind", "i_j", "block")
    big = Schedule(k, Target(threads_per_block=16, warps_per_core=1)).fuse("i", "j").split("i_j", 2)
    _rejects(big, r"32 blocks x 16 threads = 512 > 255", "bind", "i_j.o", "block")


def test_smem_over_budget():
    k, _, _ = case("mlp_block", "down")                           # act 4x16 @ w_down 16x8
    s = Schedule(k).split("i", 4).bind("i.o", "block").bind("i.i", "lane")
    s.cache_read("w_down", "j")                                    # 16 words: fine
    _rejects(s, r"shared memory over budget: 16 \+ 64 words > 64", "cache_read", "act", "j")


def test_warp_scope_budget_counts_every_warp():
    k, bufs, ref = case("mlp_block", "down")                      # act 4x16 @ w_down 16x8
    w = Schedule(k, Target(4, 2)).split("i", 2).bind("i.o", "block").bind("i.i", "warp")
    w.split("j", 2).reorder("j.i", "j.o").bind("j.i", "lane")
    w.cache_read("w_down", "j.o", scope="warp")                   # 16 x 2 words per warp, x 2 warps = 64
    assert np.array_equal(w.run(bufs)["down"], ref)
    _rejects(w, r"over budget: 64 \+ 32 words > 64 per block \(32 per warp at WARPS_PER_CORE=2\)",
             "cache_read", "act", "j.o", scope="warp")


def test_smem_primitives_come_after_binding():
    k, bufs, ref = case("mlp_block", "gate")
    s = Schedule(k).split("i", 2).bind("i.o", "block").cache_read("w_gate", "j")
    _rejects(s, "bind threads before cache_read", "bind", "i.i", "lane")
    assert np.array_equal(s.finalize().run(bufs)["gate"], ref)


def test_pad_mask_rules():
    k, bufs, ref = case("attention", "kT")                        # 8x4
    _rejects(Schedule(case("mlp_block", "gate")[0]), "needs a spatial loop", "split", "k", 3, "pad")
    s = Schedule(k).split("d0", 3, "pad")                         # 8 = 3*3 - 1 wrapped iteration
    assert np.array_equal(s.run(bufs)["kT"], ref)
    s = Schedule(k).split("d0", 3, "mask")
    assert np.array_equal(s.run(bufs)["kT"], ref)
    assert any(isinstance(o, t.GuardOp) for o in s.kernel.func.walk())
    # mask cannot wrap a barrier
    g, gb, gr = case("mlp_block", "gate")
    s2 = Schedule(g).split("i", 4).split("j", 4).reorder("i.o", "j.o", "j.i", "i.i").fuse("i.o", "j.o", name="b") \
        .bind("b", "block").bind("j.i", "lane").cache_read("h", "i.i")
    _rejects(s2, "barrier under a guard", "split", "i.i", 3, "mask")
    _rejects(s2, "updates a buffer it also reads", "split", "i.i", 3, "pad")   # smem staging is state
    s2.split("k", 3, "mask")                                    # reduction tail: mask, no barrier inside k
    assert np.array_equal(s2.run(gb)["gate"], gr)


def test_order_and_structure_errors():
    k, bufs, ref = case("attention", "p")
    s = Schedule(k)
    _rejects(s, "only statement", "bind", "i", "block")            # three stages
    _rejects(s, "read outside", "compute_at", "max.i", "i")        # sum also reads max
    _rejects(s, "comes before|inside the stage", "compute_at", "sum.i", "max.j")
    _rejects(s, "not a perfect nest", "reorder", "i", "max.j")
    s.compute_at("sum.i", "i").compute_at("max.i", "i").bind("i", "block").replicate("lane", 4)
    _rejects(s, "not a reduction", "cross_lane_reduce", "j")
    s.cross_lane_reduce("max.j")
    _rejects(s, "lane is already bound", "replicate", "warp", 1)
    _rejects(s, "is written by the kernel", "cache_read", "p", "j")
    _rejects(s, "unknown primitive", "frobnicate")
    s.distribute("j")
    assert np.array_equal(s.run(bufs)["p"], ref)


def test_finalize_rejects_non_canonical():
    k, bufs, ref = case("mlp_block", "gate")
    s = Schedule(k).split("i", 2).bind("i.o", "block").replicate("lane", 4)
    fin = s.finalize()                                           # redundant lanes: legal
    assert np.array_equal(fin.run(bufs)["gate"], ref)
    s = Schedule(k).split("i", 2).bind("i.o", "block").bind("i.i", "lane").interchange("j", "k").unroll("j", 1)
    assert np.array_equal(s.run(bufs)["gate"], ref)
    with pytest.raises(ScheduleError, match="16 register-buffer words per thread > 8"):
        s.finalize()
    with pytest.raises(ScheduleError, match="not fully unrolled"):
        canonicalize(s.kernel, Target(max_reg_words=64))


def test_canonical_form_of_an_unscheduled_kernel():
    k, bufs, ref = case("mlp_block", "h")                         # two stages and temps, single thread
    fin = canonicalize(k)
    bnd = u.bound(fin.func)
    assert [u.tag_of(bnd[x]) for x in ("block", "warp", "lane")] == ["_block", "_warp", "_lane"]
    assert set(fin.temps) == {"h.rs", "h.k"}
    assert np.array_equal(fin.run(bufs)["h"], ref)


def test_target_from_params():
    from anvil.runtime import GpuParams

    tg = Target.from_params(GpuParams(THREADS_PER_BLOCK=8, WARPS_PER_CORE=2))
    assert (tg.threads_per_block, tg.lanes_per_warp) == (8, 4)


@pytest.mark.parametrize("at", ["i", "j"])
def test_compute_at_every_consumer_loop(at):
    k, bufs, ref = case("mlp_block", "h")
    s = Schedule(k).compute_at("stat.i", at)
    assert not s.kernel.temps and np.array_equal(s.run(bufs)["h"], ref)
    k, bufs, ref = case("attention", "p")
    s = Schedule(k).compute_at("sum.i", at).compute_at("max.i", at)
    assert not s.kernel.temps and np.array_equal(s.run(bufs)["p"], ref)
    if at == "i":
        _check(s.split("i", 2).bind("i.o", "block").bind("i.i", "lane"), bufs, ref, "p")


def test_compute_at_a_batch_loop_keeps_a_register_row():
    """At an outer loop the region is a whole column of rows: a reg buffer of that shape, its loop unrolled."""
    from anvil.ir.graph import GraphBuilder
    from anvil.interp import graph as gi
    from anvil.lower.graph_to_tensor import lower

    rng = np.random.default_rng(2)
    g = GraphBuilder("b")
    p = g.softmax_lut("p", g.input("x", [2, 3, 3]), g.const("e", [256], "i32"), 50, True)
    g.output(p)
    x, e = rng.integers(-128, 128, (2, 3, 3)), rng.integers(1, 1000, 256)
    ref = gi.run(g.module(), {"x": x}, {"e": e}).values["p"]
    k = lower(g.module()).kernel("p")
    s = Schedule(k).compute_at("sum.b0", "b0").compute_at("max.b0", "b0")
    assert np.array_equal(s.run({"x": x, "e": e})["p"], ref)
    regs = [o for o in s.kernel.func.walk() if isinstance(o, t.AllocOp) and t.space_of(o.result.type) == "reg"
            and o.result.type.get_shape() == (1, 3)]
    assert len(regs) == 2
    s.bind("b0", "block")
    with pytest.raises(ScheduleError, match="'i', which is not fully unrolled"):
        s.finalize()
    fin = s.unroll("i").finalize()
    assert np.array_equal(fin.run({"x": x, "e": e})["p"], ref)
