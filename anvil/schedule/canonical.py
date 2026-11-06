"""Canonical form for codegen (SG3b): `canonicalize` adds missing block/warp/lane levels as extent-1 loops and
`validate` checks every rule listed in anvil.schedule's docstring."""
import math

from ..ir import tensor as t
from . import ir_util as u
from .target import ScheduleError, Target

LEVELS = ("block", "warp", "lane")


def canonicalize(kernel, target: Target = Target()):
    """A validated copy of the kernel in canonical form (the input is unchanged)."""
    k = kernel.clone()
    f = k.func
    bnd = u.bound(f)
    parent = f.body.block
    for lvl in LEVELS:
        if lvl not in bnd:
            keep = ("global",) if lvl == "block" else ("global", "smem")
            ops = [o for o in parent.ops if not (isinstance(o, t.AllocOp) and t.space_of(o.result.type) in keep)]
            tag = f"_{lvl}"
            u.check_tags_free(f, tag)
            L = t.ForOp(0, 1, bind=lvl, tag=tag)
            parent.add_op(L)
            u.move_ops(ops, L.body.block)          # top-level smem allocs move into the block with the code
            k.kinds[tag] = "spatial"
            bnd[lvl] = L
        parent = bnd[lvl].body.block
    f.verify()
    validate(k, target)
    return k


def _fail(msg):
    raise ScheduleError(f"not canonical: {msg}")


def validate(k, target: Target = Target()):
    """Raise ScheduleError unless `k` is in canonical form (see anvil.schedule) and fits `target`."""
    f = k.func
    bnd = u.bound(f)
    if set(bnd) != set(LEVELS):
        _fail(f"needs block, warp and lane loops, has {sorted(bnd)} (use canonicalize)")
    parent = f.body.block
    for lvl in LEVELS:
        L = bnd[lvl]
        allowed = ("global",) if lvl == "block" else ("smem",)
        stmts = [o for o in parent.ops if o is not L]
        bad = [o for o in stmts if not (isinstance(o, t.AllocOp) and t.space_of(o.result.type) in allowed)]
        if L.parent_block() is not parent or bad:
            _fail(f"the {lvl} loop must be the only statement (besides {allowed[0]} allocs) of its parent")
        parent = L.body.block
    lane = bnd["lane"]
    for op in f.walk():
        if isinstance(op, t.ForOp):
            lb, ub, step = op.bounds
            if lb != 0 or step != 1:
                _fail(f"loop {u.tag_of(op)!r} is not normalised (lb 0, step 1)")
            if u.extent(op) % int(op.unroll.value.data):
                _fail(f"loop {u.tag_of(op)!r}: unroll does not divide the extent")
            if op is not lane and lane.is_ancestor(op) and op.bind.data != "seq":
                _fail(f"bound loop {u.tag_of(op)!r} inside the lane loop")
        if isinstance(op, t.BarrierOp):
            if not lane.is_ancestor(op):
                _fail("barrier outside the lane loop")
            if u.under_guard(op):
                _fail("barrier under a guard")
        if isinstance(op, t.AllocOp):
            sp = t.space_of(op.result.type)
            if sp == "reg" and not lane.is_ancestor(op):
                _fail("register buffer outside the lane loop")
            if sp != "reg" and lane.is_ancestor(op):
                _fail(f"{sp} buffer allocated inside the lane loop")
    nb, nw, nl = (u.extent(bnd[x]) for x in LEVELS)
    if nb * target.threads_per_block > target.max_threads:
        _fail(f"{nb} blocks x {target.threads_per_block} threads > {target.max_threads} per launch")
    if nw > target.warps_per_core:
        _fail(f"{nw} warps > WARPS_PER_CORE={target.warps_per_core}")
    if nl > target.lanes_per_warp:
        _fail(f"{nl} lanes > {target.lanes_per_warp} lanes per warp")
    smem = sum(math.prod(a.result.type.get_shape()) * (nw if bnd["warp"].is_ancestor(a) else 1)
               for a in u.smem_allocs(f))
    if smem > target.smem_words:
        _fail(f"{smem} words of shared memory > {target.smem_words}")
    regs = [o for o in lane.walk() if isinstance(o, t.AllocOp)]
    words = sum(math.prod(a.result.type.get_shape()) for a in regs)
    if words > target.max_reg_words:
        _fail(f"{words} register-buffer words per thread > {target.max_reg_words}")
    loop_of = {op.iv: op for op in u.loops(f)}
    for op in f.walk():
        if isinstance(op, (t.LoadOp, t.StoreOp)) and u.space(op.buf) == "reg":
            used = op.map.data.used_dims() if hasattr(op.map.data, "used_dims") else range(len(op.indices))
            for p in used:
                L = loop_of[op.indices[p]]
                if int(L.unroll.value.data) != u.extent(L):
                    _fail(f"register buffer indexed by loop {u.tag_of(L)!r}, which is not fully unrolled")
    return k
