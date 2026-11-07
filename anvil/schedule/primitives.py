"""Schedule primitives: in-place rewrites of a Kernel's anvil.tensor func, each checking its legality first.

Every function takes (kernel, target, ...) and raises ScheduleError when the rewrite is illegal; `Schedule`
runs them on a clone so a rejected primitive leaves the schedule unchanged. Loops are named by tag.
"""
import itertools
import math

from xdsl.dialects.builtin import AffineMapAttr, DenseArrayBase, i64
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap

from ..ir import tensor as t
from . import ir_util as u
from .ir_util import ScheduleError, const, dim

TAILS = ("mask", "pad")
ORDER = {"block": 0, "warp": 1, "lane": 2}


def _seq(loop, what):
    if loop.bind.data != "seq":
        raise ScheduleError(f"{what}: loop {u.tag_of(loop)!r} is bound to {loop.bind.data}; only seq loops can be "
                            f"rewritten (bind last)")


def _kind(k, tag):
    return k.kinds.get(tag, "spatial")


def _guard_op(ivs, exprs, ubs):
    return t.GuardOp(operands=[list(ivs)],
                     properties={"map": AffineMapAttr(AffineMap(len(ivs), 0, tuple(exprs))),
                                 "ub": DenseArrayBase.from_list(i64, list(ubs))},
                     regions=[Region(Block())])


def _wrap_in_guard(block, ivs, exprs, ubs):
    ops = list(block.ops)
    g = _guard_op(ivs, exprs, ubs)
    block.add_op(g)
    u.move_ops(ops, g.body.block)
    return g


def _idempotent(loop, what):
    """pad recomputes wrapped iterations: legal iff no state outlives an iteration (pure outputs only)."""
    inside = set(loop.walk())
    stored = {op.buf for op in inside if isinstance(op, t.StoreOp) and op.buf.owner not in inside}
    loaded = {op.buf for op in inside if isinstance(op, t.LoadOp)}
    if stored & loaded:
        raise ScheduleError(f"{what}: tail='pad' recomputes iterations, but the loop updates a buffer it also reads "
                            f"(a reduction); use tail='mask'")


def _tail(k, loop, tail, n, ivs, expr, what):
    """Apply the ragged-tail policy to `loop`'s body whose old iv becomes expr(ivs) over [0, n)."""
    if tail not in TAILS:
        raise ScheduleError(f"{what}: tail must be one of {TAILS}")
    if tail == "pad":
        if _kind(k, u.tag_of(loop)) != "spatial":
            raise ScheduleError(f"{what}: tail='pad' needs a spatial loop")
        _idempotent(loop, what)
        return expr % n
    if u.has_barrier(loop):
        raise ScheduleError(f"{what}: tail='mask' would put a barrier under a guard; use tail='pad'")
    return expr


# --------------------------------------------------------------------------------------- loop structure

def split(k, tgt, tag, factor, tail="mask", outer=None, inner=None):
    f = k.func
    L = u.find(f, tag)
    _seq(L, "split")
    n = u.extent(L)
    if not 1 <= factor <= n:
        raise ScheduleError(f"split: factor {factor} outside [1, {n}]")
    outer, inner = outer or f"{tag}.o", inner or f"{tag}.i"
    u.check_tags_free(f, outer, inner)
    ragged = n % factor != 0
    expr = dim(0) * factor + dim(1)
    if ragged:
        new_expr = _tail(k, L, tail, n, None, expr, "split")
    else:
        new_expr = expr
    Lo = t.ForOp(0, -(-n // factor), tag=outer)
    Li = t.ForOp(0, factor, tag=inner)
    Lo.body.block.add_op(Li)
    L.parent_block().insert_op_before(Lo, L)
    u.move_ops(u.body(L), Li.body.block)
    u.subst(L.iv, [Lo.iv, Li.iv], new_expr)
    if ragged and tail == "mask":
        _wrap_in_guard(Li.body.block, [Lo.iv, Li.iv], [expr], [n])
    L.detach()
    L.erase()
    kind = k.kinds.pop(tag, "spatial")
    k.kinds[outer] = k.kinds[inner] = kind


def _chain(f, tags, what):
    ls = [u.find(f, tg) for tg in tags]
    if len(set(tags)) != len(tags):
        raise ScheduleError(f"{what}: repeated loop")
    for L in ls:
        _seq(L, what)
    top = min(ls, key=lambda L: len(u.ancestors(L)))
    chain = [top]
    while len(chain) < len(ls):
        ops = u.body(chain[-1])
        if len(ops) != 1 or not isinstance(ops[0], t.ForOp):
            raise ScheduleError(f"{what}: loops {list(tags)} are not a perfect nest")
        chain.append(ops[0])
    if {u.tag_of(c) for c in chain} != set(tags):
        raise ScheduleError(f"{what}: loops {list(tags)} are not a perfect nest")
    return chain


def reorder(k, tgt, *tags):
    f = k.func
    chain = _chain(f, tags, "reorder")
    props = {u.tag_of(c): dict(c.properties) for c in chain}
    old_iv = {u.tag_of(c): c.iv for c in chain}
    for c, tg in zip(chain, tags):
        for name, val in props[tg].items():
            c.properties[name] = val
    mapping = {old_iv[tg]: c.iv for c, tg in zip(chain, tags)}
    tmp = {v: w for v, w in mapping.items() if v is not w}
    u.replace_value(u.body(chain[-1]), tmp)


def fuse(k, tgt, outer, inner, name=None):
    f = k.func
    Lo, Li = _chain(f, [outer, inner], "fuse")
    name = name or f"{outer}_{inner}"
    if name not in (outer, inner):
        u.check_tags_free(f, name)
    no, ni = u.extent(Lo), u.extent(Li)
    u.set_prop(Lo, "ub", no * ni)
    u.set_prop(Lo, "unroll", 1)
    u.set_prop(Lo, "tag", name)
    u.move_ops(u.body(Li), Lo.body.block)
    u.subst(Lo.iv, [Lo.iv], dim(0) // ni)
    u.subst(Li.iv, [Lo.iv], dim(0) % ni)
    Li.detach()
    Li.erase()
    kinds = {k.kinds.pop(outer, "spatial"), k.kinds.pop(inner, "spatial")}
    k.kinds[name] = "reduce" if "reduce" in kinds else "spatial"


def tile(k, tgt, outer, inner, f_outer, f_inner, tail="mask"):
    """split both loops of a perfect nest and reorder to outer.o, inner.o, outer.i, inner.i."""
    _chain(k.func, [outer, inner], "tile")
    split(k, tgt, outer, f_outer, tail)
    split(k, tgt, inner, f_inner, tail)
    reorder(k, tgt, f"{outer}.o", f"{inner}.o", f"{outer}.i", f"{inner}.i")


def unroll(k, tgt, tag, factor=None):
    L = u.find(k.func, tag)
    _seq(L, "unroll")
    n = u.extent(L)
    factor = n if factor is None else factor
    if factor < 1 or n % factor:
        raise ScheduleError(f"unroll: factor {factor} must divide the extent {n}")
    u.set_prop(L, "unroll", factor)


def interchange(k, tgt, spatial, reduce):
    """Register tiling: for s { regs; pre; for k {b}; post }  ->  regs[s]; for s {pre}; for k { for s {b} };
    for s {post}. The s loops are fully unrolled (they index registers)."""
    f = k.func
    S, K = u.find(f, spatial), u.find(f, reduce)
    _seq(S, "interchange")
    _seq(K, "interchange")
    if _kind(k, spatial) != "spatial" or _kind(k, reduce) != "reduce":
        raise ScheduleError(f"interchange: needs a spatial loop around a reduction loop")
    if K.parent_op() is not S:
        raise ScheduleError(f"interchange: {reduce!r} is not directly in the body of {spatial!r}")
    if u.has_barrier(S):
        raise ScheduleError("interchange: loop holds a barrier")
    ops = u.body(S)
    kpos = ops.index(K)
    regs = [o for o in ops if isinstance(o, t.AllocOp)]
    if any(t.space_of(o.result.type) != "reg" for o in regs) or any(ops.index(o) > kpos for o in regs):
        raise ScheduleError("interchange: only register allocs before the reduction may sit in the loop body")
    pre = [o for o in ops[:kpos] if o not in regs]
    post = ops[kpos + 1:]
    parts = [pre, [K], post]
    defs = [{r for o in p for x in o.walk() for r in x.results} for p in parts]
    for i, p in enumerate(parts):
        for o in p:
            for x in o.walk():
                for v in x.operands:
                    if any(v in defs[j] for j in range(3) if j != i):
                        raise ScheduleError("interchange: a value flows between the parts of the loop body other "
                                            "than through a register buffer")
    reg_bufs = {o.result for o in regs}
    k_stores = {o.buf for o in K.walk() if isinstance(o, t.StoreOp)}
    if k_stores - reg_bufs:
        raise ScheduleError("interchange: the reduction stores to memory other than its registers")
    post_stores = {o.buf for p in post for o in p.walk() if isinstance(o, t.StoreOp)}
    early_loads = {o.buf for p in pre + [K] for o in p.walk() if isinstance(o, t.LoadOp)}
    if post_stores & early_loads:
        raise ScheduleError("interchange: the epilogue writes a buffer the reduction reads")
    n = u.extent(S)
    init_tag, epi_tag = f"{spatial}.init", f"{spatial}.epi"
    u.check_tags_free(f, init_tag, epi_tag)
    # expand the registers by a leading [n] dimension indexed by s
    for a in regs:
        ty = a.result.type
        na = t.AllocOp(properties=dict(a.properties),
                       result_types=[t.buffer_type([n, *ty.get_shape()], t.dtype_of(ty), "reg")])
        S.parent_block().insert_op_before(na, S)
        for use in list(a.result.uses):
            op = use.operation
            idx = list(op.indices)
            m = op.map.data
            m2 = AffineMap(len(idx) + 1, 0, (dim(len(idx)),) + tuple(m.results))
            if isinstance(op, t.LoadOp):
                op.operands = [na.result, *idx, S.iv]
            else:
                op.operands = [op.value, na.result, *idx, S.iv]
            op.properties["map"] = AffineMapAttr(m2)
        a.detach()
        a.erase()
    new = []
    if pre:
        P = t.ForOp(0, n, tag=init_tag, unroll=n)
        u.move_ops(pre, P.body.block)
        new.append(P)
        k.kinds[init_tag] = "spatial"
    Sk = t.ForOp(0, n, tag=spatial, unroll=n)
    u.move_ops(u.body(K), Sk.body.block)
    K.body.block.add_op(Sk)
    if post:
        E = t.ForOp(0, n, tag=epi_tag, unroll=n)
        u.move_ops(post, E.body.block)
        k.kinds[epi_tag] = "spatial"
    K.detach()
    S.parent_block().insert_op_before(K, S)
    if pre:
        S.parent_block().insert_op_before(P, K)
    if post:
        S.parent_block().insert_op_before(E, S)
    for loop in [x for x in (P if pre else None, Sk, E if post else None) if x is not None]:
        u.replace_value([loop], {S.iv: loop.iv})
    S.detach()
    S.erase()


# ------------------------------------------------------------------------------------------ binding

def _level_parent(f, bnd):
    """The block a loop bound at the next level must be the only statement of."""
    for b in ("lane", "warp", "block"):
        if b in bnd:
            return bnd[b].body.block, b
    return f.body.block, None


def _statements(block):
    return [o for o in block.ops if not (isinstance(o, t.AllocOp) and t.space_of(o.result.type) in ("global", "smem"))]


def _check_extent(tgt, kind, n):
    if kind == "block" and n * tgt.threads_per_block > tgt.max_threads:
        raise ScheduleError(f"bind: {n} blocks x {tgt.threads_per_block} threads = {n * tgt.threads_per_block} "
                            f"> {tgt.max_threads} threads per launch")
    if kind == "warp" and n > tgt.warps_per_core:
        raise ScheduleError(f"bind: {n} warps > WARPS_PER_CORE={tgt.warps_per_core}")
    if kind == "lane" and n > tgt.lanes_per_warp:
        raise ScheduleError(f"bind: {n} lanes > {tgt.lanes_per_warp} lanes per warp "
                            f"(THREADS_PER_BLOCK={tgt.threads_per_block} / WARPS_PER_CORE={tgt.warps_per_core})")


def _check_bind_order(f, bnd, kind, what):
    if kind not in ORDER:
        raise ScheduleError(f"{what}: bind must be one of {tuple(ORDER)}")
    if kind in bnd:
        raise ScheduleError(f"{what}: a loop is already bound to {kind}")
    deeper = [b for b in bnd if ORDER[b] > ORDER[kind]]
    if deeper:
        raise ScheduleError(f"{what}: {deeper[0]} is already bound; bind block, then warp, then lane")
    if u.smem_allocs(f):
        raise ScheduleError(f"{what}: bind threads before cache_read / cross_lane_reduce (the kernel already uses "
                            f"shared memory)")


def bind(k, tgt, tag, kind):
    f = k.func
    L = u.find(f, tag)
    _seq(L, "bind")
    bnd = u.bound(f)
    _check_bind_order(f, bnd, kind, "bind")
    if _kind(k, tag) != "spatial":
        raise ScheduleError(f"bind: {tag!r} is a reduction loop; use cross_lane_reduce")
    parent, pk = _level_parent(f, bnd)
    if L.parent_block() is not parent or _statements(parent) != [L]:
        where = "the kernel's top level" if pk is None else f"the body of the {pk} loop"
        raise ScheduleError(f"bind: {tag!r} must be the only statement in {where} (split/reorder/fuse/compute_at "
                            f"it into place)")
    _check_extent(tgt, kind, u.extent(L))
    u.set_prop(L, "bind", kind)


def replicate(k, tgt, kind, n, tag=None):
    """Wrap the innermost thread level's body in a new warp/lane loop whose iv is unused: every thread runs the
    same code (then `distribute` / `cross_lane_reduce` split work across them)."""
    f = k.func
    tag = tag or f"_{kind}"
    if kind not in ("warp", "lane"):
        raise ScheduleError("replicate: kind must be warp or lane")
    bnd = u.bound(f)
    _check_bind_order(f, bnd, kind, "replicate")
    u.check_tags_free(f, tag)
    _check_extent(tgt, kind, n)
    parent, _ = _level_parent(f, bnd)
    ops = _statements(parent)
    if not ops:
        raise ScheduleError("replicate: nothing to replicate")
    L = t.ForOp(0, n, bind=kind, tag=tag)
    parent.add_op(L)
    u.move_ops(ops, L.body.block)
    k.kinds[tag] = "spatial"


def _threads(f, scope="block"):
    """(ivs, AffineExpr tid over them, T) for the threads sharing a block (or a warp)."""
    bnd = u.bound(f)
    lane, warp = bnd.get("lane"), bnd.get("warp")
    le = u.extent(lane) if lane is not None else 1
    if scope == "warp":
        return ([lane.iv], dim(0), le) if lane is not None else ([], const(0), 1)
    ivs, e = [], const(0)
    if warp is not None:
        ivs.append(warp.iv)
        e = dim(0) * le
    if lane is not None:
        ivs.append(lane.iv)
        e = e + dim(len(ivs) - 1)
    return ivs, e, (u.extent(warp) if warp is not None else 1) * le


def _inside_threads(f, L, what):
    bnd = u.bound(f)
    inner = bnd.get("lane") or bnd.get("warp")
    if inner is not None and not inner.is_ancestor(L):
        raise ScheduleError(f"{what}: {u.tag_of(L)!r} must be inside the thread loops")
    if u.under_guard(L):
        raise ScheduleError(f"{what}: {u.tag_of(L)!r} is under a guard (barriers there would diverge)")
    return inner


def _seq_trips_between(inner, L):
    """True when L runs more than once per thread (it or an enclosing seq loop below the thread loops loops)."""
    chain = u.ancestors(L) + [L]
    if inner is not None:
        chain = chain[chain.index(inner) + 1:]
    return any(isinstance(c, t.ForOp) and u.extent(c) > 1 for c in chain)


def _smem_budget(f, tgt, extra, what):
    used = 0
    bnd = u.bound(f)
    for a in u.smem_allocs(f):
        w = math.prod(a.result.type.get_shape())
        warp = bnd.get("warp")
        if warp is not None and warp.is_ancestor(a):
            w *= u.extent(warp)
        used += w
    if used + extra > tgt.smem_words:
        raise ScheduleError(f"{what}: shared memory over budget: {used} + {extra} words > {tgt.smem_words} "
                            f"per block ({tgt.smem_words // tgt.warps_per_core} per warp at "
                            f"WARPS_PER_CORE={tgt.warps_per_core})")


def _smem_alloc(f, shape, dtype, scope):
    bnd = u.bound(f)
    if scope == "warp":
        if "warp" not in bnd:
            raise ScheduleError("scope='warp' needs a warp loop")
        blk = bnd["warp"].body.block
    elif "block" in bnd:
        blk = bnd["block"].body.block
    else:
        blk = f.body.block
    a = t.AllocOp(result_types=[t.buffer_type(shape, dtype, "smem")])
    first = next((o for o in blk.ops if not isinstance(o, t.AllocOp)), None)
    if first is None:
        blk.add_op(a)
    else:
        blk.insert_op_before(a, first)
    return a.result


def distribute(k, tgt, tag, tail="mask"):
    """Spread a spatial seq loop inside replicated thread loops over the threads: iv = o * T + tid."""
    f = k.func
    L = u.find(f, tag)
    _seq(L, "distribute")
    if _kind(k, tag) != "spatial":
        raise ScheduleError("distribute: reduction loops go through cross_lane_reduce")
    inner = _inside_threads(f, L, "distribute")
    tivs, tid, T = _threads(f)
    if inner is None or T == 1:
        raise ScheduleError("distribute: no thread loops to distribute over (replicate or bind first)")
    inside = set(L.walk())
    if any(v in tivs for o in inside for v in o.operands):
        raise ScheduleError(f"distribute: {tag!r} already depends on the thread index")
    stored = {o.buf for o in inside if isinstance(o, t.StoreOp)}
    if any(u.space(b) == "reg" and b.owner not in inside for b in stored):
        raise ScheduleError(f"distribute: {tag!r} updates a register that outlives it")
    n = u.extent(L)
    ragged = n % T != 0
    expr = dim(0) * T + tid.replace_dims_and_symbols([dim(i + 1) for i in range(len(tivs))], [])
    new = _tail(k, L, tail, n, None, expr, "distribute") if ragged else expr
    u.set_prop(L, "ub", -(-n // T))
    u.set_prop(L, "unroll", 1)
    ivs = [L.iv, *tivs]
    u.subst(L.iv, ivs, new)
    if ragged and tail == "mask":
        _wrap_in_guard(L.body.block, ivs, [expr], [n])
    later = [o for o in f.walk() if isinstance(o, t.LoadOp) and o.buf in stored and o not in inside]
    if later:
        if _seq_trips_between(inner, L):
            L.parent_block().insert_op_before(t.BarrierOp(), L)
        nxt = L.next_op
        if nxt is None:
            L.parent_block().add_op(t.BarrierOp())
        else:
            L.parent_block().insert_op_before(t.BarrierOp(), nxt)


def _reduction(L):
    ops = u.body(L)
    st = ops[-1] if ops else None
    if not (isinstance(st, t.StoreOp) and u.space(st.buf) == "reg" and st.map.data.num_dims == 0):
        raise ScheduleError(f"cross_lane_reduce: {u.tag_of(L)!r} does not end in a scalar register update")
    acc = st.buf
    bop = st.value.owner
    kinds = {t.AddOp: "add", t.MaxOp: "max", t.MinOp: "min"}
    if type(bop) not in kinds:
        raise ScheduleError("cross_lane_reduce: the update is not add/max/min")
    lds = [x.owner for x in (bop.lhs, bop.rhs) if isinstance(x.owner, t.LoadOp) and x.owner.buf == acc]
    if len(lds) != 1:
        raise ScheduleError("cross_lane_reduce: the update does not read the accumulator once")
    others = [o for o in L.walk() if isinstance(o, (t.LoadOp, t.StoreOp)) and o.buf == acc and o not in (st, lds[0])]
    if others:
        raise ScheduleError("cross_lane_reduce: the accumulator is used elsewhere in the loop")
    init = None
    for o in L.parent_block().ops:
        if o is L:
            break
        if isinstance(o, t.StoreOp) and o.buf == acc:
            init = o
    if init is None or not isinstance(init.value.owner, t.ConstOp):
        raise ScheduleError("cross_lane_reduce: no constant initialisation of the accumulator before the loop")
    c = int(init.value.owner.value.value.data)
    kind = kinds[type(bop)]
    if kind == "add" and c != 0:
        raise ScheduleError("cross_lane_reduce: an add reduction must start at 0")
    return acc, kind, c


def cross_lane_reduce(k, tgt, tag, tail="mask"):
    """Each thread reduces a strided slice into its accumulator, stores the partial to smem[tid], BAR, and every
    thread folds the T partials in thread order (exact for integer add/max/min), BAR."""
    f = k.func
    L = u.find(f, tag)
    _seq(L, "cross_lane_reduce")
    if _kind(k, tag) != "reduce":
        raise ScheduleError(f"cross_lane_reduce: {tag!r} is not a reduction loop")
    inner = _inside_threads(f, L, "cross_lane_reduce")
    tivs, tid, T = _threads(f)
    if inner is None or T == 1:
        raise ScheduleError("cross_lane_reduce: no thread loops to reduce across (replicate or bind first)")
    if u.has_barrier(L):
        raise ScheduleError("cross_lane_reduce: the loop holds a barrier")
    if any(v in tivs for o in L.walk() for v in o.operands):
        raise ScheduleError(f"cross_lane_reduce: {tag!r} already depends on the thread index")
    acc, kind, c = _reduction(L)
    xtag = f"{tag}.x"
    u.check_tags_free(f, xtag)
    _smem_budget(f, tgt, T, "cross_lane_reduce")
    n = u.extent(L)
    expr = dim(0) * T + tid.replace_dims_and_symbols([dim(i + 1) for i in range(len(tivs))], [])
    u.set_prop(L, "ub", -(-n // T))
    u.set_prop(L, "unroll", 1)
    ivs = [L.iv, *tivs]
    u.subst(L.iv, ivs, expr)
    if n % T:
        _wrap_in_guard(L.body.block, ivs, [expr], [n])
    red = _smem_alloc(f, [T], "i32", "block")
    z = lambda: (0,)  # noqa: E731
    tmap = AffineMap(len(tivs), 0, (tid,))

    def tail_code(b):
        b.store(b.load(acc, [], z), red, tivs, tmap)
        b.barrier()
        b.store(b.const(c), acc, [], z)
        with b.loop(0, T, tag=xtag) as x:
            b.store(b.binary(kind, b.load(acc, [], z), b.load(red, [x])), acc, [], z)
        b.barrier()

    nxt = L.next_op
    if nxt is None:
        u.build(tail_code, block=L.parent_block())
    else:
        u.build(tail_code, before=nxt)
    k.kinds[xtag] = "reduce"


# ---------------------------------------------------------------------------------------- memory

def _domain_max(space_, exprs, trips):
    """Max of each affine expr over the outer ivs' domain, or None if the domain is too large to enumerate."""
    outer = space_.outer
    if math.prod(trips.get(v, 1) for v in outer) > 4096 or any(v not in trips for v in outer):
        return None
    m = AffineMap(len(outer), 0, tuple(exprs))
    best = None
    for pt in itertools.product(*[range(trips[v]) for v in outer]):
        vals = m.eval(list(pt), [])
        best = vals if best is None else [max(a, b) for a, b in zip(best, vals)]
    return best


def _check_box_in_domain(space_, bx, extents, trips, stage):
    """compute_at recomputes the whole box per iteration, unguarded: every box must lie inside the stage's
    iteration domain for every value of the outer ivs. A consumer under a ragged guard (split(..., "mask")) reads
    a box that runs past the domain on its last iteration; computing it there would read out of bounds (A2-M8)."""
    outer = space_.outer
    if any(v not in trips for v in outer) or math.prod(trips[v] for v in outer) > 1 << 16:
        raise ScheduleError(f"compute_at: cannot prove the region of {stage!r} stays inside its domain")
    m = AffineMap(len(outer), 0, tuple(u.rest_expr(r, lo) for r, lo, _ in bx))
    for pt in itertools.product(*[range(trips[v]) for v in outer]):
        for base, (_, _, e), n in zip(m.eval(list(pt), []), bx, extents):
            if base < 0 or base + e > n:
                raise ScheduleError(f"compute_at: the region of {stage!r} one iteration reads ([{base}, {base + e}) "
                                    f"of [0, {n})) leaves the stage's domain (a guarded/ragged consumer); split "
                                    f"with tail='pad' or compute at another loop")


def _trips(f):
    return {op.iv: u.extent(op) for op in u.loops(f)}


def cache_read(k, tgt, sym, at, scope="block"):
    """Stage the block of global buffer `sym` that one iteration of loop `at` reads (over every thread sharing the
    smem buffer) into shared memory: cooperative copy, BAR, the loop body reading smem, BAR."""
    f = k.func
    L = u.find(f, at)
    _seq(L, "cache_read")
    inner_t = _inside_threads(f, L, "cache_read")
    allocs = [o for o in f.body.block.ops if isinstance(o, t.AllocOp) and o.sym is not None and o.sym.data == sym]
    if not allocs:
        raise ScheduleError(f"cache_read: no global buffer {sym!r}")
    buf = allocs[0].result
    if any(isinstance(o, t.StoreOp) and o.buf == buf for o in f.walk()):
        raise ScheduleError(f"cache_read: {sym!r} is written by the kernel")
    if scope not in ("block", "warp"):
        raise ScheduleError("cache_read: scope must be block or warp")
    loads = [o for o in L.walk() if isinstance(o, t.LoadOp) and o.buf == buf]
    if not loads:
        raise ScheduleError(f"cache_read: {at!r} does not read {sym!r}")
    tivs, tid, T = _threads(f, scope)
    trips = _trips(f)
    inner = {o.iv for o in L.walk() if isinstance(o, t.ForOp) and o is not L} | set(tivs)
    sp = u.Space()
    bx = u.box(sp, loads, inner, trips)
    ext = [e for _, _, e in bx]
    W = math.prod(ext)
    nwarps = u.extent(u.bound(f)["warp"]) if scope == "warp" and "warp" in u.bound(f) else 1
    _smem_budget(f, tgt, W * nwarps, "cache_read")
    shape = buf.type.get_shape()
    bases = [u.rest_expr(r, lo) for r, lo, _ in bx]
    mx = _domain_max(sp, [b + (e - 1) for b, e in zip(bases, ext)], trips)
    need = [mx is None or mx[r] >= shape[r] for r in range(len(ext))]
    smem = _smem_alloc(f, ext, t.dtype_of(buf.type), scope)
    for ld in loads:
        ivs, m = u.local_map(sp, ld, inner, [lo for _, lo, _ in bx])
        ld.operands = [smem, *ivs]
        ld.properties["map"] = AffineMapAttr(m)
    outer = list(sp.outer)
    no = -(-W // T)
    ctag = f"{at}.{sym}.copy"
    u.check_tags_free(f, ctag)
    strides = [math.prod(ext[r + 1:]) for r in range(len(ext))]

    def copy(b):
        with b.loop(0, no, tag=ctag) as e:
            ivs = outer + [e] + tivs
            no_, ne = len(outer), len(tivs)
            ide = dim(no_) * T + tid.replace_dims_and_symbols([dim(no_ + 1 + i) for i in range(ne)], [])
            coords = [ide if len(ext) == 1 else ((ide // strides[r]) if r == 0 else
                                                 (ide // strides[r]) % ext[r]) for r in range(len(ext))]
            gidx = [bases[r] + coords[r] for r in range(len(ext))]
            conds, ubs = ([ide], [W]) if W % T else ([], [])
            for r in range(len(ext)):
                if need[r]:
                    conds.append(gidx[r])
                    ubs.append(shape[r])

            def body():
                v = b.load(buf, ivs, AffineMap(len(ivs), 0, tuple(gidx)))
                b.store(v, smem, ivs, AffineMap(len(ivs), 0, tuple(coords)))
            if conds:
                with b.guard(ivs, AffineMap(len(ivs), 0, tuple(conds)), ubs):
                    body()
            else:
                body()
        b.barrier()

    first = next(iter(L.body.block.ops), None)
    if first is None:
        raise ScheduleError("cache_read: empty loop")
    u.build(copy, before=first)
    if _seq_trips_between(inner_t, L):
        L.body.block.add_op(t.BarrierOp())
    k.kinds[ctag] = "spatial"


def compute_at(k, tgt, stage, at):
    """Move the top-level stage `stage` (a perfect spatial nest writing kernel temps) into loop `at`, computing only
    the region of the temps one iteration of `at` reads; the temps become registers of that shape."""
    f = k.func
    S = u.find(f, stage)
    _seq(S, "compute_at")
    if S.parent_op() is not f:
        raise ScheduleError(f"compute_at: {stage!r} is not a top-level stage")
    temps = {}
    for o in S.walk():
        if isinstance(o, t.StoreOp) and u.space(o.buf) == "global" and o.buf.owner.sym is not None \
                and o.buf.owner.sym.data in k.temps:
            temps.setdefault(o.buf, []).append(o)
    if not temps:
        raise ScheduleError(f"compute_at: stage {stage!r} writes no kernel temp")
    rank = {len(b.type.get_shape()) for b in temps}
    if len(rank) != 1:
        raise ScheduleError("compute_at: the stage's temps differ in rank")
    m = rank.pop()
    chain = [S]
    while len(chain) < m:
        ops = u.body(chain[-1])
        if len(ops) != 1 or not isinstance(ops[0], t.ForOp):
            raise ScheduleError(f"compute_at: stage {stage!r} is not a perfect nest over its temps' dims")
        chain.append(ops[0])
    civs = [c.iv for c in chain]
    for c in chain:
        _seq(c, "compute_at")
        if _kind(k, u.tag_of(c)) != "spatial":
            raise ScheduleError("compute_at: stage loops over the temps must be spatial")
    for b, sts in temps.items():
        for st in sts:
            if list(st.indices) != civs or st.map.data != AffineMap.identity(m):
                raise ScheduleError("compute_at: temps must be stored at [stage ivs] (identity)")
    L = u.find(f, at)
    if L.bind.data not in ("seq", "lane"):
        raise ScheduleError(f"compute_at: {at!r} is bound to {L.bind.data}; compute at a seq or lane loop")
    if S.is_ancestor(L):
        raise ScheduleError("compute_at: the loop is inside the stage itself")
    top = (u.ancestors(L) + [L])[0]
    tops = list(f.body.block.ops)
    if tops.index(top) < tops.index(S):
        raise ScheduleError("compute_at: the consumer comes before the stage")
    loads = [o for o in f.walk() if isinstance(o, t.LoadOp) and o.buf in temps]
    if any(not L.is_ancestor(o) for o in loads):
        raise ScheduleError(f"compute_at: a temp of {stage!r} is read outside {at!r}")
    if any(isinstance(o, t.StoreOp) and o.buf in temps and not S.is_ancestor(o) for o in f.walk()):
        raise ScheduleError("compute_at: a temp is also written outside the stage")
    if not loads:
        raise ScheduleError("compute_at: the temps are never read")
    trips = _trips(f)
    inner = {o.iv for o in L.walk() if isinstance(o, t.ForOp) and o is not L}
    sp = u.Space()
    bx = u.box(sp, loads, inner, trips)
    ext = [e for _, _, e in bx]
    _check_box_in_domain(sp, bx, [u.extent(c) for c in chain], trips, stage)
    for c in chain:
        k.kinds.pop(u.tag_of(c), None)
    # build: reg temps, then loops over the box dims of extent > 1 (reusing the chain's tags)
    first = next(iter(L.body.block.ops))
    regs = {}
    for b in temps:
        a = t.AllocOp(result_types=[t.buffer_type(ext, t.dtype_of(b.type), "reg")])
        L.body.block.insert_op_before(a, first)
        regs[b] = a.result
    blk, eivs = L.body.block, []
    for r, c in enumerate(chain):
        if ext[r] > 1:
            nl = t.ForOp(0, ext[r], tag=u.tag_of(c), unroll=ext[r])
            blk.insert_op_before(nl, first) if blk is L.body.block else blk.add_op(nl)
            blk = nl.body.block
            eivs.append(nl.iv)
            k.kinds[u.tag_of(c)] = "spatial"
        else:
            eivs.append(None)
    moved = u.body(chain[-1])
    if blk is L.body.block:
        u.move_ops(moved, blk, first)
    else:
        u.move_ops(moved, blk)
    live = [v for v in eivs if v is not None]
    for b, sts in temps.items():
        for st in sts:
            st.operands = [st.value, regs[b], *live]
            st.properties["map"] = AffineMapAttr(AffineMap(len(live), 0, tuple(
                dim(live.index(e)) if e is not None else const(0) for e in eivs)))
    outer = list(sp.outer)
    for r, c in enumerate(chain):
        base = u.rest_expr(bx[r][0], bx[r][1])
        if eivs[r] is None:
            u.subst(c.iv, outer, base)
        else:
            u.subst(c.iv, outer + [eivs[r]], base + dim(len(outer)))
    lo = [x for _, x, _ in bx]
    for ld in loads:
        ivs, mm = u.local_map(sp, ld, inner, lo)
        ld.operands = [regs[ld.buf], *ivs]
        ld.properties["map"] = AffineMapAttr(mm)
    S.detach()
    S.erase()
    for b in temps:
        k.temps.pop(b.owner.sym.data, None)
        b.owner.detach()
        b.owner.erase()


PRIMITIVES = {
    "split": split, "reorder": reorder, "fuse": fuse, "tile": tile, "unroll": unroll, "interchange": interchange,
    "bind": bind, "replicate": replicate, "distribute": distribute, "cross_lane_reduce": cross_lane_reduce,
    "cache_read": cache_read, "compute_at": compute_at,
}
