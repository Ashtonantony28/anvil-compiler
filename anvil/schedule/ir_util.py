"""Tensor-IR plumbing shared by the schedule primitives: loop lookup, induction-variable substitution through
affine maps, building ops in place, and the linear decomposition behind bound inference."""
from xdsl.dialects.builtin import AffineMapAttr, IntegerAttr, StringAttr, i64
from xdsl.ir import Block
from xdsl.ir.affine import (
    AffineBinaryOpExpr,
    AffineBinaryOpKind,
    AffineConstantExpr,
    AffineDimExpr,
    AffineExpr,
    AffineMap,
)

from ..ir import tensor as t
from .target import ScheduleError

MAPPED = (t.LoadOp, t.StoreOp, t.ApplyOp, t.GuardOp)
BINDS = ("block", "warp", "lane")


def dim(i):
    return AffineExpr.dimension(i)


def const(c):
    return AffineExpr.constant(int(c))


def loops(func):
    return [op for op in func.walk() if isinstance(op, t.ForOp)]


def tag_of(loop):
    return None if loop.tag is None else loop.tag.data


def find(func, tag) -> t.ForOp:
    hits = [op for op in loops(func) if tag_of(op) == tag]
    if len(hits) != 1:
        have = [tag_of(op) for op in loops(func)]
        raise ScheduleError(f"no loop tagged {tag!r}" if not hits else f"{len(hits)} loops tagged {tag!r}"
                            f" (loops: {have})")
    return hits[0]


def extent(loop) -> int:
    lb, ub, step = loop.bounds
    return -(-(ub - lb) // step)


def set_prop(op, name, value):
    if isinstance(value, str):
        op.properties[name] = StringAttr(value)
    else:
        op.properties[name] = IntegerAttr(int(value), i64)


def bound(func) -> dict:
    """bind -> ForOp for the block/warp/lane loops of the kernel."""
    out = {}
    for op in loops(func):
        b = op.bind.data
        if b in BINDS:
            if b in out:
                raise ScheduleError(f"two loops bound to {b}")
            out[b] = op
    return out


def ancestors(op):
    """Enclosing ops (loops, guards) from the outermost, below the func."""
    out, p = [], op.parent_op()
    while p is not None and not isinstance(p, t.FuncOp):
        out.append(p)
        p = p.parent_op()
    return out[::-1]


def under_guard(op) -> bool:
    return any(isinstance(a, t.GuardOp) for a in ancestors(op))


def body(op):
    return list(op.regions[0].block.ops)


def has_barrier(op) -> bool:
    return any(isinstance(o, t.BarrierOp) for o in op.walk())


def alloc_of(buf):
    return buf.owner


def space(buf):
    return t.space_of(buf.type)


def smem_allocs(func):
    return [op for op in func.walk() if isinstance(op, t.AllocOp) and t.space_of(op.result.type) == "smem"]


def check_tags_free(func, *tags):
    have = {tag_of(op) for op in loops(func)}
    for tg in tags:
        if tg in have:
            raise ScheduleError(f"loop tag {tg!r} already exists")


def move_ops(ops, block: Block, before=None):
    for o in ops:
        o.detach()
        if before is None:
            block.add_op(o)
        else:
            block.insert_op_before(o, before)


def build(fn, *, before=None, block=None):
    """Run fn(TensorBuilder) into a scratch block, then move the ops before `before` (or to the end of `block`)."""
    tb = t.TensorBuilder("_")
    scratch = Block()
    tb._blocks = [scratch]
    res = fn(tb)
    ops = list(scratch.ops)
    if before is not None:
        move_ops(ops, before.parent_block(), before)
    else:
        move_ops(ops, block)
    return res


# ------------------------------------------------------------------------------------- IV substitution

def _set_indices(op, new_idx, new_map):
    if isinstance(op, t.LoadOp):
        op.operands = [op.buf, *new_idx]
    elif isinstance(op, t.StoreOp):
        op.operands = [op.value, op.buf, *new_idx]
    else:
        op.operands = list(new_idx)
    op.properties["map"] = AffineMapAttr(new_map)


def remap(op, repl: dict):
    """Rewrite op's index operands: repl maps an SSA index value to (list of SSA values, AffineExpr over them)."""
    if isinstance(op, t.CastOp):
        new = t.ApplyOp(operands=[[op.x]], properties={"map": AffineMapAttr(AffineMap.identity(1))},
                        result_types=[t.i32])
        op.parent_block().insert_op_before(new, op)
        op.result.replace_all_uses_with(new.result)
        op.detach()
        op.erase()
        remap(new, repl)
        return
    if not isinstance(op, MAPPED):
        raise ScheduleError(f"an induction variable is used by {op.name}, which has no index map")
    old = list(op.indices)
    new_idx = []
    for v in old:
        if v not in repl and v not in new_idx:
            new_idx.append(v)
    for v in old:
        if v in repl:
            for w in repl[v][0]:
                if w not in new_idx:
                    new_idx.append(w)
    exprs = []
    for v in old:
        if v in repl:
            ivs, e = repl[v]
            exprs.append(e.replace_dims_and_symbols([dim(new_idx.index(w)) for w in ivs], []))
        else:
            exprs.append(dim(new_idx.index(v)))
    m = op.map.data.replace_dims_and_symbols(exprs, [], len(new_idx), 0)
    _set_indices(op, new_idx, m)


def subst(iv, ivs, expr):
    """Replace every use of index value `iv` by affine `expr` over `ivs`."""
    users = []
    for u in list(iv.uses):
        if u.operation not in users:
            users.append(u.operation)
    for op in users:
        remap(op, {iv: (list(ivs), expr)})


def replace_value(root_ops, mapping: dict):
    """Plain operand replacement (no map change) in every op below root_ops."""
    for r in root_ops:
        for op in r.walk():
            if any(o in mapping for o in op.operands):
                op.operands = [mapping.get(o, o) for o in op.operands]


# ------------------------------------------------------------------------------- linear decomposition

def _linear(e, inner):
    """e -> (coef {dim: c}, const, rest) with rest the non-constant part free of `inner` dims, or None."""
    if isinstance(e, AffineConstantExpr):
        return {}, e.value, None
    if isinstance(e, AffineDimExpr):
        return ({e.position: 1}, 0, None) if e.position in inner else ({}, 0, e)
    if isinstance(e, AffineBinaryOpExpr):
        if e.kind == AffineBinaryOpKind.Add:
            a, b = _linear(e.lhs, inner), _linear(e.rhs, inner)
            if a is None or b is None:
                return None
            coef = dict(a[0])
            for k, v in b[0].items():
                coef[k] = coef.get(k, 0) + v
            rest = a[2] if b[2] is None else (b[2] if a[2] is None else a[2] + b[2])
            return coef, a[1] + b[1], rest
        if e.kind == AffineBinaryOpKind.Mul:
            c, x = (e.lhs, e.rhs) if isinstance(e.lhs, AffineConstantExpr) else (e.rhs, e.lhs)
            if not isinstance(c, AffineConstantExpr):
                return None
            a = _linear(x, inner)
            if a is None:
                return None
            return ({k: v * c.value for k, v in a[0].items()}, a[1] * c.value,
                    None if a[2] is None or c.value == 0 else a[2] * c.value)
        if not (set(e.used_dims()) & inner):
            return {}, 0, e
    return None


class Access:
    """One map result of an access, split as rest(outer SSA values) + const + sum coef * inner iv."""

    def __init__(self, coef, const, rest):
        self.coef, self.const, self.rest = coef, const, rest


class Space:
    """Collects outer SSA values so rest expressions from different ops compare structurally."""

    def __init__(self):
        self.outer = []

    def gdim(self, v):
        if v not in self.outer:
            self.outer.append(v)
        return dim(self.outer.index(v))

    def decompose(self, op, inner: set):
        """Per map result of an indexed op: Access, or ScheduleError if an inner iv enters non-linearly."""
        idx = list(op.indices)
        inner_pos = {p for p, v in enumerate(idx) if v in inner}
        out = []
        for r in op.map.data.results:
            d = _linear(r, inner_pos)
            if d is None:
                raise ScheduleError(f"index {r} of {op.name} is not linear in the loops it ranges over")
            coef = {}
            for p, c in d[0].items():
                coef[idx[p]] = coef.get(idx[p], 0) + c
            rest = None
            if d[2] is not None:
                rest = d[2].replace_dims_and_symbols(
                    [self.gdim(v) if p not in inner_pos else const(0) for p, v in enumerate(idx)], [])
            out.append(Access({k: v for k, v in coef.items() if v}, d[1], rest))
        return out


def box(space_, loads, inner, trips):
    """Union box of the loads' footprints over the inner ivs: per dim (rest, lo, ext)."""
    per = [space_.decompose(ld, inner) for ld in loads]
    rank = len(per[0])
    out = []
    for r in range(rank):
        accs = [p[r] for p in per]
        rest = accs[0].rest
        for a in accs[1:]:
            if a.rest != rest:
                raise ScheduleError("accesses in the region do not share one base (offsets differ non-constantly)")
        lo = min(a.const + sum(min(0, c) * (trips[v] - 1) for v, c in a.coef.items()) for a in accs)
        hi = max(a.const + sum(max(0, c) * (trips[v] - 1) for v, c in a.coef.items()) for a in accs)
        out.append((rest, lo, hi - lo + 1))
    return out


def local_map(space_, op, inner, lo):
    """(ivs, AffineMap) indexing a box-shaped local copy: result r = linear inner part + const - lo[r]."""
    accs = space_.decompose(op, inner)
    ivs = []
    for a in accs:
        for v in a.coef:
            if v not in ivs:
                ivs.append(v)
    res = []
    for a, l in zip(accs, lo):
        e = const(a.const - l)
        for v, c in a.coef.items():
            e = e + dim(ivs.index(v)) * c
        res.append(e)
    return ivs, AffineMap(len(ivs), 0, tuple(res))


def rest_expr(rest, lo):
    return const(lo) if rest is None else rest + const(lo)
