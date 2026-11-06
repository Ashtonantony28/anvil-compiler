"""Integer interpreter for anvil.tensor: loop nests over numpy buffers, scalars through anvil.semantics.

Loops run sequentially in iteration order whatever their binding, which is the semantics a schedule must
preserve. Parallel loops (bind warp | lane) whose body holds top-level barriers run phase by phase: every iteration
runs up to the first barrier, then every iteration runs to the next, and so on, each iteration keeping its own
registers. That is exactly the BAR contract, so a schedule that stages through SMEM is checked here as on the GPU.
"""
import numpy as np

from .. import semantics as sem
from ..ir import tensor as t

PARALLEL = ("warp", "lane")


class TensorInterpError(RuntimeError):
    pass


def _func(m):
    if isinstance(m, t.FuncOp):
        return m
    fs = [op for op in m.ops if isinstance(op, t.FuncOp)]
    if len(fs) != 1:
        raise ValueError(f"expected one anvil.tensor.func, found {len(fs)}")
    return fs[0]


def _i(attr):
    return int(attr.value.data)


class _Interp:
    def __init__(self, buffers):
        self.buffers = buffers

    def index(self, env, op):
        idx = op.map.data.eval([env[v] for v in op.indices], [])
        shape = op.buf.type.get_shape()
        if any(not 0 <= i < n for i, n in zip(idx, shape)):
            raise IndexError(f"{op.name}: index {idx} outside {list(shape)}")
        return tuple(idx)

    def store_value(self, buf_type, value, what):
        if t.dtype_of(buf_type) == "i8":
            sem.check_i8(value, what)
        return sem.check_i32(value, what)

    def block(self, ops, env):
        for op in ops:
            self.op(op, env)

    def op(self, op, env):
        v = env.__getitem__
        if isinstance(op, t.AllocOp):
            ty = op.result.type
            shape = tuple(ty.get_shape())
            if t.space_of(ty) == "global":
                name = op.sym.data
                if name not in self.buffers:
                    self.buffers[name] = np.zeros(shape, dtype=np.int64)
                buf = self.buffers[name] = np.asarray(self.buffers[name], dtype=np.int64)
                if buf.shape != shape:
                    raise ValueError(f"buffer {name}: shape {buf.shape} != declared {shape}")
                self.store_value(ty, buf, name)
                env[op.result] = buf
            else:
                env[op.result] = np.zeros(shape, dtype=np.int64)
        elif isinstance(op, t.ForOp):
            self.loop(op, env)
        elif isinstance(op, t.LoadOp):
            env[op.result] = np.int64(v(op.buf)[self.index(env, op)])
        elif isinstance(op, t.StoreOp):
            val = self.store_value(op.buf.type, v(op.value), f"store to {op.buf.type}")
            v(op.buf)[self.index(env, op)] = val
        elif isinstance(op, t.GatherOp):
            env[op.result] = np.int64(sem.lut_lookup(v(op.buf), v(op.idx), _i(op.offset)))
        elif isinstance(op, t.ConstOp):
            env[op.result] = sem.check_i32(_i(op.value), "const")
        elif isinstance(op, t.CastOp):
            env[op.result] = sem.check_i32(v(op.x), "cast")
        elif isinstance(op, t.AddOp):
            env[op.result] = sem.add_i32(v(op.lhs), v(op.rhs))
        elif isinstance(op, t.SubOp):
            env[op.result] = sem.sub_i32(v(op.lhs), v(op.rhs))
        elif isinstance(op, t.MulOp):
            env[op.result] = sem.mul_i32(v(op.lhs), v(op.rhs))
        elif isinstance(op, t.MaxOp):
            env[op.result] = sem.max_i32(v(op.lhs), v(op.rhs))
        elif isinstance(op, t.MinOp):
            env[op.result] = sem.min_i32(v(op.lhs), v(op.rhs))
        elif isinstance(op, t.DivOp):
            env[op.result] = sem.div_floor(v(op.lhs), v(op.rhs))
        elif isinstance(op, t.SraOp):
            env[op.result] = sem.sra_i32(v(op.lhs), v(op.rhs))
        elif isinstance(op, t.CmpOp):
            env[op.result] = sem.cmp(op.pred.data, v(op.lhs), v(op.rhs))
        elif isinstance(op, t.SelectOp):
            env[op.result] = v(op.a) if v(op.cond) else v(op.b)
        elif isinstance(op, t.RequantOp):
            s = _i(op.s) + (0 if op.extra is None else int(v(op.extra)))
            env[op.result] = np.int64(sem.requant(v(op.acc), _i(op.M), s))
        elif isinstance(op, t.RsqrtIndexOp):
            idx, k = sem.rsqrt_index(sem.check_i32(v(op.ss), "ss"), _i(op.n))
            env[op.idx], env[op.k] = np.int64(idx), np.int64(k)
        elif isinstance(op, t.BarrierOp):
            pass        # outside a parallel loop's phase split there is nothing to wait for
        else:
            raise NotImplementedError(op.name)

    def loop(self, op, env):
        lb, ub, step = op.bounds
        ops = list(op.body.block.ops)
        iters = range(lb, ub, step)
        if op.bind.data in PARALLEL:
            for o in ops:
                for inner in o.walk():
                    if inner is not o and isinstance(inner, t.BarrierOp):
                        raise TensorInterpError("a barrier below the top level of a parallel loop is not modelled")
        if op.bind.data in PARALLEL and any(isinstance(o, t.BarrierOp) for o in ops):
            phases, cur = [], []
            for o in ops:
                if isinstance(o, t.BarrierOp):
                    phases.append(cur)
                    cur = []
                else:
                    cur.append(o)
            phases.append(cur)
            envs = [dict(env) for _ in iters]
            for e, i in zip(envs, iters):
                e[op.iv] = i
            for ph in phases:
                for e in envs:
                    self.block(ph, e)
            return
        for i in iters:
            env[op.iv] = i
            self.block(ops, env)


def run(m, buffers: dict) -> dict:
    """Run the func; `buffers` binds global allocs by sym (missing ones start at zero). Returns the buffers,
    as int64 arrays, updated in place."""
    buffers = {k: np.array(v, dtype=np.int64) for k, v in buffers.items()}
    _Interp(buffers).block(list(_func(m).body.block.ops), {})
    return buffers
