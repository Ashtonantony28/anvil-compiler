"""Integer interpreter for anvil.tensor: loop nests over numpy buffers, scalars through anvil.semantics.

Loops run sequentially in iteration order whatever their binding, which is the semantics a schedule must
preserve. Parallel loops (bind warp | lane) that contain a barrier anywhere below them (directly, or inside seq loops,
or inside nested warp x lane loops) run phase by phase, as BAR does on the GPU: every thread (one iteration of each
enclosing parallel loop, with its own registers) runs up to its next barrier; once every thread of the block waits,
they all continue. A barrier only some threads reach (inside a guard) is a divergence error. Outside any parallel loop
a barrier is a no-op.

Race check (on by default): inside one block's parallel loops, within one barrier phase, a thread reading an smem or
global word another thread stored, two threads storing different values to one word, or a thread overwriting with a
different value a word another thread read, raises TensorInterpError. These are exactly the orderings the GPU leaves
unspecified (lanes in lockstep, warps interleaved), so a schedule that passes here does not depend on issue order.
"""
import numpy as np

from .. import semantics as sem
from ..ir import tensor as t

PARALLEL = ("warp", "lane")
_TID, _TRK = "__tid__", "__trk__"


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


class _Tracker:
    """Accesses of one block's threads to shared words (smem and global) within the current barrier phase."""

    def __init__(self):
        self.words = {}             # (id(buf), idx) -> [last writer tid | None, readers set]

    def new_phase(self):
        self.words.clear()

    def read(self, buf, idx, tid, what):
        rec = self.words.get((id(buf), idx))
        if rec is None:
            self.words[(id(buf), idx)] = [None, {tid}]
            return
        if rec[0] is not None and rec[0] != tid:
            raise TensorInterpError(f"race: {what} {list(idx)} read by thread {tid} after thread {rec[0]} stored it "
                                    f"in the same barrier phase")
        rec[1].add(tid)

    def write(self, buf, idx, tid, value, what):
        key = (id(buf), idx)
        rec = self.words.get(key)
        if rec is None:
            self.words[key] = [tid, set()]
            return
        same = int(buf[idx]) == int(value)
        if rec[0] is not None and rec[0] != tid and not same:
            raise TensorInterpError(f"race: threads {rec[0]} and {tid} store different values to {what} {list(idx)} "
                                    f"in the same barrier phase")
        if not same and any(r != tid for r in rec[1]):
            raise TensorInterpError(f"race: thread {tid} overwrites {what} {list(idx)} that another thread read in "
                                    f"the same barrier phase")
        rec[0] = tid


class _Interp:
    def __init__(self, buffers, check_races=True):
        self.buffers = buffers
        self.check_races = check_races
        self._bar = {}

    def has_barrier(self, op):
        k = id(op)
        if k not in self._bar:
            self._bar[k] = any(isinstance(o, t.BarrierOp) for o in op.walk())
        return self._bar[k]

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

    def _tracked(self, env, op):
        return self.check_races and _TRK in env and t.space_of(op.buf.type) != "reg"

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
        elif isinstance(op, t.GuardOp):
            if self.guard(op, env):
                self.block(op.body.block.ops, env)
        elif isinstance(op, t.LoadOp):
            idx = self.index(env, op)
            if self._tracked(env, op):
                env[_TRK].read(v(op.buf), idx, env[_TID], f"{t.space_of(op.buf.type)} word")
            env[op.result] = np.int64(v(op.buf)[idx])
        elif isinstance(op, t.StoreOp):
            val = self.store_value(op.buf.type, v(op.value), f"store to {op.buf.type}")
            idx = self.index(env, op)
            if self._tracked(env, op):
                env[_TRK].write(v(op.buf), idx, env[_TID], val, f"{t.space_of(op.buf.type)} word")
            v(op.buf)[idx] = val
        elif isinstance(op, t.GatherOp):
            env[op.result] = np.int64(sem.lut_lookup(v(op.buf), v(op.idx), _i(op.offset)))
        elif isinstance(op, t.ConstOp):
            env[op.result] = sem.check_i32(_i(op.value), "const")
        elif isinstance(op, t.CastOp):
            env[op.result] = sem.check_i32(v(op.x), "cast")
        elif isinstance(op, t.ApplyOp):
            env[op.result] = sem.check_i32(op.map.data.eval([env[x] for x in op.indices], [])[0], "apply")
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

    def guard(self, op, env):
        vals = op.map.data.eval([env[x] for x in op.indices], [])
        return all(x < u for x, u in zip(vals, op.ub_list))

    # -------------------------------------------------------------------------------------------- loops

    def _thread_env(self, env, op, i):
        e = dict(env)
        e[op.iv] = i
        e[_TID] = env.get(_TID, ()) + (i,)
        return e

    def loop(self, op, env):
        lb, ub, step = op.bounds
        ops = list(op.body.block.ops)
        iters = range(lb, ub, step)
        if op.bind.data not in PARALLEL:
            for i in iters:
                env[op.iv] = i
                self.block(ops, env)
            return
        own = _TRK not in env
        if own:
            env[_TRK] = _Tracker()
        try:
            if self.has_barrier(op):
                children = [self.gblock(ops, self._thread_env(env, op, i)) for i in iters]
                for _ in self.drive(children):
                    env[_TRK].new_phase()
            else:
                tid = env.get(_TID, ())
                for i in iters:
                    env[op.iv], env[_TID] = i, tid + (i,)
                    self.block(ops, env)
                if tid:
                    env[_TID] = tid
                else:
                    env.pop(_TID, None)
        finally:
            if own:
                env.pop(_TRK, None)

    def drive(self, children):
        """Run threads to their next barrier, together; yield once per block-wide barrier."""
        live = children
        while True:
            waiting, done = [], 0
            for c in live:
                try:
                    next(c)
                    waiting.append(c)
                except StopIteration:
                    done += 1
            if not waiting:
                return
            if done:
                raise TensorInterpError(f"barrier divergence: {len(waiting)} threads wait at a barrier that "
                                        f"{done} finished threads never reach")
            yield
            live = waiting

    def gblock(self, ops, env):
        for op in ops:
            if isinstance(op, t.BarrierOp):
                yield
            elif isinstance(op, t.ForOp) and self.has_barrier(op):
                yield from self.gloop(op, env)
            elif isinstance(op, t.GuardOp) and self.has_barrier(op):
                if self.guard(op, env):
                    yield from self.gblock(list(op.body.block.ops), env)
            else:
                self.op(op, env)

    def gloop(self, op, env):
        lb, ub, step = op.bounds
        ops = list(op.body.block.ops)
        if op.bind.data in PARALLEL:
            yield from self.drive([self.gblock(ops, self._thread_env(env, op, i)) for i in range(lb, ub, step)])
        else:
            for i in range(lb, ub, step):
                env[op.iv] = i
                yield from self.gblock(ops, env)


def run(m, buffers: dict, *, check_races: bool = True) -> dict:
    """Run the func; `buffers` binds global allocs by sym (missing ones start at zero). Returns the buffers,
    as int64 arrays, updated in place."""
    buffers = {k: np.array(v, dtype=np.int64) for k, v in buffers.items()}
    _Interp(buffers, check_races).block(list(_func(m).body.block.ops), {})
    return buffers
