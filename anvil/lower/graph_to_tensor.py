"""anvil.graph -> anvil.tensor: one kernel (one anvil.tensor.func, later one launch) per graph op.

    prog = lower(graph_module)                    # Program: kernels in graph order, views, buffer table
    bufs = prog.run({"x": x}, consts)             # tensor-interpreter run of every kernel, = graph interpreter

Default loop nests are unscheduled: every loop is `seq`, has lb 0 / step 1, a unique `tag`, and a kind in
`Kernel.kinds` ("spatial": iterations independent, each writes its own outputs; "reduce": iterations accumulate
into a register). Schedules (anvil.schedule) rewrite them.

    qmatmul / qlinear  loops b*, i, j (spatial), k (reduce):  acc = sum a*w; c = [lut[]](requant(acc, M, s))
    lut, lut_mul, add_requant, transpose   loops d0..dn (spatial) over the output, one store per element
    rmsnorm_int  stage `stat` (stat.b*, stat.i spatial; stat.j reduce) writes per-row temps <sym>.rs (rsqrt LUT
                 value) and <sym>.k (shift); stage `norm` (b*, i, j) reads them
    softmax_lut  stage `max` -> temp <sym>.max, stage `sum` -> temp <sym>.sum, stage `norm` (b*, i, j)
    reshape      no kernel: a view. Consumers load the source buffer through a linearize/delinearize map.

Temps are kernel-private global scratch buffers (`Kernel.temps`); `compute_at` turns them into registers.
Reductions use one canonical pattern that schedule primitives recognise: a `reg` buffer of shape [1], an init
`store const -> acc` before the reduce loop, and as the reduce loop's last op `store op(load acc, x) -> acc`
with op in add | max.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir.affine import AffineExpr, AffineMap

from ..interp import tensor as ti
from ..ir import graph as g
from ..ir import tensor as t


@dataclass
class Kernel:
    name: str                                   # graph op sym = func sym_name
    op: str                                     # graph op kind, e.g. "qmatmul"
    module: ModuleOp                            # builtin.module holding one anvil.tensor.func
    reads: list = field(default_factory=list)   # global syms read (graph values, consts)
    writes: list = field(default_factory=list)  # global syms written (graph values)
    temps: dict = field(default_factory=dict)   # scratch sym -> (shape, dtype)
    kinds: dict = field(default_factory=dict)   # loop tag -> "spatial" | "reduce"

    @property
    def func(self) -> t.FuncOp:
        return next(op for op in self.module.ops if isinstance(op, t.FuncOp))

    def clone(self) -> "Kernel":
        return Kernel(self.name, self.op, self.module.clone(), list(self.reads), list(self.writes),
                      dict(self.temps), dict(self.kinds))

    def run(self, buffers: dict, *, check_races=True) -> dict:
        return ti.run(self.module, buffers, check_races=check_races)


@dataclass
class Program:
    name: str
    kernels: list                               # Kernel, in graph order
    buffers: dict                               # sym -> (shape, dtype) for every graph value with storage
    views: dict                                 # view sym -> (source sym, shape)
    inputs: list
    outputs: list
    const_values: dict                          # inline graph consts

    def kernel(self, name) -> Kernel:
        return next(k for k in self.kernels if k.name == name)

    def run(self, inputs: dict, consts: dict | None = None, kernels: dict | None = None, *,
            check_races=True) -> dict:
        """Run every kernel in order on one buffer dict; `kernels` replaces kernels by name (scheduled ones).
        Returns sym -> array for every graph value (views materialised) plus temps."""
        bufs = {**self.const_values, **(consts or {}), **inputs}
        bufs = {k: np.asarray(v, dtype=np.int64) for k, v in bufs.items()}
        for k in self.kernels:
            k = (kernels or {}).get(k.name, k)
            bufs = k.run(bufs, check_races=check_races)
        for v, (src, shape) in self.views.items():
            bufs[v] = bufs[src].reshape(shape)
        return bufs


# --------------------------------------------------------------------------------------------- helpers

def amap(n, fn) -> AffineMap:
    """AffineMap over n dims: fn(list of dim exprs) -> list of result exprs."""
    dims = [AffineExpr.dimension(i) for i in range(n)]
    return AffineMap(n, 0, tuple(fn(dims)))


def _strides(shape):
    return [math.prod(shape[r + 1:]) for r in range(len(shape))]


def view_map(view_shape, src_shape) -> AffineMap:
    """Row-major reshape: view index -> source index."""
    vs, ss = _strides(view_shape), _strides(src_shape)

    def fn(d):
        lin = sum((d[r] * vs[r] for r in range(len(d))), AffineExpr.constant(0))
        return [lin if len(ss) == 1 else (lin // ss[q] if q == 0 else (lin // ss[q]) % src_shape[q])
                for q in range(len(ss))]
    return amap(len(view_shape), fn)


class _KB:
    """Kernel builder: a TensorBuilder plus global allocs by sym, views and loop kinds."""

    def __init__(self, name, opname, lowering):
        self.b = t.TensorBuilder(name)
        self.name, self.opname, self.L = name, opname, lowering
        self.allocs, self.kinds, self.temps = {}, {}, {}
        self.reads, self.writes = [], []

    def buf(self, sym, shape, dtype):
        if sym not in self.allocs:                 # global allocs always sit at the top of the func
            top = self.b.func.body.block
            op = t.AllocOp(properties={"sym": StringAttr(sym)}, result_types=[t.buffer_type(shape, dtype)])
            first = next((o for o in top.ops if not isinstance(o, t.AllocOp)), None)
            top.add_op(op) if first is None else top.insert_op_before(op, first)
            self.allocs[sym] = op.result
        return self.allocs[sym]

    def src(self, value):
        """(alloc, map or None) to read graph value `value` through its view chain."""
        sym = self.L.sym(value)
        base, shape = self.L.resolve(sym)
        ty = g.qt(value)
        if base not in self.reads:
            self.reads.append(base)
        bshape, dtype = self.L.buffers[base]
        a = self.buf(base, bshape, dtype)
        return a, (None if base == sym else view_map(ty.dims, bshape))

    def dst(self, value):
        sym = self.L.sym(value)
        ty = g.qt(value)
        self.writes.append(sym)
        return self.buf(sym, ty.dims, ty.dtype)

    def temp(self, suffix, shape):
        sym = f"{self.name}.{suffix}"
        self.temps[sym] = (list(shape), "i32")
        return self.buf(sym, shape, "i32")

    def load(self, src, idx, m=None):
        a, vm = src
        n = len(idx)
        m = m if m is not None else AffineMap.identity(n)
        if vm is not None:
            m = vm.compose(m)
        return self.b.load(a, idx, m)

    def loops(self, extents, tags, kind="spatial"):
        """Nested seq loops; returns (list of ivs, exit callback)."""
        ctxs, ivs = [], []
        for n, tag in zip(extents, tags):
            c = self.b.loop(0, n, tag=tag)
            ivs.append(c.__enter__())
            ctxs.append(c)
            self.kinds[tag] = kind
        return ivs, ctxs

    @staticmethod
    def close(ctxs):
        for c in reversed(ctxs):
            c.__exit__(None, None, None)

    def acc(self, init):
        a = self.b.alloc(None, [1], "i32", "reg")
        self.b.store(self.b.const(init), a, [], lambda: (0,))
        return a

    def acc_load(self, a):
        return self.b.load(a, [], lambda: (0,))

    def acc_update(self, a, kind, x):
        self.b.store(self.b.binary(kind, self.acc_load(a), x), a, [], lambda: (0,))

    def kernel(self):
        return Kernel(self.name, self.opname, self.b.module(), self.reads, self.writes, self.temps, self.kinds)


def _i(attr):
    return int(attr.value.data)


# -------------------------------------------------------------------------------------------- lowering

class _Lowering:
    def __init__(self, func):
        self.func = func
        self.names, self.buffers, self.views, self.consts = {}, {}, {}, {}
        self.inputs, self.outputs, self.kernels = [], [], []
        self._n = 0

    def sym(self, value):
        if value not in self.names:
            s = value.owner.sym_str if isinstance(value.owner, g._GraphOp) else None
            if s is None:
                s = f"t{self._n}"
                self._n += 1
            self.names[value] = s
        return self.names[value]

    def resolve(self, sym):
        shape = None
        while sym in self.views:
            src, sh = self.views[sym]
            shape = shape or sh
            sym = src
        return sym, shape

    def run(self):
        for op in self.func.body.block.ops:
            if isinstance(op, g.OutputOp):
                self.outputs = [self.sym(v) for v in op.values]
                continue
            res = op.results[0]
            sym, ty = self.sym(res), g.qt(res)
            if isinstance(op, g.ReshapeOp):
                self.views[sym] = (self.sym(op.x), ty.dims)
                continue
            self.buffers[sym] = (ty.dims, ty.dtype)
            if isinstance(op, g.InputOp):
                self.inputs.append(sym)
            elif isinstance(op, g.ConstOp):
                if op.value is not None:
                    self.consts[sym] = np.asarray(op.value.get_values(), dtype=np.int64).reshape(ty.dims)
            else:
                self.kernels.append(LOWER[type(op)](self, op, sym))
        return Program(self.func.sym_name.data, self.kernels, self.buffers, self.views, self.inputs,
                       self.outputs, self.consts)


def _matmul(L, op, sym):
    k = _KB(sym, "qmatmul" if isinstance(op, g.QMatmulOp) else "qlinear", L)
    b = k.b
    a, w = k.src(op.a), k.src(op.b)
    lut = None if op.lut is None else k.src(op.lut)
    c = k.dst(op.result)
    adims, (K, N) = g.qt(op.a).dims, g.qt(op.b).dims
    batch, M = adims[:-2], adims[-2]
    ivs, ctx = k.loops(batch + [M, N], [f"b{n}" for n in range(len(batch))] + ["i", "j"])
    acc = k.acc(0)
    kv, kc = k.loops([K], ["k"], "reduce")
    nb = len(batch)
    av = k.load(a, ivs[:-1] + kv, amap(nb + 2, lambda d: d))
    wv = k.load(w, [kv[0], ivs[-1]])
    k.acc_update(acc, "add", b.mul(av, wv))
    k.close(kc)
    r = b.requant(k.acc_load(acc), _i(op.M), _i(op.s))
    if lut is not None:
        r = b.gather(lut[0], r, 128)
    b.store(r, c, ivs)
    k.close(ctx)
    return k.kernel()


def _elementwise(opname, body):
    def lower(L, op, sym):
        k = _KB(sym, opname, L)
        out = k.dst(op.result)
        dims = g.qt(op.result).dims
        ivs, ctx = k.loops(dims, [f"d{n}" for n in range(len(dims))])
        k.b.store(body(k, op, ivs), out, ivs)
        k.close(ctx)
        return k.kernel()
    return lower


def _lut(k, op, ivs):
    return k.b.gather(k.src(op.table)[0], k.load(k.src(op.x), ivs), 128)


def _lut_mul(k, op, ivs):
    b = k.b
    sl = b.gather(k.src(op.table)[0], k.load(k.src(op.x), ivs), 128)
    return b.requant(b.mul(sl, k.load(k.src(op.y), ivs)), _i(op.M), _i(op.s))


def _add_requant(k, op, ivs):
    b = k.b
    x = b.mul(k.load(k.src(op.a), ivs), b.const(_i(op.Ma)))
    y = b.mul(k.load(k.src(op.b), ivs), b.const(_i(op.Mb)))
    return b.requant(b.add(x, y), 1, _i(op.s))


def _transpose(k, op, ivs):
    p = op.perm_list
    return k.load(k.src(op.x), ivs, amap(len(ivs), lambda d: [d[p.index(q)] for q in range(len(p))]))


def _rows(dims):
    return dims[:-2], dims[-2], dims[-1]


def _rmsnorm(L, op, sym):
    k = _KB(sym, "rmsnorm_int", L)
    b = k.b
    x, w, lut = k.src(op.x), k.src(op.w), k.src(op.lut)
    out = k.dst(op.result)
    dims = g.qt(op.x).dims
    batch, T, D = _rows(dims)
    nb, n = len(batch), g.qt(op.lut).dims[0]
    rs, kk = k.temp("rs", batch + [T]), k.temp("k", batch + [T])
    ivs, ctx = k.loops(batch + [T], [f"stat.b{q}" for q in range(nb)] + ["stat.i"])
    ss = k.acc(0)
    jv, jc = k.loops([D], ["stat.j"], "reduce")
    xv = k.load(x, ivs + jv)
    k.acc_update(ss, "add", b.mul(xv, xv))
    k.close(jc)
    idx, sh = b.rsqrt_index(k.acc_load(ss), n)
    b.store(b.gather(lut[0], idx), rs, ivs)
    b.store(sh, kk, ivs)
    k.close(ctx)
    ivs, ctx = k.loops(batch + [T, D], [f"b{q}" for q in range(nb)] + ["i", "j"])
    xw = b.mul(k.load(x, ivs), k.load(w, ivs[-1:]))
    row = ivs[:-1]
    b.store(b.requant(b.mul(xw, b.load(rs, row)), _i(op.M), _i(op.s), extra=b.load(kk, row)), out, ivs)
    k.close(ctx)
    return k.kernel()


def _softmax(L, op, sym):
    k = _KB(sym, "softmax_lut", L)
    b = k.b
    x, lut = k.src(op.x), k.src(op.lut)[0]
    out = k.dst(op.result)
    batch, T, T2 = _rows(g.qt(op.x).dims)
    nb, one, causal = len(batch), _i(op.one), bool(op.causal.value.data)
    tmax, tsum = k.temp("max", batch + [T]), k.temp("sum", batch + [T])

    def keep(i, j):
        return b.cmp("le", b.apply([j], lambda j: (j,)), b.apply([i], lambda i: (i,))) if causal else None

    def exp(row, j):
        kp = keep(row[-1], j)
        dv = b.add(b.sub(k.load(x, row + [j]), b.load(tmax, row)), b.const(255))
        if kp is None:
            return b.gather(lut, dv)
        return b.select(kp, b.gather(lut, b.select(kp, dv, b.const(255))), b.const(0))

    def stage(name):
        return k.loops(batch + [T], [f"{name}.b{q}" for q in range(nb)] + [f"{name}.i"])

    row, ctx = stage("max")
    mx = k.acc(-128)
    jv, jc = k.loops([T2], ["max.j"], "reduce")
    v = k.load(x, row + jv)
    kp = keep(row[-1], jv[0])
    k.acc_update(mx, "max", v if kp is None else b.select(kp, v, b.const(-128)))
    k.close(jc)
    b.store(k.acc_load(mx), tmax, row)
    k.close(ctx)

    row, ctx = stage("sum")
    tot = k.acc(0)
    jv, jc = k.loops([T2], ["sum.j"], "reduce")
    k.acc_update(tot, "add", exp(row, jv[0]))
    k.close(jc)
    b.store(k.acc_load(tot), tsum, row)
    k.close(ctx)

    ivs, ctx = k.loops(batch + [T, T2], [f"b{q}" for q in range(nb)] + ["i", "j"])
    row = ivs[:-1]
    total = b.load(tsum, row)
    num = b.add(b.mul(exp(row, ivs[-1]), b.const(one)), b.div(total, b.const(2)))
    b.store(b.div(num, total), out, ivs)
    k.close(ctx)
    return k.kernel()


LOWER = {
    g.QMatmulOp: _matmul, g.QLinearOp: _matmul,
    g.LutOp: _elementwise("lut", _lut), g.LutMulOp: _elementwise("lut_mul", _lut_mul),
    g.AddRequantOp: _elementwise("add_requant", _add_requant), g.TransposeOp: _elementwise("transpose", _transpose),
    g.RmsNormIntOp: _rmsnorm, g.SoftmaxLutOp: _softmax,
}


def lower(module) -> Program:
    """Lower the graph func of `module` (or a graph FuncOp) to one tensor kernel per op."""
    func = module if isinstance(module, g.FuncOp) else g.funcs(module)[0]
    return _Lowering(func).run()
