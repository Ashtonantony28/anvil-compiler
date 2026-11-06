"""Integer interpreter for anvil.graph: whole-tensor numpy evaluation, every op through anvil.semantics."""
from dataclasses import dataclass, field

import numpy as np

from .. import semantics as sem
from ..ir import graph as g


@dataclass
class GraphResult:
    outputs: list                                   # graph.output operands, in order
    values: dict = field(default_factory=dict)      # sym -> array for every op that carries a sym


def _func(m):
    if isinstance(m, g.FuncOp):
        return m
    fs = g.funcs(m)
    if len(fs) != 1:
        raise ValueError(f"expected one anvil.graph.func, found {len(fs)}")
    return fs[0]


def _i(attr):
    return int(attr.value.data)


def _check_type(op, val):
    t = g.qt(op.results[0])
    if list(val.shape) != t.dims:
        raise ValueError(f"{op.name} {op.sym_str!r}: shape {list(val.shape)} != declared {t.dims}")
    (sem.check_i8 if t.dtype == "i8" else sem.check_i32)(val, f"{op.name} {op.sym_str!r}")
    return val


def run(m, inputs: dict, consts: dict | None = None) -> GraphResult:
    """Evaluate the graph. `inputs` binds graph.input by sym; `consts` binds graph.const without an inline value."""
    consts = consts or {}
    env, res = {}, GraphResult([])
    for op in _func(m).body.block.ops:
        v = lambda x: env[x]  # noqa: E731
        if isinstance(op, g.InputOp):
            val = np.asarray(inputs[op.sym_str], dtype=np.int64)
        elif isinstance(op, g.ConstOp):
            if op.value is not None:
                val = np.asarray(op.value.get_values(), dtype=np.int64).reshape(g.qt(op.result).dims)
            else:
                val = np.asarray(consts[op.sym_str], dtype=np.int64)
        elif isinstance(op, (g.QMatmulOp, g.QLinearOp)):
            lut = None if op.lut is None else v(op.lut)
            val = sem.qmatmul(v(op.a), v(op.b), _i(op.M), _i(op.s), lut)
        elif isinstance(op, g.LutOp):
            val = sem.lut_i8(v(op.x), v(op.table))
        elif isinstance(op, g.LutMulOp):
            val = sem.swiglu_lut(v(op.x), v(op.y), v(op.table), _i(op.M), _i(op.s))
        elif isinstance(op, g.RmsNormIntOp):
            val = sem.rmsnorm_int(v(op.x), v(op.w), v(op.lut), _i(op.M), _i(op.s))
        elif isinstance(op, g.SoftmaxLutOp):
            val = sem.softmax_lut(v(op.x), v(op.lut), _i(op.one), bool(op.causal.value.data))
        elif isinstance(op, g.AddRequantOp):
            val = sem.add_requant(v(op.a), v(op.b), _i(op.Ma), _i(op.Mb), _i(op.s))
        elif isinstance(op, g.TransposeOp):
            val = np.transpose(v(op.x), op.perm_list)
        elif isinstance(op, g.ReshapeOp):
            val = v(op.x).reshape(g.qt(op.result).dims)
        elif isinstance(op, g.OutputOp):
            res.outputs = [v(x) for x in op.values]
            continue
        else:
            raise NotImplementedError(op.name)
        env[op.results[0]] = _check_type(op, val)
        if op.sym_str is not None:
            res.values[op.sym_str] = env[op.results[0]]
    return res
