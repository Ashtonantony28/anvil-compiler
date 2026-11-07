"""Integer interpreter for anvil.graph: whole-tensor numpy evaluation, every op through anvil.semantics.

    run(module, {"x": x}, consts)              one sample: x has the graph input's shape
    run(module, {"x": xs}, consts, batch=True) xs has a leading batch dim; every value keeps it (consts do not).
                                               Same ops, broadcast over the batch: bit-identical to stacking
                                               per-sample runs (tests/interp), and how SG5a measures 20k+ inputs.
"""
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


def _check_type(op, val, lead=()):
    t = g.qt(op.results[0])
    if list(val.shape) != list(lead) + t.dims:
        raise ValueError(f"{op.name} {op.sym_str!r}: shape {list(val.shape)} != declared {t.dims}")
    (sem.check_i8 if t.dtype == "i8" else sem.check_i32)(val, f"{op.name} {op.sym_str!r}")
    return val


def run(m, inputs: dict, consts: dict | None = None, *, batch: bool = False) -> GraphResult:
    """Evaluate the graph. `inputs` binds graph.input by sym; `consts` binds graph.const without an inline value.
    With batch=True every input (and so every non-const value) carries one extra leading batch dim."""
    consts = consts or {}
    env, res = {}, GraphResult([])
    lead, nb, is_const = (), 0, set()
    if batch:
        first = next(iter(inputs.values()))
        lead, nb = (int(np.shape(first)[0]),), 1
    for op in _func(m).body.block.ops:
        v = lambda x: env[x]  # noqa: E731
        if isinstance(op, g.InputOp):
            val = np.asarray(inputs[op.sym_str], dtype=np.int64)
        elif isinstance(op, g.ConstOp):
            is_const.add(op.results[0])
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
            k = 0 if op.x in is_const else nb
            val = np.transpose(v(op.x), list(range(k)) + [p + k for p in op.perm_list])
        elif isinstance(op, g.ReshapeOp):
            k = () if op.x in is_const else lead
            val = v(op.x).reshape(list(k) + g.qt(op.result).dims)
        elif isinstance(op, g.OutputOp):
            res.outputs = [v(x) for x in op.values]
            continue
        else:
            raise NotImplementedError(op.name)
        if isinstance(op, (g.TransposeOp, g.ReshapeOp)) and op.x in is_const:
            is_const.add(op.results[0])
        env[op.results[0]] = _check_type(op, val, () if op.results[0] in is_const else lead)
        if op.sym_str is not None:
            res.values[op.sym_str] = env[op.results[0]]
    return res
