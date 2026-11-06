"""anvil.graph: quantized tensor ops on SSA values of type !anvil.graph.qtensor<shape x dtype[, scale]>.

Every value is a whole tensor (int8 activations / weights, int32 LUTs) with its quantization scale as type
metadata. Requant constants (M, s) are op properties, so the graph is self-describing; weights and LUTs are
`const` ops bound by symbol (or carried inline as a dense attribute). Semantics: anvil.semantics, op by op.

    func  {sym_name}                      container, one block
    input {sym}                    -> t   graph input
    const {sym[, value]}           -> t   weight / LUT / any constant tensor
    qmatmul (a, b[, lut]) {M, s}   -> i8  requant(a @ b, M, s), optional fused lut[. + 128] epilogue
    qlinear (x, w[, lut]) {M, s}   -> i8  qmatmul whose w is a const (x @ W, W is (in, out))
    lut     (x, table)             -> i8  table[x + 128], 256 int8 entries
    lut_mul (x, y, table) {M, s}   -> i8  requant(table[x + 128] * y, M, s)          (SwiGLU)
    rmsnorm_int (x, w, lut) {M, s} -> i8  golden.rmsnorm_int over the last axis
    softmax_lut (x, lut) {one, causal} -> golden.softmax_lut over the last axis
    add_requant (a, b) {Ma, Mb, s} -> i8  clamp((a*Ma + b*Mb + 2^(s-1)) >> s)         (residual)
    transpose (x) {perm}, reshape (x)     layout ops
    output (values...)                    graph results
"""
from collections.abc import Sequence

from xdsl.dialects.builtin import (
    ArrayAttr,
    BoolAttr,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    FloatAttr,
    IntAttr,
    IntegerAttr,
    IntegerType,
    NoneAttr,
    StringAttr,
    TensorType,
    f64,
    i32,
    i64,
)
from xdsl.ir import Block, Dialect, ParametrizedAttribute, Region, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import NoTerminator
from xdsl.utils.exceptions import VerifyException

DTYPES = {"i8": 8, "i32": 32}


@irdl_attr_definition
class QTensorType(ParametrizedAttribute, TypeAttribute):
    """!anvil.graph.qtensor<4x8xi8, 3.7e-02> (scale optional: LUTs and requant-free int32 tensors)."""
    name = "anvil.graph.qtensor"
    shape: ArrayAttr[IntAttr]
    elem: IntegerType
    scale: FloatAttr | NoneAttr

    def __init__(self, shape: Sequence[int], dtype: str = "i8", scale: float | None = None):
        if dtype not in DTYPES:
            raise ValueError(f"dtype {dtype!r} not in {sorted(DTYPES)}")
        super().__init__(ArrayAttr([IntAttr(int(d)) for d in shape]), IntegerType(DTYPES[dtype]),
                         NoneAttr() if scale is None else FloatAttr(float(scale), f64))

    @property
    def dims(self) -> list[int]:
        return [d.data for d in self.shape.data]

    @property
    def dtype(self) -> str:
        return f"i{self.elem.width.data}"

    @property
    def qscale(self) -> float | None:
        return None if isinstance(self.scale, NoneAttr) else self.scale.value.data

    def verify(self):
        if self.elem.width.data not in (8, 32):
            raise VerifyException(f"qtensor dtype must be i8 or i32, got {self.elem}")
        if any(d < 1 for d in self.dims):
            raise VerifyException(f"qtensor dims must be positive, got {self.dims}")

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string("".join(f"{d}x" for d in self.dims))
            printer.print_attribute(self.elem)
            if self.qscale is not None:
                printer.print_string(f", {self.qscale:.17e}")

    @classmethod
    def parse_parameters(cls, parser: AttrParser):
        with parser.in_angle_brackets():
            dims, elem = parser.parse_ranked_shape()
            scale = NoneAttr()
            if parser.parse_optional_punctuation(",") is not None:
                scale = FloatAttr(parser.parse_float(), f64)
        return (ArrayAttr([IntAttr(d) for d in dims]), elem, scale)


def qt(value) -> QTensorType:
    t = value.type
    if not isinstance(t, QTensorType):
        raise VerifyException(f"expected a qtensor, got {t}")
    return t


def _int(attr) -> int:
    return attr.value.data


def _check_shift(op, *names):
    for n in names:
        s = _int(getattr(op, n))
        if not 0 <= s <= 31:
            raise VerifyException(f"{op.name}: {n}={s} outside [0, 31]")


def _expect(op, cond, msg):
    if not cond:
        raise VerifyException(f"{op.name}: {msg}")


def _lut256_i8(op, value):
    t = qt(value)
    _expect(op, t.dims == [256] and t.dtype == "i8", f"LUT must be 256 x i8, got {t.dims} x {t.dtype}")


class _GraphOp(IRDLOperation):
    sym = opt_prop_def(StringAttr)

    @property
    def sym_str(self) -> str | None:
        return None if self.sym is None else self.sym.data


@irdl_op_definition
class FuncOp(IRDLOperation):
    name = "anvil.graph.func"
    sym_name = prop_def(StringAttr)
    body = region_def("single_block")
    traits = traits_def(NoTerminator())

    def __init__(self, sym_name: str, ops: Sequence[IRDLOperation] = ()):
        super().__init__(properties={"sym_name": StringAttr(sym_name)}, regions=[Region(Block(list(ops)))])


@irdl_op_definition
class InputOp(_GraphOp):
    name = "anvil.graph.input"
    result = result_def(QTensorType)


@irdl_op_definition
class ConstOp(_GraphOp):
    name = "anvil.graph.const"
    value = opt_prop_def(DenseIntOrFPElementsAttr)
    result = result_def(QTensorType)

    def verify_(self):
        _expect(self, self.sym is not None or self.value is not None, "needs a sym or an inline value")
        if self.value is not None:
            _expect(self, list(self.value.get_shape()) == qt(self.result).dims, "value shape != result shape")


class _MatmulBase(_GraphOp):
    a = operand_def(QTensorType)
    b = operand_def(QTensorType)
    lut = opt_operand_def(QTensorType)
    M = prop_def(IntegerAttr)
    s = prop_def(IntegerAttr)
    result = result_def(QTensorType)

    def verify_(self):
        a, b, r = qt(self.a), qt(self.b), qt(self.result)
        _expect(self, a.dtype == b.dtype == r.dtype == "i8", "operands and result must be i8")
        _expect(self, len(a.dims) >= 2 and len(b.dims) == 2, f"shapes {a.dims} @ {b.dims}: need (..., M, K) @ (K, N)")
        _expect(self, a.dims[-1] == b.dims[0], f"contraction mismatch {a.dims} @ {b.dims}")
        _expect(self, r.dims == a.dims[:-1] + [b.dims[1]], f"result {r.dims} != {a.dims[:-1] + [b.dims[1]]}")
        _check_shift(self, "s")
        if self.lut is not None:
            _lut256_i8(self, self.lut)


@irdl_op_definition
class QMatmulOp(_MatmulBase):
    name = "anvil.graph.qmatmul"


@irdl_op_definition
class QLinearOp(_MatmulBase):
    name = "anvil.graph.qlinear"

    def verify_(self):
        super().verify_()
        _expect(self, isinstance(self.b.owner, ConstOp), "weight operand must be an anvil.graph.const")


@irdl_op_definition
class LutOp(_GraphOp):
    name = "anvil.graph.lut"
    x = operand_def(QTensorType)
    table = operand_def(QTensorType)
    result = result_def(QTensorType)

    def verify_(self):
        _expect(self, qt(self.x).dtype == qt(self.result).dtype == "i8", "x and result must be i8")
        _expect(self, qt(self.x).dims == qt(self.result).dims, "shape changes")
        _lut256_i8(self, self.table)


@irdl_op_definition
class LutMulOp(_GraphOp):
    name = "anvil.graph.lut_mul"
    x = operand_def(QTensorType)
    y = operand_def(QTensorType)
    table = operand_def(QTensorType)
    M = prop_def(IntegerAttr)
    s = prop_def(IntegerAttr)
    result = result_def(QTensorType)

    def verify_(self):
        x, y, r = qt(self.x), qt(self.y), qt(self.result)
        _expect(self, x.dtype == y.dtype == r.dtype == "i8", "operands and result must be i8")
        _expect(self, x.dims == y.dims == r.dims, f"shapes {x.dims}, {y.dims} -> {r.dims}")
        _lut256_i8(self, self.table)
        _check_shift(self, "s")


@irdl_op_definition
class RmsNormIntOp(_GraphOp):
    name = "anvil.graph.rmsnorm_int"
    x = operand_def(QTensorType)
    w = operand_def(QTensorType)
    lut = operand_def(QTensorType)
    M = prop_def(IntegerAttr)
    s = prop_def(IntegerAttr)
    result = result_def(QTensorType)

    def verify_(self):
        x, w, lut, r = qt(self.x), qt(self.w), qt(self.lut), qt(self.result)
        _expect(self, x.dtype == w.dtype == r.dtype == "i8", "x, w and result must be i8")
        _expect(self, w.dims == x.dims[-1:], f"weight {w.dims} != [{x.dims[-1]}]")
        _expect(self, len(lut.dims) == 1 and lut.dtype == "i32", "rsqrt LUT must be 1-D i32")
        _expect(self, r.dims == x.dims, "shape changes")
        _check_shift(self, "s")


@irdl_op_definition
class SoftmaxLutOp(_GraphOp):
    name = "anvil.graph.softmax_lut"
    x = operand_def(QTensorType)
    lut = operand_def(QTensorType)
    one = prop_def(IntegerAttr)
    causal = prop_def(BoolAttr)
    result = result_def(QTensorType)

    def verify_(self):
        x, lut, r = qt(self.x), qt(self.lut), qt(self.result)
        _expect(self, x.dtype == "i8" and len(x.dims) >= 2, "scores must be i8 with rank >= 2")
        _expect(self, lut.dims == [256] and lut.dtype == "i32", "exp LUT must be 256 x i32")
        _expect(self, r.dims == x.dims, "shape changes")
        one = _int(self.one)
        _expect(self, one >= 1 and (r.dtype == "i32" or one <= 127), f"one={one} does not fit result {r.dtype}")


@irdl_op_definition
class AddRequantOp(_GraphOp):
    name = "anvil.graph.add_requant"
    a = operand_def(QTensorType)
    b = operand_def(QTensorType)
    Ma = prop_def(IntegerAttr)
    Mb = prop_def(IntegerAttr)
    s = prop_def(IntegerAttr)
    result = result_def(QTensorType)

    def verify_(self):
        a, b, r = qt(self.a), qt(self.b), qt(self.result)
        _expect(self, a.dtype == b.dtype == r.dtype == "i8", "operands and result must be i8")
        _expect(self, a.dims == b.dims == r.dims, f"shapes {a.dims} + {b.dims} -> {r.dims}")
        _check_shift(self, "s")


@irdl_op_definition
class TransposeOp(_GraphOp):
    name = "anvil.graph.transpose"
    x = operand_def(QTensorType)
    perm = prop_def(DenseArrayBase)
    result = result_def(QTensorType)

    @property
    def perm_list(self) -> list[int]:
        return list(self.perm.get_values())

    def verify_(self):
        x, r, p = qt(self.x), qt(self.result), self.perm_list
        _expect(self, sorted(p) == list(range(len(x.dims))), f"perm {p} is not a permutation of rank {len(x.dims)}")
        _expect(self, r.dims == [x.dims[i] for i in p] and r.dtype == x.dtype, "result type mismatch")


@irdl_op_definition
class ReshapeOp(_GraphOp):
    name = "anvil.graph.reshape"
    x = operand_def(QTensorType)
    result = result_def(QTensorType)

    def verify_(self):
        import math

        x, r = qt(self.x), qt(self.result)
        _expect(self, math.prod(x.dims) == math.prod(r.dims) and x.dtype == r.dtype, "reshape changes size or dtype")


@irdl_op_definition
class OutputOp(IRDLOperation):
    name = "anvil.graph.output"
    values = var_operand_def(QTensorType)


Graph = Dialect("anvil.graph", [FuncOp, InputOp, ConstOp, QMatmulOp, QLinearOp, LutOp, LutMulOp, RmsNormIntOp,
                                SoftmaxLutOp, AddRequantOp, TransposeOp, ReshapeOp, OutputOp], [QTensorType])


# ----------------------------------------------------------------------------------------------------- builder

def _iattr(v):
    return IntegerAttr(int(v), i64)


class GraphBuilder:
    """Appends verified graph ops to a func; every method returns the result SSA value.

        g = GraphBuilder("mlp"); x = g.input("x", [4, 8], scale=0.03); ...; g.output(out); module = g.module()
    """

    def __init__(self, sym_name: str):
        self.func = FuncOp(sym_name)
        self.block = self.func.body.block

    def _add(self, op):
        self.block.add_op(op)
        op.verify()
        return op.results[0] if op.results else op

    @staticmethod
    def _props(sym, **kw):
        p = {k: v for k, v in kw.items() if v is not None}
        if sym is not None:
            p["sym"] = StringAttr(sym)
        return p

    def input(self, sym, shape, dtype="i8", scale=None):
        return self._add(InputOp(properties=self._props(sym), result_types=[QTensorType(shape, dtype, scale)]))

    def const(self, sym, shape, dtype="i8", scale=None, value=None):
        dense = None
        if value is not None:
            import numpy as np

            flat = [int(v) for v in np.asarray(value).ravel()]
            dense = DenseIntOrFPElementsAttr.from_list(TensorType(i32, list(shape)), flat)
        return self._add(ConstOp(properties=self._props(sym, value=dense),
                                 result_types=[QTensorType(shape, dtype, scale)]))

    def _mm(self, cls, sym, a, b, m, s, scale, lut):
        out = qt(a).dims[:-1] + [qt(b).dims[1]]
        return self._add(cls(operands=[a, b, [lut] if lut is not None else []],
                             properties=self._props(sym, M=_iattr(m), s=_iattr(s)),
                             result_types=[QTensorType(out, "i8", scale)]))

    def qmatmul(self, sym, a, b, m, s, scale=None, lut=None):
        return self._mm(QMatmulOp, sym, a, b, m, s, scale, lut)

    def qlinear(self, sym, x, w, m, s, scale=None, lut=None):
        return self._mm(QLinearOp, sym, x, w, m, s, scale, lut)

    def lut(self, sym, x, table, scale=None):
        return self._add(LutOp(operands=[x, table], properties=self._props(sym),
                               result_types=[QTensorType(qt(x).dims, "i8", scale)]))

    def lut_mul(self, sym, x, y, table, m, s, scale=None):
        return self._add(LutMulOp(operands=[x, y, table], properties=self._props(sym, M=_iattr(m), s=_iattr(s)),
                                  result_types=[QTensorType(qt(x).dims, "i8", scale)]))

    def rmsnorm_int(self, sym, x, w, lut, m, s, scale=None):
        return self._add(RmsNormIntOp(operands=[x, w, lut], properties=self._props(sym, M=_iattr(m), s=_iattr(s)),
                                      result_types=[QTensorType(qt(x).dims, "i8", scale)]))

    def softmax_lut(self, sym, x, lut, one, causal=True, scale=None, dtype="i8"):
        return self._add(SoftmaxLutOp(operands=[x, lut],
                                      properties=self._props(sym, one=_iattr(one), causal=BoolAttr.from_bool(causal)),
                                      result_types=[QTensorType(qt(x).dims, dtype, scale)]))

    def add_requant(self, sym, a, b, ma, mb, s, scale=None):
        return self._add(AddRequantOp(operands=[a, b],
                                      properties=self._props(sym, Ma=_iattr(ma), Mb=_iattr(mb), s=_iattr(s)),
                                      result_types=[QTensorType(qt(a).dims, "i8", scale)]))

    def transpose(self, sym, x, perm=None, scale=None):
        t = qt(x)
        perm = list(perm) if perm is not None else list(range(len(t.dims)))[::-1]
        sc = t.qscale if scale is None else scale
        return self._add(TransposeOp(operands=[x], properties=self._props(sym, perm=DenseArrayBase.from_list(i64, perm)),
                                     result_types=[QTensorType([t.dims[i] for i in perm], t.dtype, sc)]))

    def reshape(self, sym, x, shape, scale=None):
        t = qt(x)
        return self._add(ReshapeOp(operands=[x], properties=self._props(sym),
                                   result_types=[QTensorType(shape, t.dtype, t.qscale if scale is None else scale)]))

    def output(self, *values):
        self._add(OutputOp(operands=[list(values)]))

    def module(self):
        """The builtin.module holding the func (built once; later calls return the same module)."""
        from . import module

        if self.func.parent is None:
            self._module = module(self.func)
        return self._module


def funcs(module) -> list[FuncOp]:
    return [op for op in module.ops if isinstance(op, FuncOp)]


