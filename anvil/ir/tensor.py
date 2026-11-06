"""anvil.tensor: loop nests over buffers in global / smem / reg memory, with affine index expressions.

Buffers are builtin memrefs whose memory space is a string: memref<4x8xi8, "global">. Loops have constant bounds
and a binding (seq | block | warp | lane) plus an unroll hint; the induction variable is an `index` block argument.
Loads and stores take index operands and an affine map from them to the buffer's indices, so schedule primitives
(split, reorder, tile, bind, cache_read to smem, compute_at, unroll) are rewrites of loop ops and of these maps.
Scalar data is i32 with checked int32 semantics (anvil.semantics); i8 buffers store int8 values.

    func   {sym_name}                            container, one block
    alloc  {sym?}                       -> memref global buffers bind to image segments by sym; smem/reg are fresh
    for    {lb, ub, step, bind, unroll, tag?}    region with one index argument
    load   (buf, idx...) {map}          -> i32   buf[map(idx)]
    store  (v, buf, idx...) {map}                buf[map(idx)] = v
    gather (buf, v) {offset}            -> i32   buf[v + offset], data-dependent (LUTs)
    const  {value}                      -> i32
    cast   (index)                      -> i32
    apply  (idx...) {map}               -> i32   one-result affine map of indices as a scalar (causal masks)
    guard  (idx...) {map, ub}                    region runs iff map(idx)[r] < ub[r] for every r (ragged-edge mask)
    add sub mul max min div sra (a, b)  -> i32   checked int32 (div: floor of non-negatives)
    cmp    (a, b) {pred}                -> i32   0/1, signed
    select (c, a, b)                    -> i32
    requant (acc[, extra]) {M, s}       -> i32   golden.requant with shift s (+ extra)
    rsqrt_index (ss) {n}                -> i32, i32
    barrier                                      all lanes/warps of the block (BAR)

Barriers may sit anywhere below a block's warp x lane loops (also inside seq loops there), never inside a guard.
"""
from collections.abc import Sequence

from xdsl.dialects.builtin import (
    AffineMapAttr,
    DenseArrayBase,
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
    StringAttr,
    i32,
    i64,
)
from xdsl.ir import Block, Dialect, Region
from xdsl.ir.affine import AffineMap
from xdsl.irdl import (
    IRDLOperation,
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
from xdsl.traits import NoTerminator
from xdsl.utils.exceptions import VerifyException

SPACES = ("global", "smem", "reg")
BINDS = ("seq", "block", "warp", "lane")
PREDS = ("eq", "ne", "lt", "le", "gt", "ge")
index = IndexType()


def buffer_type(shape: Sequence[int], dtype: str = "i32", space: str = "global") -> MemRefType:
    if space not in SPACES:
        raise ValueError(f"memory space {space!r} not in {SPACES}")
    return MemRefType(IntegerType({"i8": 8, "i32": 32}[dtype]), list(shape), memory_space=StringAttr(space))


def space_of(t: MemRefType) -> str:
    return t.memory_space.data


def dtype_of(t: MemRefType) -> str:
    return f"i{t.element_type.width.data}"


def _expect(op, cond, msg):
    if not cond:
        raise VerifyException(f"{op.name}: {msg}")


def _int(a):
    return a.value.data


def _check_buf(op, t):
    _expect(op, isinstance(t, MemRefType), f"expected a memref, got {t}")
    _expect(op, isinstance(t.memory_space, StringAttr) and t.memory_space.data in SPACES,
            f"memory space must be one of {SPACES}")
    _expect(op, isinstance(t.element_type, IntegerType) and t.element_type.width.data in (8, 32), "dtype must be i8/i32")


def _check_map(op, m: AffineMap, n_idx, rank):
    _expect(op, m.num_dims == n_idx and m.num_symbols == 0, f"map {m} takes {m.num_dims} dims, {n_idx} indices given")
    _expect(op, len(m.results) == rank, f"map {m} has {len(m.results)} results, buffer rank {rank}")


@irdl_op_definition
class FuncOp(IRDLOperation):
    name = "anvil.tensor.func"
    sym_name = prop_def(StringAttr)
    body = region_def("single_block")
    traits = traits_def(NoTerminator())

    def __init__(self, sym_name: str, ops=()):
        super().__init__(properties={"sym_name": StringAttr(sym_name)}, regions=[Region(Block(list(ops)))])


@irdl_op_definition
class AllocOp(IRDLOperation):
    name = "anvil.tensor.alloc"
    sym = opt_prop_def(StringAttr)
    result = result_def(MemRefType)

    def verify_(self):
        _check_buf(self, self.result.type)
        if space_of(self.result.type) == "global":
            _expect(self, self.sym is not None, "global buffers need a sym")


@irdl_op_definition
class ForOp(IRDLOperation):
    name = "anvil.tensor.for"
    lb = prop_def(IntegerAttr)
    ub = prop_def(IntegerAttr)
    step = prop_def(IntegerAttr)
    bind = prop_def(StringAttr)
    unroll = prop_def(IntegerAttr)
    tag = opt_prop_def(StringAttr)
    body = region_def("single_block")
    traits = traits_def(NoTerminator())

    def __init__(self, lb: int, ub: int, step: int = 1, bind: str = "seq", unroll: int = 1, tag: str | None = None,
                 ops=()):
        props = {"lb": IntegerAttr(lb, i64), "ub": IntegerAttr(ub, i64), "step": IntegerAttr(step, i64),
                 "bind": StringAttr(bind), "unroll": IntegerAttr(unroll, i64)}
        if tag is not None:
            props["tag"] = StringAttr(tag)
        super().__init__(properties=props, regions=[Region(Block(list(ops), arg_types=[index]))])

    @property
    def iv(self):
        return self.body.block.args[0]

    @property
    def bounds(self) -> tuple[int, int, int]:
        return _int(self.lb), _int(self.ub), _int(self.step)

    def verify_(self):
        lb, ub, step = self.bounds
        _expect(self, step >= 1 and ub >= lb, f"bounds [{lb}, {ub}) step {step}")
        _expect(self, self.bind.data in BINDS, f"bind must be one of {BINDS}")
        _expect(self, _int(self.unroll) >= 1, "unroll must be >= 1")
        b = self.body.block
        _expect(self, len(b.args) == 1 and isinstance(b.args[0].type, IndexType), "body takes one index argument")


@irdl_op_definition
class LoadOp(IRDLOperation):
    name = "anvil.tensor.load"
    buf = operand_def(MemRefType)
    indices = var_operand_def(IndexType)
    map = prop_def(AffineMapAttr)
    result = result_def(i32)

    def verify_(self):
        _check_buf(self, self.buf.type)
        _check_map(self, self.map.data, len(self.indices), len(self.buf.type.get_shape()))


@irdl_op_definition
class StoreOp(IRDLOperation):
    name = "anvil.tensor.store"
    value = operand_def(i32)
    buf = operand_def(MemRefType)
    indices = var_operand_def(IndexType)
    map = prop_def(AffineMapAttr)

    def verify_(self):
        _check_buf(self, self.buf.type)
        _check_map(self, self.map.data, len(self.indices), len(self.buf.type.get_shape()))


@irdl_op_definition
class GatherOp(IRDLOperation):
    name = "anvil.tensor.gather"
    buf = operand_def(MemRefType)
    idx = operand_def(i32)
    offset = prop_def(IntegerAttr)
    result = result_def(i32)

    def verify_(self):
        _check_buf(self, self.buf.type)
        _expect(self, len(self.buf.type.get_shape()) == 1, "gather needs a 1-D buffer")


@irdl_op_definition
class ConstOp(IRDLOperation):
    name = "anvil.tensor.const"
    value = prop_def(IntegerAttr)
    result = result_def(i32)


@irdl_op_definition
class CastOp(IRDLOperation):
    name = "anvil.tensor.cast"
    x = operand_def(IndexType)
    result = result_def(i32)


class _Binary(IRDLOperation):
    lhs = operand_def(i32)
    rhs = operand_def(i32)
    result = result_def(i32)

    def __init__(self, lhs, rhs):
        super().__init__(operands=[lhs, rhs], result_types=[i32])


@irdl_op_definition
class AddOp(_Binary):
    name = "anvil.tensor.add"


@irdl_op_definition
class SubOp(_Binary):
    name = "anvil.tensor.sub"


@irdl_op_definition
class MulOp(_Binary):
    name = "anvil.tensor.mul"


@irdl_op_definition
class MaxOp(_Binary):
    name = "anvil.tensor.max"


@irdl_op_definition
class MinOp(_Binary):
    name = "anvil.tensor.min"


@irdl_op_definition
class DivOp(_Binary):
    name = "anvil.tensor.div"


@irdl_op_definition
class SraOp(_Binary):
    name = "anvil.tensor.sra"


@irdl_op_definition
class CmpOp(IRDLOperation):
    name = "anvil.tensor.cmp"
    lhs = operand_def(i32)
    rhs = operand_def(i32)
    pred = prop_def(StringAttr)
    result = result_def(i32)

    def verify_(self):
        _expect(self, self.pred.data in PREDS, f"pred must be one of {PREDS}")


@irdl_op_definition
class SelectOp(IRDLOperation):
    name = "anvil.tensor.select"
    cond = operand_def(i32)
    a = operand_def(i32)
    b = operand_def(i32)
    result = result_def(i32)


@irdl_op_definition
class RequantOp(IRDLOperation):
    name = "anvil.tensor.requant"
    acc = operand_def(i32)
    extra = opt_operand_def(i32)
    M = prop_def(IntegerAttr)
    s = prop_def(IntegerAttr)
    result = result_def(i32)

    def verify_(self):
        _expect(self, 0 <= _int(self.s) <= 31, "s outside [0, 31]")


@irdl_op_definition
class RsqrtIndexOp(IRDLOperation):
    name = "anvil.tensor.rsqrt_index"
    ss = operand_def(i32)
    n = prop_def(IntegerAttr)
    idx = result_def(i32)
    k = result_def(i32)


@irdl_op_definition
class BarrierOp(IRDLOperation):
    name = "anvil.tensor.barrier"


@irdl_op_definition
class ApplyOp(IRDLOperation):
    name = "anvil.tensor.apply"
    indices = var_operand_def(IndexType)
    map = prop_def(AffineMapAttr)
    result = result_def(i32)

    def verify_(self):
        _check_map(self, self.map.data, len(self.indices), 1)


@irdl_op_definition
class GuardOp(IRDLOperation):
    name = "anvil.tensor.guard"
    indices = var_operand_def(IndexType)
    map = prop_def(AffineMapAttr)
    ub = prop_def(DenseArrayBase)
    body = region_def("single_block")
    traits = traits_def(NoTerminator())

    @property
    def ub_list(self) -> list[int]:
        return [int(x) for x in self.ub.get_values()]

    def verify_(self):
        _check_map(self, self.map.data, len(self.indices), len(self.ub_list))
        _expect(self, not self.body.block.args, "guard body takes no arguments")


TensorDialect = Dialect("anvil.tensor", [
    FuncOp, AllocOp, ForOp, LoadOp, StoreOp, GatherOp, ConstOp, CastOp, AddOp, SubOp, MulOp, MaxOp, MinOp, DivOp,
    SraOp, CmpOp, SelectOp, RequantOp, RsqrtIndexOp, BarrierOp, ApplyOp, GuardOp,
], [])

BINARY = {"add": AddOp, "sub": SubOp, "mul": MulOp, "max": MaxOp, "min": MinOp, "div": DivOp, "sra": SraOp}


# ----------------------------------------------------------------------------------------------------- builder

class TensorBuilder:
    """Insertion-point builder: `with b.loop(0, 4, bind="lane") as i:` nests; scalar methods return SSA values.

        b = TensorBuilder("k"); x = b.alloc("x", [4, 8], "i8"); with b.loop(0, 4) as i: ...; m = b.module()
    """

    def __init__(self, sym_name: str):
        self.func = FuncOp(sym_name)
        self._blocks = [self.func.body.block]

    def _add(self, op):
        self._blocks[-1].add_op(op)
        return op

    def _v(self, op):
        self._add(op)
        return op.results[0]

    def alloc(self, sym, shape, dtype="i32", space="global"):
        props = {} if sym is None else {"sym": StringAttr(sym)}
        return self._v(AllocOp(properties=props, result_types=[buffer_type(shape, dtype, space)]))

    def _nest(self, op, ret):
        b = self

        class _Ctx:
            def __enter__(self_):
                b._add(op)
                b._blocks.append(op.regions[0].block)
                return ret(op)

            def __exit__(self_, *exc):
                b._blocks.pop()

        return _Ctx()

    def loop(self, lb, ub, step=1, bind="seq", unroll=1, tag=None):
        return self._nest(ForOp(lb, ub, step, bind, unroll, tag), lambda op: op.iv)

    def guard(self, idx, m, ub):
        """`with b.guard([i, j], lambda i, j: (i * 4 + j,), [n]):` runs the body iff every map result < its ub."""
        idx = list(idx)
        mm = m if isinstance(m, AffineMap) else AffineMap.from_callable(m)
        return self._nest(GuardOp(operands=[idx], properties={"map": AffineMapAttr(mm),
                                                              "ub": DenseArrayBase.from_list(i64, list(ub))},
                                  regions=[Region(Block())]), lambda op: None)

    def apply(self, idx, m):
        idx = list(idx)
        mm = m if isinstance(m, AffineMap) else AffineMap.from_callable(m)
        return self._v(ApplyOp(operands=[idx], properties={"map": AffineMapAttr(mm)}, result_types=[i32]))

    @staticmethod
    def _map(m, n_idx, rank):
        if m is None:
            if n_idx != rank:
                raise ValueError("an identity map needs one index per buffer dimension")
            m = AffineMap.identity(rank)
        elif not isinstance(m, AffineMap):
            m = AffineMap.from_callable(m)
        return AffineMapAttr(m)

    def load(self, buf, idx, m=None):
        idx = list(idx)
        return self._v(LoadOp(operands=[buf, idx], properties={"map": self._map(m, len(idx), len(buf.type.get_shape()))},
                              result_types=[i32]))

    def store(self, value, buf, idx, m=None):
        idx = list(idx)
        self._add(StoreOp(operands=[value, buf, idx],
                          properties={"map": self._map(m, len(idx), len(buf.type.get_shape()))}))

    def gather(self, buf, v, offset=0):
        return self._v(GatherOp(operands=[buf, v], properties={"offset": IntegerAttr(offset, i64)}, result_types=[i32]))

    def const(self, value):
        return self._v(ConstOp(properties={"value": IntegerAttr(int(value), i64)}, result_types=[i32]))

    def cast(self, iv):
        return self._v(CastOp(operands=[iv], result_types=[i32]))

    def binary(self, kind, a, b):
        return self._v(BINARY[kind](a, b))

    def __getattr__(self, kind):
        if kind in BINARY:
            return lambda a, b: self.binary(kind, a, b)
        raise AttributeError(kind)

    def cmp(self, pred, a, b):
        return self._v(CmpOp(operands=[a, b], properties={"pred": StringAttr(pred)}, result_types=[i32]))

    def select(self, c, a, b):
        return self._v(SelectOp(operands=[c, a, b], result_types=[i32]))

    def requant(self, acc, m, s, extra=None):
        return self._v(RequantOp(operands=[acc, [] if extra is None else [extra]],
                                 properties={"M": IntegerAttr(int(m), i64), "s": IntegerAttr(int(s), i64)},
                                 result_types=[i32]))

    def rsqrt_index(self, ss, n):
        op = self._add(RsqrtIndexOp(operands=[ss], properties={"n": IntegerAttr(int(n), i64)}, result_types=[i32, i32]))
        return op.idx, op.k

    def barrier(self):
        self._add(BarrierOp())

    def module(self):
        """The builtin.module holding the func (built once; later calls return the same module)."""
        from . import module

        if self.func.parent is None:
            self._module = module(self.func)
        return self._module
