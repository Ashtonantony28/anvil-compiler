"""Anvil's three xDSL dialects (anvil.graph, anvil.tensor, anvil.mir) and their textual form.

    text = print_ir(module); module2 = parse_ir(text)       # MLIR textual round trip
"""
import io

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.parser import Parser
from xdsl.printer import Printer


def context():
    from . import graph, mir, tensor

    ctx = Context()
    for d in (Builtin, graph.Graph, tensor.TensorDialect, mir.Mir):
        ctx.load_dialect(d)
    return ctx


def print_ir(op):
    s = io.StringIO()
    Printer(stream=s).print_op(op)
    return s.getvalue() + "\n"


def parse_ir(text):
    module = Parser(context(), text).parse_module()
    module.verify()
    return module


def module(*ops):
    m = ModuleOp(list(ops))
    m.verify()
    return m
