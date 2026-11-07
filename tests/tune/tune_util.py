"""Shared helpers for tests/tune: the xf fixture programs as graph IR with every buffer's reference values."""
import functools

from anvil.lower.graph_to_tensor import lower
from anvil.runtime import GpuParams
from anvil.tune import Task
from anvil.tune.g3_report import fixture_model, mm_model


@functools.cache
def program(block):
    """(lowered Program, values of every buffer, (module, consts), fixture dict)."""
    (module, consts), d = mm_model() if block == "mm1x8x8" else fixture_model(block)
    prog = lower(module)
    return prog, prog.run({"x": d["inputs"]["x"]}, consts), (module, consts), d


def task(block, name, params=GpuParams(), **kw):
    prog, vals, _, _ = program(block)
    return Task.from_program(prog, name, vals, params, **kw)
