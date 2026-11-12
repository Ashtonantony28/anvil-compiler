"""Anvil's second target: Vortex-V, an RV64IM core with the Xmlacc int8 extension (github.com/Ashtonantony28/vortex-v).

    from anvil import vortexv
    prog = vortexv.compile(module, consts)            # an anvil.graph module (+ consts bound by sym)
    res = prog.run({"x": x})                          # Vortex-V's golden ISS
    res.outputs, res.values["scores"], res.instret
    prog.image({"x": x})                              # (stub words, data) for Vortex-V's RTL harness

The same graph IR that lowers onto tiny-gpu lowers here onto one RISC-V core: matmuls use ``mlacc.macc`` (eight
int8 products per instruction) and every requant is ``mul`` + ``mlacc.rqs``, which is bit-identical to Anvil's
requant (see ``codegen``). ``mutant="floor"`` (rqs replaced by a plain shift) and ``mutant="scalar"`` (macc replaced
by RV64IM code) exist for the tests and the Xmlacc speed-up measurement.
"""
from dataclasses import dataclass, field

import numpy as np

from ..ir import graph as g
from . import _vv
from .codegen import CODE_BASE, DATA_BASE, Compiler

__all__ = ["compile", "Program", "Result", "available", "CODE_BASE", "DATA_BASE"]

available = _vv.available


@dataclass
class Result:
    outputs: list
    values: dict = field(default_factory=dict)
    instret: int = 0


def _func(m):
    if isinstance(m, g.FuncOp):
        return m
    fs = g.funcs(m)
    if len(fs) != 1:
        raise ValueError(f"expected one anvil.graph.func, found {len(fs)}")
    return fs[0]


class Program:
    def __init__(self, comp: Compiler):
        rv = comp.e.rv
        self._rv = rv
        self.words = comp.e.a.assemble()
        self.inputs = comp.inputs
        self.syms = comp.syms
        self.views = comp.view
        self.outputs = comp.outputs
        self.bufs = comp.lay.bufs
        self.const_data = comp.const_data
        self.data_end = comp.lay.next
        stub = rv.Asm(base=rv.RESET_PC)
        stub.la(5, CODE_BASE)
        stub.jalr(0, 5, 0)
        self.stub = stub.assemble()
        if CODE_BASE + 4 * len(self.words) > DATA_BASE:
            raise ValueError(f"program of {len(self.words)} instructions overruns the data region")

    @property
    def n_instructions(self):
        return len(self.words)

    def _bytes(self, inputs):
        """{addr: bytes} for the code, the constants and the inputs."""
        code = b"".join(w.to_bytes(4, "little") for w in self.words)
        out = {CODE_BASE: code}
        for name, raw in self.const_data.items():
            out[self.bufs[name].addr] = raw
        for sym, b in self.inputs.items():
            if sym not in inputs:
                raise KeyError(f"missing input {sym!r}")
            x = np.asarray(inputs[sym], dtype=np.int64)
            if list(x.shape) != b.dims:
                raise ValueError(f"input {sym!r}: shape {list(x.shape)} != {b.dims}")
            out[b.addr] = b.pack(x)
        return out

    def image(self, inputs):
        """(stub words at RESET_PC, data as (addr, 8, value) tuples) in the format of Vortex-V's harness."""
        data = []
        for addr, raw in self._bytes(inputs).items():
            raw = raw + b"\0" * (-len(raw) % 8)
            data += [(addr + i, 8, int.from_bytes(raw[i:i + 8], "little")) for i in range(0, len(raw), 8)]
        return self.stub, data

    def decode(self, load):
        """Read every named value back with `load(addr, nbytes) -> unsigned int`."""
        def read(b):
            raw = bytearray()
            for i in range(0, b.nbytes, 8):
                raw += load(b.addr + i, 8).to_bytes(8, "little")
            return b.unpack(raw)
        values = {sym: read(b) for sym, b in self.syms.items()}
        for sym, (b, how) in self.views.items():
            v = read(b)
            values[sym] = np.swapaxes(v, -1, -2) if how == "swap" else v
        return [read(b) for b in self.outputs], values

    def run(self, inputs, max_steps=50_000_000):
        iss = _vv.iss()
        rv = self._rv
        mem = iss.Memory(base=rv.RESET_PC)
        mem.load_words(self.stub, base=rv.RESET_PC)
        for addr, raw in self._bytes(inputs).items():
            mem.load_image(raw, base=addr)
        hart = iss.Hart(mem=mem, reset_pc=rv.RESET_PC, tohost_addr=rv.TOHOST_ADDR)
        r = hart.run(max_steps=max_steps)
        if not (r["halted"] and r["halt_code"] == (True, 0)):
            raise RuntimeError(f"Vortex-V ISS did not finish: {r}")
        outs, values = self.decode(mem.load)
        return Result(outs, values, r["steps"])


def compile(module, consts=None, *, mutant=None) -> Program:
    """Lower an anvil.graph module to a Vortex-V program."""
    if mutant not in (None, "floor", "scalar"):
        raise ValueError(f"unknown mutant {mutant!r}")
    comp = Compiler(_func(module), consts, mutant)
    comp.run()
    return Program(comp)
