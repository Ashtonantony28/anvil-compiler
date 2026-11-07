"""anvil.isa_ext: ISA proposals as a model-only study (SG8 / G5). Never edits sim/.

    from anvil.isa_ext import enabled, run_job, PROPOSALS
    with enabled({"MIN", "MAD"}):
        ck = anvil.codegen.compile_kernel(...)      # codegen + backend emit the new instructions where they pay
    r = run_job(ck_job)                             # isasim fork: memory, flags, races, tier-1 cycles

Modules
    isa      extended opcode table (5-bit opcode, 17-bit words; base opcodes unchanged), encode/decode, semantics
    isasim   fork of sim/xf/isasim.py running extended programs, recording tier-1 instruction streams
    run      Job runner on the fork (+ cross-block race check, tier-1 timing) and anvil.tune's execute contract
    backend  MIR ops, idiom switches, MIR peepholes (MAD fusion, AND for mod 2^k, SHL rounding) and `enabled`
    study    `python3 -m anvil.isa_ext.study`: every proposal on the MLP block, attention and the SG5 model
"""
from .backend import enabled
from .isa import PROPOSALS
from .run import run_job

__all__ = ["PROPOSALS", "enabled", "run_job"]
