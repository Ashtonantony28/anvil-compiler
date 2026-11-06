"""Helpers shared by the backend tests: run MIR (virtual, with R12 = args_base) on the MIR interpreter and compiled
programs on xf.isasim."""
from anvil.backend import compile_mir
from anvil.interp import mir as mi
from anvil.ir import mir
from xf import isasim

ADDR = 12


def with_args(f: mir.FuncOp, args_base: int) -> mir.FuncOp:
    """The backend input `f` made runnable on the MIR interpreter: R12 = args_base first, as the prologue does."""
    return mir.FuncOp(f.sym_name.data, [mir.Const32Op("R12", args_base)] + [op.clone() for op in f.instrs])


def interp(f, data, threads, args_base=0, **kw):
    return mi.run(with_args(f, args_base), data, threads, addr_bits=ADDR, **kw).memory


def sim(words, data, threads, *, flags_clean=True, **kw):
    r = isasim.run(words, data, threads, word_bits=32, addr_bits=ADDR, **kw)
    if flags_clean:
        assert not r.divergent, r.divergent
        assert not r.store_races, r.store_races
        assert not r.stores_conflict, r.stores_conflict
    return r


def compile_and_check(f, data, threads, *, args_base, spill_slot=None, ignore=(), **kw):
    """Compile f; check isasim(compiled) == MIR interpreter(f) outside the `ignore` address ranges and
    == MIR interpreter(allocated func) everywhere. Returns (program, memory)."""
    p = compile_mir(f, args_base=args_base, spill_slot=spill_slot)
    want = interp(f, data, threads, args_base, **kw)
    got = sim(p.words, data, threads, **kw).memory
    alloc = mi.run(p.func, data, threads, addr_bits=ADDR, **kw).memory
    assert alloc == got
    skip = set()
    for lo, hi in ignore:
        skip |= set(range(lo, hi))
    bad = [a for a in range(len(want)) if a not in skip and want[a] != got[a]]
    assert not bad, f"{len(bad)} words differ, first at {bad[0]}: want {want[bad[0]]} got {got[bad[0]]}"
    return p, got
