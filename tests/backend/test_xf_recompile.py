"""SG2 accept: every xf kernel, decoded to MIR, de-allocated to virtual registers and recompiled, runs the full mlp
and attention launch chains on isasim with golden-equal memory (and memory equal to the hand-written chain)."""
import numpy as np
import pytest

from anvil.backend import compile_mir
from anvil.backend.dealloc import devirtualize
from anvil.interp import mir as mi
from anvil.ir import mir
from xf import attention, fixture, isa, mlp
from xf.asm import assemble
from xf.kernels import KERNEL_DIR, program

from backend_util import ADDR, sim, with_args

CHAIN_KERNELS = ["matmul_naive", "matmul_tiled", "relu", "requant", "residual_add", "rmsnorm", "softmax", "swiglu",
                 "transpose"]


def args_base_of(words):
    return isa.decode(words[0])[1]["imm"] * 256 + isa.decode(words[4])[1]["imm"]


def recompile(words, name, seed=None):
    return compile_mir(devirtualize(mir.decode(words, name), seed=seed), args_base=args_base_of(words))


@pytest.mark.parametrize("name", CHAIN_KERNELS)
def test_devirtualized_kernel_is_virtual_and_recompiles(name):
    words = program(name, 0x2A7)
    v = devirtualize(mir.decode(words, name))
    regs = {getattr(op, k).data for op in v.instrs for k in op.READS + op.WRITES}
    assert not regs & {f"R{k}" for k in range(12)}, "physical registers left"
    assert not any(isinstance(op, mir.ConstOp) and op.rd.data == "R12" for op in v.instrs)
    p = recompile(words, name)
    assert assemble(p.asm_text) == p.words
    assert p.words[:6] == words[:6], "ARGS prologue identical to xf.kernels.program's"
    assert len(p.words) <= 256 and p.stats["spilled_vregs"] == 0


def _chain(block, mod, kind, tpb, warps, seed=None):
    d = fixture.load(fixture.FIXTURE_DIR / f"{block}_0.json")
    img = d["image"]
    img.write("x", d["inputs"]["x"])
    chain = mod.launches(img, kind)
    mem_new, mem_old = list(img.words), list(img.words)
    kw = dict(threads_per_block=tpb, warps_per_core=warps)
    sizes = []
    for out, (prog, threads) in chain:
        p = recompile(prog, out, seed)
        sizes.append((len(prog), len(p.words)))
        old = sim(prog, mem_old, threads, flags_clean=False, **kw)
        new = sim(p.words, mem_new, threads, flags_clean=False, **kw)
        assert (new.divergent, new.store_races) == (old.divergent, old.store_races) == ([], []), out
        mem_new, mem_old = new.memory, old.memory
        assert mem_new == mem_old, out
    for k, v in d["golden"].items():
        assert np.array_equal(img.read(k, mem_new), v), k
    return sizes


@pytest.mark.parametrize("block,mod", [("mlp_block", mlp), ("attention", attention)])
@pytest.mark.parametrize("kind", ["naive", "tiled"])
@pytest.mark.parametrize("tpb,warps", [(4, 1), (8, 2), (16, 4)])
def test_recompiled_chains_golden_equal(block, mod, kind, tpb, warps):
    _chain(block, mod, kind, tpb, warps)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_recompiled_chains_with_shuffled_virtual_names(seed):
    """Allocation must not depend on the de-allocator's register numbering."""
    _chain("mlp_block", mlp, "tiled", 4, 1, seed)
    _chain("attention", attention, "naive", 8, 2, seed)


@pytest.mark.parametrize("block,mod,kind", [("mlp_block", mlp, "tiled"), ("attention", attention, "naive")])
def test_mir_interp_before_and_after_allocation(block, mod, kind):
    """C4 level 3 on every launch of a chain: virtual MIR == allocated MIR == isasim (multi-warp config)."""
    d = fixture.load(fixture.FIXTURE_DIR / f"{block}_0.json")
    img = d["image"]
    img.write("x", d["inputs"]["x"])
    chain = mod.launches(img, kind)             # allocates the args segments
    mem = list(img.words)
    kw = dict(threads_per_block=8, warps_per_core=2)
    for out, (prog, threads) in chain:
        base = args_base_of(prog)
        v = devirtualize(mir.decode(prog, out))
        p = compile_mir(v, args_base=base)
        a = mi.run(with_args(v, base), mem, threads, addr_bits=ADDR, **kw).memory
        b = mi.run(p.func, mem, threads, addr_bits=ADDR, **kw).memory
        c = sim(p.words, mem, threads, **kw).memory
        assert a == b == c, out
        mem = c


def test_code_size_vs_hand_written():
    """Recompiled kernels stay within 10% (+2 words) of the hand-written ones (stats in the SG2 report)."""
    for name in CHAIN_KERNELS:
        words = assemble((KERNEL_DIR / f"{name}.asm").read_text())
        n = len(recompile(words, name).words)
        assert n <= len(words) * 1.1 + 2, (name, len(words), n)
