"""Predication strategies, thread activity, barriers, the SIMT-stack depth proof, the smem sink and launch
splitting, on hand-built canonical kernels and on lowered graphs (isasim, bit-exact vs the tensor interpreter)."""
import numpy as np
import pytest

from anvil.codegen import CodegenError, compile_kernel, kernel_job, lower_kernel, program_job
from anvil.interp import graph as gi
from anvil.ir import tensor as t
from anvil.ir.graph import GraphBuilder
from anvil.lower.graph_to_tensor import Kernel, lower
from anvil.runtime import run
from anvil.schedule import Schedule, canonicalize, templates as T
from codegen_util import check_kernel, bad_flags, params, schedule, target

RNG = np.random.default_rng(7)


def kernel(name, build, p, writes=("y",)):
    b = t.TensorBuilder(name)
    build(b)
    k = Kernel(name, "custom", b.module(), [], list(writes), {}, {})
    return canonicalize(k, target(p))


def grid(b, B, W, L):
    """Open block / warp / lane loops; returns the context managers' ivs via a generator protocol."""
    return b.loop(0, B, bind="block", tag="blk"), b.loop(0, W, bind="warp", tag="w"), b.loop(0, L, bind="lane", tag="l")


def build_guarded(n_guards, B=2, W=1, L=4, smem=None, uniform=False):
    """y[g] = x[g] * 2 + 1 under `n_guards` nested lane-varying guards g < bound_k (bound decreasing)."""
    def build(b):
        x = b.alloc("x", [B * W * L])
        y = b.alloc("y", [B * W * L])
        cb, cw, cl = grid(b, B, W, L)
        with cb as i:
            s = b.alloc(None, [smem], "i32", "smem") if smem else None
            with cw as w, cl as j:
                lin = lambda i, w, j: (i * (W * L) + w * L + j,)  # noqa: E731
                ctxs = []
                for g in range(n_guards):
                    if uniform:
                        c = b.guard([i], lambda i: (i,), [B - 1 if g == 0 else B])
                    else:
                        c = b.guard([j], lambda j: (j,), [L - 1 - (g % 2)])
                    c.__enter__()
                    ctxs.append(c)
                v = b.add(b.mul(b.load(x, [i, w, j], lin), b.const(2)), b.const(1))
                if s is not None:
                    b.store(v, s, [w, j], lambda w, j: (w * L + j,))
                    v = b.load(s, [w, j], lambda w, j: (w * L + j,))
                b.store(v, y, [i, w, j], lin)
                for c in reversed(ctxs):
                    c.__exit__(None, None, None)
    return build


def bufs_for(n):
    return {"x": RNG.integers(-1000, 1000, n), "y": np.full(n, 7)}


@pytest.mark.parametrize("simt", [0, 1])
def test_lane_varying_guards_mask_and_sync(simt):
    p = params(4, 1, simt)
    k = kernel("g", build_guarded(2), p)
    ck, r = check_kernel(k, bufs_for(8), p, mir_level=not simt)
    reg = ck.code.stats["regions"]
    assert (reg["sync"], reg["mask"]) == ((2, 0) if simt else (0, 2))
    assert ck.code.stats["sink"] == (not simt)


def test_sync_nesting_is_capped_at_8_stack_entries():
    p = params(4, 1, 1)
    k = kernel("deep", build_guarded(6), p)
    ck, r = check_kernel(k, bufs_for(8), p)          # isasim raises if the stack ever exceeds 8 entries
    st = ck.code.stats
    assert st["sync_depth"] == 4 and st["simt_stack_entries"] == 8
    assert st["regions"]["sync"] == 4 and st["regions"]["mask"] == 2


def test_forced_predication_modes():
    p1 = params(4, 1, 1)
    k = kernel("g", build_guarded(1), p1)
    ck, _ = check_kernel(k, bufs_for(8), p1, predication="mask")
    assert ck.code.stats["regions"]["mask"] == 1
    ck, _ = check_kernel(k, bufs_for(8), p1, predication="sync")
    assert ck.code.stats["regions"]["sync"] == 1
    with pytest.raises(CodegenError, match="SIMT_STACK=1"):
        lower_kernel(k, params(4, 1, 0), predication="sync")
    with pytest.raises(CodegenError, match="needs mask"):
        lower_kernel(kernel("deep", build_guarded(5), p1), p1, predication="sync")


def test_uniform_guard_is_a_branch():
    p = params(4, 1, 0)
    k = kernel("u", build_guarded(2, B=3, uniform=True), p)
    ck, _ = check_kernel(k, bufs_for(12), p, mir_level=True)
    reg = ck.code.stats["regions"]
    assert reg["uniform"] >= 1 and reg["mask"] == 0 and not ck.code.stats["sink"]


def test_static_guards_emit_nothing():
    p = params(4, 1, 0)

    def build(b):
        x, y = b.alloc("x", [8]), b.alloc("y", [8])
        cb, cw, cl = grid(b, 2, 1, 4)
        with cb as i, cw, cl as j:
            with b.guard([i, j], lambda i, j: (i * 4 + j,), [8]):          # always true
                b.store(b.load(x, [i, j], lambda i, j: (i * 4 + j,)), y, [i, j], lambda i, j: (i * 4 + j,))
            with b.guard([j], lambda j: (j + 4,), [4]):                     # never true
                b.store(b.const(5), y, [i, j], lambda i, j: (i * 4 + j,))
    k = kernel("s", build, p)
    ck, _ = check_kernel(k, bufs_for(8), p, mir_level=True)
    assert ck.code.stats["regions"]["static"] == 2 and ck.code.stats["regions"]["mask"] == 0


def exchange_kernel(p):
    """W < WARPS_PER_CORE warps and Lx < lanes per warp lanes exchange words through smem between barriers."""
    lpw = p.THREADS_PER_BLOCK // p.WARPS_PER_CORE
    L, W = max(1, lpw - 1), max(1, p.WARPS_PER_CORE - 1)

    def build(b):
        n = 3 * W * L
        x, y = b.alloc("x", [n]), b.alloc("y", [n])
        cb, cw, cl = grid(b, 3, W, L)
        with cb as i:
            s = b.alloc(None, [W * L], "i32", "smem")
            with cw as w, cl as j:
                g = lambda i, w, j: (i * W * L + w * L + j,)  # noqa: E731
                b.store(b.load(x, [i, w, j], g), s, [w, j], lambda w, j: (w * L + j,))
                b.barrier()
                # read the neighbour's word: (w*L + j + 1) mod (W*L)
                v = b.load(s, [w, j], lambda w, j: ((w * L + j + 1) % (W * L),))
                b.barrier()
                b.store(b.sub(v, b.const(3)), y, [i, w, j], g)
    return kernel("act", build, p), bufs_for(3 * W * L)


@pytest.mark.parametrize("wpc,simt", [(1, 0), (1, 1), (2, 0), (2, 1), (4, 0), (4, 1)])
def test_inactive_lanes_and_warps_with_barriers(wpc, simt):
    p = params(8, wpc, simt)
    k, bufs = exchange_kernel(p)
    ck, _ = check_kernel(k, bufs, p, mir_level=not simt)
    reg = ck.code.stats["regions"]
    assert reg["uniform"] == (wpc > 1) and reg["sync" if simt else "mask"] == 1


def test_masked_smem_store_without_a_spare_word():
    """At SIMT_STACK=0 a masked smem store needs one spare smem word; with all 64 in use codegen says so, and
    SIMT_STACK=1 compiles the same kernel with SYNC predication instead."""
    k0 = kernel("full", build_guarded(1, smem=64, L=4), params(4, 1, 0))
    with pytest.raises(CodegenError, match="spare smem word"):
        lower_kernel(k0, params(4, 1, 0))
    p1 = params(4, 1, 1)
    ck, _ = check_kernel(kernel("full", build_guarded(1, smem=64, L=4), p1), bufs_for(8), p1)
    assert ck.code.stats["regions"]["sync"] == 1
    p = params(4, 1, 0)
    ck, _ = check_kernel(kernel("part", build_guarded(1, smem=8, L=4), p), bufs_for(8), p)
    assert ck.code.stats["mask_smem_stores"] == 1 and ck.code.smem_words == 9


def big_matmul(M=16, K=8, N=32):
    g = GraphBuilder("big")
    c = g.qmatmul("c", g.input("a", [M, K]), g.const("w", [K, N]), 3, 6)
    g.output(c)
    m = g.module()
    vals = {"a": RNG.integers(-128, 128, (M, K)), "w": RNG.integers(-128, 128, (K, N))}
    ref = gi.run(m, {"a": vals["a"]}, {"w": vals["w"]}).values
    return lower(m), vals, ref


SPLIT_CASES = [("matmul_naive", c) for c in [(4, 1, 0), (4, 2, 1), (8, 2, 0), (8, 4, 1), (2, 1, 1), (8, 4, 0)]]
SPLIT_CASES += [("matmul_tiled", c) for c in [(4, 1, 0), (4, 2, 1), (8, 2, 0), (8, 4, 1), (8, 4, 0)]]


@pytest.mark.parametrize("tpl,cfg", SPLIT_CASES, ids=str)
def test_launch_splitting(tpl, cfg):
    """B x THREADS_PER_BLOCK > 255: the block loop runs as several launches of one program, each with its own
    args block (block_offset)."""
    prog, vals, ref = big_matmul()
    p = params(*cfg)
    k = prog.kernels[0]
    fin = schedule(k, tpl, target(p)).finalize()
    ck = compile_kernel(fin, p)
    blocks = ck.code.blocks
    assert blocks * p.THREADS_PER_BLOCK > 255 or tpl == "matmul_tiled"
    if blocks * p.THREADS_PER_BLOCK > 255:
        assert ck.launches == -(-blocks // (255 // p.THREADS_PER_BLOCK)) >= 2
        assert ("block_offset", None) in ck.slots
    jp = program_job(prog, {k.name: ck}, vals, p)
    r = run(jp.job)
    assert not bad_flags(r, p.SIMT_STACK)
    assert not jp.stray_writes(r.memory, unchanged=[s for s in jp.buffers if s != "c"])   # A1-M3 (kills M02)
    assert np.array_equal(jp.read(r.memory, "c"), ref["c"])
    assert all(ln.threads <= 255 for ln in jp.job.launches)


def test_launch_split_kernel_job_matches_tensor_interpreter():
    p = params(4, 1, 0)

    def build(b):
        x, y = b.alloc("x", [70, 8]), b.alloc("y", [70, 8])
        cb, cw, cl = grid(b, 140, 1, 4)
        with cb as i, cw, cl as j:
            m = lambda i, j: ((i * 4 + j) // 8, (i * 4 + j) % 8)  # noqa: E731
            b.store(b.max(b.load(x, [i, j], m), b.const(0)), y, [i, j], m)
    k = kernel("relu", build, p)
    ck, r = check_kernel(k, {"x": RNG.integers(-99, 99, (70, 8)), "y": np.zeros((70, 8))}, p, mir_level=True)
    assert ck.launches == 3 and [ln.threads for ln in kernel_job(ck, {}).job.launches] == [252, 252, 56]
