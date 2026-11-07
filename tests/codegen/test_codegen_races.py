"""Masked-off lanes and cross-block races (AUDIT-A1 M1): under mask predication, lanes that are inactive or fail a
lane-varying guard must load nothing outside the buffers active lanes read, so no block of a launch reads a word
another block writes. anvil.runtime.check_races finds such races in an isasim address trace."""
import numpy as np
import pytest

from anvil.codegen import compile_kernel, kernel_job
from anvil.ir import tensor as t
from anvil.lower.graph_to_tensor import Kernel
from anvil.runtime import Job, Launch, check_races, run
from anvil.schedule import Schedule, canonicalize
from codegen_util import OPS, case, check_kernel, params, target


def ragged(kernel, tgt, factor):
    """Fuse the spatial loops of a matmul/elementwise kernel, split by `factor` with a masked tail, bind block x
    lane: with factor < THREADS_PER_BLOCK lanes >= factor are inactive, and the last block has a ragged guard."""
    s = Schedule(kernel, tgt)
    loops = [x[0] for x in s.loops() if x[3] == "spatial" and (x[0] in ("i", "j") or x[0].startswith(("b", "d")))]
    name = loops[0]
    for nxt in loops[1:]:
        s.fuse(name, nxt, name=f"{name}_{nxt}")
        name = f"{name}_{nxt}"
    s.split(name, factor, "mask", outer="blk", inner="thr").bind("blk", "block").bind("thr", "lane")
    return s.finalize()


EW = [c for c in OPS if c[1] in ("gate", "down", "act", "out", "kT", "scores", "o", "y")]


@pytest.mark.parametrize("factor", [3, 2])
@pytest.mark.parametrize("block,name", EW)
def test_masked_lanes_do_not_race_across_blocks(block, name, factor):
    k, bufs, _ = case(block, name)
    p = params(4, 1, 0)
    fin = ragged(k, target(p), factor)
    ck, _ = check_kernel(fin, bufs, p, predication="mask")
    vals = {s: v for s, v in bufs.items() if s in ck.buffers}
    out = [s for s in ck.buffers if s in k.writes]
    for r in [s for s in ck.buffers if s not in out]:
        # each read buffer in turn directly followed by the output: an out-of-bounds load past its end would read
        # words another block is writing (the shape of the SG6 repro)
        order = [s for s in ck.buffers if s != r and s not in out] + [r] + out
        addr, a = {}, 0
        for s in order:
            addr[s] = a
            a += int(np.prod(ck.buffers[s][0]))
        plan = kernel_job(ck, vals, addresses=addr)
        rr = check_races(plan.job)
        assert rr.races == [], (r, rr.races[:4])
        assert rr.result.memory == run(plan.job).memory      # the traced run is the plain isasim run


def test_check_races_finds_a_cross_block_read_of_another_blocks_output():
    """Block b stores y[b] and reads y[(b + 1) % 2]: a read-write race between the two blocks (isasim, which runs blocks
    in order, gives a deterministic but wrong-by-design answer)."""
    p = params(4, 1, 0)

    def build(b):
        x = b.alloc("x", [2])
        y = b.alloc("y", [2])
        z = b.alloc("z", [2])
        with b.loop(0, 2, bind="block", tag="blk") as i, b.loop(0, 1, bind="warp", tag="w"), \
                b.loop(0, 1, bind="lane", tag="l"):
            b.store(b.load(x, [i], lambda i: (i,)), y, [i], lambda i: (i,))
            b.store(b.load(y, [i], lambda i: ((i + 1) % 2,)), z, [i], lambda i: (i,))
    tb = t.TensorBuilder("race")
    build(tb)
    k = canonicalize(Kernel("race", "custom", tb.module(), ["x"], ["y", "z"], {}, {}), target(p))
    ck = compile_kernel(k, p)
    plan = kernel_job(ck, {"x": np.array([5, 6])})
    rr = check_races(plan.job)
    ya = plan.addresses["y"]
    assert {(a, kind) for _, a, kind, _ in rr.races} == {(ya, "read-write"), (ya + 1, "read-write")}
    assert rr.result.memory == run(plan.job).memory


def test_check_races_write_write_needs_different_values():
    from xf import isa
    # every thread of every block stores its blockIdx to word 10 (different values across blocks), then RET
    prog = [isa.encode("CONST", rd=0, imm=10), isa.encode("STR", rs=0, rt=13), isa.encode("RET")]
    same = [isa.encode("CONST", rd=0, imm=10), isa.encode("CONST", rd=1, imm=7), isa.encode("STR", rs=0, rt=1),
            isa.encode("RET")]
    job = Job([0] * 16, [Launch(prog, 8)])
    assert [(a, kind) for _, a, kind, _ in check_races(job).races] == [(10, "write-write")]
    assert check_races(Job([0] * 16, [Launch(same, 8)])).races == []
