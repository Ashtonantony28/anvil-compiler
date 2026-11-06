"""anvil.mir interpreter: agrees with xf.isasim on allocated programs (the xf kernels, hand-written MIR, random
straight-line code) and runs virtual-register MIR with the same results."""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from anvil.interp import mir as mi
from anvil.ir import mir, module, parse_ir, print_ir
from xf import attention, fixture, isasim, mlp

ADDR = 12


def _isasim(f, data, threads, **kw):
    r = isasim.run(mir.encode(f), data, threads, word_bits=32, addr_bits=ADDR, **kw)
    assert not r.divergent and not r.store_races
    return r.memory


@pytest.mark.parametrize("block,mod", [("mlp_block", mlp), ("attention", attention)])
@pytest.mark.parametrize("kind", ["naive", "tiled"])
@pytest.mark.parametrize("tpb,warps", [(4, 1), (8, 2), (16, 4)])
def test_xf_launch_chains_agree_with_isasim_and_golden(block, mod, kind, tpb, warps):
    """Every xf kernel, decoded to MIR, runs the whole launch chain bit-exactly as isasim and the golden."""
    d = fixture.load(fixture.FIXTURE_DIR / f"{block}_0.json")
    img = d["image"]
    img.write("x", d["inputs"]["x"])
    chain = mod.launches(img, kind)
    mem_m, mem_i = list(img.words), list(img.words)
    for out, (prog, threads) in chain:
        f = mir.decode(prog, out)
        mem_m = mi.run(f, mem_m, threads, addr_bits=ADDR, threads_per_block=tpb, warps_per_core=warps).memory
        mem_i = isasim.run(prog, mem_i, threads, word_bits=32, addr_bits=ADDR, threads_per_block=tpb,
                           warps_per_core=warps).memory
        assert mem_m == mem_i, out
    for k, v in d["golden"].items():
        assert np.array_equal(img.read(k, mem_m), v), k


def sum_loop(reg):
    """out[tid] = sum_{i<n} (tid + i) over a counted equality loop; `reg(k)` names register k."""
    r = reg
    return [
        mir.ConstOp(r(0), 0),                  # acc
        mir.ConstOp(r(1), 0),                  # i
        mir.ConstOp(r(2), 5),                  # n
        mir.ConstOp(r(3), 1),
        mir.LabelOp("top"),
        mir.AddOp(r(4), "R15", r(1)),
        mir.AddOp(r(0), r(0), r(4)),
        mir.AddOp(r(1), r(1), r(3)),
        mir.CmpOp(r(1), r(2)),
        mir.BrOp(0b100, "top"),                # loop while i != n
        mir.ConstOp(r(5), 100),
        mir.MulOp(r(6), "R13", "R14"),
        mir.AddOp(r(6), r(6), "R15"),
        mir.AddOp(r(5), r(5), r(6)),           # 100 + global thread id
        mir.StrOp(r(5), r(0)),
        mir.RetOp(),
    ]


def test_hand_written_loop_physical_vs_isasim():
    f = mir.func("sum", sum_loop(lambda k: f"R{k}"))
    want = _isasim(f, [], 6)
    got = mi.run(f, [], 6, addr_bits=ADDR).memory
    assert got == want
    assert got[100:106] == [5 * t + 10 for t in (0, 1, 2, 3, 0, 1)]


def test_virtual_registers_and_pseudo_ops():
    """Same program on virtual registers (+ sreg, const32, labels): same memory as the allocated version."""
    ops = sum_loop(lambda k: f"v{k}")
    ops = [mir.SregOp("v20", "threadIdx")] + [
        mir.AddOp(o.rd.data, "v20", o.rt.data) if isinstance(o, mir.AddOp) and o.rs.data == "R15" else o for o in ops]
    ops.insert(0, mir.Const32Op("v30", -70000))     # a negative wide constant, wrapped
    ops.insert(-1, mir.ConstOp("v31", 200))
    ops.insert(-1, mir.StrOp("v31", "v30"))
    f = parse_ir(print_ir(module(mir.func("v", ops)))).body.block.first_op
    got = mi.run(f, [], 6, addr_bits=ADDR).memory
    want = _isasim(mir.func("sum", sum_loop(lambda k: f"R{k}")), [], 6)
    assert got[:200] == want[:200] and got[200] == (1 << 32) - 70000


def test_barrier_across_warps_vs_isasim():
    """Lane t writes SMEM[t], BAR, reads SMEM[t ^ 4]-ish neighbour across the warp boundary (warps of 4 lanes)."""
    R = lambda k: f"R{k}"  # noqa: E731
    ops = [mir.ConstOp(R(0), 10), mir.MulOp(R(1), "R15", R(0)), mir.StsOp("R15", R(1)), mir.BarOp(),
           mir.ConstOp(R(2), 4), mir.AddOp(R(3), "R15", R(2)), mir.ConstOp(R(4), 7), mir.SubOp(R(5), R(4), "R15"),
           mir.LdsOp(R(6), R(5)),               # read lane 7 - t: half the reads cross warps
           mir.MulOp(R(7), "R13", "R14"), mir.AddOp(R(7), R(7), "R15"), mir.StrOp(R(7), R(6)), mir.RetOp()]
    f = mir.func("bar", ops)
    for warps in (1, 2):
        want = _isasim(f, [], 16, threads_per_block=8, warps_per_core=warps)
        got = mi.run(f, [], 16, addr_bits=ADDR, threads_per_block=8, warps_per_core=warps).memory
        assert got == want and got[:8] == [10 * (7 - t) for t in range(8)]


ALU = sorted(mir.ALU)


@st.composite
def straight_line(draw):
    """Random ALU/LDR code on R0..R10 seeded from memory; R11 = 1 and R12 the store pointer at the end."""
    data = draw(st.lists(st.integers(0, (1 << 32) - 1), min_size=11, max_size=11))
    ops = [mir.ConstOp("R11", 1), mir.ConstOp("R12", 0)]
    for k in range(11):           # R_k = data[k]
        ops += [mir.LdrOp(f"R{k}", "R12"), mir.AddOp("R12", "R12", "R11")]
    for _ in range(draw(st.integers(0, 25))):
        kind = draw(st.sampled_from(ALU + ["LDR", "CONST"]))
        d = f"R{draw(st.integers(0, 10))}"
        s, t = (f"R{draw(st.sampled_from([*range(11), 13, 14, 15]))}" for _ in range(2))
        if kind == "LDR":
            ops.append(mir.LdrOp(d, s))
        elif kind == "CONST":
            ops.append(mir.ConstOp(d, draw(st.integers(0, 255))))
        else:
            ops.append(mir.ALU[kind](d, s, t))
    # R12 = 64 + 11 * (blockIdx * blockDim + threadIdx), then store R0..R10
    ops += [mir.MulOp("R12", "R13", "R14"), mir.AddOp("R12", "R12", "R15"), mir.NopOp()]
    ops += [mir.ConstOp("R11", 11), mir.MulOp("R12", "R12", "R11"), mir.ConstOp("R11", 64),
            mir.AddOp("R12", "R12", "R11"), mir.ConstOp("R11", 1)]
    for k in range(11):
        ops += [mir.StrOp("R12", f"R{k}"), mir.AddOp("R12", "R12", "R11")]
    ops.append(mir.RetOp())
    return mir.func("rand", ops), data


@settings(max_examples=40, deadline=None)
@given(straight_line(), st.integers(1, 10))
def test_random_straight_line_vs_isasim(case, threads):
    f, data = case
    assert mi.run(f, data, threads, addr_bits=ADDR).memory == _isasim(f, data, threads)


def test_divergence_raises():
    f = mir.func("div", [mir.ConstOp("v0", 1), mir.CmpOp("R15", "v0"), mir.BrOp(0b010, "end"), mir.NopOp(),
                         mir.LabelOp("end"), mir.RetOp()])
    with pytest.raises(mi.DivergenceError):
        mi.run(f, [], 4)
    mi.run(f, [], 1)          # one lane: uniform


def test_undefined_virtual_register_and_runaway():
    with pytest.raises(mi.MirInterpError, match="read before written"):
        mi.run(mir.func("u", [mir.AddOp("v0", "v1", "R0"), mir.RetOp()]), [], 1)
    with pytest.raises(mi.MirInterpError, match="past the last"):
        mi.run(mir.func("r", [mir.NopOp()]), [], 1)
    with pytest.raises(mi.MirInterpError, match="steps"):
        mi.run(mir.func("inf", [mir.LabelOp("a"), mir.CmpOp("R0", "R0"), mir.BrOp(0b010, "a")]), [], 1, max_steps=100)
