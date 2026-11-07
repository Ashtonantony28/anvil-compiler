"""Audit A1 (compiler correctness, backend/ISA consistency): strict-xfail repros for every critical/high/medium
finding of AUDIT-A1.md. Each test states the correct behaviour and passes once the finding is fixed (then drop its
xfail marker). Tests that need a cycle-level simulator are marked `rtl` (Verilator) and run under `make check_rtl`."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "schedule"))
sys.path.insert(0, str(HERE.parent / "codegen"))

from anvil.codegen import CodegenError, compile_kernel, kernel_job  # noqa: E402
from anvil.runtime import GpuParams, run  # noqa: E402
from anvil.schedule import Schedule, Target  # noqa: E402

import codegen_util as cu  # noqa: E402

# --------------------------------------------------------------------------------------------- shared builders

# The SG6 tuner candidate that showed "isasim and RTL disagree on scratch words" (1x8x8 qlinear `y`, 2 active
# lanes of 4, mask predication), as a plain schedule trace.
Y_TRACE = [["split", ["i", 1], {"outer": "io", "inner": "ii"}], ["split", ["j", 2], {"outer": "jo", "inner": "ji"}],
           ["reorder", ["io", "jo", "ii", "ji"], {}], ["fuse", ["ii", "ji"], {"name": "r"}],
           ["interchange", ["r", "k"], {}], ["fuse", ["io", "jo"], {"name": "io_jo"}],
           ["split", ["io_jo", 2, "mask"], {"outer": "io_jo.blk", "inner": "io_jo.thr"}],
           ["bind", ["io_jo.blk", "block"], {}], ["bind", ["io_jo.thr", "lane"], {}]]


def _y_kernel_job(p=GpuParams()):
    import json

    from anvil.lower.graph_to_tensor import lower
    from anvil.tune.g3_report import mm_model
    (m, consts), d = mm_model()
    k = lower(m).kernel("y")
    s = Schedule.replay(k, json.dumps(Y_TRACE), Target.from_params(p, max_threads=4096))
    ck = compile_kernel(s.finalize(), p, const_args=None)
    return ck, kernel_job(ck, {"x": d["inputs"]["x"], "w": consts["w"]})


# --------------------------------------------------------------------------------------------- A1-M1

@pytest.mark.rtl
@pytest.mark.xfail(strict=True, reason="A1-M1: masked-off lanes load out-of-bounds words that other cores are "
                                       "writing in the same launch (cross-core read/write race)")
def test_a1_m1_masked_off_lanes_do_not_race_other_cores():
    """Every word of the job (sink scratch included) is a deterministic function of the inputs: the cycle-level
    simulator, where blocks run concurrently on two cores, ends with exactly isasim's memory. Today block 1's
    inactive lanes 2-3 read w[7, 8..11] = y[0..3] while core 0 is storing y[0..3]."""
    ck, jp = _y_kernel_job()
    ri, rv = run(jp.job, "isasim"), run(jp.job, "verilator")
    assert rv.memory[:jp.end] == ri.memory[:jp.end]


@pytest.mark.xfail(strict=True, reason="A1-M1: codegen emits global loads for inactive lanes at addresses outside "
                                       "every buffer the kernel reads (canonical rule 2 says they must not load)")
def test_a1_m1_inactive_lane_loads_stay_inside_read_buffers():
    """isasim-free form of A1-M1 (runs in make check): trace the allocated program per thread; threads whose lane is
    outside the bound lane extent (lanes 2-3 here) must load nothing outside the buffers the kernel reads. Today
    block 1's inactive lanes load y[0..3], the output that core 0 is writing in the same launch."""
    ck, jp = _y_kernel_job()
    reads = _inactive_lane_load_addresses(ck, jp)
    a = jp.addresses
    inputs = set(range(a["x"], a["x"] + 8)) | set(range(a["w"], a["w"] + 64))
    assert reads <= inputs, sorted(reads - inputs)[:8]


def _inactive_lane_load_addresses(ck, jp):
    """Global addresses loaded by threads whose lane is >= the bound lane extent, from a per-lane MIR trace of the
    allocated program (anvil.interp.mir semantics, re-implemented minimally for LDR/ALU/branches)."""
    from anvil import semantics as sem
    from anvil.backend.lir import lift
    code = [i for i in lift(cu._patched(ck.program.func, jp.args[0][2]))]
    labels, pc = {}, 0
    prog = []
    for i in code:
        if i.op == "label":
            labels[i.sym] = len(prog)
        else:
            prog.append(i)
    mem = list(jp.job.data) + [0] * ((1 << 12) - len(jp.job.data))
    tpb, lx = ck.params.THREADS_PER_BLOCK, 2
    out = set()
    alu = {"add": sem.w_add, "sub": sem.w_sub, "mul": sem.w_mul, "div": sem.w_div, "sra": sem.w_sra,
           "max": sem.w_max}
    for b in range(ck.code.blocks):
        for t in range(tpb):
            regs = {"R13": b, "R14": tpb, "R15": t}
            pc, nz, steps = 0, False, 0
            while pc < len(prog) and steps < 100000:
                i, steps = prog[pc], steps + 1
                pc += 1
                g = lambda r: regs.get(r, 0)  # noqa: E731
                if i.op == "const":
                    regs[i.d] = i.imm
                elif i.op in alu:
                    regs[i.d] = alu[i.op](g(i.a), g(i.b))
                elif i.op == "ldr":
                    a = g(i.a) & 0xFFF
                    if t >= lx:
                        out.add(a)
                    regs[i.d] = mem[a]
                elif i.op == "lds":
                    regs[i.d] = 0                          # (no smem in this kernel)
                elif i.op == "cmp":
                    nz = g(i.a) != g(i.b)
                elif i.op == "br":
                    if nz and i.imm & 0b100:
                        pc = labels[i.sym]
                elif i.op == "ret":
                    break
    return out


# --------------------------------------------------------------------------------------------- A1-M2

@pytest.mark.xfail(strict=True, reason="A1-M2: the code generator accepts WORD_BITS < 32 and emits 32-bit code")
@pytest.mark.parametrize("bits", [16, 24])
def test_a1_m2_word_bits_below_32_rejected(bits):
    """Anvil's codegen, constants and requant idioms assume 32-bit words; a narrower GpuParams must be rejected
    (today it compiles and the lut kernel returns wrong values with no error)."""
    k, bufs, _ = cu.case("lut", "y")
    p = GpuParams(WORD_BITS=bits, DATA_MEM_DATA_BITS=bits)
    fin = cu.schedule(k, "elementwise", cu.target(p)).finalize()
    with pytest.raises(CodegenError):
        compile_kernel(fin, p)


# --------------------------------------------------------------------------------------------- A1-M3

@pytest.mark.xfail(strict=True, reason="A1-M3: the codegen/driver test oracles compare only the kernel's named "
                                       "buffers, so out-of-bounds writes elsewhere in memory go unnoticed")
def test_a1_m3_check_kernel_catches_stray_writes(monkeypatch):
    """codegen_util.check_kernel (the C4 level-4 oracle of tests/codegen) must fail when the run changes a word that
    belongs to no buffer of the kernel (here the first word past the job's data). Today it only
    reads the kernel's buffers back, which is why seeded mutant M02 (an extra block in the last split launch,
    writing past `c`) survives most launch-splitting configurations."""
    k, bufs, _ = cu.case("lut", "y")
    p = cu.params(4, 1, 0)
    fin = cu.schedule(k, "elementwise", cu.target(p)).finalize()
    real = cu.run

    def stray(job, backend="isasim"):
        r = real(job, backend)
        r.memory[len(job.data)] ^= 0x1234                 # first word past the job: no buffer, args or scratch
        return r
    monkeypatch.setattr(cu, "run", stray)
    with pytest.raises(AssertionError):
        cu.check_kernel(fin, bufs, p)


# --------------------------------------------------------------------------------------------- mutant killers
# Passing tests added by the audit: each kills a seeded mutant that `make check` let survive (AUDIT-A1.md, table).

def test_a1_rmsnorm_register_shift_31_rounds_like_golden():
    """rmsnorm_int whose requant shift s + k reaches 31 (k = 0: small rows): the register-shift rounding term must
    be 2^30, not SRA(2^31, 1) = -2^30 (kills mutant M15)."""
    from anvil.interp import graph as gi
    from anvil.ir.graph import GraphBuilder
    from anvil.lower.graph_to_tensor import lower
    rng = np.random.default_rng(31)
    g = GraphBuilder("rms31")
    x = g.input("x", [3, 4])
    y = g.rmsnorm_int("y", x, g.const("w", [4]), g.const("r", [256], dtype="i32"), 1, 31)
    g.output(y)
    m = g.module()
    vals = {"x": rng.integers(-3, 4, (3, 4)), "w": rng.integers(-128, 128, 4),
            "r": rng.integers(1 << 20, 1 << 22, 256)}
    want = gi.run(m, {"x": vals["x"]}, {"w": vals["w"], "r": vals["r"]}).values["y"]
    prog = lower(m)
    for tpl in ("rows_per_thread", "row_per_block"):
        p = cu.params(4, 1, 0)
        fin = cu.schedule(prog.kernel("y"), tpl, cu.target(p)).finalize()
        ck = compile_kernel(fin, p)
        jp = kernel_job(ck, {s: v for s, v in vals.items() if s in ck.buffers})
        r = run(jp.job, "isasim")
        assert np.array_equal(jp.read(r.memory, "y"), want), tpl


@pytest.mark.parametrize("simt", [0, 1])
def test_a1_warp_scope_cache_read_with_two_warps(simt):
    """cache_read(scope="warp") with W = 2 bound warps: each warp stages its own row of h, so warp-private smem
    buffers must be laid out at offset + w * size (kills mutant M20, which gives every warp the same words;
    no make-check test bound more than one warp around a warp-scope buffer)."""
    k, bufs, _ = cu.case("mlp_block", "gate")
    p = cu.params(8, 2, simt)
    s = Schedule(k, cu.target(p))
    s.split("i", 2, outer="io", inner="ii").split("j", 4, outer="jo", inner="ji").reorder("io", "jo", "ii", "ji")
    s.fuse("io", "jo", name="b").bind("b", "block").bind("ii", "warp").bind("ji", "lane")
    s.split("k", 4, outer="ko", inner="ki")
    s.cache_read(k.reads[0], "ko", scope="warp")
    ck, _ = cu.check_kernel(s.finalize(), bufs, p)
    assert ck.code.smem_words >= 2 * 4


@pytest.mark.parametrize("op", ["rmsnorm", "softmax"])
@pytest.mark.parametrize("cfg", [(4, 1, 0), (8, 2, 0), (4, 1, 1)])
def test_a1_row_per_block_ragged_rows(op, cfg):
    """row_per_block on rows whose length (5) is not a multiple of the lanes: the cross-lane partials accumulate
    under a lane-varying guard, so at SIMT_STACK=0 the register stores must be predicated (kills mutant M30; every
    make-check row kernel has 8- or 16-wide rows, a multiple of the lane count)."""
    from anvil.interp import graph as gi
    from anvil.ir.graph import GraphBuilder
    from anvil.lower.graph_to_tensor import lower
    rng = np.random.default_rng(5)
    g = GraphBuilder("ragged")
    x = g.input("x", [5, 5])
    if op == "rmsnorm":
        y = g.rmsnorm_int("y", x, g.const("w", [5]), g.const("r", [64], dtype="i32"), 3, 9)
        vals = {"w": rng.integers(-128, 128, 5), "r": rng.integers(0, 1 << 12, 64)}
    else:
        y = g.softmax_lut("y", x, g.const("r", [256], dtype="i32"), 127, causal=True)
        vals = {"r": rng.integers(1, 1 << 12, 256)}
    g.output(y)
    m = g.module()
    vals["x"] = rng.integers(-128, 128, (5, 5))
    want = gi.run(m, {"x": vals["x"]}, {s: v for s, v in vals.items() if s != "x"}).values["y"]
    p = cu.params(*cfg)
    fin = cu.schedule(lower(m).kernel("y"), "row_per_block", cu.target(p)).finalize()
    ck = compile_kernel(fin, p)
    jp = kernel_job(ck, {s: v for s, v in vals.items() if s in ck.buffers})
    r = run(jp.job, "isasim")
    assert not cu.flags_clean(r, p.SIMT_STACK)
    assert np.array_equal(jp.read(r.memory, "y"), want)
