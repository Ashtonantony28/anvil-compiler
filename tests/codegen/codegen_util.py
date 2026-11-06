"""Helpers for the codegen tests: schedule an op, compile it, run it on isasim (and the MIR interpreter) and compare
with the tensor interpreter (C4 levels 2-4)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "schedule"))

from anvil.codegen import compile_kernel, kernel_job  # noqa: E402
from anvil.interp import mir as mi  # noqa: E402
from anvil.ir import mir  # noqa: E402
from anvil.runtime import GpuParams, run  # noqa: E402
from anvil.schedule import Schedule, Target, templates as T  # noqa: E402
from sched_cases import OPS, case  # noqa: E402,F401

CONFIGS = [(4, w, s) for w in (1, 2, 4) for s in (0, 1)]      # (THREADS_PER_BLOCK, WARPS_PER_CORE, SIMT_STACK)


def params(tpb=4, wpc=1, simt=0, **kw):
    return GpuParams(THREADS_PER_BLOCK=tpb, WARPS_PER_CORE=wpc, SIMT_STACK=simt, **kw)


def target(p: GpuParams, **kw):
    return Target.from_params(p, max_threads=kw.pop("max_threads", 4096), **kw)


def templates_for(op):
    if op in ("qmatmul", "qlinear"):
        return ["matmul_naive", "matmul_tiled"]
    if op in ("rmsnorm_int", "softmax_lut"):
        return ["rows_per_thread", "row_per_block", "none"]
    return ["elementwise", "none"]


def schedule(kernel, tpl, tgt):
    s = Schedule(kernel, tgt)
    return T.TEMPLATES[tpl](s) if tpl != "none" else s


def flags_clean(r, simt):
    bad = {k: r.flags[k] for k in ("store_races", "stores_conflict") if r.flags[k]}
    if not simt and r.flags["divergent"]:
        bad["divergent"] = r.flags["divergent"]
    return bad


def mir_memory(ck, plan, *, allocated=False):
    """Run every launch of `plan` on the MIR interpreter (virtual MIR, or the allocated program)."""
    p = ck.params
    mem = list(plan.job.data) + [0] * ((1 << p.DATA_MEM_ADDR_BITS) - len(plan.job.data))
    f = ck.program.func if allocated else ck.code.func
    for (_, i, base), ln in zip(plan.args, plan.job.launches):
        g = mir.FuncOp(f.sym_name.data, [mir.Const32Op("R12", base)] + [op.clone() for op in f.instrs]) \
            if not allocated else _patched(f, base)
        mem = mi.run(g, mem, ln.threads, threads_per_block=p.THREADS_PER_BLOCK,
                     warps_per_core=p.WARPS_PER_CORE, addr_bits=p.DATA_MEM_ADDR_BITS).memory
    return mem


def _patched(f, base):
    ops = [op.clone() for op in f.instrs]
    ops[0] = mir.ConstOp("R12", base >> 8)
    ops[4] = mir.ConstOp("R0", base & 0xFF)
    return mir.FuncOp(f.sym_name.data, ops)


def check_kernel(fin, bufs, p, *, predication="auto", mir_level=False, backend="isasim", compile_kw=None):
    """Compile the canonical kernel `fin`, run it, and assert every global buffer equals the tensor interpreter's.
    Returns (CompiledKernel, RunResult)."""
    want = fin.run(dict(bufs))
    ck = compile_kernel(fin, p, predication=predication, **(compile_kw or {}))
    vals = {s: v for s, v in bufs.items() if s in ck.buffers}
    plan = kernel_job(ck, vals)
    r = run(plan.job, backend)
    if backend == "isasim":
        bad = flags_clean(r, p.SIMT_STACK)
        assert not bad, bad
    for s in ck.buffers:
        got = plan.read(r.memory, s)
        assert np.array_equal(got, np.asarray(want[s]).reshape(got.shape)), (s, got, want[s])
    if mir_level:
        n = len(plan.job.data)
        for allocated in (False, True):
            m = mir_memory(ck, plan, allocated=allocated)
            assert [w & 0xFFFFFFFF for w in m[:plan.scratch]] == [w & 0xFFFFFFFF for w in r.memory[:plan.scratch]], \
                f"MIR interpreter (allocated={allocated}) differs from isasim"
            assert len(m) >= n
    return ck, r
