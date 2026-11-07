"""anvil.codegen: canonical tensor kernels -> anvil.mir -> compiled launches and runnable Jobs (SG3b).

    ck = compile_kernel(sch.finalize(), GpuParams(...))          # one kernel: MIR, compile_mir, launch geometry
    ck.code.func                  the virtual-register MIR (anvil.interp.mir runs it with R12 = args base)
    ck.program                    anvil.backend.Program (words bound to args_base 0; launches patch words 0/4)
    plan = program_job(prog, {name: ck, ...}, values, params)    # every kernel of a lowered Program -> Job
    res = anvil.runtime.run(plan.job, "isasim"); plan.read(res.memory, "out")

Lowering strategy (index arithmetic, loops, predication, memory) is documented in `anvil.codegen.lower`.

Args block of one launch (Anvil ABI v1, see anvil.backend; one block per launch, `ck.slots` names each word)
--------------------------------------------------------------------------------------------------------
    args[0 .. nb-1]   absolute address of every global buffer of the kernel, in func order (its global allocs:
                      graph values, consts and Kernel.temps)
    block_offset      only when the block loop is split: the first block iv of this launch (launch i runs blocks
                      i*bpl .. i*bpl + bpl - 1, bpl = 255 // THREADS_PER_BLOCK, the last one possibly fewer)
    sink              only when a masked global store exists: base of `threads` per-thread sink words
                      (thread g = blockIdx*blockDim + threadIdx writes word sink + g instead of a masked store)
    spill             only when the program spills: base of `threads * frame_words` spill words (anvil.backend)
Every compile-time constant (requant M and s, LUT offsets, strides, extents) is an immediate of the program.

Launch splitting: blocks x THREADS_PER_BLOCK above 255 (allowed when the schedule's Target.max_threads is raised)
becomes ceil(B / bpl) launches of one program with per-launch args blocks; launches are sequential, so the
scratch words (sink, spill) are shared by all launches and kernels of a job.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..backend import Program as MachineProgram
from ..backend import compile_mir
from ..ir import tensor as t
from ..runtime import GpuParams, Job, Launch
from .affine import CodegenError
from .lower import HW_MAX_THREADS, PREDICATION, KernelCode, _Gen

__all__ = ["compile_kernel", "lower_kernel", "CompiledKernel", "KernelCode", "CodegenError", "JobPlan",
           "program_job", "kernel_job", "bump_allocate", "patch_args_base", "PREDICATION", "HW_MAX_THREADS"]


def lower_kernel(kernel, params: GpuParams = GpuParams(), *, predication: str = "auto",
                 const_args: int | None = 3) -> KernelCode:
    """Canonical tensor kernel -> KernelCode (virtual-register MIR + args slots + geometry). `const_args`: word
    constants that take at least this many instructions to build are read from the args block instead."""
    return _Gen(kernel, params, predication, const_args).run()


def patch_args_base(words, args_base: int) -> list[int]:
    """Rebind a compiled program to another args block (ARGS prologue words 0 and 4, as xf.kernels.program)."""
    from xf import isa
    if not 0 <= args_base < 1 << 16:
        raise CodegenError(f"args_base {args_base} outside 0..65535")
    w = list(words)
    w[0] = isa.encode("CONST", rd=12, imm=args_base >> 8)
    w[4] = isa.encode("CONST", rd=0, imm=args_base & 0xFF)
    return w


@dataclass
class CompiledKernel:
    name: str
    code: KernelCode
    program: MachineProgram
    params: GpuParams
    slots: list                                 # code.slots + ("spill", None) when the program spills
    buffers: dict = field(default_factory=dict)  # sym -> (shape, dtype) of every global buffer it addresses

    @property
    def launches(self) -> int:
        return self.code.launches

    def threads(self, i: int) -> int:
        bpl, B = self.code.blocks_per_launch, self.code.blocks
        return (min(B, (i + 1) * bpl) - i * bpl) * self.code.threads_per_block

    @property
    def max_threads(self) -> int:
        return self.threads(0)

    @property
    def frame_words(self) -> int:
        return self.program.stats["frame_words"]

    @property
    def scratch_words(self) -> int:
        """Words of shared scratch (sink + spill areas) one launch needs."""
        n = self.max_threads
        return (n if ("sink", None) in self.slots else 0) + n * self.frame_words


    def args(self, addresses: dict, i: int, scratch: int) -> list[int]:
        """The args block of launch i."""
        out = []
        sink_words = self.max_threads if ("sink", None) in self.slots else 0
        for kind, sym in self.slots:
            if kind == "buf":
                if sym not in addresses:
                    raise CodegenError(f"{self.name}: no address for buffer {sym!r}")
                out.append(int(addresses[sym]))
            elif kind == "block_offset":
                out.append(i * self.code.blocks_per_launch)
            elif kind == "sink":
                out.append(scratch)
            elif kind == "spill":
                out.append(scratch + sink_words)
            elif kind == "const":
                out.append(sym)
        return out

    def launch(self, args_base: int, i: int) -> Launch:
        return Launch(patch_args_base(self.program.words, args_base), self.threads(i))


def compile_kernel(kernel, params: GpuParams = GpuParams(), *, predication: str = "auto",
                   const_args: int | None = 3, **compile_kw) -> CompiledKernel:
    """Canonical tensor kernel (Schedule.finalize()) -> compiled program + args layout."""
    code = lower_kernel(kernel, params, predication=predication, const_args=const_args)
    spill_slot = len(code.slots)
    prog = compile_mir(code.func, args_base=0, spill_slot=spill_slot, name=code.name, **compile_kw)
    slots = list(code.slots) + ([("spill", None)] if prog.stats["frame_words"] else [])
    bufs = {op.sym.data: (list(op.result.type.get_shape()), t.dtype_of(op.result.type))
            for op in kernel.func.body.block.ops if isinstance(op, t.AllocOp)}
    return CompiledKernel(kernel.name, code, prog, params, slots, bufs)


# ------------------------------------------------------------------------------------------------- jobs

def bump_allocate(buffers: dict, start: int = 0) -> dict:
    """sym -> address, buffers (sym -> (shape, dtype)) packed in order from `start` (a test allocator; SG4b
    owns the real memory planner)."""
    out, a = {}, start
    for sym, (shape, _) in buffers.items():
        out[sym] = a
        a += math.prod(shape)
    return out


@dataclass
class JobPlan:
    job: Job
    addresses: dict                     # buffer sym -> address
    buffers: dict                       # buffer sym -> (shape, dtype)
    args: list = field(default_factory=list)          # [(kernel name, launch index, args_base)]
    scratch: int = 0
    end: int = 0                                        # first free word

    def stray_writes(self, memory, *, unchanged=(), scratch_ok: bool = True) -> list[int]:
        """Addresses whose final word differs from the job's initial memory although they belong to no buffer of
        the plan: args blocks, gaps and every word past the job (AUDIT-A1 M3: a check of the named buffers alone
        cannot see a store outside them), plus the words of the buffers named in `unchanged` (e.g. consts). The
        shared sink/spill words [scratch, end) are exempt with scratch_ok."""
        mask = (1 << self.job.params.WORD_BITS) - 1
        inside = np.zeros(len(memory), dtype=bool)
        keep = set(unchanged)
        for sym, (shape, _) in self.buffers.items():
            if sym in self.addresses and sym not in keep:
                a = self.addresses[sym]
                inside[a:a + math.prod(shape)] = True
        if scratch_ok:
            inside[self.scratch:self.end] = True
        init = np.zeros(len(memory), dtype=np.int64)
        init[:len(self.job.data)] = np.asarray(self.job.data, dtype=np.int64) & mask
        got = np.asarray(memory, dtype=np.int64) & mask
        return [int(a) for a in np.flatnonzero((got != init) & ~inside)]

    def read(self, memory, sym) -> np.ndarray:
        shape, _ = self.buffers[sym]
        a, n = self.addresses[sym], math.prod(shape)
        w = np.array(memory[a:a + n], dtype=np.int64) & 0xFFFFFFFF
        return np.where(w >= 1 << 31, w - (1 << 32), w).reshape(shape)


def _job(order, buffers, values, params, addresses, mem_words):
    addresses = dict(addresses) if addresses is not None else bump_allocate(buffers)
    missing = [s for s in buffers if s not in addresses]
    if missing:
        raise CodegenError(f"no address for buffers {missing}")
    end = max([addresses[s] + math.prod(buffers[s][0]) for s in buffers] + [0])
    launches, arg_list = [], []
    blocks = []
    for ck in order:
        for i in range(ck.launches):
            blocks.append((ck, i, end))
            end += len(ck.slots)
    scratch = end
    end += max([ck.scratch_words for ck in order] + [0])
    if end > mem_words:
        raise CodegenError(f"job needs {end} words of data memory > {mem_words}")
    data = [0] * end
    mask = (1 << params.WORD_BITS) - 1
    for sym, v in values.items():
        if sym not in buffers:
            continue
        a = np.asarray(v, dtype=np.int64).ravel()
        if a.size != math.prod(buffers[sym][0]):
            raise CodegenError(f"{sym}: {a.size} values for shape {buffers[sym][0]}")
        base = addresses[sym]
        data[base:base + a.size] = [int(x) & mask for x in a]
    for ck, i, base in blocks:
        data[base:base + len(ck.slots)] = [int(x) & mask for x in ck.args(addresses, i, scratch)]
        launches.append(ck.launch(base, i))
        arg_list.append((ck.name, i, base))
    return JobPlan(Job(data, launches, params), addresses, dict(buffers), arg_list, scratch, end)


def kernel_job(ck: CompiledKernel, values: dict, *, addresses: dict | None = None) -> JobPlan:
    """A Job running one compiled kernel; `values` binds its global buffers by sym (missing ones are zero)."""
    return _job([ck], ck.buffers, values, ck.params, addresses, 1 << ck.params.DATA_MEM_ADDR_BITS)


def program_job(program, kernels: dict, values: dict, params: GpuParams | None = None, *,
                addresses: dict | None = None) -> JobPlan:
    """A Job running every kernel of a lowered anvil.lower Program in order. `kernels` maps kernel name ->
    CompiledKernel (each kernel of the program must be present); `values` binds inputs and consts by sym;
    `addresses` maps every buffer sym (graph values with storage and kernel temps) to an address, default a bump
    allocation in program order."""
    order = []
    for k in program.kernels:
        if k.name not in kernels:
            raise CodegenError(f"no compiled kernel for {k.name!r}")
        order.append(kernels[k.name])
    params = params or order[0].params
    buffers = {s: (list(sh), dt) for s, (sh, dt) in program.buffers.items()}
    for k in program.kernels:
        for s, (sh, dt) in k.temps.items():
            buffers[s] = (list(sh), dt)
    for ck in order:
        for s, v in ck.buffers.items():
            if s not in buffers:
                raise CodegenError(f"{ck.name}: buffer {s!r} is not in the program")
    return _job(order, buffers, values, params, addresses, 1 << params.DATA_MEM_ADDR_BITS)
