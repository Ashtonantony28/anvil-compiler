"""Contract C2: one `run()` over the ISA model, the iverilog RTL build and (if adopted) a Verilator build.

Memory persists across a Job's launches and the GPU resets between them, as in
`sim/test/helpers/runner.py::run_launches`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

# Parameter defaults of `module gpu` in sim/src/gpu.sv (tests/runtime checks this table against the source).
GPU_SV_DEFAULTS = {
    "WORD_BITS": 8, "DATA_MEM_ADDR_BITS": 8, "DATA_MEM_DATA_BITS": 8, "NUM_CORES": 2, "THREADS_PER_BLOCK": 4,
    "ICACHE_LINES": 0, "COALESCE": 0, "PIPELINED": 0, "DCACHE_LINES": 0, "MEDIC": 0, "WARPS_PER_CORE": 1,
    "WARP_POLICY": 0, "SIMT_STACK": 0,
}
WARP_POLICIES = {0: "lrr", 1: "gto", 2: "gtlrr"}
PROGRAM_WORDS = 256
DEFAULT_TIMEOUT_CYCLES = 2_000_000


@dataclass(frozen=True)
class GpuParams:
    """`module gpu` parameters. Field names are the RTL's; the defaults are the xf build (32-bit words, 4096 words)."""
    WORD_BITS: int = 32
    DATA_MEM_ADDR_BITS: int = 12
    DATA_MEM_DATA_BITS: int = 32
    NUM_CORES: int = 2
    THREADS_PER_BLOCK: int = 4
    ICACHE_LINES: int = 0
    COALESCE: int = 0
    PIPELINED: int = 0
    DCACHE_LINES: int = 0
    MEDIC: int = 0
    WARPS_PER_CORE: int = 1
    WARP_POLICY: int = 0
    SIMT_STACK: int = 0

    def __post_init__(self):
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"GpuParams.{f.name} = {v!r}: not a non-negative int")
        if self.WARP_POLICY not in WARP_POLICIES:
            raise ValueError(f"WARP_POLICY {self.WARP_POLICY} not in {sorted(WARP_POLICIES)}")
        if self.WARPS_PER_CORE < 1 or self.THREADS_PER_BLOCK % self.WARPS_PER_CORE:
            raise ValueError(f"{self.WARPS_PER_CORE} warps do not divide {self.THREADS_PER_BLOCK} threads per block")
        if not 1 <= self.WORD_BITS <= 32 or not 1 <= self.DATA_MEM_DATA_BITS <= 32:
            raise ValueError("WORD_BITS and DATA_MEM_DATA_BITS must be 1..32")
        if not 1 <= self.DATA_MEM_ADDR_BITS <= 16:
            raise ValueError("DATA_MEM_ADDR_BITS must be 1..16")

    def plusargs(self) -> list[str]:
        """iverilog switches `-Pgpu.NAME=V` for every field that differs from the gpu.sv default, so the list alone
        builds this configuration (GpuParams() gives the Makefile's XF_PARAMS)."""
        return [f"-Pgpu.{k}={v}" for k, v in self.as_dict().items() if v != GPU_SV_DEFAULTS[k]]

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)

    def key(self) -> str:
        """Stable text form, e.g. for cache keys."""
        return ",".join(f"{k}={v}" for k, v in self.as_dict().items())


@dataclass
class Launch:
    program: list[int]    # 16-bit instruction words, at most 256
    threads: int          # 1..255


@dataclass
class Job:
    data: list[int]
    launches: list[Launch]
    params: GpuParams = field(default_factory=GpuParams)
    mem_latency: int = 0

    def to_dict(self) -> dict:
        return {"data": [int(w) for w in self.data],
                "launches": [{"program": [int(w) for w in ln.program], "threads": int(ln.threads)}
                             for ln in self.launches],
                "params": self.params.as_dict(), "mem_latency": int(self.mem_latency)}

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        return cls(list(d["data"]), [Launch(list(ln["program"]), ln["threads"]) for ln in d["launches"]],
                   GpuParams(**d["params"]), d.get("mem_latency", 0))


@dataclass
class LaunchResult:
    cycles: int | None                       # None on isasim
    counters: dict[str, int] = field(default_factory=dict)


@dataclass
class RunResult:
    memory: list[int]                        # final data memory (2**DATA_MEM_ADDR_BITS words), unsigned
    launches: list[LaunchResult]
    flags: dict = field(default_factory=dict)   # isasim only: divergent, stores_conflict, store_races, race_words


def validate(job: Job) -> None:
    """Raise ValueError for a Job no backend can run; the backends rely on it."""
    p = job.params
    if not isinstance(p, GpuParams):
        raise ValueError(f"Job.params is {type(p).__name__}, not GpuParams")
    if len(job.data) > 1 << p.DATA_MEM_ADDR_BITS:
        raise ValueError(f"{len(job.data)} data words exceed 2**{p.DATA_MEM_ADDR_BITS}")
    if job.mem_latency < 0:
        raise ValueError(f"mem_latency {job.mem_latency} < 0")
    if not job.launches:
        raise ValueError("Job has no launches")
    from xf.kernels import check_block   # the simulator's own blockDim rule for kernels with max_block
    for i, ln in enumerate(job.launches):
        if not 1 <= ln.threads <= 255:
            raise ValueError(f"launch {i}: threads {ln.threads} not in 1..255")
        if not 1 <= len(ln.program) <= PROGRAM_WORDS:
            raise ValueError(f"launch {i}: program has {len(ln.program)} words, need 1..{PROGRAM_WORDS}")
        bad = [w for w in ln.program if not 0 <= int(w) <= 0xFFFF]
        if bad:
            raise ValueError(f"launch {i}: program words outside 16 bits: {bad[:4]}")
        check_block(getattr(ln.program, "max_block", None), p.THREADS_PER_BLOCK, f"#{i}")


def run(job: Job, backend: str = "isasim", *, timeout_cycles: int = DEFAULT_TIMEOUT_CYCLES) -> RunResult:
    """Run every launch of `job` on one data memory. backend: "isasim" | "rtl" | "verilator".

    timeout_cycles bounds each launch on the cycle-level backends (a hung kernel raises TimeoutError there)."""
    validate(job)
    if backend == "isasim":
        from .isasim_backend import run_isasim
        return run_isasim(job)
    if backend == "rtl":
        from .rtl import run_rtl
        return run_rtl(job, timeout_cycles)
    if backend == "verilator":
        from .verilator import run_verilator
        return run_verilator(job, timeout_cycles)
    raise ValueError(f"unknown backend {backend!r}: isasim, rtl or verilator")
