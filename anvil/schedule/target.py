"""Machine limits a schedule must respect (sim/src/gpu.sv parameters, docs/isa.md)."""
from dataclasses import dataclass


class ScheduleError(ValueError):
    """An illegal schedule primitive or a kernel outside the canonical form. The kernel is left unchanged."""


@dataclass(frozen=True)
class Target:
    threads_per_block: int = 4      # THREADS_PER_BLOCK = %blockDim
    warps_per_core: int = 1         # WARPS_PER_CORE: a block is this many warps of consecutive threads
    num_cores: int = 2              # NUM_CORES (blocks are dispatched across cores; no schedule limit)
    smem_words: int = 64            # core.SMEM_SIZE, per block, shared by its warps
    max_threads: int = 255          # blocks x threads_per_block per kernel; 255 = one launch (8-bit thread count).
                                    # Raise it to let anvil.codegen split the block loop over several launches.
    max_reg_words: int = 8          # register-buffer words per thread (13 GPRs minus addressing/temps)

    def __post_init__(self):
        if self.threads_per_block % self.warps_per_core:
            raise ValueError(f"{self.warps_per_core} warps do not divide {self.threads_per_block} threads")

    @property
    def lanes_per_warp(self) -> int:
        return self.threads_per_block // self.warps_per_core

    @classmethod
    def from_params(cls, params, **kw) -> "Target":
        """From an anvil.runtime.GpuParams (or anything with THREADS_PER_BLOCK, WARPS_PER_CORE, NUM_CORES)."""
        return cls(params.THREADS_PER_BLOCK, params.WARPS_PER_CORE, params.NUM_CORES, **kw)
