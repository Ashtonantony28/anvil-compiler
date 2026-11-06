"""anvil.schedule: schedule primitives as anvil.tensor rewrites, the replayable `Schedule` trace, and templates.

    from anvil.schedule import Schedule, Target, templates
    sch = Schedule(kernel, Target(threads_per_block=4, warps_per_core=1))
    sch.fuse("i", "j").split("i_j", 4).bind("i_j.o", "block").bind("i_j.i", "lane")
    kernel = sch.finalize()                     # canonical + validated: what SG3b compiles
    again = Schedule.replay(base_kernel, sch.to_json())

Primitives (legality-checked; a rejected one raises ScheduleError and changes nothing):
    split(loop, factor, tail="mask"|"pad")      ragged tails: mask = guard; pad = recompute iteration (i mod n),
                                                legal only for spatial loops that keep no state across iterations
    reorder(*loops)        permute a perfect nest of seq loops
    fuse(outer, inner)     perfect pair -> one loop, ivs = f floordiv n_inner, f mod n_inner
    tile(a, b, fa, fb)     split both, reorder to a.o, b.o, a.i, b.i
    unroll(loop, factor=extent)
    interchange(s, k)      register tiling: hoist reduction k above spatial s, accumulators become reg[s] (unrolled)
    bind(loop, block|warp|lane)   outer to inner; the loop must be the only statement at its level; spatial;
                           blocks x THREADS_PER_BLOCK <= 255, warps <= WARPS_PER_CORE, lanes <= lanes per warp
    replicate(warp|lane, n)       new thread loop around the current level's body, every thread runs it
    distribute(loop, tail)        spread a spatial loop under replicated threads over them (iv = o*T + tid);
                           inserts BAR when other threads read what it stores
    cross_lane_reduce(loop)       per-thread partial -> smem[tid] -> BAR -> every thread folds the T partials
                           in thread order -> BAR (add from 0, max/min from any constant)
    cache_read(sym, at, scope="block"|"warp")   stage what one iteration of `at` reads of a read-only global
                           buffer into smem: cooperative strided copy (guarded at buffer edges), BAR, body, BAR
                           when `at` repeats; budget 64 words per block, warp-scope buffers count x warps
    compute_at(stage, at)  move a top-level stage writing kernel temps into loop `at`, for the region one
                           iteration reads; temps become reg buffers of that region's shape
Order rules: compute_at before binding (multi-stage kernels cannot bind), binding (block, warp, lane) before the
smem primitives (cache_read, cross_lane_reduce).

Templates (anvil.schedule.templates): matmul_naive, matmul_tiled (mirror sim/kernels/matmul_{naive,tiled}.asm),
elementwise, rows_per_thread, row_per_block; default_schedule(kernel, target).

Canonical form SG3b may assume (`Schedule.finalize()` / `canonicalize` produce it and `validate` enforces it):

 1. Shape. func body = global allocs + exactly one `for bind=block` (B blocks). Its body = smem allocs (block
    shared) + exactly one `for bind=warp` (W warps). Its body = smem allocs (private per warp: lay out at
    warp * size) + exactly one `for bind=lane` (Lx lanes). Missing levels appear as extent-1 loops tagged
    _block / _warp / _lane. Everything else (seq loops, guards, scalar ops, loads/stores, reg allocs, barriers)
    is inside the lane loop: the per-thread program.
 2. Threads. blockIdx = block iv. threadIdx t has warp w = t // (THREADS_PER_BLOCK / WARPS_PER_CORE) and lane
    l = t mod that; a thread is active iff w < W and l < Lx. Inactive threads must not load or store, but must
    execute every barrier (or have finished: BAR waits only for unfinished warps). The launch is B *
    THREADS_PER_BLOCK <= 255 threads; W <= WARPS_PER_CORE; Lx <= THREADS_PER_BLOCK / WARPS_PER_CORE.
 3. Loops. Every loop has constant bounds, lb = 0, step = 1, a unique tag, and `unroll` dividing its extent.
    All trip counts are uniform across threads (no data-dependent control flow exists).
 4. Indices. Loads, stores, guards and `apply` use affine maps of loop ivs with constant coefficients,
    + - * by constants, floordiv / mod by positive constants. `gather` is the only data-dependent address.
 5. Memory. global buffers bind to image segments by sym (graph values, consts, and kernel temps listed in
    Kernel.temps that the memory planner must allocate). smem words per block (sum of block-level buffers +
    warp-level buffers x W) <= 64. reg buffers live in the lane loop, total <= Target.max_reg_words words,
    and every index of a reg access depends only on fully unrolled loops (unroll == extent), so each element
    is a virtual register.
 6. Barriers. Only inside the lane loop, never under a guard; possibly inside seq loops (uniform trip counts).
    Shared-memory and global accesses are race-free between barriers (the tensor interpreter checks it).
 7. Guards. `guard` regions hold no barrier; their condition is an affine bound (map(idx) < ub), i.e. a
    uniform-or-per-thread predicate for the codegen's predication strategy.
"""
from . import templates
from .canonical import canonicalize, validate
from .primitives import PRIMITIVES
from .schedule import Schedule
from .target import ScheduleError, Target

__all__ = ["Schedule", "Target", "ScheduleError", "PRIMITIVES", "canonicalize", "validate", "templates"]
