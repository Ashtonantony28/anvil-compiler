"""Named schedule templates: functions Schedule -> Schedule built only from primitives, so the trace replays.

    matmul_naive   one thread per output element, operands straight from global memory (matmul_naive.asm)
    matmul_tiled   4 x G output tile per G-thread group, 4 register accumulators, A staged through smem in G-wide
                   K chunks with BAR before and after each chunk (matmul_tiled.asm, G = 4)
    elementwise    flatten, one thread per element
    rows_per_thread  rmsnorm/softmax: stages computed at the row loop, one thread per row
    row_per_block    rmsnorm/softmax: one block per row, lanes split the columns, reductions across lanes in smem
    default_schedule(kernel, target)  the default per op kind
"""
from .schedule import Schedule
from .target import ScheduleError, Target


def _flat_threads(s: Schedule, loops, tail="mask"):
    """Fuse `loops` (a perfect nest, outer first) into one, then split into block x [warp x] lane."""
    tgt = s.target
    name = loops[0]
    for nxt in loops[1:]:
        s.fuse(name, nxt, name=f"{name}_{nxt}")
        name = f"{name}_{nxt}"
    tpb, le = tgt.threads_per_block, tgt.lanes_per_warp
    n = dict((x[0], x[1]) for x in s.loops())[name]
    if n <= tpb:
        s.split(name, n, tail, outer=f"{name}.blk", inner=f"{name}.thr")
    else:
        s.split(name, tpb, tail, outer=f"{name}.blk", inner=f"{name}.thr")
    s.bind(f"{name}.blk", "block")
    thr = f"{name}.thr"
    nthr = dict((x[0], x[1]) for x in s.loops())[thr]
    if nthr > le:
        s.split(thr, le, tail, outer=f"{name}.w", inner=f"{name}.l")
        s.bind(f"{name}.w", "warp")
        s.bind(f"{name}.l", "lane")
    else:
        s.bind(thr, "lane")
    return s


def matmul_naive(s: Schedule) -> Schedule:
    if s.kernel.op not in ("qmatmul", "qlinear"):
        raise ScheduleError("matmul_naive: not a matmul kernel")
    batch = [x[0] for x in s.loops() if x[0].startswith("b") and x[3] == "spatial"]
    return _flat_threads(s, batch + ["i", "j"])


def matmul_tiled(s: Schedule, tile: int = 4) -> Schedule:
    """C tile of `tile` rows x G columns per group of G = tile threads (one block, or warps x lanes of it)."""
    if s.kernel.op not in ("qmatmul", "qlinear"):
        raise ScheduleError("matmul_tiled: not a matmul kernel")
    tgt = s.target
    ext = {x[0]: x[1] for x in s.loops()}
    M, N, K = ext["i"], ext["j"], ext["k"]
    G = tile
    if N % G or M % tile:
        raise ScheduleError(f"matmul_tiled: needs M % {tile} == 0 and N % {G} == 0, got M={M} N={N}")
    if G > tgt.threads_per_block:
        raise ScheduleError(f"matmul_tiled: a {G}-thread group does not fit THREADS_PER_BLOCK={tgt.threads_per_block}")
    batch = [x[0] for x in s.loops() if x[0].startswith("b") and x[3] == "spatial"]
    s.split("i", tile).split("j", G).reorder("i.o", "j.o", "j.i", "i.i")
    blk = "i.o"
    for b in batch[::-1]:
        s.fuse(b, blk, name=f"{b}_{blk}")
        blk = f"{b}_{blk}"
    s.fuse(blk, "j.o", name="tile")
    s.bind("tile", "block")
    if G > tgt.lanes_per_warp:
        s.split("j.i", tgt.lanes_per_warp, outer="j.w", inner="j.l").bind("j.w", "warp").bind("j.l", "lane")
    else:
        s.bind("j.i", "lane")
    s.interchange("i.i", "k")
    s.split("k", G)
    s.cache_read(s.kernel.reads[0], "k.o")
    if K >= G and K % G == 0:
        s.unroll("k.i")
    return s


def elementwise(s: Schedule) -> Schedule:
    return _flat_threads(s, [x[0] for x in s.loops()])


def _row_stages(s: Schedule):
    op = s.kernel.op
    if op == "rmsnorm_int":
        return ["stat"]
    if op == "softmax_lut":
        return ["sum", "max"]      # sum reads max: compute sum into the row loop first
    raise ScheduleError(f"not a row-wise kernel: {op}")


def _compute_rows(s: Schedule):
    stages = _row_stages(s)
    batch = [x[0] for x in s.loops() if x[0].startswith("b") and "." not in x[0]]
    for st in stages:
        sb = [x[0] for x in s.loops() if x[0].startswith(f"{st}.b")]
        s.compute_at(sb[0] if sb else f"{st}.i", "i")
    return batch


def rows_per_thread(s: Schedule) -> Schedule:
    batch = _compute_rows(s)
    return _flat_threads(s, batch + ["i"])


def row_per_block(s: Schedule) -> Schedule:
    batch = _compute_rows(s)
    name = "i"
    for b in batch[::-1]:
        s.fuse(b, name, name=f"{b}_{name}")
        name = f"{b}_{name}"
    s.bind(name, "block")
    tgt = s.target
    if tgt.warps_per_core > 1:
        s.replicate("warp", tgt.warps_per_core)
    s.replicate("lane", tgt.lanes_per_warp)
    for tag, _, bind, kind, _ in list(s.loops()):
        if bind == "seq" and kind == "reduce" and not tag.endswith(".x"):
            s.cross_lane_reduce(tag)
    s.distribute("j", "pad")
    return s


def default_schedule(kernel, target: Target = Target()) -> Schedule:
    s = Schedule(kernel, target)
    op = kernel.op
    if op in ("qmatmul", "qlinear"):
        return matmul_naive(s)
    if op in ("rmsnorm_int", "softmax_lut"):
        return rows_per_thread(s)
    return elementwise(s)


TEMPLATES = {"matmul_naive": matmul_naive, "matmul_tiled": matmul_tiled, "elementwise": elementwise,
             "rows_per_thread": rows_per_thread, "row_per_block": row_per_block}
