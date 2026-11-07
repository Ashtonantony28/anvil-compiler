"""Search spaces: per-op sketches over the schedule primitives.

A *sketch* is a parameterized recipe that builds a `Schedule` from primitives only (so its trace replays) plus
the codegen knobs a candidate compiles with. A *candidate* is (sketch name, parameter assignment); it is a pure
value (hashable, JSON-serializable, ordered), so a search over candidates is reproducible.

    sketches_for(kernel, target, params)            -> [Sketch] applicable to the kernel
    sk.domain                                       -> {param: [choices]} (every choice is a legal *value*;
                                                       a combination may still be rejected when built)
    build_schedule(kernel, target, cand)            -> Schedule (raises ScheduleError when illegal)
    compile_options(cand)                           -> kwargs for anvil.codegen.compile_kernel

Sketches (every one also takes the shared knobs `U` and `ca`):
    tpl     a named template of anvil.schedule.templates (the driver's presets are all reachable here)
    mm      matmuls: ti x tj register tile per thread (interchange; ti*tj <= Target.max_reg_words), thread order
            (row- or column-major over output tiles), lanes per block, k unroll, optional smem staging of A or B
            per k-chunk of kc (cache_read at the chunk loop)
    ew      all-spatial kernels (elementwise, lut, add_requant, transpose): v elements per thread, consecutive
            ("inner") or strided across the grid ("strided"), lanes per block, ragged tails masked or padded
    rows    rmsnorm/softmax: stats computed at the row loop; one thread per r rows ("thread"), or one block per row
            with n lanes splitting the columns through cross-lane reductions ("block")
Shared knobs: U = unroll every innermost sequential loop by its largest divisor <= U (1: no unrolling);
ca = codegen `const_args` (None: every constant built inline; c: constants that take >= c instructions come
from the args block). With SIMT_STACK=1 a `pred` knob picks the predication strategy.

Pruning before measurement (the measurer reports these as invalid, not as measurements): primitive legality
(ScheduleError), the canonical form, the smem budget, Target.max_reg_words, codegen limits (CodegenError) and
the 256-word program memory (ProgramTooLarge).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..ir import tensor as t
from ..schedule import Schedule, ScheduleError, Target
from ..schedule import ir_util as u
from ..schedule import templates as T

MATMUL = ("qmatmul", "qlinear")
ROWS = ("rmsnorm_int", "softmax_lut")
CA = (None, 2, 3, 4)
UNROLL = (1, 2, 4, 8, 16)


@dataclass(frozen=True, order=True)
class Candidate:
    sketch: str
    params: tuple                   # sorted ((name, value), ...)

    @classmethod
    def make(cls, sketch, **params):
        return cls(sketch, tuple(sorted(params.items())))

    @property
    def p(self) -> dict:
        return dict(self.params)

    @property
    def key(self) -> str:
        return json.dumps([self.sketch, self.params], sort_keys=True, default=str)

    def to_dict(self) -> dict:
        return {"sketch": self.sketch, "params": self.p}

    @classmethod
    def from_dict(cls, d) -> "Candidate":
        return cls.make(d["sketch"], **d["params"])

    def with_(self, **kw) -> "Candidate":
        return Candidate.make(self.sketch, **{**self.p, **kw})


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _ext(s: Schedule) -> dict:
    return {x[0]: x[1] for x in s.loops()}


def _batch(s: Schedule, exclude=()):
    return [x[0] for x in s.loops() if x[0].startswith("b") and "." not in x[0] and x[3] == "spatial"
            and x[0] not in exclude]


# ------------------------------------------------------------------------------------------ shared knobs

def unroll_inner(s: Schedule, U: int) -> Schedule:
    """Unroll every innermost seq loop (no loop inside it) that is not unrolled yet by its largest divisor <= U."""
    if U <= 1:
        return s
    for L in u.loops(s.kernel.func):
        if L.bind.data != "seq" or int(L.unroll.value.data) != 1:
            continue
        if any(isinstance(o, t.ForOp) and o is not L for o in L.walk()):
            continue
        n = u.extent(L)
        f = max(d for d in _divisors(n) if d <= U)
        if f > 1:
            s.unroll(u.tag_of(L), f)
    return s


def _threads(s: Schedule, name: str, lanes: int, tail="mask"):
    """Split loop `name` into block x [warp x] lane with `lanes` threads per block and bind it."""
    tgt = s.target
    n = _ext(s)[name]
    L = max(1, min(lanes, n, tgt.threads_per_block))
    s.split(name, L, tail, outer=f"{name}.blk", inner=f"{name}.thr")
    s.bind(f"{name}.blk", "block")
    if L > tgt.lanes_per_warp:
        s.split(f"{name}.thr", tgt.lanes_per_warp, tail, outer=f"{name}.w", inner=f"{name}.l")
        s.bind(f"{name}.w", "warp").bind(f"{name}.l", "lane")
    else:
        s.bind(f"{name}.thr", "lane")
    return s


def _fuse_all(s: Schedule, loops):
    name = loops[0]
    for nxt in loops[1:]:
        s.fuse(name, nxt, name=f"{name}_{nxt}")
        name = f"{name}_{nxt}"
    return name


# ----------------------------------------------------------------------------------------------- sketches

class Sketch:
    name = ""

    def __init__(self, kernel, target: Target, params=None):
        self.kernel, self.target, self.gp = kernel, target, params
        self.ext = _ext(Schedule(kernel, target))

    def shared(self) -> dict:
        d = {"U": list(UNROLL), "ca": list(CA)}
        if self.gp is not None and getattr(self.gp, "SIMT_STACK", 0):
            d["pred"] = ["auto", "mask"]
        return d

    @property
    def domain(self) -> dict:
        raise NotImplementedError

    def seeds(self) -> list:
        """Hand-picked starting points (the driver's existing choices) tried before random sampling."""
        return []

    def build(self, s: Schedule, p: dict) -> Schedule:
        raise NotImplementedError


class TemplateSketch(Sketch):
    name = "tpl"

    def templates(self):
        op = self.kernel.op
        if op in MATMUL:
            return ["matmul_naive", "matmul_tiled"]
        if op in ROWS:
            return ["rows_per_thread", "row_per_block"]
        return ["default"]

    @property
    def domain(self):
        return {"template": self.templates(), **self.shared()}

    def seeds(self):
        return [Candidate.make("tpl", template=n, U=1, ca=3, **({"pred": "auto"} if "pred" in self.shared() else {}))
                for n in self.templates()]

    def build(self, s, p):
        name = p["template"]
        if name == "default":
            return T.default_schedule(s.base, s.target)
        return T.TEMPLATES[name](s)


class MatmulSketch(Sketch):
    name = "mm"

    @property
    def domain(self):
        M, N, K = self.ext["i"], self.ext["j"], self.ext["k"]
        R = self.target.max_reg_words
        return {"ti": [d for d in _divisors(M) if d <= R], "tj": [d for d in _divisors(N) if d <= R],
                "order": ["ij", "ji"], "lanes": list(range(1, self.target.threads_per_block + 1)),
                "ku": _divisors(K), "stage": ["none", "a", "b"], "kc": _divisors(K), **self.shared()}

    def build(self, s, p):
        ti, tj, ku = p["ti"], p["tj"], p["ku"]
        if ti * tj > s.target.max_reg_words:
            raise ScheduleError(f"mm: {ti}x{tj} accumulators > {s.target.max_reg_words} register words")
        batch = _batch(s, ("i", "j", "k"))
        s.split("i", ti, outer="io", inner="ii").split("j", tj, outer="jo", inner="ji")
        a, b = ("io", "jo") if p["order"] == "ij" else ("jo", "io")
        s.reorder(a, b, "ii", "ji")
        s.fuse("ii", "ji", name="r")
        s.interchange("r", "k")
        thr = _fuse_all(s, batch + [a, b])
        _threads(s, thr, p["lanes"])
        if p["stage"] != "none":
            kc = p["kc"]
            if kc % ku:
                raise ScheduleError("mm: ku must divide kc when staging")
            s.split("k", kc, outer="k.o", inner="k.i")
            reads = s.kernel.reads
            s.cache_read(reads[0] if p["stage"] == "a" else reads[1], "k.o")
            if ku > 1:
                s.unroll("k.i", ku)
        elif ku > 1:
            s.unroll("k", ku)
        return s


class ElementwiseSketch(Sketch):
    name = "ew"

    @classmethod
    def applies(cls, kernel, target):
        s = Schedule(kernel, target)
        return all(x[3] == "spatial" for x in s.loops()) and not kernel.temps

    @property
    def domain(self):
        n = 1
        for x in self.ext.values():
            n *= x
        vs = sorted(set([d for d in _divisors(n) if d <= 32] + [3, 5, 6]))
        return {"v": vs, "mode": ["inner", "strided"], "lanes": list(range(1, self.target.threads_per_block + 1)),
                "tail": ["mask", "pad"], **self.shared()}

    def build(self, s, p):
        name = _fuse_all(s, [x[0] for x in s.loops()])
        n = _ext(s)[name]
        v, tail = p["v"], p["tail"]
        if v > n:
            raise ScheduleError("ew: more elements per thread than elements")
        cap = (s.target.max_threads // s.target.threads_per_block) * s.target.threads_per_block
        if -(-n // v) > cap:
            raise ScheduleError("ew: too many threads")
        if v > 1 and p["mode"] == "inner":
            s.split(name, v, tail, outer=f"{name}.par", inner=f"{name}.seq")
            name = f"{name}.par"
        elif v > 1:
            T_ = -(-n // v)
            s.split(name, T_, tail, outer=f"{name}.seq", inner=f"{name}.par")
            s.reorder(f"{name}.par", f"{name}.seq")
            name = f"{name}.par"
        _threads(s, name, p["lanes"], tail)
        return s


class RowsSketch(Sketch):
    name = "rows"

    @property
    def domain(self):
        rows = self.ext["i"]
        for b in _batch(Schedule(self.kernel, self.target), ("i", "j")):
            rows *= self.ext[b]
        return {"mode": ["thread", "block"], "r": _divisors(rows), "lanes": list(range(1, self.target.threads_per_block + 1)),
                "tail": ["mask", "pad"], **self.shared()}

    def _compute_rows(self, s):
        stages = {"rmsnorm_int": ["stat"], "softmax_lut": ["sum", "max"]}[s.kernel.op]
        batch = [x[0] for x in s.loops() if x[0].startswith("b") and "." not in x[0]]
        for st in stages:
            sb = [x[0] for x in s.loops() if x[0].startswith(f"{st}.b")]
            s.compute_at(sb[0] if sb else f"{st}.i", "i")
        return batch

    def build(self, s, p):
        batch = self._compute_rows(s)
        name = _fuse_all(s, batch + ["i"])
        tgt = s.target
        if p["mode"] == "thread":
            if p["r"] > 1:
                s.split(name, p["r"], outer=f"{name}.par", inner=f"{name}.seq")
                name = f"{name}.par"
            return _threads(s, name, p["lanes"])
        if p["r"] != 1:
            raise ScheduleError("rows: block mode is one row per block")
        n = p["lanes"]
        if n < 2:
            raise ScheduleError("rows: block mode needs >= 2 lanes")
        s.bind(name, "block")
        if n > tgt.lanes_per_warp:
            if n % tgt.lanes_per_warp:
                raise ScheduleError("rows: lanes must fill whole warps")
            s.replicate("warp", n // tgt.lanes_per_warp)
            s.replicate("lane", tgt.lanes_per_warp)
        else:
            s.replicate("lane", n)
        for tag, _, bind, kind, _ in list(s.loops()):
            if bind == "seq" and kind == "reduce" and not tag.endswith(".x"):
                s.cross_lane_reduce(tag)
        s.distribute("j", p["tail"])
        return s


def sketches_for(kernel, target: Target, params=None) -> list:
    out = [TemplateSketch(kernel, target, params)]
    if kernel.op in MATMUL:
        out.append(MatmulSketch(kernel, target, params))
    elif kernel.op in ROWS:
        out.append(RowsSketch(kernel, target, params))
    elif ElementwiseSketch.applies(kernel, target):
        out.append(ElementwiseSketch(kernel, target, params))
    return out


SKETCHES = {"tpl": TemplateSketch, "mm": MatmulSketch, "ew": ElementwiseSketch, "rows": RowsSketch}


def build_schedule(kernel, target: Target, cand: Candidate) -> Schedule:
    """The candidate's Schedule (primitives only: `.to_json()` replays it). Raises ScheduleError if illegal."""
    p = cand.p
    sk = SKETCHES[cand.sketch](kernel, target)
    s = sk.build(Schedule(kernel, target), p)
    return unroll_inner(s, p.get("U", 1))


def compile_options(cand: Candidate) -> dict:
    p = cand.p
    out = {"const_args": p.get("ca", 3)}
    if "pred" in p:
        out["predication"] = p["pred"]
    return out
