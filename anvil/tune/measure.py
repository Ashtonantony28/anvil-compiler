"""Candidate evaluation: build (schedule -> canonical kernel -> program -> Job), verify on isasim, time on Verilator,
with a persistent measurement cache and a JSONL measurement log.

    task = Task.from_program(prog, kernel_name, values, params)       # one kernel + its inputs + reference outputs
    b = build(task, cand)                                            # Built | Invalid (pruned before measuring)
    raw = execute(b.job, "verilator")                                # run isasim, then the timing backend
    rec = finish(task, b, raw)                                       # verdict: bit-exact + no flags, cycles

Cache (`MeasureCache`, default `.anvil-cache/measure/`): one JSON record per key, where
    key = sha256(program words, args layout (slots, buffer addresses, args-block bases and words), GpuParams.key(),
                 mem_latency, timing backend)
Cycles do not depend on data values (uniform trip counts, no data-dependent control flow), so a record holds the
cycles of the program at that layout plus the hash of the data it was verified on; a hit with other data is
re-verified on isasim (cheap) but not re-timed. Every new measurement is appended to `log.jsonl` with the
kernel, candidate, trace, program stats and counters: training data for SG7's cost models.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..backend.lir import BackendError
from ..codegen import CodegenError, compile_kernel, kernel_job
from ..runtime import GpuParams, run
from ..schedule import ScheduleError, Target
from .space import Candidate, build_schedule, compile_options

FLAGS = ("divergent", "stores_conflict", "store_races")


def default_cache_dir() -> Path:
    from ..runtime.rtl import CACHE
    return Path(os.environ.get("ANVIL_MEASURE_CACHE", CACHE / "measure"))


def canonical_ir(kernel) -> str:
    """The kernel's IR text with buffer and func names replaced by their order of appearance: the program a
    schedule produces does not depend on names (the frontends name weights differently)."""
    import re

    from ..ir import print_ir
    names = {}

    def sub(m):
        n = names.setdefault(m.group(2), f"s{len(names)}")
        return f'{m.group(1)}"{n}"'
    return re.sub(r'(sym(?:_name)? = )"([^"]*)"', sub, print_ir(kernel.module))


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class Task:
    """One kernel to tune: the unscheduled kernel, the machine, inputs for every buffer it reads and the
    reference contents of every buffer it writes (the tensor interpreter of the unscheduled kernel)."""
    kernel: object
    target: Target
    params: GpuParams
    values: dict
    ref: dict
    mem_latency: int = 0
    backend: str = "verilator"      # timing backend; "isasim" times by warp issues (a proxy for fast tests)

    @classmethod
    def from_program(cls, program, name, values: dict, params: GpuParams, target: Target | None = None, **kw):
        """`values`: every buffer of the program (e.g. Program.run's result)."""
        k = program.kernel(name)
        target = target or Target.from_params(params, max_threads=4096)
        ins = {s: np.asarray(values[s]) for s in k.reads if s in values}
        ref = {s: np.asarray(values[s]) for s in k.writes}
        return cls(k, target, params, ins, ref, **kw)

    @property
    def signature(self) -> str:
        """Identity of what is tuned: the kernel IR up to names, the schedule target, the GPU and the timing
        backend (anvil.tune.db key)."""
        ident = [canonical_ir(self.kernel), repr(self.target), self.params.key(), self.mem_latency]
        if self.backend != "verilator":         # verilator and rtl time identically (ADR-001)
            ident.append(self.backend)
        return _sha(ident)[:20]

    @property
    def ir_hash(self) -> str:
        """The kernel IR alone (the same kernel at other GPU params shares it)."""
        return _sha(canonical_ir(self.kernel))[:20]

    @property
    def data_hash(self) -> str:
        return _sha({s: np.asarray(v).astype(np.int64).ravel().tolist() for s, v in sorted(self.values.items())})[:16]


@dataclass
class Invalid:
    cand: Candidate
    stage: str                  # "schedule" | "codegen" | "backend"
    error: str


@dataclass
class Built:
    cand: Candidate
    trace: str                  # Schedule.to_json()
    compile_kw: dict
    words: int
    stats: dict
    threads: int
    launches: int
    smem_words: int
    key: str                    # measurement-cache key
    job: object = field(repr=False, default=None)
    plan: object = field(repr=False, default=None)   # codegen JobPlan (addresses, read())

    def features(self) -> dict:
        """Static features of the built program (for ranking hooks / cost models)."""
        return {"words": self.words, "threads": self.threads, "launches": self.launches,
                "smem_words": self.smem_words, **{k: v for k, v in self.stats.items() if isinstance(v, (int, float))}}


def build(task: Task, cand: Candidate):
    """Candidate -> Built, or Invalid when the schedule, codegen or backend rejects it (pruned, never measured)."""
    try:
        s = build_schedule(task.kernel, task.target, cand)
        fin = s.finalize()
    except ScheduleError as e:
        return Invalid(cand, "schedule", str(e)[:300])
    ckw = compile_options(cand)
    try:
        ck = compile_kernel(fin, task.params, **ckw)
    except CodegenError as e:
        return Invalid(cand, "codegen", str(e)[:300])
    except BackendError as e:                      # ProgramTooLarge (256 words) and other backend limits
        return Invalid(cand, "backend", str(e)[:300])
    try:
        jp = kernel_job(ck, {s_: task.values[s_] for s_ in ck.buffers if s_ in task.values})
    except CodegenError as e:
        return Invalid(cand, "codegen", str(e)[:300])
    job = jp.job
    job.mem_latency = task.mem_latency
    layout = {"slots": [list(map(str, x)) for x in ck.slots], "addresses": sorted(jp.addresses.items()),
              "args": [(n, i, b, job.data[b:b + len(ck.slots)]) for n, i, b in jp.args], "end": jp.end}
    key = _sha({"words": [ln.program for ln in job.launches], "threads": [ln.threads for ln in job.launches],
                "layout": layout, "params": task.params.key(), "mem_latency": task.mem_latency,
                "backend": task.backend})
    return Built(cand, s.to_json(), ckw, ck.program.stats["words"], dict(ck.program.stats),
                 sum(ck.threads(i) for i in range(ck.launches)), ck.launches, ck.code.smem_words, key, job, jp)


def execute(job, backend="verilator", *, timing=True, scratch=None) -> dict:
    """Run on isasim (results + race/divergence flags) and, when `timing`, on the timing backend. The two memories
    must agree below `scratch` (the sink/spill words hold don't-care values of masked-off lanes, which isasim and
    the RTL may compute differently)."""
    t0 = time.time()
    ri = run(job, "isasim")
    out = {"memory": ri.memory, "flags": {k: len(ri.flags.get(k, [])) for k in FLAGS}, "cycles": None,
           "counters": {}, "launch_cycles": None}
    if timing and not any(out["flags"].values()) and backend == "isasim":
        # proxy timing for fast tests: warp instruction issues, not cycles
        out["launch_cycles"] = [lr.counters["warp_issues_total"] for lr in ri.launches]
        out["cycles"] = sum(out["launch_cycles"])
    elif timing and not any(out["flags"].values()):
        rv = run(job, backend)
        out["cycles"] = sum(lr.cycles for lr in rv.launches)
        out["launch_cycles"] = [lr.cycles for lr in rv.launches]
        cs = {}
        for lr in rv.launches:
            for c, v in lr.counters.items():
                cs[c] = cs.get(c, 0) + v
        out["counters"] = cs
        n = len(job.data) if scratch is None else scratch
        out["timing_memory_equal"] = rv.memory[:n] == ri.memory[:n]
        out["scratch_differs"] = rv.memory[n:len(job.data)] != ri.memory[n:len(job.data)]
    out["seconds"] = time.time() - t0
    return out


def verdict(task: Task, b: Built, memory) -> tuple[bool, str]:
    for s, want in task.ref.items():
        got = b.plan.read(memory, s)
        if not np.array_equal(got.reshape(np.shape(want)), want):
            return False, f"{s} differs from the reference"
    return True, ""


class MeasureCache:
    """Persistent measurement records under `root` (one JSON per key) plus an append-only JSONL log."""

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root is not None else default_cache_dir()
        self.hits = self.misses = 0

    def _path(self, key) -> Path:
        return self.root / "db" / key[:2] / f"{key}.json"

    @property
    def log_path(self) -> Path:
        return self.root / "log.jsonl"

    def get(self, key):
        p = self._path(key)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                return None
        return None

    def put(self, key, rec: dict):
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(rec, sort_keys=True))
        tmp.replace(p)

    def log(self, entry: dict):
        self.root.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, sort_keys=True, default=str) + "\n"
        with open(self.log_path, "a") as f:             # one write per line (O_APPEND): safe across processes
            f.write(line)


def record(task: Task, b: Built, raw: dict) -> dict:
    """The verdict of one execution as a cache record."""
    ok, why = verdict(task, b, raw["memory"])
    flags = raw["flags"]
    if any(flags.values()):
        ok, why = False, f"isasim flags {flags}"
    if ok and raw.get("timing_memory_equal") is False:
        ok, why = False, "timing backend memory differs from isasim"
    return {"ok": ok, "why": why, "cycles": raw["cycles"] if ok else None, "launch_cycles": raw.get("launch_cycles"),
            "counters": raw.get("counters", {}), "flags": flags, "verified_on": task.data_hash,
            "words": b.words, "threads": b.threads,
            "scratch_differs": bool(raw.get("scratch_differs", False))}
