"""Search: random seeding, then evolution over sketch parameters, measured through a parallel pool and the cache.

    res = tune_kernel(task, budget=64, seed=0, workers=4)        # TuneResult: best candidate, trace, cycles, ...
    res.best, res.trace, res.compile_kw, res.cycles, res.default_cycles, res.measured, res.history

Per generation: candidates are built in the pool (schedule, canonical form, codegen, backend; illegal ones are
pruned and do not count against the budget), handed to the ranking hook, and the ones it keeps are measured
(isasim verdict, then Verilator cycles) unless the cache already holds them. Generation 0 is the sketches' seed
points (the driver's default and template schedules, so a tuned kernel is never slower than those) plus uniform
random samples; later generations mutate (one or two knobs re-drawn) and cross over (same sketch) the best
`population` candidates found so far, plus a few fresh random samples. The budget counts measured candidates
(cache hits included); the search is deterministic for a seed: candidates are drawn from a seeded RNG, results are
consumed in submission order and ties break on (cycles, program words, candidate key).

Ranking hook (SG7): `ranker(task, builts) -> builts to measure, in order`. Default `measure_all`. A ranker sees
`Built.features()` (static program features) and the candidate; `top_k(score, k)` builds one from a score.
"""
from __future__ import annotations

import multiprocessing as mp
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

from .measure import Built, Invalid, MeasureCache, Task, build, execute, record
from .space import SKETCHES, Candidate, sketches_for

MAX_WORKERS = 4


def measure_all(task, builts):
    return list(builts)


def top_k(score, k):
    """A ranker that measures the k built candidates with the lowest score(task, built)."""
    def rank(task, builts):
        return sorted(builts, key=lambda b: (score(task, b), b.cand.key))[:k]
    return rank


# ------------------------------------------------------------------------------------------------- pool

_TASK = None


def _w_build(cand):
    return build(_TASK, cand)


def _w_exec(args):
    job, timing, scratch = args
    return execute(job, _TASK.backend, timing=timing, scratch=scratch)


class Pool:
    """Up to 4 forked workers sharing one Task (inherited at fork); `workers <= 1` runs inline."""

    def __init__(self, task: Task, workers: int = MAX_WORKERS):
        self.task, self.workers = task, max(1, min(int(workers), MAX_WORKERS))
        self.ex = None

    def __enter__(self):
        global _TASK
        _TASK = self.task
        if self.workers > 1:
            self.ex = ProcessPoolExecutor(self.workers, mp_context=mp.get_context("fork"))
        return self

    def __exit__(self, *exc):
        global _TASK
        if self.ex is not None:
            self.ex.shutdown()
        _TASK = None

    def map(self, fn, items):
        items = list(items)
        if self.ex is None or len(items) <= 1:
            return [fn(x) for x in items]
        return list(self.ex.map(fn, items))

    def build(self, cands):
        return self.map(_w_build, cands)

    def execute(self, jobs_timing):
        return self.map(_w_exec, jobs_timing)


# ------------------------------------------------------------------------------------------------ search

@dataclass
class TuneResult:
    signature: str
    kernel: str
    op: str
    best: Candidate | None
    trace: str | None
    compile_kw: dict
    cycles: int | None
    words: int | None
    default_cycles: int | None             # the driver's default schedule (tpl seed) at the same params
    measured: int = 0                      # candidates measured (cache hits included)
    timed: int = 0                         # of which actually run on the timing backend
    cache_hits: int = 0
    invalid: int = 0                       # pruned before measurement
    failed: int = 0                        # built but not bit-exact / flagged on isasim (should be 0)
    seconds: float = 0.0
    seed: int = 0
    budget: int = 0
    history: list = field(default_factory=list)      # [(candidate key, cycles | None)] in measurement order

    def to_entry(self) -> dict:
        return {"kernel": self.kernel, "op": self.op, "candidate": self.best.to_dict() if self.best else None,
                "trace": self.trace, "compile_kw": self.compile_kw, "cycles": self.cycles, "words": self.words,
                "default_cycles": self.default_cycles, "measured": self.measured, "timed": self.timed,
                "invalid": self.invalid, "seconds": round(self.seconds, 2), "seed": self.seed, "budget": self.budget}


def _draw(rng, sk, cand_sketch):
    return Candidate.make(cand_sketch, **{k: rng.choice(v) for k, v in sk.domain.items()})


def _mutate(rng, sk, parent: Candidate):
    dom = sk.domain
    p = parent.p
    for k in rng.sample(sorted(dom), min(len(dom), rng.choice((1, 1, 2)))):
        choices = [c for c in dom[k] if c != p.get(k)] or dom[k]
        p[k] = rng.choice(choices)
    return Candidate.make(parent.sketch, **p)


def _cross(rng, a: Candidate, b: Candidate):
    pa, pb = a.p, b.p
    return Candidate.make(a.sketch, **{k: (pa if rng.random() < 0.5 else pb)[k] for k in pa})


def _rank_key(c, rec):
    return (rec["cycles"], rec["words"], c.key)


def tune_kernel(task: Task, *, budget: int = 64, seed: int = 0, workers: int = MAX_WORKERS, ranker=None,
                cache: MeasureCache | None = None, population: int = 8, generation: int = 16,
                log: bool = True, extra: list | None = None) -> TuneResult:
    """Tune one kernel. `extra`: candidates to evaluate in generation 0 (e.g. a previously best one)."""
    t0 = time.time()
    ranker = ranker or measure_all
    cache = cache if cache is not None else MeasureCache()
    sks = {sk.name: sk for sk in sketches_for(task.kernel, task.target, task.params)}
    rng = random.Random(f"anvil-tune/{seed}/{task.signature}")
    sig, dh = task.signature, task.data_hash
    seen, results = set(), {}               # candidate key -> record
    res = TuneResult(sig, task.kernel.name, task.kernel.op, None, None, {}, None, None, None, seed=seed,
                     budget=budget)
    attempts = 0
    names = sorted(sks)
    weights = [1 if n == "tpl" else 4 for n in names]

    def fresh():
        n = rng.choices(names, weights)[0]
        return _draw(rng, sks[n], n)

    def evaluate(pool, cands):
        nonlocal attempts
        cands = [c for c in cands if c.key not in seen]
        for c in cands:
            seen.add(c.key)
        attempts += len(cands)
        builts = pool.build(cands)
        valid = []
        for b in builts:
            if isinstance(b, Invalid):
                res.invalid += 1
            else:
                valid.append(b)
        chosen = ranker(task, valid)[:max(0, budget - res.measured)]
        todo, recs = {}, {}
        for b in chosen:
            hit = cache.get(b.key)
            if hit is not None and hit.get("verified_on") == dh:
                recs[b.key] = hit
            elif b.key not in todo:
                todo[b.key] = (b, hit)
        runs = pool.execute([(b.job, hit is None or not hit.get("ok"), b.plan.scratch) for b, hit in todo.values()])
        for (b, hit), raw in zip(todo.values(), runs):
            rec = record(task, b, raw)
            if hit is not None and hit.get("ok") and raw["cycles"] is None and rec["ok"]:
                rec.update(cycles=hit["cycles"], launch_cycles=hit.get("launch_cycles"),
                           counters=hit.get("counters", {}))
            else:
                res.timed += 1
            cache.put(b.key, rec)
            recs[b.key] = rec
            if log:
                cache.log({"time": time.time(), "signature": sig, "kernel": task.kernel.name, "op": task.kernel.op,
                           "params": task.params.key(), "mem_latency": task.mem_latency, "key": b.key,
                           "candidate": b.cand.to_dict(), "trace": b.trace, "compile_kw": b.compile_kw,
                           "features": b.features(), "ok": rec["ok"], "why": rec["why"], "cycles": rec["cycles"],
                           "launch_cycles": rec["launch_cycles"], "counters": rec["counters"],
                           "seconds": raw.get("seconds")})
        res.cache_hits += len(chosen) - len(todo)
        for b in chosen:
            rec = recs[b.key]
            res.measured += 1
            if not rec["ok"]:
                res.failed += 1
            results[b.cand.key] = (b, rec)
            res.history.append((b.cand.key, rec["cycles"]))

    def ranked():
        ok = [(b, r) for b, r in results.values() if r["ok"]]
        return sorted(ok, key=lambda br: _rank_key(br[0].cand, br[1]))

    with Pool(task, workers) as pool:
        gen0 = []
        for n in names:
            gen0 += sks[n].seeds()
        gen0 += list(extra or [])
        n0 = len(gen0) + generation
        while len(gen0) < n0:
            gen0.append(fresh())
        evaluate(pool, gen0)
        stall = 0
        while res.measured < budget and attempts < 50 * max(budget, 1) and stall < 20:
            top = ranked()[:population]
            gen, tries = [], 0
            want = min(generation, budget - res.measured + generation // 2)
            while len(gen) < want and tries < 40 * generation:
                tries += 1
                r = rng.random()
                if not top or r < 0.15:
                    c = fresh()
                else:
                    a = min(rng.sample(range(len(top)), min(2, len(top))))
                    pa = top[a][0].cand
                    mates = [b.cand for b, _ in top if b.cand.sketch == pa.sketch and b.cand != pa]
                    if mates and r < 0.4:
                        c = _cross(rng, pa, rng.choice(mates))
                        if rng.random() < 0.5:
                            c = _mutate(rng, sks[c.sketch], c)
                    else:
                        c = _mutate(rng, sks[pa.sketch], pa)
                if c.key not in seen and c not in gen:
                    gen.append(c)
            before = res.measured
            evaluate(pool, gen)
            stall = stall + 1 if res.measured == before else 0

    order = ranked()
    if order:
        b, rec = order[0]
        res.best, res.trace, res.compile_kw = b.cand, b.trace, dict(b.compile_kw)
        res.cycles, res.words = rec["cycles"], rec["words"]
    dflt = sks["tpl"].seeds()[0].key
    if dflt in results and results[dflt][1]["ok"]:
        res.default_cycles = results[dflt][1]["cycles"]
    res.seconds = time.time() - t0
    return res
