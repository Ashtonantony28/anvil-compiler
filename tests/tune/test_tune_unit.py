"""SG6 fast tests (make check: no Verilog simulator). Timing uses the isasim proxy (warp issues), so these test the
tuner's machinery: sketch legality, pruning, cache keys and hits, the JSONL log, seeded reproducibility, the pool,
the ranking hook and the driver's schedules="tuned"."""
import json
import random

import numpy as np
import pytest

from anvil.driver import compile
from anvil.ir import print_ir
from anvil.runtime import GpuParams
from anvil.schedule import Schedule, ScheduleError
from anvil.tune import (BestDB, Built, Candidate, Invalid, MeasureCache, TunedPolicy, build, build_schedule,
                        sketches_for, top_k, tune_kernel)
from anvil.tune.measure import Task
from tune_util import program, task

KERNELS = [("mlp_block", "h"), ("mlp_block", "gate"), ("mlp_block", "act"), ("mlp_block", "down"),
           ("mlp_block", "out"), ("attention", "kT"), ("attention", "scores"), ("attention", "p"),
           ("attention", "o"), ("mm1x8x8", "y")]


@pytest.mark.parametrize("block,name", KERNELS)
def test_sketch_points_are_legal_or_rejected(block, name):
    """Random points of every sketch: either a ScheduleError, or a schedule that interprets bit-exactly (C4 level 2,
    races checked) before and after finalize and whose trace replays to the same IR."""
    prog, vals, _, _ = program(block)
    k = prog.kernel(name)
    t = task(block, name)
    rng = random.Random(f"{block}/{name}")
    ok = 0
    for sk in sketches_for(k, t.target, t.params):
        cands = sk.seeds() + [Candidate.make(sk.name, **{p: rng.choice(v) for p, v in sk.domain.items()})
                              for _ in range(12)]
        for c in cands:
            try:
                s = build_schedule(k, t.target, c)
            except ScheduleError:
                continue
            for w in k.writes:
                assert np.array_equal(s.run(vals)[w], vals[w]), c
            try:
                fin = s.finalize()
            except ScheduleError:
                continue
            for w in k.writes:
                assert np.array_equal(fin.run(vals)[w], vals[w]), c
            again = Schedule.replay(k, s.to_json(), t.target)
            assert print_ir(again.kernel.module) == print_ir(s.kernel.module)
            ok += 1
    assert ok >= 3


def test_seeds_are_the_driver_schedules():
    """Every template seed builds; the first is the driver's default schedule (same program)."""
    from anvil.codegen import compile_kernel
    from anvil.schedule import templates as T
    t = task("mlp_block", "gate")
    sk = sketches_for(t.kernel, t.target, t.params)[0]
    seeds = sk.seeds()
    assert [c.p["template"] for c in seeds] == ["matmul_naive", "matmul_tiled"]
    b = build(t, seeds[0])
    want = compile_kernel(T.default_schedule(t.kernel, t.target).finalize(), t.params)
    assert b.job.launches[0].program == want.launch(b.plan.args[0][2], 0).program


def test_pruning_before_measurement():
    t = task("mlp_block", "gate")
    too_many_regs = Candidate.make("mm", ti=4, tj=4, order="ij", lanes=4, ku=1, stage="none", kc=1, U=1, ca=3)
    r = build(t, too_many_regs)
    assert isinstance(r, Invalid) and r.stage == "schedule"
    huge = Candidate.make("mm", ti=4, tj=2, order="ij", lanes=4, ku=8, stage="none", kc=1, U=16, ca=None)
    r = build(t, huge)
    assert isinstance(r, Invalid) and r.stage == "backend" and "256" in r.error
    bad_tile = Candidate.make("tpl", template="matmul_tiled", U=1, ca=3)
    assert isinstance(build(task("mm1x8x8", "y"), bad_tile), Invalid)        # M=1 is not a multiple of 4


def test_cache_keys():
    t = task("mlp_block", "act")
    c = Candidate.make("ew", v=4, mode="strided", lanes=4, tail="mask", U=4, ca=3)
    a, b = build(t, c), build(t, c)
    assert isinstance(a, Built) and a.key == b.key
    assert build(t, c.with_(ca=None)).key != a.key                           # other program
    assert build(task("mlp_block", "act", GpuParams(DCACHE_LINES=16)), c).key != a.key
    assert build(task("mlp_block", "act", mem_latency=2), c).key != a.key
    assert build(task("mlp_block", "act", backend="isasim"), c).key != a.key
    # tail only matters for ragged splits: 64 elements, 4 per thread -> the same program
    assert build(t, c.with_(tail="pad")).key == a.key


def test_cache_hits_and_log(tmp_path):
    t = task("mlp_block", "out", backend="isasim")
    cache = MeasureCache(tmp_path)
    r1 = tune_kernel(t, budget=12, seed=1, workers=1, cache=cache)
    assert r1.measured == 12 and r1.timed > 0 and r1.failed == 0
    r2 = tune_kernel(t, budget=12, seed=1, workers=1, cache=MeasureCache(tmp_path))
    assert r2.timed == 0 and r2.cache_hits == r2.measured == 12
    assert r2.history == r1.history and r2.best == r1.best
    lines = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert len(lines) == r1.timed
    e = lines[0]
    for f in ("signature", "kernel", "op", "params", "key", "candidate", "trace", "compile_kw", "features", "ok",
              "cycles", "counters"):
        assert f in e
    assert e["features"]["words"] > 0


def test_reproducible_with_seed(tmp_path):
    t = task("mlp_block", "gate", backend="isasim")
    a = tune_kernel(t, budget=20, seed=7, workers=4, cache=MeasureCache(tmp_path / "a"))
    b = tune_kernel(t, budget=20, seed=7, workers=1, cache=MeasureCache(tmp_path / "b"))
    assert a.best == b.best and a.history == b.history and a.cycles == b.cycles
    assert a.cycles <= a.default_cycles
    c = tune_kernel(t, budget=20, seed=8, workers=4, cache=MeasureCache(tmp_path / "a"))
    assert c.history != a.history


def test_ranking_hook(tmp_path):
    t = task("mlp_block", "gate", backend="isasim")
    seen = []

    def by_words(task_, b):
        seen.append(b.features())
        return b.words

    r = tune_kernel(t, budget=6, seed=0, workers=1, ranker=top_k(by_words, 2), cache=MeasureCache(tmp_path))
    assert r.measured <= 6 and seen and all("words" in f and "threads" in f for f in seen)
    assert all(c is not None for _, c in r.history)
    none = tune_kernel(t, budget=6, seed=0, workers=1, ranker=lambda task_, bs: [], cache=MeasureCache(tmp_path))
    assert none.measured == 0 and none.best is None


def test_driver_tuned_search_then_replay(tmp_path):
    """schedules="tuned": search on a miss (stored), replay on a hit; golden-equal on isasim."""
    _, _, model, d = program("mm1x8x8")
    db = tmp_path / "best.json"
    kw = dict(budget=10, seed=0, workers=2, backend="isasim", cache=MeasureCache(tmp_path / "m"))
    cm = compile(model, schedules="tuned", tune={**kw, "db": BestDB(write=db, committed=False)})
    assert cm.tuning["y"]["how"] == "searched" and cm.schedules["y"][0] == "tuned"
    r = cm.run(d["inputs"]["x"], "isasim", quantized=True)
    assert np.array_equal(r.ints["y"], d["golden"]["y"])
    cm2 = compile(model, schedules="tuned", tune={**kw, "search": False, "db": BestDB(write=db, committed=False)})
    assert cm2.tuning["y"]["how"] == "db"
    assert cm2.kernels["y"].program.words == cm.kernels["y"].program.words
    cm3 = compile(model, schedules="tuned", tune={**kw, "search": False,
                                                  "db": BestDB(write=tmp_path / "empty.json", committed=False)})
    assert cm3.tuning["y"]["how"] == "default"


@pytest.mark.parametrize("block", ["mlp_block", "attention"])
def test_committed_best_schedules_replay(block):
    """The committed best schedules (anvil/tune/best/xf-bench.json) cover every kernel of both fixtures at the
    default params, replay and run golden-equal on isasim with no race/divergence flags."""
    _, _, model, d = program(block)
    cm = compile(model, schedules="tuned", tune={"search": False, "db": BestDB(write="/nonexistent/x.json")})
    assert {v["how"] for v in cm.tuning.values()} == {"db"}, cm.tuning
    r = cm.run(d["inputs"]["x"], "isasim", quantized=True, keep="all")
    for seg, want in d["golden"].items():
        assert np.array_equal(r.ints[seg], want), seg
    assert not any(r.result.flags[f] for f in ("divergent", "stores_conflict", "store_races"))


def test_signature_ignores_names_not_constants():
    """Signatures are the kernel IR up to buffer/func names (the torch and image frontends name weights
    differently); requant constants still count (gate != up); other params share ir_hash, not signature."""
    import re
    from anvil.ir import parse_ir
    from anvil.lower.graph_to_tensor import Kernel
    t = task("mlp_block", "gate")
    k = t.kernel
    renamed = Kernel(k.name, k.op, parse_ir(re.sub(r'"w_gate"', '"gate.w"', print_ir(k.module))), k.reads, k.writes,
                     k.temps, k.kinds)
    t2 = Task(renamed, t.target, t.params, t.values, t.ref)
    assert t2.signature == t.signature
    assert task("mlp_block", "up").signature != t.signature
    other = task("mlp_block", "gate", GpuParams(DCACHE_LINES=16))
    assert other.signature != t.signature and other.ir_hash == t.ir_hash
