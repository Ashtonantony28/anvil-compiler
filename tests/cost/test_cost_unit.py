"""SG7 fast tests (make check, no Verilog simulator): the profiler against isasim, tier-1 timing rules on tiny
programs (expected cycles derived from the RTL FSMs; tests/cost/test_cost_rtl.py checks them against Verilator),
deterministic features, the tier-2 model's persistence, the frozen split, and the ranking/observe hooks of
tune_kernel."""
import json
import random
import tempfile

import numpy as np
import pytest

from anvil.cost import FEATURES, ActiveRanker, closed_form, featurize, predict, profile
from anvil.cost.dataset import CONFIGS, DATA, HELDOUT_CONFIGS, HELDOUT_KERNELS, classes, config_params
from anvil.runtime import GpuParams, run
from anvil.tune import MeasureCache, tune_kernel
from cost_util import asm_job, builts, straight_line

KERNELS = ["xf_mlp.h", "xf_mlp.gate", "xf_mlp.act", "xf_attention.kT", "xf_attention.scores", "xf_attention.p",
           "syn.add_5x8", "syn.softmax_2x4x4"]


@pytest.mark.parametrize("kid", KERNELS)
@pytest.mark.parametrize("W", [1, 2, 4])
def test_profile_matches_isasim(kid, W):
    """The profiler's final memory and warp-instruction count equal xf.isasim's on tuner candidates."""
    params = GpuParams(WARPS_PER_CORE=W)
    _, bs = builts(kid, params, n=2)
    for b in bs:
        prof = profile(b.job)
        ri = run(b.job, "isasim")
        n = b.plan.scratch
        assert prof.memory[:n] == list(ri.memory[:n])
        assert sum(lp.issues for lp in prof.launches) == sum(lr.counters["warp_issues_total"] for lr in ri.launches)


# Expected launch cycles of straight-line programs, from the FSM state durations (module docstring of
# anvil.cost.analytic): FETCH 6+L, DECODE 1, REQUEST 1, WAIT (1 | SMEM 3 | global 6+L), EXECUTE 1, UPDATE 1;
# the launch adds 2 (first FETCH) + 3 (tail). One block, one core, WARPS_PER_CORE=1.
PER_OP = {"ADD R1, R1, R1": 11, "CONST R1, #7": 11, "CMP R1, R2": 11, "BAR": 11, "LDS R1, R2": 13,
          "STS R2, R1": 13, "LDR R1, %threadIdx": 16, "STR %threadIdx, R1": 16}


@pytest.mark.parametrize("op", sorted(PER_OP))
@pytest.mark.parametrize("L", [0, 3])
@pytest.mark.parametrize("n", [0, 1, 5])
def test_tier1_straight_line_rules(op, L, n):
    lat = 2 * L if op.startswith(("LDR", "STR")) else L
    want = 5 + (n * (PER_OP[op] + lat) + 11 + L)
    for p in (GpuParams(), GpuParams(NUM_CORES=1), GpuParams(COALESCE=1)):
        assert predict(asm_job((op + "\n") * n + "RET\n", 4, p, L)).cycles == want


def test_tier1_rules_pipelined_caches_warps():
    """PIPELINED drops DECODE; an L1-D hit waits 4, a miss or a write-through 8+L; WARPS_PER_CORE>1 adds the IDLE
    issue slot and re-fetches a yielding load; blocks on one core are 3 cycles apart; contention on the program
    channel (one channel, 5+L cycles per fetch) delays the second core's first fetch."""
    assert predict(asm_job("ADD R1, R1, R1\n" * 4 + "RET\n", 4, GpuParams(PIPELINED=1))).cycles == 5 + 5 * 10
    d = GpuParams(DCACHE_LINES=16)
    assert predict(asm_job("LDR R1, R2\n" * 3 + "RET\n", 4, d)).cycles == 5 + 18 + 2 * 14 + 11
    assert predict(asm_job("STR R2, R1\n" * 3 + "RET\n", 4, d, 3)).cycles == 5 + 3 * 24 + 14
    w2 = GpuParams(WARPS_PER_CORE=2)
    assert predict(asm_job("ADD R1, R1, R1\nRET\n", 1, w2)).cycles == 5 + 12 + 12
    assert predict(asm_job("LDR R1, R2\nRET\n", 1, w2, 3)).cycles == 5 + 35 + 15
    assert predict(asm_job("RET\n", 12, GpuParams(NUM_CORES=1))).cycles == 5 + 3 * 11 + 2 * 3
    assert predict(asm_job("RET\n", 8, GpuParams(NUM_CORES=2))).cycles == 16 + 5
    assert predict(asm_job("RET\n", 8, GpuParams(NUM_CORES=2), 3)).cycles == 19 + 8


def test_tier1_closed_form_is_exact_without_contention():
    """One block, one warp, no shared-resource conflicts: the closed form and the event model agree."""
    rng = random.Random(3)
    for i in range(20):
        src = straight_line(rng, rng.randrange(1, 30))
        for p in (GpuParams(), GpuParams(PIPELINED=1), GpuParams(ICACHE_LINES=16)):
            job = asm_job(src, 4, p, i % 4)
            assert closed_form(profile(job), p, i % 4) == predict(job).cycles


def test_features_deterministic():
    for kid in ("xf_mlp.gate", "syn.softmax_2x4x4"):
        for p in (GpuParams(), GpuParams(WARPS_PER_CORE=4, WARP_POLICY=1, DCACHE_LINES=16)):
            _, bs = builts(kid, p, n=2)
            for b in bs:
                x1, t1 = featurize(b.job, b.features())
                x2, t2 = featurize(b.job, b.features())
                assert x1.shape == (len(FEATURES),) and np.isfinite(x1).all()
                assert np.array_equal(x1, x2) and t1.cycles == t2.cycles


def test_split_is_frozen():
    """The committed split equals the module's constants, and held-out configs/kernels never reach training."""
    split = json.loads((DATA / "split.json").read_text())
    assert split == {"heldout_kernels": list(HELDOUT_KERNELS), "heldout_configs": list(HELDOUT_CONFIGS)}
    cfg = json.loads((DATA / "configs.json").read_text())
    assert set(cfg) == set(CONFIGS)
    for cid, c in cfg.items():
        assert c["params"] == CONFIGS[cid][0] and c["mem_latency"] == CONFIGS[cid][1]
        assert (c["split"] == "test") == (cid in HELDOUT_CONFIGS)
    assert all(len(v) >= 3 for v in classes().values())
    kern = json.loads((DATA / "kernels.json").read_text())
    assert {k for k, v in kern.items() if v["split"] == "test"} == set(HELDOUT_KERNELS)
    total = sum(p.stat().st_size for p in DATA.iterdir())
    assert total < 25e6


def test_dataset_features_reproduce():
    """Rebuilding a dataset program from its candidate and featurizing it at a row's config gives the row's features
    (the tuner's ranker sees exactly what the model was trained on)."""
    pytest.importorskip("lightgbm")
    from anvil.cost.dataset import load
    from anvil.cost.learned import row_features
    from anvil.tune import Candidate, build
    from cost_util import task
    ds = load()
    rows = [r for r in ds.rows if r["pid"].startswith(("xf_mlp.gate|w1|", "syn.add_5x8|w2|")) and r["cycles"]]
    assert rows
    for r in rows[::7][:6]:
        prog = ds.programs[r["pid"]]
        params, lat = config_params(r["cid"])
        t = task(prog["kid"], GpuParams(WARPS_PER_CORE=prog["W"]))
        b = build(t, Candidate.from_dict(prog["candidate"]))
        b.job.params, b.job.mem_latency = params, lat
        x, t1 = featurize(b.job, b.features())
        assert t1.cycles == r["t1"]
        assert np.allclose(x, row_features(ds, r))


def test_cost_model_roundtrip_and_determinism(tmp_path):
    pytest.importorskip("lightgbm")
    from anvil.cost.learned import CostModel
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 100, (300, len(FEATURES)))
    X[:, FEATURES.index("t1_cycles")] = rng.uniform(100, 1000, 300)
    X[:, FEATURES.index("t1_closed")] = X[:, FEATURES.index("t1_cycles")]
    y = np.log(X[:, FEATURES.index("t1_cycles")] * (1 + 0.01 * X[:, 0]))
    a = CostModel("tier2", seed=1, rounds=30).fit(X, y)
    b = CostModel("tier2", seed=1, rounds=30).fit(X, y)
    assert np.array_equal(a.predict(X), b.predict(X))
    a.save(tmp_path / "m.txt.gz")
    c = CostModel.load(tmp_path / "m.txt.gz")
    assert c.variant == "tier2" and np.allclose(c.predict(X), a.predict(X))
    u = a.updated(X[:4], y[:4] + 0.5)
    assert np.mean(u.predict_log(X[:4]) - a.predict_log(X[:4])) > 0.1


class Spy:
    """A ranker that keeps the 2 smallest programs and records the observe hook's calls."""

    def __init__(self):
        self.calls, self.seen = [], []

    def __call__(self, task, builts):
        self.calls.append(len(builts))
        return sorted(builts, key=lambda b: (b.words, b.cand.key))[:2]

    def observe(self, task, b, rec):
        self.seen.append((b.key, rec["ok"], rec["cycles"]))


def test_ranker_and_observe_hooks_in_tune_kernel():
    from cost_util import task
    t = task("xf_mlp.out", backend="isasim")
    spy = Spy()
    with tempfile.TemporaryDirectory() as d:
        res = tune_kernel(t, budget=6, seed=0, workers=1, ranker=spy, cache=MeasureCache(d), log=False)
    assert res.measured == len(spy.seen) <= 6 and res.measured >= 4
    assert all(ok and c is not None for _, ok, c in spy.seen)
    assert [c for _, c in res.history] == [c for _, _, c in spy.seen]


@pytest.mark.parametrize("model", ["tier1", "tier2"])
def test_active_ranker_plugs_into_tune_kernel(model):
    """ActiveRanker measures only its top-k per round and learns from what tune_kernel reports."""
    from cost_util import task
    if model == "tier2":
        pytest.importorskip("lightgbm")
        if not (DATA / "tier2-seed0.txt.gz").exists():
            pytest.skip("no committed tier-2 model")
        rk = ActiveRanker(k=2)
    else:
        rk = ActiveRanker(model="tier1", k=2)
    t = task("syn.add_5x8", backend="isasim")
    with tempfile.TemporaryDirectory() as d:
        res = tune_kernel(t, budget=6, seed=0, workers=1, ranker=rk, cache=MeasureCache(d), log=False)
    assert res.best is not None and res.measured <= 6
    assert rk.calls >= 3                       # two per round
    if model == "tier2":
        assert len(rk.y) == res.measured and rk.cur is not rk.model


def test_active_ranker_ranks_by_tier1():
    """With the exact tier-1 model, the ranker's first pick is the fastest of the generation (isasim-free check:
    the prediction is the event model)."""
    from cost_util import task
    p = GpuParams()
    _, bs = builts("xf_mlp.act", p, n=6)
    t = task("xf_mlp.act", p)
    rk = ActiveRanker(model="tier1", k=6)
    order = rk(t, bs)
    pred = [predict(b.job).cycles for b in order]
    assert pred == sorted(pred)
