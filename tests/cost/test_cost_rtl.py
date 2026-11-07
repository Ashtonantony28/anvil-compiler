"""SG7 RTL tests: tier 1 against Verilator, the committed dataset against fresh Verilator runs, and the G4 numbers
recomputed from the committed dataset and models (python3 -m pytest -q -m rtl tests/cost)."""
import random

import numpy as np
import pytest

from anvil.cost import predict
from anvil.cost.dataset import CONFIGS, config_params
from anvil.runtime import GpuParams, run
from cost_util import asm_job, builts, straight_line

pytestmark = pytest.mark.rtl

PARAMS = {cid: config_params(cid) for cid in CONFIGS}


def _verilator(job):
    return sum(lr.cycles for lr in run(job, "verilator").launches)


@pytest.mark.parametrize("cid", sorted(CONFIGS))
def test_tier1_exact_on_straight_line_programs(cid):
    """Random straight-line programs (ALU, CONST, CMP, LDR/STR to per-lane and shared words, LDS/STS, BAR), 1-24
    threads: tier 1 equals Verilator's cycles exactly at every dataset configuration."""
    params, lat = PARAMS[cid]
    rng = random.Random(cid)
    for i in range(12):
        src = straight_line(rng, rng.randrange(1, 25))
        threads = rng.choice([1, 3, 4, 5, 8, 13, 16, 24])
        job = asm_job(src, threads, params, lat)
        assert predict(job).cycles == _verilator(job), (cid, threads, src)


@pytest.mark.parametrize("cid", ["c00", "c03", "c05", "c09", "c13", "c14", "h0", "h3"])
def test_tier1_exact_on_kernels(cid):
    """Tuner candidates of loop kernels (matmul, rows, elementwise): tier 1 equals Verilator."""
    params, lat = PARAMS[cid]
    for kid in ("xf_mlp.gate", "xf_attention.p", "syn.add_5x8", "syn.rmsnorm_2x16"):
        _, bs = builts(kid, GpuParams(WARPS_PER_CORE=params.WARPS_PER_CORE), n=2, seed=1)
        for b in bs:
            b.job.params, b.job.mem_latency = params, lat
            assert predict(b.job).cycles == _verilator(b.job), (cid, kid, b.cand)


@pytest.fixture(scope="module")
def ds():
    pytest.importorskip("lightgbm")
    from anvil.cost.dataset import load
    return load()


def test_dataset_rows_remeasure(ds):
    """A sample of committed rows rebuilt from their candidates and re-run on Verilator gives the recorded cycles."""
    from anvil.tune import Candidate, Task, build
    from cost_util import specs
    sp = specs()
    rows = [r for r in ds.ok_rows() if ds.programs[r["pid"]]["kid"] in sp]
    rng = random.Random(0)
    for r in rng.sample(rows, 12):
        prog = ds.programs[r["pid"]]
        s = sp[prog["kid"]]
        t = Task.from_program(s.program, s.name, s.values, GpuParams(WARPS_PER_CORE=prog["W"]))
        b = build(t, Candidate.from_dict(prog["candidate"]))
        b.job.params, b.job.mem_latency = PARAMS[r["cid"]]
        assert _verilator(b.job) == r["cycles"], r["pid"]


def test_g4_tau_heldout(ds):
    """G4: Kendall tau >= 0.8 between predicted and measured cycles on the frozen held-out split (held-out kernels x
    held-out configs), per config, pooled and within every kernel, for tier 1 and the committed tier-2 model."""
    from anvil.cost.g4_report import TAU_TARGET
    from anvil.cost.learned import CostModel, design, tau_table
    from anvil.cost.g4_report import model_path
    rows = ds.select("test", "test")
    assert len(rows) > 1000 and all(ds.is_test_config(r["cid"]) for r in rows)
    X, y, meta = design(ds, rows)
    yt = np.exp(y)
    for name, pred in (("tier1", np.array([r["t1"] for r in rows], float)),
                       ("tier2", CostModel.load(model_path("tier2", 0)).predict(X))):
        tab = tau_table(meta, yt, pred)
        assert set(tab) == set(ds.split["heldout_configs"])
        for cid, d in tab.items():
            assert d["pooled"] >= TAU_TARGET, (name, cid, d["pooled"])
            assert d["within_mean"] >= TAU_TARGET, (name, cid, d["within_mean"])


def test_g4_failures_reported(ds):
    """No dataset row or program is silently dropped: every failure carries a reason."""
    for r in ds.rows:
        assert (r["cycles"] is None) == bool(r["err"])
    for p in ds.programs.values():
        assert p["ok"] or p["why"]


def test_g4_guided_vs_random(ds):
    """G4: the tier-2-guided search (committed models, seeds 0-2) reaches within 5 % of the pool best with <= 10 % of
    the measurements random search needs, summed over the held-out pools."""
    from anvil.cost import active as A
    from anvil.cost.g4_report import RATIO_TARGET, SEEDS, model_path
    from anvil.cost.learned import CostModel, design
    pools = A.pools(ds, "test", "test")
    assert len(pools) >= 20
    models = [CostModel.load(model_path("tier2", s)) for s in SEEDS]
    rnd = guided = 0.0
    for rows in pools.values():
        X, _, _ = design(ds, rows)
        cyc = [r["cycles"] for r in rows]
        best = min(cyc)
        rnd += A.expected_random(cyc)
        for m in models:
            n = None
            for budget in (2, 5, len(cyc)):          # stop early: only the count to target matters here
                n = A.to_target(A.guided_curve(m, X, cyc, budget), best)
                if n is not None:
                    break
            guided += n / len(models)
    assert guided / rnd <= RATIO_TARGET, (guided, rnd)
