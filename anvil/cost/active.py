"""Model-guided tuning: a ranking hook for `anvil.tune.tune_kernel` with active learning, and the offline
tuning-curve experiment of the G4 report.

    from anvil.cost import ActiveRanker
    res = tune_kernel(task, budget=16, ranker=ActiveRanker(k=2))     # measure only the predicted top-k per round

ActiveRanker(task, builts): featurizes every built candidate at the task's GpuParams and mem_latency (profile ->
tier-1 event model + closed form -> features), predicts its cycles with the tier-2 model (default: the committed
data/cost/tier2-seed0.txt.gz; `model="tier1"` ranks by the tier-1 event model alone) and returns the k best.
`observe(task, built, record)` (called by tune_kernel after each measurement) adds the measured cycles; once two
points of the task are measured the model is boosted a few small trees on them (CostModel.updated) before the next
round. Features are cached per candidate program.

Offline experiment (`simulate`): a pool = every measured candidate of one held-out kernel at one held-out config.
Random search draws uniformly without replacement; the model-guided search measures the model's top-1 each round
and updates as above. Both stop counting at the first measurement within 5 % of the pool's best.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

from .features import FEATURES, featurize

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "data" / "cost"
TARGET = 1.05


def default_model(seed: int = 0, variant: str = "tier2"):
    from .learned import CostModel
    return CostModel.load(MODELS / f"{variant}-seed{seed}.txt.gz")


class ActiveRanker:
    def __init__(self, model=None, k: int = 2, update: bool = True, update_rounds: int = 10):
        if model is None:
            model = default_model()
        self.model, self.k, self.update, self.update_rounds = model, k, update, update_rounds
        self.cur = model
        self.feats = {}               # built key -> feature vector
        self.X, self.y = [], []
        self.calls = 0

    def _features(self, task, b):
        if b.key not in self.feats:
            job = b.job
            job.params, job.mem_latency = task.params, task.mem_latency
            self.feats[b.key] = featurize(job, b.features())[0]
        return self.feats[b.key]

    def predict(self, task, builts) -> np.ndarray:
        X = np.stack([self._features(task, b) for b in builts])
        if self.cur == "tier1":
            return X[:, FEATURES.index("t1_cycles")]
        return self.cur.predict(X)

    def __call__(self, task, builts):
        self.calls += 1
        if not builts:
            return []
        p = self.predict(task, builts)
        order = sorted(range(len(builts)), key=lambda i: (p[i], builts[i].cand.key))
        return [builts[i] for i in order[:self.k]]

    def observe(self, task, built, rec):
        if not rec.get("ok") or rec.get("cycles") is None or self.cur == "tier1":
            return
        self.X.append(self._features(task, built))
        self.y.append(math.log(rec["cycles"]))
        if self.update and len(self.y) >= 2:
            self.cur = self.model.updated(np.stack(self.X), np.array(self.y), self.update_rounds)


# ------------------------------------------------------------------------------------------ offline curves

def pools(ds, kernels="test", configs="test", min_size=8):
    """{(kid, cid): [row, ...]} of measured candidates."""
    out = {}
    for r in ds.select(kernels, configs):
        out.setdefault((ds.programs[r["pid"]]["kid"], r["cid"]), []).append(r)
    return {k: sorted(v, key=lambda r: r["pid"]) for k, v in sorted(out.items()) if len(v) >= min_size}


def random_curve(cycles, seed):
    rng = random.Random(seed)
    order = list(range(len(cycles)))
    rng.shuffle(order)
    return [cycles[i] for i in order]


def expected_random(cycles, target=TARGET) -> float:
    """E[draws without replacement until one within target x best] = (N + 1) / (m + 1)."""
    best = min(cycles)
    m = sum(c <= best * target for c in cycles)
    return (len(cycles) + 1) / (m + 1)


def guided_curve(model, X, cycles, budget=None, update=True, rounds=10, tier1=False):
    """Measured cycles in the order a model-guided search measures them (top-1 per round, updated online)."""
    n = len(cycles)
    budget = n if budget is None else min(budget, n)
    left = list(range(n))
    cur = model
    seen, ys, out = [], [], []
    for _ in range(budget):
        if tier1:
            p = X[left, FEATURES.index("t1_cycles")]
        else:
            p = cur.predict(X[left])
        j = left[int(np.argmin(p))]
        left.remove(j)
        out.append(cycles[j])
        seen.append(j)
        ys.append(math.log(cycles[j]))
        if update and not tier1 and len(ys) >= 2:
            cur = model.updated(X[seen], np.array(ys), rounds)
    return out


def to_target(curve, best, target=TARGET):
    for i, c in enumerate(curve):
        if c <= best * target:
            return i + 1
    return None


def best_so_far(curve, best, upto):
    out, b = [], math.inf
    for i in range(upto):
        if i < len(curve):
            b = min(b, curve[i])
        out.append(b / best)
    return out
