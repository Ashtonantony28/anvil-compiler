"""Tier 2: gradient-boosted trees (LightGBM) on program + config + tier-1 features, trained on the G4 dataset.

    X, y, meta = design(ds, rows)                 # feature matrix (features.FEATURES order), log cycles, row info
    m = CostModel("tier2").fit(X, y)              # or CostModel.load(path)
    cycles = m.predict(X)

Variants (the target is a log-ratio to a tier-1 prior, so the trees learn what the prior misses):
  tier2         all features; target log(cycles / tier-1 event model)
  tier2_static  no event-model output (only the program, the config and the closed form); target
                log(cycles / closed form): what a learned model gets without the event-level evaluation
Training is deterministic for a seed (single thread, deterministic LightGBM).
"""
from __future__ import annotations

import gzip
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .features import FEATURES, config_features, tier1_features, vector

EVENT_FEATURES = ("t1_cycles", "t1_log", "t1_fetch_frac", "t1_icache_hit_frac", "t1_dcache_hit_frac",
                  "t1_grab_wait", "t1_yields", "t1_idle_stall", "t1_mem_wait")
VARIANTS = {
    "tier2": {"features": list(FEATURES), "prior": "t1_cycles"},
    "tier2_static": {"features": [f for f in FEATURES if f not in EVENT_FEATURES], "prior": "t1_closed"},
}
LGB_PARAMS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "verbose": -1, "num_threads": 1, "deterministic": True, "force_row_wise": True}
ROUNDS = 400


def row_features(ds, r) -> np.ndarray:
    """The FEATURES vector of one dataset row (program features + its config + its recorded tier-1 outputs)."""
    prog = ds.programs[r["pid"]]
    params, lat = ds.params(r["cid"])
    pf = prog["pf"]
    cf = config_features(params, lat, pf)
    t1 = SimpleNamespace(cycles=r["t1"], breakdown=r["t1_bd"])
    return vector(pf, cf, tier1_features(t1, r["closed"]))


def design(ds, rows):
    X = np.stack([row_features(ds, r) for r in rows]) if rows else np.zeros((0, len(FEATURES)))
    y = np.log(np.array([r["cycles"] for r in rows], dtype=np.float64))
    meta = [{"pid": r["pid"], "cid": r["cid"], "kid": ds.programs[r["pid"]]["kid"]} for r in rows]
    return X, y, meta


class CostModel:
    def __init__(self, variant: str = "tier2", seed: int = 0, rounds: int = ROUNDS):
        self.variant, self.seed, self.rounds = variant, seed, rounds
        spec = VARIANTS[variant]
        self.features = spec["features"]
        self.idx = [FEATURES.index(f) for f in self.features]
        self.prior = FEATURES.index(spec["prior"])
        self.booster = None

    def _prior(self, X):
        return np.log(np.maximum(1.0, X[:, self.prior]))

    def fit(self, X, y):
        import lightgbm as lgb
        params = {**LGB_PARAMS, "seed": self.seed, "bagging_seed": self.seed, "feature_fraction_seed": self.seed}
        d = lgb.Dataset(X[:, self.idx], y - self._prior(X), feature_name=self.features, free_raw_data=False)
        self.booster = lgb.train(params, d, num_boost_round=self.rounds)
        return self

    def predict_log(self, X) -> np.ndarray:
        X = np.atleast_2d(X)
        return self._prior(X) + self.booster.predict(X[:, self.idx])

    def predict(self, X) -> np.ndarray:
        return np.exp(self.predict_log(X))

    def updated(self, X, y, rounds: int = 10):
        """A copy corrected on measured points (X, log cycles): the active-learning update inside a tuning run. The
        mean residual becomes a bias and `rounds` small, strongly shrunk trees (4 leaves) fit what remains, on top of
        the trained model (a few points of one kernel at one config)."""
        import lightgbm as lgb
        new = CostModel(self.variant, self.seed, self.rounds)
        Xs = np.atleast_2d(X)
        base = self.booster.predict(Xs[:, self.idx])
        r = np.asarray(y) - self._prior(Xs) - base
        bias = float(np.mean(r)) if len(r) else 0.0
        delta = None
        if len(r) >= 2:
            params = {**LGB_PARAMS, "num_leaves": 4, "min_data_in_leaf": 1, "min_sum_hessian_in_leaf": 0.0,
                      "learning_rate": 0.3, "bagging_fraction": 1.0, "bagging_freq": 0, "feature_fraction": 1.0,
                      "seed": self.seed, "min_data_in_bin": 1, "verbose": -1}
            d = lgb.Dataset(Xs[:, self.idx], r - bias, feature_name=self.features, init_score=np.zeros(len(r)),
                            params={"min_data_in_bin": 1, "verbose": -1})
            delta = lgb.train(params, d, num_boost_round=rounds)
        new.booster = _Stacked(self.booster, delta, bias)
        return new

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as f:
            f.write(f"# anvil.cost {self.variant} seed={self.seed} rounds={self.rounds}\n")
            f.write(self.booster.model_to_string())

    @classmethod
    def load(cls, path):
        import lightgbm as lgb
        with gzip.open(path, "rt") as f:
            head = f.readline()
            text = f.read()
        kv = dict(t.split("=") for t in head.split()[3:])
        m = cls(head.split()[2], int(kv["seed"]), int(kv["rounds"]))
        m.booster = lgb.Booster(model_str=text)
        return m


class _Stacked:
    """base booster + bias + delta booster (fit to the base's residuals)."""

    def __init__(self, base, delta, bias=0.0):
        self.base, self.delta, self.bias = base, delta, bias

    def predict(self, X):
        out = self.base.predict(X) + self.bias
        return out if self.delta is None else out + self.delta.predict(X)


# ---------------------------------------------------------------------------------------------- metrics

def kendall(a, b) -> float:
    from scipy.stats import kendalltau
    if len(a) < 2:
        return float("nan")
    t = kendalltau(a, b).statistic
    return float(t) if t == t else float("nan")


def tau_table(meta, y_true, y_pred):
    """Per config: pooled tau over every (kernel, candidate) row, the mean of per-kernel taus (ranking the candidates
    of one kernel: what the tuner needs) and the worst per-kernel tau."""
    out = {}
    cids = sorted({m["cid"] for m in meta})
    for cid in cids:
        ix = [i for i, m in enumerate(meta) if m["cid"] == cid]
        pooled = kendall(y_true[ix], y_pred[ix])
        per = {}
        for kid in sorted({meta[i]["kid"] for i in ix}):
            jx = [i for i in ix if meta[i]["kid"] == kid]
            if len(jx) >= 3:
                per[kid] = kendall(y_true[jx], y_pred[jx])
        vals = [v for v in per.values() if not math.isnan(v)]
        out[cid] = {"n": len(ix), "pooled": pooled, "within_mean": float(np.mean(vals)) if vals else float("nan"),
                    "within_min": float(np.min(vals)) if vals else float("nan"), "kernels": len(vals),
                    "per_kernel": per}
    return out
