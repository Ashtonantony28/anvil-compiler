"""The sweep's ranking hook: tier2_static (anvil.cost, G4 as stated after audit A) inside anvil.tune.tune_kernel.

`StaticRanker` is `anvil.cost.ActiveRanker` with the committed tier2_static model and features computed without the
tier-1 event model (profile + closed form only: what tier2_static was trained on; the event-model columns of the
feature vector are filled with the closed form and never read by the model). Per tuning round it measures the
predicted top-k; every measurement updates the model online (ActiveRanker.observe).

SIMT_STACK=1: the profiler models lane-0 branch semantics only, so candidates are featurized at the same params
with SIMT_STACK=0 (an approximate ranking; the Verilator measurement that decides is exact). A candidate whose
profile fails is ranked last.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from ..cost.active import ActiveRanker, default_model
from ..cost.features import config_features, program_features, tier1_features, vector


class _T1:
    def __init__(self, cycles):
        self.cycles, self.breakdown = cycles, {}


def static_features(job, stats=None):
    from ..cost.analytic import closed_form
    from ..cost.profile import profile
    p = job.params
    if p.SIMT_STACK:
        p = dataclasses.replace(p, SIMT_STACK=0)
        job = dataclasses.replace(job, params=p)
    prof = profile(job)
    pf = program_features(prof, stats)
    cf = config_features(job.params, job.mem_latency, pf)
    closed = closed_form(prof, job.params, job.mem_latency)
    return vector(pf, cf, tier1_features(_T1(closed), closed))


class StaticRanker(ActiveRanker):
    def __init__(self, k: int = 2, seed: int = 0, **kw):
        super().__init__(model=default_model(seed, "tier2_static"), k=k, **kw)
        self.failed = set()

    def _features(self, task, b):
        if b.key not in self.feats:
            job = b.job
            job.params, job.mem_latency = task.params, task.mem_latency
            try:
                self.feats[b.key] = static_features(job, b.features())
            except Exception:                      # noqa: BLE001 - unprofilable: rank last
                self.failed.add(b.key)
                self.feats[b.key] = None
        return self.feats[b.key]

    def predict(self, task, builts) -> np.ndarray:
        xs = [self._features(task, b) for b in builts]
        ok = [i for i, x in enumerate(xs) if x is not None]
        out = np.full(len(builts), math.inf)
        if ok:
            out[ok] = self.cur.predict(np.stack([xs[i] for i in ok]))
        return out

    def observe(self, task, built, rec):
        if self.feats.get(built.key) is None:
            return
        super().observe(task, built, rec)
