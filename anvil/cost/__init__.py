"""anvil.cost: cost models of tiny-gpu-extended kernels (SG7, G4).

    from anvil.cost import predict, profile, featurize, CostModel, ActiveRanker
    t = predict(job)                         # tier 1: cycles from the RTL's FSM state durations (no training)
    x, t1 = featurize(job, stats)            # tier-2 feature vector (anvil.cost.features.FEATURES)
    m = CostModel.load("data/cost/tier2-seed0.txt.gz"); m.predict(x)
    tune_kernel(task, ranker=ActiveRanker(k=2))     # model-guided tuning with active learning

Modules
    profile    dynamic instruction streams (per launch / block / warp, global addresses per lane)
    analytic   tier 1: the event-level FSM timing model (`predict`) and a contention-free closed form
    features   tier-2 features: program, dynamic profile, configuration, tier-1 outputs
    dataset    the Verilator dataset (kernels x sketch points x GpuParams configs) with a frozen split
               (`python3 -m anvil.cost.dataset` regenerates data/cost/)
    learned    tier 2: LightGBM on the features, log-ratio target over a tier-1 prior; Kendall tau tables
    active     ActiveRanker (tune_kernel's ranking hook + observe hook) and the offline tuning curves
    g4_report  `python3 -m anvil.cost.g4_report` trains the committed models and writes docs/g4-report.md
"""
from .active import ActiveRanker
from .analytic import Timing, closed_form, predict, predict_profile
from .features import FEATURES, featurize
from .learned import CostModel
from .profile import Profile, profile

__all__ = ["ActiveRanker", "Timing", "closed_form", "predict", "predict_profile", "FEATURES", "featurize",
           "CostModel", "Profile", "profile"]
