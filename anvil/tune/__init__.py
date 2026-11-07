"""anvil.tune: the autotuner (SG6) - sketches over the schedule primitives, evolutionary search, a persistent
measurement cache with a JSONL log, a parallel measurement pool and a ranking hook.

    from anvil.tune import Task, tune_kernel, TunedPolicy
    task = Task.from_program(prog, "gate", values, GpuParams())   # values: Program.run(...) on some inputs
    res = tune_kernel(task, budget=64, seed=0, workers=4)         # res.best, res.trace, res.cycles, res.history
    cm = anvil.compile(model, ..., schedules="tuned", tune={"budget": 64, "seed": 0})

Modules
    space    sketches (tpl, mm, ew, rows) and `Candidate`; every candidate builds a Schedule from primitives only,
             so its trace replays (`Schedule.replay`) without the tuner
    measure  build (pruning: legality, canonical form, smem budget, registers, codegen, the 256-word program
             limit) -> isasim verdict (bit-exact vs the tensor interpreter, no divergence / race / store-conflict
             flags, C4 level 4) -> Verilator cycles; `MeasureCache` keyed by (program, args layout, GpuParams,
             mem_latency, backend) under .anvil-cache/measure/ with `log.jsonl` (SG7 training data)
    search   `tune_kernel`: seeds (the driver's templates) + random, then mutation/crossover of the best; a fork
             pool of <= 4 workers; `ranker(task, builts) -> builts to measure` (default `measure_all`; SG7)
    db       best schedules per kernel signature: committed anvil/tune/best/*.json, user .anvil-cache/tune/
    policy   `TunedPolicy`, the driver's schedules="tuned"
    g3_report  `python3 -m anvil.tune.g3_report` regenerates docs/g3-report.md and anvil/tune/best/xf-bench.json
"""
from .db import BestDB
from .measure import Built, Invalid, MeasureCache, Task, build
from .policy import TunedPolicy
from .search import TuneResult, measure_all, top_k, tune_kernel
from .space import Candidate, build_schedule, compile_options, sketches_for

__all__ = ["BestDB", "Built", "Invalid", "MeasureCache", "Task", "build", "TunedPolicy", "TuneResult", "measure_all",
           "top_k", "tune_kernel", "Candidate", "build_schedule", "compile_options", "sketches_for"]
