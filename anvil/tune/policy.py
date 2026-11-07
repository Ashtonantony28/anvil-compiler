"""The driver's `schedules="tuned"` policy: per kernel, the best schedule found by anvil.tune.

    cm = anvil.compile(model, ..., schedules="tuned", tune={"budget": 96, "seed": 0})
    cm = anvil.compile(model, ..., schedules=TunedPolicy(params, search=False))     # replay only, never search
    cm.tuning[kernel] -> {"how": "db" | "searched" | "default", "cycles": ..., "entry": ...}

For each kernel the policy looks the kernel's signature (kernel IR up to buffer names, Target, GpuParams) up in
the best-schedule store (anvil.tune.db: the committed `anvil/tune/best/*.json`, then
`.anvil-cache/tune/best.json`); on a hit the stored sketch candidate is rebuilt on this kernel (its trace names the
tuned kernel's buffers) and its codegen options apply. On a miss it searches (`tune_kernel` with `budget`, `seed`,
`workers`, `ranker`) unless `search=False`, stores the result, and otherwise falls back to the default schedule.
The best candidates of kernels already selected with the same op, and the stored best of the same kernel IR at
other GPU params, seed each new search (transfer, e.g. between the q/k/v projections; a search is reproducible
for a seed and a store). Candidates are verified on inputs drawn from `inputs` (graph input sym -> int array) or, by
default, seeded uniform int8 inputs pushed through the tensor interpreter of the unscheduled program.
"""
from __future__ import annotations

import numpy as np

from ..schedule import Schedule, templates as T
from .db import BestDB
from .measure import MeasureCache, Task
from .search import tune_kernel


class TunedPolicy:
    def __init__(self, params=None, *, budget: int = 64, seed: int = 0, workers: int = 4, search: bool = True,
                 db: BestDB | None = None, ranker=None, cache: MeasureCache | None = None, inputs: dict | None = None,
                 input_seed: int = 1234, mem_latency: int = 0, save: bool = True,
                 backend: str = "verilator"):
        self.params, self.budget, self.seed, self.workers, self.search = params, budget, seed, workers, search
        self.db = db if db is not None else BestDB()
        self.ranker, self.cache, self.inputs, self.input_seed = ranker, cache, inputs, input_seed
        self.mem_latency, self.save, self.backend = mem_latency, save, backend
        self.values = None
        self.found = []             # best candidates found so far: seeds for later kernels (transfer)
        self.results = {}           # kernel name -> {"how", "cycles", "entry", "result"}

    def prepare(self, program, consts: dict, params, target):
        """Reference values of every buffer for verifying candidates (once per program)."""
        self.params, self.target, self.program = params, target, program
        ins = dict(self.inputs or {})
        rng = np.random.default_rng(self.input_seed)
        for s in program.inputs:
            if s not in ins:
                shape, _ = program.buffers[s]
                ins[s] = rng.integers(-128, 128, shape)
        self.values = program.run(ins, consts)

    def task(self, kernel) -> Task:
        return Task.from_program(self.program, kernel.name, self.values, self.params, self.target,
                                 mem_latency=self.mem_latency, backend=self.backend)

    def transfer(self, task) -> list:
        """Extra generation-0 candidates: the best of kernels with the same op selected earlier, then the stored
        best of the same kernel IR at other params (deterministic order)."""
        from .space import Candidate
        out = [c for op, c in self.found if op == task.kernel.op]
        ir = task.ir_hash
        for sig in sorted(self.db.entries):
            e = self.db.entries[sig]
            if e.get("ir") == ir and e.get("candidate"):
                c = Candidate.from_dict(e["candidate"])
                if c not in out:
                    out.append(c)
        return out

    def select(self, kernel, target) -> tuple[Schedule, dict, str]:
        """(Schedule, codegen keywords, how) for one kernel."""
        task = self.task(kernel)
        sig = task.signature
        entry = self.db.get(sig)
        how = "db"
        if entry is None and self.search:
            res = tune_kernel(task, budget=self.budget, seed=self.seed, workers=self.workers, ranker=self.ranker,
                              cache=self.cache, extra=self.transfer(task))
            if res.best is not None:
                entry = {**res.to_entry(), "ir": task.ir_hash}
                self.db.put(sig, entry)
                if self.save:
                    self.db.save()
                how = "searched"
            self.results[kernel.name] = {"result": res}
        if entry is None:
            self.results.setdefault(kernel.name, {}).update(how="default", cycles=None, entry=None)
            return T.default_schedule(kernel, target), {}, "default"
        if entry.get("candidate"):           # rebuild from the candidate: its trace names this kernel's buffers
            from .space import Candidate, build_schedule
            c = Candidate.from_dict(entry["candidate"])
            if (kernel.op, c) not in self.found:
                self.found.append((kernel.op, c))
            s = build_schedule(kernel, target, c)
        else:
            s = Schedule.replay(kernel, entry["trace"], target)
        self.results.setdefault(kernel.name, {}).update(how=how, cycles=entry.get("cycles"), entry=entry,
                                                        signature=sig)
        return s, dict(entry.get("compile_kw") or {}), "tuned"
