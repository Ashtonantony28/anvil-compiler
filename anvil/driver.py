"""Compile driver (SG4b): PyTorch module or anvil.graph IR -> scheduled, compiled launch chain with a static
memory plan, runnable on every C2 backend.

    cm = anvil.compile(block, (x_example,), calib, GpuParams(), schedules="tiled", names={...})
    res = cm.run(x_float, "rtl")              # quantize x, run the chain, read back and dequantize
    res.output, res.ints["out"], res.cycles   # float output, int segments, per-kernel cycles (None on isasim)
    cm.plan.report(), cm.footprint            # memory plan (anvil.memplan)

`model` is a torch.nn.Module (captured and quantized by anvil.frontend.from_torch on `calib_inputs`, extra
keywords go to quantize), an `anvil.frontend.Quantized`, an anvil.graph builtin.module (consts by `consts=`) or a
`(module, consts)` pair as `anvil.frontend.from_image` returns. The driver is op-agnostic: it lowers every graph op
of the Program (single-head xf ops today, SG5a's head-batched ops when they lower), schedules each kernel, compiles
it through anvil.codegen and lays the chain out with anvil.memplan.

Schedule selection (`schedules`):
    None | "default"      anvil.schedule.templates.default_schedule per kernel
    "tiled"               matmuls -> matmul_tiled, rmsnorm/softmax -> row_per_block, falling back to the default
                          schedule where the template does not apply
    dict                  kernel name or op kind -> template name | callable(Schedule) -> Schedule | JSON trace
                          (Schedule.to_json); unlisted kernels take the default. Explicit entries never fall back:
                          an illegal one raises ScheduleError
    callable(kernel, target) -> Schedule | None (None: default)
    "tuned"               the best schedule anvil.tune found per kernel (SG6): replayed from the best-schedule
                          store (committed anvil/tune/best/*.json, then .anvil-cache/tune/best.json), searched on a
                          miss. `tune=` takes TunedPolicy keywords: budget, seed, workers, search (False: replay or
                          default, never search), ranker, db, cache, inputs. A TunedPolicy instance works too.
                          Tuned kernels compile with their own codegen options; `cm.tuning` reports per kernel

Runs: `keep="outputs"` (the default plan) reuses dead activation buffers, so only the graph outputs (and inputs
not overwritten) can be read back; `run(..., keep="all")` uses a second plan of the same compiled programs in which
every graph value stays live, for segment-by-segment golden checks. Programs do not depend on the plan: every
address is in the per-launch args block.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import memplan
from .codegen import CodegenError, compile_kernel, program_job
from .ir import graph as g
from .lower.graph_to_tensor import lower
from .runtime import GpuParams, run as run_job
from .schedule import Schedule, ScheduleError, Target, templates as T

PRESETS = {
    "default": {},
    "tiled": {"qmatmul": "matmul_tiled", "qlinear": "matmul_tiled", "rmsnorm_int": "row_per_block",
              "softmax_lut": "row_per_block"},
}
MAX_THREADS = 4096          # Target.max_threads: kernels above 255 threads split into several launches


@dataclass
class RunOutput:
    output: np.ndarray                  # first graph output, dequantized (float)
    outputs: list                       # every graph output, dequantized
    ints: dict                          # sym -> int array of every value readable after the run
    cycles: dict                        # kernel name -> cycles summed over its launches (None on isasim)
    result: object                      # anvil.runtime.RunResult
    job: object                         # anvil.codegen.JobPlan
    _consts: frozenset = frozenset()    # const syms (must be unchanged by the run)

    def stray_writes(self, *, scratch_ok: bool = True) -> list:
        """Words the run changed that it must not have: outside every planned buffer (args blocks, gaps, past the
        job) or inside a const (weights, LUTs). Empty for a correct chain (AUDIT-A1 M3); scratch_ok exempts the
        shared sink/spill words."""
        consts = [s for s in self.job.buffers if s in self._consts]
        return self.job.stray_writes(self.result.memory, unchanged=consts, scratch_ok=scratch_ok)

    @property
    def total_cycles(self):
        c = list(self.cycles.values())
        return None if any(v is None for v in c) else sum(c)


def _graph_meta(module):
    """sym -> (scale, shape, dtype) for every graph value."""
    out = {}
    for op in g.funcs(module)[0].body.block.ops:
        if isinstance(op, g.OutputOp) or not op.results:
            continue
        ty = g.qt(op.results[0])
        out[op.sym_str] = (ty.qscale, tuple(ty.dims), ty.dtype)
    return out


def _resolve_model(model, example_inputs, calib_inputs, consts, quant_kw):
    from xdsl.dialects.builtin import ModuleOp

    from .frontend import Quantized
    graph_path = isinstance(model, (Quantized, ModuleOp)) or (
        isinstance(model, tuple) and len(model) == 2 and isinstance(model[0], ModuleOp))
    if graph_path and quant_kw:
        raise TypeError(f"anvil.compile got unexpected keyword argument(s) {sorted(quant_kw)} (extra keywords go to "
                        "quantize, which only runs for a torch module)")
    if isinstance(model, Quantized):
        return model.module, dict(model.consts), model
    if isinstance(model, tuple) and len(model) == 2 and isinstance(model[0], ModuleOp):
        return _checked_graph(model[0], dict(model[1]))
    if isinstance(model, ModuleOp):
        return _checked_graph(model, dict(consts or {}))
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(model, torch.nn.Module):
        from .frontend import from_torch
        if example_inputs is None or calib_inputs is None:
            raise ValueError("compiling a torch module needs example_inputs and calib_inputs")
        q = from_torch(model, example_inputs, calib_inputs, **quant_kw)
        return q.module, dict(q.consts), q
    raise TypeError(f"cannot compile a {type(model).__name__}: expected a torch.nn.Module, anvil.frontend.Quantized, "
                    "an anvil.graph module or (module, consts)")


def _checked_graph(module, consts):
    """A graph-IR model as given (not from quantize): every non-inline const needs its values, and the int32
    overflow proof that quantize runs must hold here too (AUDIT-A2 M4, M6)."""
    from .frontend.quantize import prove_no_overflow
    need = [op.sym_str for op in g.funcs(module)[0].body.block.ops
            if isinstance(op, g.ConstOp) and op.value is None]
    missing = [s for s in need if s not in consts]
    if missing:
        raise ValueError(f"no values for graph consts {missing}: pass consts= or a (module, consts) pair")
    unknown = sorted(set(consts) - set(need))
    if unknown:
        raise ValueError(f"consts {unknown} name no graph const of the module")
    prove_no_overflow(module, consts)
    return module, consts, None


def select_schedule(kernel, target: Target, schedules=None) -> tuple[Schedule, str]:
    """(Schedule, description) for one lowered kernel under the `schedules` policy (module docstring)."""
    if schedules is None:
        schedules = "default"
    if callable(schedules) and not isinstance(schedules, (str, dict)):
        s = schedules(kernel, target)
        return (s, "custom") if s is not None else (T.default_schedule(kernel, target), "default")
    if not isinstance(schedules, (str, dict)):
        raise TypeError(f"schedules must be a preset name, a dict, a callable or 'tuned', not {schedules!r}")
    strict = not isinstance(schedules, str)
    if isinstance(schedules, str) and schedules not in PRESETS:
        raise KeyError(f"unknown schedule preset {schedules!r}: {sorted(PRESETS)}")
    table = PRESETS[schedules] if isinstance(schedules, str) else schedules
    spec = table.get(kernel.name, table.get(kernel.op))
    if spec is None:
        return T.default_schedule(kernel, target), "default"
    try:
        if isinstance(spec, str) and spec in T.TEMPLATES:
            return T.TEMPLATES[spec](Schedule(kernel, target)), spec
        if isinstance(spec, str) and not spec.lstrip().startswith("["):
            raise ScheduleError(f"{kernel.name}: {spec!r} is neither a template ({sorted(T.TEMPLATES)}) nor a "
                                "JSON schedule trace")
        if isinstance(spec, (str, list)):
            return Schedule.replay(kernel, spec, target), "trace"
        if callable(spec):
            return spec(Schedule(kernel, target)), getattr(spec, "__name__", "custom")
    except ScheduleError:
        if strict:
            raise
        return T.default_schedule(kernel, target), "default"
    raise TypeError(f"{kernel.name}: schedule spec {spec!r} is not a template name, trace or callable")


class CompiledModel:
    """A compiled launch chain: lowered Program, per-kernel schedules and programs, memory plan."""

    def __init__(self, module, consts, params: GpuParams, schedules=None, *, quantized=None, target_kw=None,
                 compile_kw=None, tune=None):
        if isinstance(params, dict):
            params = GpuParams(**params)
        if not isinstance(params, GpuParams):
            raise TypeError(f"params must be a GpuParams (or a dict of its fields), not {type(params).__name__}")
        self.module, self.consts, self.params, self.quantized = module, consts, params, quantized
        self.program = lower(module)
        if not self.program.kernels:
            raise ValueError("the graph has no operations to compile (its outputs are its inputs)")
        self.in_dtypes = {op.sym_str: g.qt(op.results[0]).dtype for op in g.funcs(module)[0].body.block.ops
                          if isinstance(op, g.InputOp)}
        self.meta = _graph_meta(module)
        self.target = Target.from_params(params, **{"max_threads": MAX_THREADS, **(target_kw or {})})
        self.schedules, self.kernels, self.tuning = {}, {}, {}
        policy = None
        if schedules == "tuned" or type(schedules).__name__ == "TunedPolicy":
            from .tune.policy import TunedPolicy
            policy = schedules if isinstance(schedules, TunedPolicy) else TunedPolicy(params, **(tune or {}))
            policy.prepare(self.program, {**self.program.const_values, **consts}, params, self.target)
        for k in self.program.kernels:
            ckw = dict(compile_kw or {})
            if policy is not None:
                s, tkw, how = policy.select(k, self.target)
                ckw.update(tkw)
            else:
                s, how = select_schedule(k, self.target, schedules)
            self.schedules[k.name] = (how, s.to_json())
            self.kernels[k.name] = compile_kernel(s.finalize(), params, **ckw)
        if isinstance(schedules, dict):
            names = {k.name for k in self.program.kernels} | {k.op for k in self.program.kernels}
            unknown = sorted(set(schedules) - names)
            if unknown:
                raise KeyError(f"schedules name no kernel or op kind of the graph: {unknown} (kernels "
                               f"{[k.name for k in self.program.kernels]})")
        if policy is not None:
            self.tuning = policy.results
        self._plans = {}
        self.plan = self.memory_plan("outputs")

    # ------------------------------------------------------------------------------------------ memory
    def memory_plan(self, keep="outputs") -> memplan.MemPlan:
        if keep not in self._plans:
            self._plans[keep] = memplan.plan_program(self.program, self.kernels, self.params, keep=keep)
        return self._plans[keep]

    @property
    def footprint(self) -> int:
        return self.plan.footprint

    # -------------------------------------------------------------------------------- (de)quantization
    @property
    def inputs(self) -> list:
        return list(self.program.inputs)

    @property
    def outputs(self) -> list:
        return list(self.program.outputs)

    def scale(self, sym):
        return self.meta[sym][0]

    def quantize_input(self, x, sym=None):
        """Float array -> integers by the input's scale (symmetric, round half to even, saturating at the input's
        dtype: int8, or int16 for a wide residual input, ADR-004)."""
        from . import semantics as sem
        sym = sym or self.inputs[0]
        s = self.scale(sym)
        if s is None:
            raise ValueError(f"input {sym!r} has no scale: pass integer inputs with quantized=True")
        return sem.clamp_dtype(np.rint(np.asarray(x, dtype=np.float64) / s), self.meta[sym][2]).astype(np.int64)

    def dequantize(self, sym, ints):
        s = self.scale(sym)
        return np.asarray(ints, dtype=np.float64) * (1.0 if s is None else s)

    # ------------------------------------------------------------------------------------------- jobs
    def _bind_inputs(self, x, quantized):
        xs = list(x) if isinstance(x, (list, tuple)) else [x]
        if len(xs) != len(self.inputs):
            raise ValueError(f"{len(xs)} inputs for graph inputs {self.inputs}")
        out = {}
        for sym, v in zip(self.inputs, xs):
            v = np.asarray(v)
            shape = tuple(self.meta[sym][1])
            if v.shape != shape:
                squeeze = lambda sh: tuple(d for d in sh if d != 1)   # noqa: E731
                if v.size != math.prod(shape) or (v.ndim > 1 and squeeze(v.shape) != squeeze(shape)):
                    # only a flat array or extra size-1 dims are reshaped; a transposed input is an error (A2-L8)
                    raise ValueError(f"input {sym!r}: shape {v.shape} != {shape}")
                v = v.reshape(shape)
            if quantized:
                out[sym] = self._check_int_input(sym, v)
            else:
                if not np.all(np.isfinite(np.asarray(v, dtype=np.float64))):
                    raise ValueError(f"input {sym!r} has NaN or infinite values")
                out[sym] = self.quantize_input(v, sym)
        return out

    def _check_int_input(self, sym, v):
        """quantized=True: integer values inside the input's own dtype (every overflow bound assumes it, A2-M5)."""
        if v.dtype.kind not in "iub":
            raise ValueError(f"input {sym!r}: quantized=True needs integer values, got dtype {v.dtype}")
        dt = self.in_dtypes.get(sym, "i8")
        bits = {"i8": 8, "i16": 16, "i32": 32}.get(dt, 32)
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        v = np.asarray(v, dtype=np.int64)
        if v.size and (v.min() < lo or v.max() > hi):
            raise ValueError(f"input {sym!r}: values in [{v.min()}, {v.max()}] outside {dt} [{lo}, {hi}]")
        return v

    def job(self, x, *, quantized=False, keep="outputs"):
        """The JobPlan for one input (anvil.codegen.program_job at this model's memory plan)."""
        mp = self.memory_plan(keep)
        values = {**self.program.const_values, **self.consts, **self._bind_inputs(x, quantized)}
        jp = program_job(self.program, self.kernels, values, self.params, addresses=mp.addresses)
        want = [(n, i, b) for n, i, b, _ in mp.args_blocks]
        if jp.args != want or jp.end != mp.footprint:      # codegen and the planner must agree on the layout
            raise CodegenError(f"job layout differs from the memory plan: end {jp.end} vs {mp.footprint}")
        return jp

    def readable(self, keep="outputs") -> list:
        """Graph values whose final contents a run with this plan returns."""
        mp = self.memory_plan(keep)
        n = len(self.program.kernels)
        root = lambda s: self.program.views[s][0] if s in self.program.views else s
        out = []
        for s in list(self.program.buffers) + list(self.program.views):
            r = root(s)
            b = mp.bufs.get(r)
            if b is not None and b.end >= n and b.kind != "const":
                out.append(s)
        return out

    def run(self, x, backend: str = "isasim", *, quantized=False, keep="outputs", **run_kw) -> RunOutput:
        """Quantize `x` (unless quantized=True: integer inputs), run the chain on `backend`, read back."""
        jp = self.job(x, quantized=quantized, keep=keep)
        res = run_job(jp.job, backend, **run_kw)
        ints = {}
        for s in self.readable(keep):
            src, shape = (self.program.views[s] if s in self.program.views else (s, None))
            v = jp.read(res.memory, src)
            ints[s] = v.reshape(shape) if shape is not None else v
        cycles, li = {}, 0
        for k in self.program.kernels:
            ck = self.kernels[k.name]
            lrs = res.launches[li:li + ck.launches]
            li += ck.launches
            cycles[k.name] = None if any(r.cycles is None for r in lrs) else sum(r.cycles for r in lrs)
        outs = [self.dequantize(s, ints[s]) for s in self.outputs]
        consts = frozenset(set(self.program.const_values) | set(self.consts))
        return RunOutput(outs[0], outs, ints, cycles, res, jp, consts)

    def summary(self) -> str:
        rows = [f"{k.name:>10} {k.op:<12} {self.schedules[k.name][0]:<14} {self.kernels[k.name].program.stats['words']:>4} words "
                f"{self.kernels[k.name].launches} launch(es)" for k in self.program.kernels]
        return "\n".join(rows + [self.plan.report()])


def compile(model, example_inputs=None, calib_inputs=None, params: GpuParams | None = None, schedules=None, *,
            consts=None, target_kw=None, compile_kw=None, tune=None, **quant_kw) -> CompiledModel:
    """anvil.compile: see the module docstring."""
    module, cs, q = _resolve_model(model, example_inputs, calib_inputs, consts, quant_kw)
    return CompiledModel(module, cs, params or GpuParams(), schedules, quantized=q, target_kw=target_kw,
                         compile_kw=compile_kw, tune=tune)
