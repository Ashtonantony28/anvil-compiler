"""The G4 dataset: Verilator cycles of many kernels x sketch points x GPU configurations, with a frozen split.

    PYTHONPATH=sim python3 -m anvil.cost.dataset [--workers 4] [--out data/cost] [--model-from-torch]
    ds = load()                                    # Dataset: kernels, configs, programs, rows (+ split)

Kernels (`suite()`): the xf MLP block and attention head kernels (graph IR from the fixture images), the SG5a model's
layer-0 kernels (d=32, 2 heads, T=16; graph IR captured once from torch and frozen in data/cost/model_graph.json.gz,
`--model-from-torch` re-captures it with NANOGPT_DIR), and synthetic single-op graphs of other shapes built from the
fixtures' constants. Per kernel and WARPS_PER_CORE class (the program depends on WARPS_PER_CORE, not on the timing
parameters) the sketch space is sampled as the tuner samples it (template seeds, then uniform draws, weighted to
non-template sketches), keeping distinct programs that build (pruned candidates are counted, not measured). Each
program is verified on isasim like a tuner candidate (bit-exact vs the tensor interpreter, no divergence/race/
store-conflict flags), profiled, and timed on Verilator at every configuration of its class; the Verilator memory
must equal isasim's below the scratch area. Rows record the measured cycles, the tier-1 predictions (event model and
closed form, with the breakdown) and the Verilator counters.

Split (frozen in SPLIT, written to split.json): held-out kernels (the whole xf attention head, five model kernels,
six synthetic ones) and held-out configurations (four; two use values never seen in training: DCACHE_LINES=8, and
GTLRR with both caches). Training uses only train kernels x train configs. Deduplicated after the build
(AUDIT-A2 M3, `dedupe_split`): a held-out kernel with the same op and shapes as a training kernel (e.g.
xf_attention.h = xf_mlp.h) is reassigned to train, and a held-out program whose feature vector equals a training
program's is dropped; split.json records both.

Files (data/cost/): kernels.json, configs.json, split.json, programs.jsonl.gz (one record per program: kernel,
class, candidate, static stats, program features), rows.jsonl.gz (one per program x config), model_graph.json.gz.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "cost"
MODEL_GRAPH = DATA / "model_graph.json.gz"

# ------------------------------------------------------------------------------------------------ configs

# id -> (GpuParams overrides, mem_latency). WARPS_PER_CORE defines the program class.
CONFIGS = {
    "c00": ({}, 0),
    "c01": ({"COALESCE": 1}, 0),
    "c02": ({"DCACHE_LINES": 16}, 0),
    "c03": ({"COALESCE": 1, "DCACHE_LINES": 16}, 0),
    "c04": ({"NUM_CORES": 1}, 0),
    "c05": ({"NUM_CORES": 4}, 0),
    "c06": ({"ICACHE_LINES": 16}, 0),
    "c07": ({"PIPELINED": 1}, 0),
    "c08": ({}, 3),
    "c09": ({"NUM_CORES": 4, "COALESCE": 1}, 3),
    "c10": ({"ICACHE_LINES": 16, "PIPELINED": 1, "DCACHE_LINES": 16}, 3),
    "c11": ({"NUM_CORES": 1, "ICACHE_LINES": 16}, 3),
    "c12": ({"WARPS_PER_CORE": 2, "WARP_POLICY": 0}, 0),
    "c13": ({"WARPS_PER_CORE": 4, "WARP_POLICY": 1}, 0),
    "c14": ({"WARPS_PER_CORE": 2, "WARP_POLICY": 2, "ICACHE_LINES": 16}, 3),
    "c15": ({"WARPS_PER_CORE": 4, "WARP_POLICY": 0, "NUM_CORES": 4, "COALESCE": 1}, 0),
    # held out
    "h0": ({"NUM_CORES": 4, "ICACHE_LINES": 16, "DCACHE_LINES": 16, "COALESCE": 1}, 3),
    "h1": ({"DCACHE_LINES": 8, "PIPELINED": 1}, 0),
    "h2": ({"WARPS_PER_CORE": 2, "WARP_POLICY": 1, "NUM_CORES": 1}, 3),
    "h3": ({"WARPS_PER_CORE": 4, "WARP_POLICY": 2, "ICACHE_LINES": 16, "DCACHE_LINES": 16}, 0),
}
HELDOUT_CONFIGS = ("h0", "h1", "h2", "h3")
HELDOUT_KERNELS = (
    "xf_attention.h", "xf_attention.q", "xf_attention.k", "xf_attention.v", "xf_attention.kT",
    "xf_attention.scores", "xf_attention.p", "xf_attention.o", "xf_attention.y",
    "model.L0.scores", "model.L0.p", "model.L0.down", "model.L0.kT", "model.L0.r1",
    "syn.qlinear_3x12x8", "syn.qlinear_8x4x12", "syn.rmsnorm_6x12", "syn.softmax_12x12", "syn.lut_mul_16x8",
    "syn.transpose_6x10")
N_CAND = {1: 40, 2: 24, 4: 24}            # distinct programs per kernel and class (train kernels)
N_CAND_HELDOUT = {1: 80, 2: 24, 4: 24}    # held-out kernels get larger pools (tuning curves)
N_CAND_MODEL = {1: 24, 2: 12, 4: 12}      # the model's kernels are 10-50x longer to simulate
N_CAND_MODEL_HELDOUT = {1: 48, 2: 12, 4: 12}


def n_candidates(kid, W):
    held = kid in HELDOUT_KERNELS
    if kid.startswith("model."):
        return (N_CAND_MODEL_HELDOUT if held else N_CAND_MODEL)[W]
    return (N_CAND_HELDOUT if held else N_CAND)[W]


def config_params(cid):
    from ..runtime import GpuParams
    over, lat = CONFIGS[cid]
    return GpuParams(**over), lat


def classes() -> dict:
    """WARPS_PER_CORE -> [config ids]."""
    out = {}
    for cid, (over, _) in CONFIGS.items():
        out.setdefault(over.get("WARPS_PER_CORE", 1), []).append(cid)
    return out


# ------------------------------------------------------------------------------------------------ kernels

@dataclass
class KernelSpec:
    kid: str
    group: str
    program: object = field(repr=False)
    name: str = ""
    values: dict = field(repr=False, default_factory=dict)

    @property
    def kernel(self):
        return self.program.kernel(self.name)


def _xf(block):
    from ..lower.graph_to_tensor import lower
    from ..tune.g3_report import fixture_model, mm_model
    (module, consts), d = mm_model() if block == "mm1x8x8" else fixture_model(block)
    prog = lower(module)
    return prog, prog.run({"x": d["inputs"]["x"]}, consts)


def _fixture_consts():
    from xf import fixture
    out = {}
    for blk in ("mlp_block", "attention"):
        img = fixture.load(fixture.FIXTURE_DIR / f"{blk}_0.json")["image"]
        for s in img.segments:
            out[f"{blk}.{s}"] = np.asarray(img.read(s)).ravel()
    return out


def _synthetic():
    """name -> (module, consts, inputs): single-op graphs at shapes the fixtures do not have."""
    from ..ir.graph import GraphBuilder
    fc = _fixture_consts()

    def c(n):
        return int(fc[n][0])
    rng = np.random.default_rng(20261107)
    out = {}

    def i8(*shape):
        return rng.integers(-128, 128, shape)

    def qlin(M, N, K):
        g = GraphBuilder(f"ql{M}x{N}x{K}")
        y = g.qlinear("y", g.input("x", [M, K]), g.const("w", [K, N]), c("mlp_block.gate_M"), c("mlp_block.gate_s"))
        g.output(y)
        return g.module(), {"w": i8(K, N)}, {"x": i8(M, K)}

    def qmm(M, N, K):
        g = GraphBuilder(f"qmm{M}x{N}x{K}")
        y = g.qmatmul("y", g.input("a", [M, K]), g.input("b", [K, N]), c("attention.scores_M"),
                      c("attention.scores_s"))
        g.output(y)
        return g.module(), {}, {"a": i8(M, K), "b": i8(K, N)}

    def lut(*shape):
        g = GraphBuilder("lut")
        y = g.lut("y", g.input("x", list(shape)), g.const("t", [256]))
        g.output(y)
        return g.module(), {"t": fc["mlp_block.silu_lut"][:256]}, {"x": i8(*shape)}

    def lut_mul(*shape):
        g = GraphBuilder("lutmul")
        y = g.lut_mul("y", g.input("a", list(shape)), g.input("b", list(shape)), g.const("t", [256]),
                      c("mlp_block.act_M"), c("mlp_block.act_s"))
        g.output(y)
        return g.module(), {"t": fc["mlp_block.silu_lut"][:256]}, {"a": i8(*shape), "b": i8(*shape)}

    def add(*shape):
        g = GraphBuilder("add")
        y = g.add_requant("y", g.input("a", list(shape)), g.input("b", list(shape)), c("mlp_block.res_Ma"),
                          c("mlp_block.res_Mb"), c("mlp_block.res_s"))
        g.output(y)
        return g.module(), {}, {"a": i8(*shape), "b": i8(*shape)}

    def rms(R, C):
        g = GraphBuilder("rms")
        lut_ = fc["mlp_block.rsqrt_lut"]
        y = g.rmsnorm_int("y", g.input("x", [R, C]), g.const("w", [C]), g.const("lut", [len(lut_)], "i32"),
                          c("mlp_block.norm_M"), c("mlp_block.norm_s"))
        g.output(y)
        w = np.resize(fc["mlp_block.norm_w"], C)
        return g.module(), {"w": w, "lut": lut_}, {"x": i8(R, C)}

    def smax(*shape):
        g = GraphBuilder("smax")
        lut_ = fc["attention.exp_lut"]
        y = g.softmax_lut("y", g.input("x", list(shape)), g.const("lut", [len(lut_)], "i32"), c("attention.one"), True)
        g.output(y)
        return g.module(), {"lut": lut_}, {"x": i8(*shape)}

    def tr(R, C):
        g = GraphBuilder("tr")
        y = g.transpose("y", g.input("x", [R, C]), [1, 0])
        g.output(y)
        return g.module(), {}, {"x": i8(R, C)}

    specs = {
        "qlinear_2x8x8": (qlin, (2, 8, 8)), "qlinear_4x4x8": (qlin, (4, 4, 8)), "qlinear_8x8x8": (qlin, (8, 8, 8)),
        "qlinear_4x8x4": (qlin, (4, 8, 4)), "qlinear_3x12x8": (qlin, (3, 12, 8)), "qlinear_6x8x16": (qlin, (6, 8, 16)),
        "qlinear_8x4x12": (qlin, (8, 4, 12)), "qlinear_2x16x16": (qlin, (2, 16, 16)),
        "qlinear_12x8x8": (qlin, (12, 8, 8)), "qmatmul_8x8x4": (qmm, (8, 8, 4)), "qmatmul_4x6x8": (qmm, (4, 6, 8)),
        "lut_4x16": (lut, (4, 16)), "lut_24": (lut, (24,)), "lut_mul_6x8": (lut_mul, (6, 8)),
        "lut_mul_16x8": (lut_mul, (16, 8)), "add_5x8": (add, (5, 8)), "add_12x12": (add, (12, 12)),
        "rmsnorm_8x8": (rms, (8, 8)), "rmsnorm_2x16": (rms, (2, 16)), "rmsnorm_6x12": (rms, (6, 12)),
        "softmax_8x8": (smax, (8, 8)), "softmax_2x4x4": (smax, (2, 4, 4)), "softmax_12x12": (smax, (12, 12)),
        "transpose_8x4": (tr, (8, 4)), "transpose_6x10": (tr, (6, 10)),
    }
    for name, (f, args) in specs.items():
        out[name] = f(*args)
    return out


def save_model_graph(path=MODEL_GRAPH, n_calib=512):
    """Capture + quantize the SG5a model from torch (NANOGPT_DIR) and freeze its graph IR and constants."""
    from .. import model as M
    from ..ir import print_ir
    nd = os.environ.get("NANOGPT_DIR")
    if not nd:
        raise RuntimeError("NANOGPT_DIR is not set")
    q = M.quantize_model(M.build_model(nd), calib=M.calib_inputs(n_calib))
    body = {"ir": print_ir(q.module), "consts": {k: np.asarray(v).astype(int).ravel().tolist() for k, v in q.consts.items()},
            "shapes": {k: list(np.shape(v)) for k, v in q.consts.items()}, "n_calib": n_calib}
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(body, f, sort_keys=True)


def _model_program():
    from ..ir import parse_ir
    from ..lower.graph_to_tensor import lower
    with gzip.open(MODEL_GRAPH, "rt") as f:
        body = json.load(f)
    module = parse_ir(body["ir"])
    consts = {k: np.asarray(v, dtype=np.int64).reshape(body["shapes"][k]) for k, v in body["consts"].items()}
    prog = lower(module)
    rng = np.random.default_rng(7)
    ins = {s: rng.integers(-128, 128, prog.buffers[s][0]) for s in prog.inputs}
    return prog, prog.run(ins, consts)


MODEL_KERNELS = ("L0.h1", "L0.q", "L0.qh", "L0.kT", "L0.scores", "L0.p", "L0.o", "L0.om.t", "L0.y", "L0.r1",
                 "L0.gate", "L0.act", "L0.down", "L0.out")


def suite(groups=("xf", "model", "syn")) -> list:
    """Every kernel of the dataset as a KernelSpec (deterministic order)."""
    from ..lower.graph_to_tensor import lower
    out = []
    if "xf" in groups:
        for blk, grp in (("mlp_block", "xf_mlp"), ("attention", "xf_attention"), ("mm1x8x8", "xf_mm")):
            prog, vals = _xf(blk)
            for k in prog.kernels:
                out.append(KernelSpec(f"{grp}.{k.name}", grp, prog, k.name, vals))
    if "model" in groups and MODEL_GRAPH.exists():
        prog, vals = _model_program()
        for n in MODEL_KERNELS:
            out.append(KernelSpec(f"model.{n}", "model", prog, n, vals))
    if "syn" in groups:
        for name, (module, consts, ins) in _synthetic().items():
            prog = lower(module)
            try:
                vals = prog.run(ins, consts)
            except OverflowError:
                continue
            out.append(KernelSpec(f"syn.{name}", "syn", prog, prog.kernels[0].name, vals))
    return out


# ------------------------------------------------------------------------------------------------ build

_SUITE = None


def _sample(spec: KernelSpec, W: int, n: int, seed: str):
    """Distinct valid programs for one kernel at WARPS_PER_CORE=W: (builts, stats)."""
    from ..runtime import GpuParams
    from ..tune import Candidate, Invalid, Task, build, sketches_for
    params = GpuParams(WARPS_PER_CORE=W)
    task = Task.from_program(spec.program, spec.name, spec.values, params)
    sks = {sk.name: sk for sk in sketches_for(task.kernel, task.target, task.params)}
    names = sorted(sks)
    weights = [1 if x == "tpl" else 4 for x in names]
    rng = random.Random(seed)
    cands = list(sks["tpl"].seeds())
    seen, progs, builts = set(), set(), []
    invalid = attempts = 0
    while len(builts) < n and attempts < 40 * n:
        if cands:
            c = cands.pop(0)
        else:
            sn = rng.choices(names, weights)[0]
            c = Candidate.make(sn, **{k: rng.choice(v) for k, v in sks[sn].domain.items()})
        if c.key in seen:
            attempts += 1
            continue
        seen.add(c.key)
        attempts += 1
        b = build(task, c)
        if isinstance(b, Invalid):
            invalid += 1
            continue
        pk = json.dumps([[ln.program for ln in b.job.launches], [ln.threads for ln in b.job.launches]])
        if pk in progs:
            continue
        progs.add(pk)
        builts.append(b)
    return task, builts, {"attempts": attempts, "invalid": invalid}


def _work(item):
    """One (kernel, W class): sample, verify, profile, time at every config of the class -> (programs, rows)."""
    ki, W, n = item
    from ..runtime import run
    from ..tune.measure import FLAGS, verdict
    from .analytic import closed_form, predict_profile
    from .features import program_features
    from .profile import profile
    spec = _SUITE[ki]
    t0 = time.time()
    task, builts, st = _sample(spec, W, n, f"anvil-cost/{spec.kid}/{W}")
    cids = classes()[W]
    programs, rows = [], []
    for j, b in enumerate(builts):
        pid = f"{spec.kid}|w{W}|{j}"
        ri = run(b.job, "isasim")
        flags = {k: len(ri.flags.get(k, [])) for k in FLAGS}
        ok, why = verdict(task, b, ri.memory)
        if any(flags.values()):
            ok, why = False, f"isasim flags {flags}"
        prec = {"pid": pid, "kid": spec.kid, "W": W, "candidate": b.cand.to_dict(), "stats": b.features(),
                "ok": ok, "why": why}
        if ok:
            tp = time.time()
            prof = profile(b.job)
            t_prof = time.time() - tp
            if prof.memory[:len(b.job.data)] != list(ri.memory[:len(b.job.data)]):
                prec.update(ok=False, why="profile memory differs from isasim")
            prec["pf"] = program_features(prof, b.features())
        programs.append(prec)
        if not prec["ok"]:
            continue
        for cid in cids:
            params, lat = config_params(cid)
            job = b.job
            job.params, job.mem_latency = params, lat
            ta = time.time()
            t1 = predict_profile(prof, params, lat)
            tm = time.time()
            cf = closed_form(prof, params, lat)
            tb = time.time()
            try:
                rv = run(job, "verilator")
                cycles = sum(lr.cycles for lr in rv.launches)
                lc = [lr.cycles for lr in rv.launches]
                cs = {}
                for lr in rv.launches:
                    for c_, v in lr.counters.items():
                        cs[c_] = cs.get(c_, 0) + v
                mem_ok = list(rv.memory[:len(job.data)]) == list(ri.memory[:len(job.data)])   # whole job (A1-M1)
                err = "" if mem_ok else "verilator memory differs from isasim"
            except Exception as e:                     # timeouts etc. are recorded as failures
                cycles, lc, cs, err = None, None, {}, f"{type(e).__name__}: {str(e)[:200]}"
            tc = time.time()
            rows.append({"pid": pid, "cid": cid, "cycles": cycles if not err else None, "launch_cycles": lc,
                         "t1": t1.cycles, "t1_launch": t1.launch_cycles, "t1_bd": t1.breakdown, "closed": cf,
                         "counters": cs, "err": err,
                         # wall time of a prediction: t_model = the headline model's inputs (functional profile of
                         # the program + closed form; tier2_static), t_tier1 = the event model, t_sim = Verilator
                         "t_model": round(t_prof + tb - tm, 4), "t_tier1": round(tm - ta, 4),
                         "t_sim": round(tc - tb, 4)})
    return spec.kid, W, programs, rows, {**st, "seconds": round(time.time() - t0, 1), "n": len(builts)}


def kernel_class(spec: KernelSpec) -> str:
    """Equivalence class of a kernel for the split: op kind and the shapes it reads and writes (constants such as
    requant M, s and LUT contents aside). Two kernels of one class compile to programs of the same shape."""
    k = spec.kernel
    bufs = spec.program.buffers
    shp = lambda syms: [list(bufs[x][0]) if x in bufs else None for x in syms]   # noqa: E731
    return json.dumps([k.op, shp(k.reads), shp(k.writes)])


def program_key(p) -> str:
    """Identity of a program for the cost model: its whole profile-feature vector (footprint aside), warps per
    core and size (AUDIT-A2 M3's twin test)."""
    pf = {k: v for k, v in p["pf"].items() if k != "footprint"}
    return json.dumps([pf, p["W"], p["stats"].get("words")], sort_keys=True)


def dedupe_split(kernels_meta, programs, rows):
    """Make the held-out kernels held out (AUDIT-A2 M3): (1) a held-out kernel whose class (op + shapes, see
    kernel_class) has a training kernel is reassigned to train; (2) a held-out program whose features equal a
    training program's is dropped with its rows. Returns (programs, rows, record of what changed)."""
    train_cls = {}
    for kid, m in kernels_meta.items():
        if m["split"] == "train":
            train_cls.setdefault(m["class"], kid)
    reassigned = {}
    for kid, m in kernels_meta.items():
        if m["split"] == "test" and m["class"] in train_cls:
            m["split"] = "train"
            reassigned[kid] = train_cls[m["class"]]
    train_keys = {program_key(p) for p in programs if p["ok"] and kernels_meta[p["kid"]]["split"] == "train"}
    drop = {p["pid"] for p in programs if p["ok"] and kernels_meta[p["kid"]]["split"] == "test"
            and program_key(p) in train_keys}
    by_kernel = {}
    for p in programs:
        if p["pid"] in drop:
            by_kernel[p["kid"]] = by_kernel.get(p["kid"], 0) + 1
    programs = [p for p in programs if p["pid"] not in drop]
    rows = [r for r in rows if r["pid"] not in drop]
    return programs, rows, {"reassigned": reassigned, "dropped_programs": len(drop),
                            "dropped_by_kernel": dict(sorted(by_kernel.items()))}


def _prebuild(cids):
    """Build the Verilator model of every configuration once, serially (the pool then only reads the cache)."""
    from ..runtime import Job, Launch, run
    for cid in cids:
        params, lat = config_params(cid)
        run(Job([0] * 16, [Launch([0xF000], 1)], params, lat), "verilator")


def build_dataset(out=DATA, workers=4, kernels=None, n_cand=None, log=print):
    global _SUITE
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _SUITE = suite()
    if kernels:
        _SUITE = [s for s in _SUITE if s.kid in kernels]
    _prebuild(list(CONFIGS))
    items = []
    for ki, spec in enumerate(_SUITE):
        for W in sorted(classes()):
            n = n_candidates(spec.kid, W)
            if n_cand:
                n = min(n, n_cand)
            items.append((ki, W, n))
    items.sort(key=lambda it: -(it[2] * (3 if _SUITE[it[0]].group == "model" else 1)))   # long jobs first
    programs, rows, info = [], [], {}
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for i, (kid, W, ps, rs, st) in enumerate(pool.imap_unordered(_work, items)):
            programs += ps
            rows += rs
            info[f"{kid}|w{W}"] = st
            log(f"[{i + 1}/{len(items)}] {kid} W={W}: {st['n']} programs, {len(rs)} rows, {st['seconds']} s "
                f"({time.time() - t0:.0f} s total)")
    programs.sort(key=lambda p: p["pid"])
    rows.sort(key=lambda r: (r["pid"], r["cid"]))
    kernels_meta = {}
    for s in _SUITE:
        k = s.kernel
        kernels_meta[s.kid] = {"group": s.group, "op": k.op, "name": s.name, "class": kernel_class(s),
                               "split": "test" if s.kid in HELDOUT_KERNELS else "train"}
    programs, rows, dedup = dedupe_split(kernels_meta, programs, rows)
    log(f"dedupe: {len(dedup['reassigned'])} held-out kernels reassigned to train (same op and shapes as a train "
        f"kernel), {dedup['dropped_programs']} held-out programs dropped (feature-identical to a train program)")
    (out / "kernels.json").write_text(json.dumps(kernels_meta, indent=1, sort_keys=True) + "\n")
    (out / "configs.json").write_text(json.dumps(
        {c: {"params": o, "mem_latency": lat, "split": "test" if c in HELDOUT_CONFIGS else "train"}
         for c, (o, lat) in CONFIGS.items()}, indent=1, sort_keys=True) + "\n")
    held = [k for k in HELDOUT_KERNELS if kernels_meta.get(k, {}).get("split") == "test"]
    (out / "split.json").write_text(json.dumps({"heldout_kernels": held, "heldout_configs": list(HELDOUT_CONFIGS),
                                                "dedupe": dedup}, indent=1) + "\n")
    with gzip.open(out / "programs.jsonl.gz", "wt") as f:
        for p in programs:
            f.write(json.dumps(p, sort_keys=True) + "\n")
    with gzip.open(out / "rows.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    (out / "build-info.json").write_text(json.dumps({"seconds": round(time.time() - t0), "jobs": info},
                                                    indent=1, sort_keys=True) + "\n")
    return programs, rows


# ------------------------------------------------------------------------------------------------ load

@dataclass
class Dataset:
    kernels: dict
    configs: dict
    programs: dict           # pid -> record
    rows: list
    split: dict

    def is_test_kernel(self, kid):
        return kid in self.split["heldout_kernels"] and self.kernels.get(kid, {}).get("split", "test") == "test"

    def is_test_config(self, cid):
        return cid in self.split["heldout_configs"]

    def params(self, cid):
        from ..runtime import GpuParams
        c = self.configs[cid]
        return GpuParams(**c["params"]), c["mem_latency"]

    def ok_rows(self):
        return [r for r in self.rows if r["cycles"] is not None]

    def select(self, kernels="train", configs="train"):
        """Rows (measured) of a split: kernels/configs in "train" | "test" | "all"."""
        def keep(flag, test):
            return flag == "all" or (flag == "test") == test
        out = []
        for r in self.ok_rows():
            kid = self.programs[r["pid"]]["kid"]
            if keep(kernels, self.is_test_kernel(kid)) and keep(configs, self.is_test_config(r["cid"])):
                out.append(r)
        return out


def load(path=DATA) -> Dataset:
    path = Path(path)
    kernels = json.loads((path / "kernels.json").read_text())
    configs = json.loads((path / "configs.json").read_text())
    split = json.loads((path / "split.json").read_text())
    with gzip.open(path / "programs.jsonl.gz", "rt") as f:
        programs = {p["pid"]: p for p in map(json.loads, f)}
    with gzip.open(path / "rows.jsonl.gz", "rt") as f:
        rows = [json.loads(ln) for ln in f]
    return Dataset(kernels, configs, programs, rows, split)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model-from-torch", action="store_true", help="re-capture data/cost/model_graph.json.gz")
    ap.add_argument("--kernels", nargs="*")
    ap.add_argument("--n-cand", type=int)
    a = ap.parse_args(argv)
    if a.model_from_torch or not MODEL_GRAPH.exists():
        save_model_graph()
    ps, rs = build_dataset(a.out, a.workers, a.kernels, a.n_cand)
    ok = sum(r["cycles"] is not None for r in rs)
    print(f"{len(ps)} programs, {len(rs)} rows ({ok} measured)")


if __name__ == "__main__":
    main()
