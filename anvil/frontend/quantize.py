"""SG4a quantization: FGraph (capture.py) + calibration inputs -> anvil.graph IR, as xf/export.py quantizes.

    q = quantize(fgraph, calib, headroom=1.5, names={"b.ln_2": "h", ...})
    q.module, q.consts                        # anvil.graph module + {const sym: int array}
    q.scales, q.bounds                        # {value sym: scale}, overflow proof {op sym: worst |numerator|}
    q.reference(x_float)                      # float activations keyed by IR sym (for C4 level 6)

Recipe (xf/export.py, contract 8): symmetric per-tensor int8. Input scales are max-abs / 127 over the
calibration inputs; every other activation scale is `headroom` * max-abs / 127 of its float value on the
fake-quantized calibration inputs (scores only where a causal softmax keeps them). Weights: max-abs / 127 per
tensor (per chunk for a split linear). Every requant (M, s) comes from `fixed_point` against the worst case over
all int8 inputs, LUTs (rsqrt, SiLU, exp) are built as xf builds them, and `prove_no_overflow` re-derives every
int32 bound from the finished graph and raises OverflowError naming the op if any can overflow.

Fusions: mul(silu(a), b) -> lut_mul; a scalar factor on a matmul folds into its multiplier; transpose keeps its
input's scale; softmax output has scale 1/one.

Multi-head (SG5a, ADR-003): split_heads -> reshape '<sym>.r' (T, nh, hd) + transpose '<sym>' (nh, T, hd);
merge_heads -> transpose '<sym>.t' (T, nh, hd) + reshape '<sym>' (T, C). Scores and p @ v are head-batched
qmatmuls; every value keeps one per-tensor scale across heads.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .. import semantics as sem
from ..ir import graph as g

INT32_MAX = sem.I32[1]
RSQRT_LEN = 1024
SOFTMAX_ONE = 127
HEADROOM = 1.5


# ------------------------------------------------------------------------------------ numeric primitives

def fixed_point(ms, bounds, k=0, *, max_shift=31):
    """(M_i, s): the largest s in [0, max_shift - k] with M_i = round(m_i * 2^s) and
    sum(bound_i * M_i) + 2^(s+k-1) <= INT32_MAX, i.e. no requant numerator acc_i*M_i (+ round) can overflow
    when |acc_i| <= bound_i. k is an extra shift the op adds at run time (rmsnorm_int's normalization steps).
    Equals xf.export.fixed_point for max_shift = 31; raises ValueError if a multiplier vanishes at the
    chosen shift and OverflowError if no shift fits."""
    if len(ms) != len(bounds) or not ms:
        raise ValueError("fixed_point needs one bound per multiplier")
    if any(m <= 0 for m in ms) or any(b < 0 for b in bounds):
        raise ValueError(f"multipliers must be positive and bounds non-negative: {ms}, {bounds}")
    for s in range(max_shift - k, -1, -1):
        mi = [round(m * (1 << s)) for m in ms]
        if sum(int(b) * m for b, m in zip(bounds, mi)) + ((1 << (s + k - 1)) if s + k else 0) <= INT32_MAX:
            if min(mi) < 1:
                raise ValueError(f"multipliers {ms} vanish at shift {s}")
            return mi, s
    raise OverflowError(f"bounds {bounds} leave no room for multipliers {ms}")


def maxabs_scale(x) -> float:
    return float(np.abs(np.asarray(x)).max()) / 127


def quantize_array(x, scale):
    return sem.clamp_i8(np.rint(np.asarray(x) / scale)).astype(np.int64)


def rsqrt_kmax(d, n=RSQRT_LEN, mag=128):
    """Normalization steps rmsnorm_int takes on the loudest row of d entries of magnitude `mag`."""
    return int(sem.rsqrt_index(d * mag * mag, n)[1])


def rmsnorm_lut(sx, sw, sh, wmax, d, eps, n=RSQRT_LEN):
    """(rsqrt_lut, s) for rmsnorm_int from x (scale sx), weight (scale sw, max |w_q| wmax) to h (scale sh):
    lut[i] = round(rsqrt(sx^2 i / d + eps) * sx sw / sh * 2^s), lut[0] = 0 (only x = 0 meets it)."""
    ss = np.arange(1, n)
    r = np.r_[0, 1 / np.sqrt(sx ** 2 * ss / d + eps) * sx * sw / sh]
    (rmax,), s = fixed_point([r.max()], [128 * wmax], rsqrt_kmax(d, n))
    lut = np.rint(r * (1 << s)).astype(np.int64)
    assert lut.max() == rmax
    return lut, s


# ----------------------------------------------------------------------- wide (i16) rmsnorm input (ADR-004)

def ss_shift_for(d, mag):
    """Least c with d * ((mag^2 + 2^(c-1)) >> c) <= INT32_MAX: the rounded shift rmsnorm applies to each square
    of a wide row so the row sum stays in int32 (0 for int8 rows at any d <= 2^17)."""
    for c in range(31):
        if d * ((mag * mag + _rnd(c)) >> c) <= INT32_MAX:
            return c
    raise OverflowError(f"no square shift fits {d} entries of magnitude {mag}")


def _xmax(ss_max, c, mag):
    """Largest |x| <= mag with (x^2 + 2^(c-1)) >> c <= ss_max (the loudest entry a row with that sum can hold)."""
    return min(mag, math.isqrt(max(0, (ss_max + 1) * (1 << c) - _rnd(c) - 1)))


def rms_xmax_table(d, n, c, mag, e=0):
    """{k: array over LUT index i of the largest |x| any row with rsqrt_index(ss) = (i, k) can contain}, for every
    k a row of d entries of magnitude <= mag can reach (ss = e + the row's shifted squares). k = 0: ss = i
    exactly; k >= 1: ss < (i + 1) 4^k, i >= n/4. Each entry's shifted square is <= ss - e. This is the tight form
    of the rmsnorm bound: |x * r[i]| is largest at large k, not at the LUT's largest entry."""
    ss_top = d * ((mag * mag + _rnd(c)) >> c) + e
    kmax = int(sem.rsqrt_index(ss_top, n)[1])
    out = {}
    for k in range(kmax + 1):
        hi = np.minimum((np.arange(n, dtype=object) + 1) * (4 ** k) - 1, ss_top)
        xm = np.array([_xmax(int(v) - e, c, mag) if v >= e else 0 for v in hi], dtype=np.int64)
        if k:
            xm[: n // 4] = 0                     # unreachable: idx >= n/4 once a row is shifted
        out[k] = xm
    return out


def rms_wide_bounds(lut, wmax, M, s, xm_table):
    """{k: worst x*w*r*M + round(s + k)} over the reachable (idx, k) of a wide rmsnorm."""
    lut = np.asarray(lut, dtype=np.int64)
    return {k: int((xm * wmax * lut).max()) * M + _rnd(s + k) for k, xm in xm_table.items()}


def rmsnorm_lut_wide(sx, sw, sh, wmax, d, eps, mag, n=RSQRT_LEN):
    """(rsqrt_lut, s, c, e) for rmsnorm_int on a wide (i16) row. The mean square is sx^2 2^c ss / d with
    ss = e + sum((x^2 + 2^(c-1)) >> c): c keeps the sum in int32 and e = round(eps d / (sx^2 2^c)) (>= 1) folds
    eps into the sum, so the LUT holds the eps-free rsqrt(sx^2 2^c i / d) sx sw / sh, exact at every shift k
    (rmsnorm_lut's eps-in-the-LUT form is exact only at k = 0: at k it adds 4^k eps, which matters once the
    input scale is small). s is the largest shift whose LUT keeps every reachable x*w*r (+ round) in int32
    under the tight bound (rms_xmax_table)."""
    c = ss_shift_for(d, mag)
    unit = sx ** 2 * (1 << c) / d                 # mean square per unit of ss
    e = max(1, int(round(eps / unit)))
    if d * ((mag * mag + _rnd(c)) >> c) + e > INT32_MAX:
        raise OverflowError(f"eps folds to {e}: the rmsnorm sum of squares cannot stay in int32 at scale {sx}")
    ss = np.arange(1, n)
    r = np.r_[0, 1 / np.sqrt(unit * ss) * sx * sw / sh]
    xm = rms_xmax_table(d, n, c, mag, e)
    kmax = max(xm)
    for s in range(31 - kmax, -1, -1):
        lut = np.rint(r * (1 << s)).astype(np.int64)
        if max(rms_wide_bounds(lut, wmax, 1, s, xm).values()) <= INT32_MAX:
            if lut[n // 4:].min() < 64:
                raise ValueError(f"rsqrt LUT too coarse at shift {s} (min entry {lut[n // 4:].min()})")
            return lut, s, c, e
    raise OverflowError("no rsqrt LUT shift fits a wide rmsnorm")


def silu_lut(s_in):
    """(lut, scale): SiLU of every int8 input at scale s_in, quantized at its own max-abs scale."""
    x = np.arange(-128, 128) * s_in
    y = x / (1 + np.exp(-x))
    s = maxabs_scale(y)
    return quantize_array(y, s), s


def exp_bits(t, one=SOFTMAX_ONE):
    """Largest E with every softmax_lut intermediate in int32 for rows of t entries <= 2^E."""
    return max(e for e in range(31) if (one + t // 2) << e <= INT32_MAX and t << e <= INT32_MAX)


def exp_lut(score_scale, t, one=SOFTMAX_ONE):
    """exp_lut[d + 255] = round(exp(d * score_scale) * 2^E), d in [-255, 0]."""
    return np.rint(np.exp(np.arange(-255, 1) * score_scale) * (1 << exp_bits(t, one))).astype(np.int64)


# ----------------------------------------------------------------------------------------- overflow proof

def _rnd(s):
    return (1 << (s - 1)) if s else 0


def _ival(attr):
    return int(attr.value.data)


def prove_no_overflow(module, consts=None) -> dict:
    """Generalized xf check_bounds over an anvil.graph module: propagate the magnitude bound of every value
    (int8 inputs 128, softmax outputs `one`, LUT outputs max|table|) and bound every int32 intermediate of
    every op over all int8 inputs. Returns {op sym: worst |intermediate|}; raises OverflowError naming every
    op whose bound exceeds INT32_MAX."""
    from ..interp.graph import _func

    consts = consts or {}
    mag, cval, worst, bad = {}, {}, {}, []
    for op in _func(module).body.block.ops:
        if isinstance(op, g.OutputOp):
            continue
        r = op.results[0]
        name = op.sym_str or op.name
        checks = {}
        if isinstance(op, g.InputOp):
            m = {"i8": 128, "i16": 1 << 15}.get(g.qt(r).dtype, 1 << 31)
        elif isinstance(op, g.ConstOp):
            v = (np.asarray(op.value.get_values(), dtype=np.int64).reshape(g.qt(r).dims) if op.value is not None
                 else np.asarray(consts[op.sym_str], dtype=np.int64))
            cval[r] = v
            m = int(np.abs(v).max()) if v.size else 0
        elif isinstance(op, (g.QMatmulOp, g.QLinearOp)):
            M, s = _ival(op.M), _ival(op.s)
            if op.b in cval:
                acc = mag[op.a] * int(np.abs(cval[op.b]).sum(axis=0).max())
            else:
                acc = g.qt(op.a).dims[-1] * mag[op.a] * mag[op.b]
            checks = {"acc": acc, "acc*M + round": acc * M + _rnd(s)}
            m = mag[op.lut] if op.lut is not None else (128 if g.qt(r).dtype == "i8" else 1 << 15)
        elif isinstance(op, g.LutOp):
            m = mag[op.table]
        elif isinstance(op, g.LutMulOp):
            M, s = _ival(op.M), _ival(op.s)
            prod = mag[op.table] * mag[op.y]
            checks = {"lut*y": prod, "lut*y*M + round": prod * M + _rnd(s)}
            m = 128
        elif isinstance(op, g.RmsNormIntOp):
            M, s = _ival(op.M), _ival(op.s)
            d, n = g.qt(op.x).dims[-1], g.qt(op.lut).dims[0]
            c, e = op.ss_shift_int, op.ss_init_int
            if g.qt(op.x).dtype == "i8" and c == 0 and e == 0:      # xf's bound (contract 8), loose in x*r
                kmax = rsqrt_kmax(d, n, mag[op.x])
                xwr = mag[op.x] * mag[op.w] * mag[op.lut]
                checks = {"sum x^2": d * mag[op.x] ** 2, "x*w*r": xwr, "x*w*r*M + round": xwr * M + _rnd(s + kmax)}
            else:                                        # wide row (ADR-004): tight bound per (idx, k)
                if op.lut not in cval:
                    raise ValueError(f"{name}: the wide rmsnorm bound needs the rsqrt LUT's values")
                xm = rms_xmax_table(d, n, c, mag[op.x], e)
                kmax = max(xm)
                lut = cval[op.lut]
                xwr = max(int((v * mag[op.w] * lut).max()) for v in xm.values())
                checks = {"x^2 + round": mag[op.x] ** 2 + _rnd(c),
                          "sum x^2": d * ((mag[op.x] ** 2 + _rnd(c)) >> c) + e, "x*w": mag[op.x] * mag[op.w],
                          "x*w*r": xwr, "x*w*r*M + round": max(rms_wide_bounds(lut, mag[op.w], M, s, xm).values())}
            if s + kmax > 31:
                bad.append(f"{name}: shift s + kmax = {s + kmax} > 31")
            m = 128
        elif isinstance(op, g.SoftmaxLutOp):
            one, emax, t = _ival(op.one), mag[op.lut], g.qt(op.x).dims[-1]
            checks = {"exp sum": t * emax, "e*one + S/2": emax * one + t * emax // 2}
            m = one
        elif isinstance(op, g.AddRequantOp):
            Ma, Mb, s = _ival(op.Ma), _ival(op.Mb), _ival(op.s)
            checks = {"a*Ma + b*Mb + round": mag[op.a] * Ma + mag[op.b] * Mb + _rnd(s)}
            m = 128 if g.qt(r).dtype == "i8" else 1 << 15
        elif isinstance(op, (g.TransposeOp, g.ReshapeOp)):
            m = mag[op.x]
            if op.x in cval:
                cval[r] = cval[op.x]
        else:
            raise NotImplementedError(f"prove_no_overflow: {op.name}")
        mag[r] = m
        if checks:
            worst[name] = max(checks.values())
            bad += [f"{name}: {k} reaches {v} > {INT32_MAX}" for k, v in checks.items() if v > INT32_MAX]
    if bad:
        raise OverflowError("int32 overflow possible: " + "; ".join(bad))
    return worst


# ----------------------------------------------------------------------------------------------- quantize

@dataclass
class Quantized:
    module: object                                  # builtin.module with one anvil.graph.func
    consts: dict                                    # const sym -> int64 array (empty when inlined)
    scales: dict                                    # value sym -> scale (None for i32 LUTs)
    bounds: dict                                    # prove_no_overflow result
    graph: object                                   # the FGraph
    syms: dict = field(default_factory=dict)        # FGraph node name -> IR sym

    def reference(self, *inputs) -> dict:
        """Float activations {IR sym: numpy array} of the captured graph on batched float inputs."""
        acts = self.graph.evaluate(list(inputs))
        return {self.syms[k]: v.detach().numpy() for k, v in acts.items() if k in self.syms}


CALIB_CHUNK = 8192


def _calibrate(fg, calib_inputs, chunk=CALIB_CHUNK, wide=()):
    """-> (input scales, {node name: max |float activation|}) on the fake-quantized calibration inputs,
    evaluated `chunk` samples at a time (max-abs is chunk-invariant, so the result does not depend on it).
    Inputs named in `wide` are i16 (scale max-abs / 32767)."""
    import torch

    if not isinstance(calib_inputs, (list, tuple)):
        calib_inputs = [calib_inputs]
    if len(calib_inputs) != len(fg.inputs):
        raise ValueError(f"{len(calib_inputs)} calibration arrays for {len(fg.inputs)} graph inputs")
    sin, deq = {}, []
    for n, c in zip(fg.inputs, calib_inputs):
        c = c.detach().numpy() if isinstance(c, torch.Tensor) else np.asarray(c)
        if tuple(c.shape[1:]) != tuple(n.shape):
            raise ValueError(f"calibration input {n.name}: shape {c.shape[1:]} != {n.shape} (plus a batch dim)")
        qmax = QMAX["i16" if n.name in wide else "i8"]
        sin[n.name] = float(np.abs(c).max()) / qmax
        deq.append(np.clip(np.rint(c / sin[n.name]), -qmax - 1, qmax) * sin[n.name])
    amax = {}
    for lo in range(0, len(deq[0]), chunk):
        with torch.no_grad():
            acts = fg.evaluate([d[lo:lo + chunk] for d in deq])
        for k, v in _act_maxabs(fg, acts).items():
            amax[k] = max(amax.get(k, 0.0), v)
    return sin, amax


def _act_maxabs(fg, acts):
    """max |activation| per node; scores feeding a causal softmax only where the mask keeps them."""
    import torch

    out = {}
    for n in fg.nodes:
        v = acts[n.name]
        if any(u.op == "softmax" and u.attrs["causal"] for u in fg.users(n)):
            keep = torch.ones(v.shape[-2], v.shape[-1], dtype=torch.bool).tril()
            v = v.masked_fill(~keep, 0)
        out[n.name] = float(v.abs().max())
    return out


QMAX = {"i8": 127, "i16": (1 << 15) - 1}


def residual_nodes(fg, branches=True) -> set:
    """Node names of the residual stream: graph inputs and adds whose every user is an rmsnorm or an add (an add
    may also be a graph output) and, with `branches`, the linears whose every user is such an add (the branch
    outputs entering the stream: attention c_proj, MLP down). These are the values `residual="i16"` widens
    (ADR-004)."""
    outs = {o.name for o in fg.outputs}
    wide = set()
    for n in fg.nodes:
        us = fg.users(n)
        if n.op in ("input", "add") and (us or n.name in outs) and all(u.op in ("rmsnorm", "add") for u in us) \
                and (n.op == "add" or n.name not in outs):
            wide.add(n.name)
    if branches:
        wide |= {n.name for n in fg.nodes if n.op == "linear" and n.name not in outs and fg.users(n)
                 and all(u.name in wide and u.op == "add" for u in fg.users(n))}
    return wide


def quantize(fg, calib_inputs, headroom=HEADROOM, *, names=None, softmax_one=SOFTMAX_ONE, rsqrt_len=RSQRT_LEN,
             inline=False, sym_name="graph", residual="i8", residual_branches=True) -> Quantized:
    """Calibrate `fg` on `calib_inputs` (one float array per graph input, leading batch dim) and lower it to
    anvil.graph. `names` renames FGraph nodes to IR syms (default: the node names); const syms are
    '<sym>.w', '<sym>.rsqrt_lut', '<sym>.silu_lut', '<sym>.exp_lut'. residual="i16" stores the residual stream
    (residual_nodes: graph inputs and residual adds feeding only rmsnorm / add) as i16 at max-abs / 32767
    (ADR-004), with the branch linears feeding it unless residual_branches=False; the default "i8" is contract 8's
    recipe, unchanged."""
    names = dict(names or {})
    unknown = set(names) - {n.name for n in fg.nodes}
    if unknown:
        raise KeyError(f"names for unknown nodes: {sorted(unknown)}")
    sym = {n.name: names.get(n.name, n.name) for n in fg.nodes}
    if len(set(sym.values())) != len(sym):
        raise ValueError("names map two nodes to one sym")
    if residual not in QMAX:
        raise ValueError(f"residual dtype {residual!r} not in {sorted(QMAX)}")
    wide = residual_nodes(fg, residual_branches) if residual != "i8" else set()
    dt = {n: ("i16" if n in wide else "i8") for n in sym}
    sin, amax = _calibrate(fg, calib_inputs, wide=wide)

    users = {n.name: fg.users(n) for n in fg.nodes}
    fused_silu = {n.name for n in fg.nodes if n.op == "silu" and users[n.name]
                  and all(u.op == "mul" for u in users[n.name]) and n not in fg.outputs}
    b = g.GraphBuilder(sym_name)
    consts, scales, ir, mag, soft = {}, {}, {}, {}, set()

    def const(name, data, dtype="i8", scale=None):
        data = np.asarray(data, dtype=np.int64)
        if not inline:
            consts[name] = data
        scales[name] = scale
        return b.const(name, list(data.shape), dtype, scale, value=data if inline else None)

    def act_scale(n):
        return headroom * (amax[n.name] / QMAX[dt[n.name]])

    def weight(w_io):
        sw = maxabs_scale(w_io)
        return quantize_array(w_io, sw), sw

    for n in fg.nodes:
        s_name = sym[n.name]
        ins = [ir.get(i.name) for i in n.inputs]
        sc_in = [scales.get(sym[i.name]) for i in n.inputs]
        if n.op == "input":
            so = sin[n.name]
            v = b.input(s_name, list(n.shape), dt[n.name], so)
        elif n.op == "rmsnorm":
            so = act_scale(n)
            d = n.shape[-1]
            wf = n.attrs["weight"] if n.attrs["weight"] is not None else np.ones(d)
            wq, sw = weight(wf)
            wmax = int(np.abs(wq).max())
            if dt[n.inputs[0].name] == "i8":
                lut, s = rmsnorm_lut(sc_in[0], sw, so, wmax, d, n.attrs["eps"], rsqrt_len)
                c = e = 0
            else:
                lut, s, c, e = rmsnorm_lut_wide(sc_in[0], sw, so, wmax, d, n.attrs["eps"], mag[n.inputs[0].name],
                                             rsqrt_len)
            v = b.rmsnorm_int(s_name, ins[0], const(f"{s_name}.w", wq, "i8", sw),
                              const(f"{s_name}.rsqrt_lut", lut, "i32"), 1, s, so, ss_shift=c,
                              ss_init=e)
        elif n.op == "linear":
            so = act_scale(n)
            w = n.attrs["weight"]
            if n.attrs["rows"] is not None:
                w = w[n.attrs["rows"][0]:n.attrs["rows"][1]]
            wq, sw = weight(w.T)
            bound = mag[n.inputs[0].name] * int(np.abs(wq).sum(axis=0).max())
            (m,), s = fixed_point([sc_in[0] * sw / so], [bound])
            v = b.qlinear(s_name, ins[0], const(f"{s_name}.w", wq, "i8", sw), m, s, so, dtype=dt[n.name])
        elif n.op == "matmul":
            so = act_scale(n)
            a_, b_ = n.inputs
            if a_.name in soft and b_.name in soft:
                raise NotImplementedError(f"{n.name}: product of two softmax outputs")
            if a_.name in soft:
                base = sc_in[1] / softmax_one
            elif b_.name in soft:
                base = sc_in[0] / softmax_one
            else:
                base = sc_in[0] * sc_in[1]
            f = n.attrs["factor"]
            if f is None:
                mult = base / so
            elif f[0] == "inv_sqrt":
                mult = base / math.sqrt(f[1]) / so
            else:
                mult = base * f[1] / so
            if mult <= 0:
                raise NotImplementedError(f"{n.name}: non-positive matmul scale factor")
            bound = n.inputs[0].shape[-1] * mag[a_.name] * mag[b_.name]
            (m,), s = fixed_point([mult], [bound])
            v = b.qmatmul(s_name, ins[0], ins[1], m, s, so)
        elif n.op == "transpose":
            so = sc_in[0]
            v = b.transpose(s_name, ins[0], n.attrs["perm"], so)
        elif n.op == "reshape":
            so = sc_in[0]
            v = b.reshape(s_name, ins[0], list(n.shape), so)
        elif n.op == "softmax":
            so = 1 / softmax_one
            t = n.shape[-1]
            lut = exp_lut(sc_in[0], t, softmax_one)
            v = b.softmax_lut(s_name, ins[0], const(f"{s_name}.exp_lut", lut, "i32"), softmax_one,
                              n.attrs["causal"], so)
            soft.add(n.name)
        elif n.op == "silu":
            if n.name in fused_silu:
                continue
            so = act_scale(n)
            x = np.arange(-128, 128) * sc_in[0]
            lut = quantize_array(x / (1 + np.exp(-x)), so)
            v = b.lut(s_name, ins[0], const(f"{s_name}.silu_lut", lut, "i8", so), so)
        elif n.op == "mul":
            a_, b_ = n.inputs
            if a_.name in fused_silu:
                gate, up = a_, b_
            elif b_.name in fused_silu:
                gate, up = b_, a_
            else:
                raise NotImplementedError(f"{n.name}: anvil.graph has no elementwise product of two activations "
                                          "(only silu(a) * b, as lut_mul)")
            so = act_scale(n)
            s_gate, s_up = scales[sym[gate.inputs[0].name]], scales[sym[up.name]]
            lut, s_silu = silu_lut(s_gate)
            (m,), s = fixed_point([s_silu * s_up / so], [mag[up.name] * int(np.abs(lut).max())])
            v = b.lut_mul(s_name, ir[gate.inputs[0].name], ir[up.name],
                          const(f"{s_name}.silu_lut", lut, "i8", s_silu), m, s, so)
        elif n.op == "add":
            so = act_scale(n)
            (ma, mb), s = fixed_point([sc_in[0] / so, sc_in[1] / so], [mag[n.inputs[0].name], mag[n.inputs[1].name]])
            v = b.add_requant(s_name, ins[0], ins[1], ma, mb, s, so, dtype=dt[n.name])
        elif n.op == "split_heads":                  # (T, C) -> view (T, nh, hd) -> transpose (nh, T, hd)
            so, nh = sc_in[0], n.attrs["nh"]
            T_, C = n.inputs[0].shape[-2:]
            r = b.reshape(f"{s_name}.r", ins[0], [*n.inputs[0].shape[:-1], nh, C // nh], so)
            rank = len(n.inputs[0].shape) + 1
            v = b.transpose(s_name, r, list(range(rank - 3)) + [rank - 2, rank - 3, rank - 1], so)
        elif n.op == "merge_heads":                  # (nh, T, hd) -> transpose (T, nh, hd) -> view (T, C)
            so, rank = sc_in[0], len(n.inputs[0].shape)
            tr = b.transpose(f"{s_name}.t", ins[0], list(range(rank - 3)) + [rank - 2, rank - 3, rank - 1], so)
            v = b.reshape(s_name, tr, list(n.shape), so)
        else:
            raise NotImplementedError(f"{n.name}: frontend op {n.op}")
        ir[n.name] = v
        scales[s_name] = so
        mag[n.name] = softmax_one if n.op == "softmax" else (mag[n.inputs[0].name] if n.op in
                                                              ("transpose", "reshape", "split_heads",
                                                               "merge_heads") else QMAX[dt[n.name]] + 1)
    b.output(*(ir[o.name] for o in fg.outputs))
    module = b.module()
    bounds = prove_no_overflow(module, consts)
    syms = {k: v for k, v in sym.items() if k in ir}
    return Quantized(module, consts, scales, bounds, fg, syms)
