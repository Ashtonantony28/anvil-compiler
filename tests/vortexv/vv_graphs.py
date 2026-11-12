"""Graphs for the Vortex-V backend tests: the xf fixture blocks and seeded random transformer-style graphs.

The random graphs exercise what the fixture blocks do not: multi-head batched matmuls with a transposed or plain
right operand, arbitrary transposes, reshapes that change the row length, int8 rows that are not a multiple of 8,
LUT epilogues, i16 results (qmatmul, add_requant) feeding an i16 rmsnorm with ss_shift/ss_init, non-causal and
causal softmax with T1 != T2, and an i32 softmax output. Constants are drawn so most graphs stay inside the int32
contract; a draw the golden interpreter rejects is redrawn.
"""
import numpy as np

from anvil.frontend import from_image
from anvil.interp import graph as gi
from anvil.ir.graph import GraphBuilder
from xf import fixture

BLOCKS = tuple(from_image.BLOCKS)
FIXTURES = [(b, i) for b in BLOCKS for i in range(4)]


def fixture_case(block, i):
    d = fixture.load(fixture.FIXTURE_DIR / f"{block}_{i}.json")
    m, c = from_image.BLOCKS[block](d["image"])
    return m, c, {"x": np.asarray(d["inputs"]["x"])}


def _rsqrt_lut(n):
    return np.round((1 << 15) / np.sqrt(np.arange(n) + 1)).astype(np.int64)


def _exp_lut():
    return np.round((1 << 14) * np.exp((np.arange(256) - 255) / 24.0)).astype(np.int64)


SD = 60.0          # rough std of an int8 activation in these graphs


def _scaled(rng, prod_sd, target_sd, lo=10, hi=20):
    """(M, s) with M / 2^s ~ target_sd / prod_sd (times a random factor): outputs spread out instead of
    saturating, so a wrong rounding or a wrong element shows up downstream."""
    s = int(rng.integers(lo, hi + 1))
    m = max(1, int(round(target_sd / prod_sd * (1 << s) * rng.uniform(0.5, 2.0))))
    return min(m, (1 << 31) - 1), s


def _mm_params(rng, k, target_sd=40.0):
    return _scaled(rng, SD * SD * np.sqrt(k), target_sd)


def _draw(rng):
    T = int(rng.integers(2, 8))
    T2 = T if rng.random() < 0.7 else int(rng.integers(2, 8))
    H = int(rng.integers(1, 4))
    dh = int(rng.integers(2, 10))
    d = int(rng.integers(3, 21))
    ff = int(rng.integers(3, 25))
    wide = rng.random() < 0.5
    g = GraphBuilder("rand")
    consts = {}

    def const(name, shape, dtype="i8", value=None):
        v = rng.integers(-128, 128, shape) if value is None else value
        consts[name] = np.asarray(v, dtype=np.int64)
        return g.const(name, list(np.shape(v)) if value is not None else list(shape), dtype)

    x = g.input("x", [T, d])
    xk = g.input("xk", [T2, d])
    nlut = int(rng.choice([64, 256, 1024]))
    rs = const("rsqrt", [nlut], "i32", _rsqrt_lut(nlut))
    h = g.rmsnorm_int("h", x, const("nw", [d]), rs, *_scaled(rng, SD * (1 << 15) / (SD * np.sqrt(d)), 40.0, 4, 14))
    hk = g.rmsnorm_int("hk", xk, const("nwk", [d]), rs, *_scaled(rng, SD * (1 << 15) / (SD * np.sqrt(d)), 40.0, 4, 14))
    q = g.qlinear("q", h, const("wq", [d, H * dh]), *_mm_params(rng, d))
    k = g.qlinear("k", hk, const("wk", [d, H * dh]), *_mm_params(rng, d))
    v = g.qlinear("v", hk, const("wv", [d, H * dh]), *_mm_params(rng, d))
    qh = g.transpose("qh", g.reshape("q3", q, [T, H, dh]), [1, 0, 2])              # (H, T, dh)
    k3 = g.reshape("k3", k, [T2, H, dh])
    if rng.random() < 0.5:                                                          # kT made by one transpose
        kT = g.transpose("kT", k3, [1, 2, 0])                                       # (H, dh, T2)
    else:                                                                           # kT = swap of (H, T2, dh)
        kT = g.transpose("kT", g.transpose("kh", k3, [1, 0, 2]), [0, 2, 1])
    sc = g.qmatmul("scores", qh, kT, *_mm_params(rng, dh))
    causal = bool(rng.random() < 0.6)
    ex = const("exp", [256], "i32", _exp_lut())
    p = g.softmax_lut("p", sc, ex, int(rng.integers(32, 128)), causal)
    vh = g.transpose("vh", g.reshape("v3", v, [T2, H, dh]), [1, 0, 2])            # (H, T2, dh)
    o = g.qmatmul("o", p, vh, *_mm_params(rng, T2))
    o2 = g.reshape("o2", g.transpose("ot", o, [1, 0, 2]), [T, H * dh])
    y = g.qlinear("y", o2, const("wo", [H * dh, d]), *_mm_params(rng, H * dh))
    rdt = "i16" if wide else "i8"
    rt = 2000.0 if wide else 40.0
    ma, s1 = _scaled(rng, SD, rt / 2, 8, 16)
    r1 = g.add_requant("r1", x, y, ma, _scaled(rng, SD, rt / 2, s1, s1)[0], s1, dtype=rdt)
    if wide:
        h2 = g.rmsnorm_int("h2", r1, const("nw2", [d]), rs, 1, int(rng.integers(12, 20)),
                           ss_shift=int(rng.integers(4, 9)), ss_init=int(rng.integers(0, 64)))
    else:
        h2 = g.rmsnorm_int("h2", r1, const("nw2", [d]), rs,
                           *_scaled(rng, SD * (1 << 15) / (SD * np.sqrt(d)), 40.0, 4, 14))
    silu = const("silu", [256])
    gate = g.qlinear("gate", h2, const("wg", [d, ff]), *_mm_params(rng, d),
                     lut=const("glut", [256]) if rng.random() < 0.5 else None)
    up = g.qlinear("up", h2, const("wu", [d, ff]), *_mm_params(rng, d))
    act = g.lut_mul("act", gate, up, silu, *_scaled(rng, SD * SD, 40.0))
    act = g.lut("act2", act, const("lut2", [256]))
    dn = g.qlinear("down", act, const("wd", [ff, d]), *_mm_params(rng, ff, 2000.0 if wide else 40.0), dtype=rdt)
    rsd = 2000.0 if wide else SD
    ma, s2 = _scaled(rng, rsd, 20.0, 8, 18)
    out = g.add_requant("out", r1, dn, ma, _scaled(rng, rsd, 20.0, s2, s2)[0], s2)
    p32 = g.softmax_lut("p32", sc, ex, int(rng.integers(1, 1 << 12)), not causal, dtype="i32")
    g.output(out, p32)
    inputs = {"x": rng.integers(-128, 128, (T, d)), "xk": rng.integers(-128, 128, (T2, d))}
    return g.module(), consts, inputs


def random_case(seed, tries=50):
    """(module, consts, inputs, golden GraphResult) for seed, redrawn until the golden model accepts it."""
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        m, c, x = _draw(rng)
        try:
            return m, c, x, gi.run(m, x, c)
        except (OverflowError, ValueError, ZeroDivisionError):
            continue
    raise RuntimeError(f"seed {seed}: no in-contract draw in {tries} tries")
