"""anvil.memplan unit tests: liveness on lowered programs, offset assignment (hypothesis: random liveness graphs
never overlap live buffers), layout regions and the address-space check."""
import types

import pytest
from hypothesis import given, settings, strategies as st

from anvil import memplan as mp
from anvil.frontend import from_image
from anvil.lower.graph_to_tensor import lower
from xf import fixture


def _bufs(spec):
    return [mp.Buf(f"b{i}", size, min(a, b), max(a, b)) for i, (size, a, b) in enumerate(spec)]


buf_specs = st.lists(st.tuples(st.integers(0, 40), st.integers(-1, 12), st.integers(-1, 12)), max_size=30)


@settings(max_examples=300, deadline=None)
@given(buf_specs)
def test_random_liveness_never_overlaps(spec):
    bufs = _bufs(spec)
    offs, top = mp.assign_offsets(bufs)
    assert not mp.check_disjoint(bufs, offs)
    assert all(0 <= offs[b.sym] and offs[b.sym] + b.size <= top for b in bufs)
    # the arena is at least the peak live size and at most the no-reuse sum
    peak = max([sum(b.size for b in bufs if b.start <= t <= b.end) for t in range(-1, 13)] + [0])
    assert peak <= top <= sum(b.size for b in bufs)


@settings(max_examples=100, deadline=None)
@given(st.lists(st.tuples(st.integers(1, 40), st.integers(0, 12)), min_size=1, max_size=20))
def test_disjoint_lifetimes_share_one_slot_size(spec):
    """Buffers live at single, distinct kernels need only the largest of them."""
    seen, bufs = set(), []
    for i, (size, t) in enumerate(spec):
        if t not in seen:
            seen.add(t)
            bufs.append(mp.Buf(f"b{i}", size, t, t))
    _, top = mp.assign_offsets(bufs)
    assert top == max(b.size for b in bufs)


@settings(max_examples=100, deadline=None)
@given(st.lists(st.tuples(st.integers(1, 30), st.integers(0, 6), st.integers(0, 6)), min_size=1, max_size=12),
       st.integers(1, 4), st.integers(0, 20))
def test_layout_regions_are_disjoint_and_ordered(spec, nlaunch, scratch):
    bufs = {b.sym: b for b in _bufs(spec)}
    bufs["w"] = mp.Buf("w", 7, -1, 7, "const")
    launches = [(f"k{i}", 0, 3 + i) for i in range(nlaunch)]
    plan = mp.plan_layout(bufs, launches, scratch, 1 << 12)
    r = plan.regions
    assert r["consts"] == (0, 7) and plan.addresses["w"] == 0
    assert r["activations"][0] == 7 and r["args"][0] == 7 + r["activations"][1]
    assert r["scratch"][0] == r["args"][0] + sum(3 + i for i in range(nlaunch))
    assert plan.footprint == r["scratch"][0] + scratch
    assert [b[2] for b in plan.args_blocks] == [r["args"][0] + sum(3 + j for j in range(i)) for i in range(nlaunch)]
    for b in bufs.values():
        assert plan.addresses[b.sym] + b.size <= r["args"][0]


def test_address_space_overflow_raises():
    bufs = {"x": mp.Buf("x", 200, -1, 2, "input"), "w": mp.Buf("w", 60, -1, 3, "const")}
    with pytest.raises(mp.MemoryPlanError, match="2\\^DATA_MEM_ADDR_BITS"):
        mp.plan_layout(bufs, [("k", 0, 4)], 0, 256)
    assert mp.plan_layout(bufs, [("k", 0, 4)], 0, 264).footprint == 264


def _prog(block):
    d = fixture.load(fixture.FIXTURE_DIR / f"{block}_0.json")
    return lower(from_image.BLOCKS[block](d["image"])[0])


def test_mlp_liveness():
    prog = _prog("mlp_block")
    live = mp.liveness(prog)
    order = [k.name for k in prog.kernels]                       # h gate up act down out
    assert order == ["h", "gate", "up", "act", "down", "out"]
    assert (live["x"].start, live["x"].end, live["x"].kind) == (-1, 5, "input")      # residual: read by out
    assert (live["h"].start, live["h"].end) == (0, 2)
    assert (live["act"].start, live["act"].end) == (3, 4)
    assert (live["out"].end, live["out"].kind) == (6, "output")
    assert live["rsqrt_lut"].kind == "const" and live["rsqrt_lut"].end == 6
    temps = [s for s, b in live.items() if b.kind == "temp"]
    assert temps and all(live[s].start == live[s].end == 0 for s in temps)          # rmsnorm stat temps
    every = mp.liveness(prog, keep="all")
    assert all(b.end == 6 for b in every.values() if b.kind != "temp")


def test_kernel_output_never_aliases_its_inputs():
    for block in ("mlp_block", "attention"):
        prog = _prog(block)
        bufs = mp.liveness(prog)
        offs, _ = mp.assign_offsets([b for b in bufs.values() if b.kind != "const"])
        for k in prog.kernels:
            for w in k.writes:
                for r in k.reads:
                    if bufs[r].kind == "const":
                        continue
                    a, b = bufs[w], bufs[r]
                    assert offs[w] + a.size <= offs[r] or offs[r] + b.size <= offs[w], (k.name, w, r)


def test_views_extend_their_source():
    """A reshape view read by a later kernel keeps its source alive (synthetic program)."""
    K = types.SimpleNamespace
    prog = K(kernels=[K(name="a", reads=["x"], writes=["y"], temps={}),
                      K(name="b", reads=["x"], writes=["z"], temps={}),
                      K(name="c", reads=["y"], writes=["u"], temps={})],
             buffers={"x": ([4], "i8"), "y": ([2, 2], "i8"), "z": ([4], "i8"), "u": ([4], "i8")},
             views={"yv": ("y", [4]), "yvv": ("yv", [1, 4])}, inputs=["x"], outputs=["yvv", "u"], const_values={})
    live = mp.liveness(prog)
    assert live["y"].end == 3 and "yv" not in live and "yvv" not in live
    assert (live["x"].start, live["x"].end) == (-1, 1) and live["z"].kind == "act"
