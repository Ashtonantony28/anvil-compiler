"""anvil.compile on graph IR (the fixture graphs built by anvil.frontend.from_image, no torch): every fixture case
golden-equal on isasim under both schedule presets and both memory plans, footprints within the xf images, the
schedule-selection hook, the address-space check and (de)quantization helpers. The torch path is in
test_g1_torch.py."""
import numpy as np
import pytest

import anvil
from anvil import driver, memplan
from anvil.frontend import from_image
from anvil.runtime import GpuParams
from anvil.schedule import Schedule, ScheduleError, templates as T
from xf import fixture

BLOCKS = ("mlp_block", "attention")
OUT = {"mlp_block": "out", "attention": "y"}
IMAGE_WORDS = {"mlp_block": 2005, "attention": 1847}


def _fx(block, i=0):
    return fixture.load(fixture.FIXTURE_DIR / f"{block}_{i}.json")


def _graph(block):
    return from_image.BLOCKS[block](_fx(block)["image"])


_CACHE = {}


def compiled(block, preset="default", params=GpuParams()):
    key = (block, preset, params)
    if key not in _CACHE:
        _CACHE[key] = anvil.compile(_graph(block), params=params, schedules=preset)
    return _CACHE[key]


def flags_clean(r):
    return not any(r.flags[k] for k in ("store_races", "stores_conflict", "divergent"))


@pytest.mark.parametrize("preset", ["default", "tiled"])
@pytest.mark.parametrize("block", BLOCKS)
def test_fixture_cases_golden_on_isasim(block, preset):
    cm = compiled(block, preset)
    for i in range(4):
        d = _fx(block, i)
        full = cm.run(d["inputs"]["x"], quantized=True, keep="all")
        assert flags_clean(full.result)
        assert set(d["golden"]) <= set(full.ints)
        for seg, want in d["golden"].items():
            assert np.array_equal(full.ints[seg], want), (i, seg)
        reuse = cm.run(d["inputs"]["x"], quantized=True)
        assert flags_clean(reuse.result)
        assert not full.stray_writes() and not reuse.stray_writes()
        assert np.array_equal(reuse.ints[OUT[block]], d["golden"][OUT[block]])
        assert np.allclose(reuse.output, d["golden"][OUT[block]] * cm.scale(OUT[block]))


@pytest.mark.parametrize("block", BLOCKS)
def test_footprint_within_xf_image(block):
    for preset in ("default", "tiled"):
        cm = compiled(block, preset)
        mp = cm.plan
        assert mp.footprint <= IMAGE_WORDS[block], mp.report()
        assert mp.regions["activations"][1] < mp.unshared             # some activation words are reused
        assert mp.footprint == sum(n for _, n in mp.regions.values())
        assert cm.memory_plan("all").footprint >= mp.footprint


def test_reuse_plan_reuses_dead_buffers():
    """MLP peak: gate, up and act live together with the residual input x (224 words of 328)."""
    mp = compiled("mlp_block").plan
    assert mp.regions["activations"][1] == 32 + 3 * 64
    a = mp.addresses
    assert a["act"] not in (a["gate"], a["up"], a["x"])                  # act is written while gate/up are read
    assert a["act"] == a["h"] or a["down"] in (a["h"], a["gate"], a["up"])  # ... but takes a dead buffer's words
    assert len(set(a.values())) < len(a)                                  # some buffer shares an address


def test_programs_do_not_depend_on_the_plan():
    cm = compiled("attention", "tiled")
    d = _fx("attention")
    j1, j2 = cm.job(d["inputs"]["x"], quantized=True), cm.job(d["inputs"]["x"], quantized=True, keep="all")
    strip = lambda w: [x for i, x in enumerate(w) if i not in (0, 4)]       # ARGS prologue words (args base)
    assert [strip(l.program) for l in j1.job.launches] == [strip(l.program) for l in j2.job.launches]
    assert j1.end < j2.end


def test_float_round_trip_and_batch():
    cm = compiled("mlp_block")
    d = _fx("mlp_block", 1)
    xq = np.asarray(d["inputs"]["x"])
    x = xq * cm.scale("x")
    assert np.array_equal(cm.quantize_input(x), xq)
    r = cm.run(x)
    assert np.array_equal(r.ints["out"], d["golden"]["out"])
    with pytest.raises(ValueError, match="inputs"):
        cm.run([x, x])
    with pytest.raises(ValueError, match="shape"):
        cm.run(np.zeros((3, 8)))


def test_schedule_selection_hook():
    g = _graph("attention")
    tiled = T.matmul_tiled
    calls = []

    def mine(s):
        calls.append(s.kernel.name)
        return tiled(s)

    trace = T.matmul_tiled(Schedule(compiled("attention").program.kernel("y"), compiled("attention").target)).to_json()
    cm = anvil.compile(g, schedules={"q": "matmul_tiled", "qmatmul": mine, "y": trace, "softmax_lut": "row_per_block"})
    how = {k: v[0] for k, v in cm.schedules.items()}
    assert how == {"h": "default", "q": "matmul_tiled", "k": "default", "v": "default", "kT": "default",
                   "scores": "mine", "p": "row_per_block", "o": "mine", "y": "trace"}
    assert calls == ["scores", "o"]
    assert cm.schedules["y"][1] == trace
    d = _fx("attention", 2)
    assert np.array_equal(cm.run(d["inputs"]["x"], quantized=True).ints["y"], d["golden"]["y"])
    # a callable policy over (kernel, target); None falls back to the default
    cm2 = anvil.compile(g, schedules=lambda k, tgt: T.row_per_block(Schedule(k, tgt)) if k.op == "rmsnorm_int" else None)
    assert cm2.schedules["h"][0] == "custom" and cm2.schedules["q"][0] == "default"


def test_explicit_illegal_schedule_raises_and_presets_fall_back():
    g = _graph("attention")
    with pytest.raises(ScheduleError):
        anvil.compile(g, schedules={"transpose": "matmul_tiled"})
    with pytest.raises(KeyError, match="preset"):
        anvil.compile(g, schedules="fastest")
    cm = compiled("attention", "tiled")                     # transpose has no tiled template: default
    assert cm.schedules["kT"][0] == "default"


def test_address_space_is_checked():
    small = GpuParams(DATA_MEM_ADDR_BITS=10)               # 1024 words < the 1024-entry rsqrt LUT + the rest
    with pytest.raises(memplan.MemoryPlanError, match="1024"):
        anvil.compile(_graph("mlp_block"), params=small)


@pytest.mark.parametrize("cfg", [dict(WARPS_PER_CORE=2), dict(THREADS_PER_BLOCK=8, WARPS_PER_CORE=2, SIMT_STACK=1)],
                         ids=str)
def test_other_configurations(cfg):
    p = GpuParams(**cfg)
    for block in BLOCKS:
        cm = compiled(block, "tiled", p)
        d = _fx(block, 3)
        r = cm.run(d["inputs"]["x"], quantized=True)
        assert np.array_equal(r.ints[OUT[block]], d["golden"][OUT[block]])
        assert not any(r.result.flags[k] for k in ("store_races", "stores_conflict"))
        assert not r.stray_writes()


def test_compile_rejects_unknown_models():
    with pytest.raises(TypeError, match="cannot compile"):
        driver.compile(42)


def test_stray_writes_catch_a_write_outside_the_plan_and_into_consts():
    """RunOutput.stray_writes (A1-M3): a word changed past the job, in an args block or in a const is reported."""
    cm = compiled("mlp_block", "default")
    d = _fx("mlp_block", 0)
    r = cm.run(d["inputs"]["x"], quantized=True)
    assert r.stray_writes() == []
    end = r.job.end
    w = r.job.addresses[next(s for s in cm.consts)]
    a = r.job.args[0][2]
    for addr in (end, a, w):
        r.result.memory[addr] ^= 0x55
        assert addr in r.stray_writes()
        r.result.memory[addr] ^= 0x55
