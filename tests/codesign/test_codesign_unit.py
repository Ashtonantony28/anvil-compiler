"""SG8 (G5): cost proxy, configuration space, Pareto/knee and the cached sweep records (fast, no RTL)."""

import pytest

from anvil.codesign import cost as hw
from anvil.codesign import space as S
from anvil.codesign.report import groups, knee, pareto
from anvil.runtime import GpuParams


def test_cost_baseline_by_hand():
    # xf build (2 cores x 4 threads, WORD 32, ADDR 12): every term of the documented formula, by hand
    b = hw.breakdown(GpuParams())
    assert b["registers"] == 2 * 4 * 16 * 32
    assert b["smem"] == 2 * 64 * 32
    assert b["lsu"] == 2 * 4 * (3 * 32 + 2 * 12 + 22)
    assert b["alu"] == 2 * 4 * 32
    assert b["nzp"] == 24 and b["decode+fetch"] == 2 * 68 and b["scheduler"] == 2 * (8 + 3 + 1 + 1)
    assert b["data_ctrl"] == 8 * 34 + 4 * (3 + 4 + 24 + 32 + 2) + 8
    assert b["prog_ctrl"] == 2 * 18 + (3 + 2 + 16 + 16 + 2) + 2
    assert b["icache"] == b["dcache"] == b["simt"] == 0
    assert hw.state_bits(GpuParams()) == sum(b.values()) == 10404


@pytest.mark.parametrize("lines,tag", [(8, 5), (16, 4), (32, 3)])
def test_cost_icache_lines(lines, tag):
    d = hw.state_bits(GpuParams(ICACHE_LINES=lines)) - hw.state_bits(GpuParams())
    assert d == 2 * lines * (1 + tag + 16)          # per core: valid + tag + 16-bit word per line (fetcher.sv:84-86)


def test_cost_dcache_and_simt_and_free_knobs():
    base = hw.state_bits(GpuParams())
    d16 = hw.state_bits(GpuParams(DCACHE_LINES=16)) - base
    assert d16 == 16 * (1 + (12 - 4) + 32) + 8 * 34 + 8 * (24 + 32 + 2) + 8 * 2
    assert hw.state_bits(GpuParams(PIPELINED=1)) == base
    assert hw.state_bits(GpuParams(WARPS_PER_CORE=2, WARP_POLICY=0)) == \
        hw.state_bits(GpuParams(WARPS_PER_CORE=2, WARP_POLICY=2))
    assert hw.state_bits(GpuParams(SIMT_STACK=1)) - base == 2 * (11 * (16 + 4) + 4 + 1 + 8)
    assert hw.state_bits(GpuParams(COALESCE=1)) - base == 4 * 8
    assert hw.state_bits(GpuParams(NUM_CORES=1)) < base < hw.state_bits(GpuParams(NUM_CORES=4))


def test_space_and_design():
    full = S.full_space()
    assert len(full) == 4032 == len(set(full)) and all(c.valid() for c in full[::97])
    d = S.design()
    assert len(d) == len(set(d)) == len({c.id for c in d}) == 55
    assert d[0] == S.Config() and d[0].id == "C2" and all(c.valid() for c in d)
    assert all(c in set(full) for c in d)
    a = S.attention_design()
    assert len(a) == 39 and set(a) <= set(d) and S.Config(mem_latency=3) in a
    m = S.model_design()
    assert len(m) == len(set(m)) and all(c.valid() for c in m)
    c = S.Config.make(3, NUM_CORES=4, ICACHE_LINES=32, WARPS=(2, 1), SIMT_STACK=1)
    assert c.id == "C4.I32.W2gto.S.L3"
    p = c.params(16)
    assert (p.NUM_CORES, p.ICACHE_LINES, p.WARPS_PER_CORE, p.WARP_POLICY, p.SIMT_STACK, p.DATA_MEM_ADDR_BITS) == \
        (4, 32, 2, 1, 1, 16)
    assert S.Config.make(0, WARPS=(1, 2)).WARP_POLICY == 0          # policy is meaningless at one warp
    assert not S.Config(WARPS_PER_CORE=3).valid() and not S.Config(ICACHE_LINES=12).valid()


def test_pareto_and_knee():
    pts = [{"config": n, "cost_bits": c, "cycles": y, "mem_latency": 0}
           for n, c, y in [("a", 1, 100), ("b", 2, 50), ("c", 3, 45), ("d", 4, 10), ("e", 5, 60), ("f", 2, 70)]]
    f = pareto(pts)
    assert [r["config"] for r in f] == ["a", "b", "c", "d"]
    assert knee(f)["config"] == "b"
    assert knee(f[:2]) is f[1]


def test_cached_sweep_records_are_consistent():
    """Every committed point record: verified chain, cost = the proxy at its params, cycles = sum of kernels."""
    from anvil.codesign.sweep import load_points
    pts = load_points()
    n = 0
    for w, recs in pts.items():
        for r in recs:
            n += 1
            p = GpuParams(**r["params"])
            assert r["cost_bits"] == hw.state_bits(p)
            assert r["cycles"] == sum(r["kernel_cycles"].values())
            assert r["verify"]["block_races"] == 0 and not any(r["verify"]["flags"].values())
            assert r["verify"]["checked"]
            for k in r["kernels"].values():
                assert k["key"] and k["isolated"] == k.get("cached_cycles")
    if n == 0:
        pytest.skip("no sweep records")
    assert groups(pts)
