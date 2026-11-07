"""SG8 (G5): a small sweep reproduces the cached numbers (rtl: Verilator).

Replays points of the committed sweep from the committed best-schedule store (no search): the compiled chains must
be bit-exact and race-free and their Verilator cycles must equal the cached records, and the cost proxy must equal
the record's."""
import json

import pytest

from anvil.codesign import cost as hw
from anvil.codesign.sweep import BEST, configs_for, point_path, run_point
from anvil.codesign.workloads import load

pytestmark = pytest.mark.rtl
POINTS = [("mlp_block", "C2"), ("mlp_block", "C2.I32.CO.PI"), ("attention", "C2.I32.CO.PI.L3"), ("attention", "C2.W2gto"),
          ("model", "C2.I32.CO.PI")]


@pytest.mark.parametrize("workload,cid", POINTS)
def test_replay_reproduces_cached_point(workload, cid, tmp_path):
    from anvil.tune import BestDB, MeasureCache
    cfg = next(c for c in configs_for(workload) if c.id == cid)
    path = point_path(workload, cfg)
    if not path.exists():
        pytest.skip(f"no cached record {path}")
    want = json.loads(path.read_text())
    db = BestDB(committed=False, paths=[BEST], write=tmp_path / "best.json")
    rec = run_point(load(workload), cfg, db=db, cache=MeasureCache(tmp_path / "m"), budget=want["budget"],
                    search=False, workers=1)
    assert {k: v["how"] for k, v in rec["kernels"].items()} == dict.fromkeys(want["kernels"], "db")
    assert rec["kernel_cycles"] == want["kernel_cycles"] and rec["cycles"] == want["cycles"]
    assert rec["cost_bits"] == want["cost_bits"] == hw.state_bits(cfg.params(want["params"]["DATA_MEM_ADDR_BITS"]))
    assert rec["verify"] == want["verify"]
