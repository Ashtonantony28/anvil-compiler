# Anvil

A tensor compiler that lowers PyTorch transformer blocks (nanoGPT Extended) onto the tiny-gpu-extended ISA.
See `docs/design.md` for the design and `PLAN.md` for status.

## Install

The simulator is a pinned submodule; Anvil imports its `xf` package from `sim/`, so put `sim` on `PYTHONPATH`
(the tests do this themselves through `tests/conftest.py`).

```bash
git submodule update --init                                  # sim/ (tiny-gpu-extended, read-only)
pip install numpy xdsl==0.75.0 pytest hypothesis             # core + tests
pip install torch cocotb==2.0.1 lightgbm                     # torch frontend, RTL runs, cost models (optional)
apt-get install -y iverilog verilator                        # plus sv2v 0.0.13: see PLAN.md "Environment"
export PYTHONPATH=sim:.
```

The torch tests and the model scripts also need nanoGPT Extended at 03dbab8 in `NANOGPT_DIR` (PLAN.md).

## Use

```python
import anvil
from anvil.runtime import GpuParams

cm = anvil.compile(block, (x_example,), calib_inputs, GpuParams(), schedules="tiled")   # torch.nn.Module
cm = anvil.compile((graph_module, consts))                                               # or anvil.graph IR
res = cm.run(x, "isasim")          # "isasim" | "verilator" | "rtl"; quantized=True takes int8 inputs
res.output, res.ints, res.cycles, res.stray_writes()
print(cm.summary())                # per-kernel schedule, program words, launches, memory plan
```

`schedules` is `"default"`, `"tiled"`, `"tuned"` (anvil.tune), a dict per kernel or op, or a callable; see the
`anvil.driver` docstring. `anvil.runtime.check_races(job)` checks a job for cross-block read/write races.

## Test

```bash
make check                                                   # unit + isasim levels, no RTL, no torch (< 5 min)
python3 -m pytest -q -m rtl tests/codegen tests/tune tests/cost tests/audit
NANOGPT_DIR=... python3 -m pytest -q -m "(rtl or torch) and not slow" tests/driver tests/tune
```

Reports: `docs/codegen-bench.md`, `docs/g1-report.md` .. `docs/g4-report.md`, each with the command that
regenerates it.
