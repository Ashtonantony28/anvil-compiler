"""Contract C2: run a Job of kernel launches on xf.isasim, the iverilog RTL or (if adopted) a Verilator build.
`check_races(job)` (anvil.runtime.races) adds the cross-block read/write race check over an isasim address trace."""
from .api import GpuParams, Job, Launch, LaunchResult, RunResult, run
from .races import RaceReport, check_races

__all__ = ["GpuParams", "Job", "Launch", "LaunchResult", "RunResult", "run", "check_races", "RaceReport"]
