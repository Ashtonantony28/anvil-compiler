"""`Schedule`: a kernel plus the trace of primitives applied to it, replayable by a tuner."""
import json

from xdsl.utils.exceptions import VerifyException

from . import ir_util as u
from .canonical import canonicalize, validate
from .primitives import PRIMITIVES
from .target import ScheduleError, Target


class Schedule:
    """    sch = Schedule(kernel, Target()); sch.fuse("i", "j").split("i_j", 4).bind("i_j.o", "block") ...
    Each primitive is applied to a clone; on ScheduleError nothing changes and the trace is not extended."""

    def __init__(self, kernel, target: Target = Target()):
        self.base = kernel
        self.kernel = kernel.clone()
        self.target = target
        self.trace = []

    def apply(self, name, /, *args, **kw):
        if name not in PRIMITIVES:
            raise ScheduleError(f"unknown primitive {name!r}")
        work = self.kernel.clone()
        try:
            PRIMITIVES[name](work, self.target, *args, **kw)
            work.func.verify()
        except VerifyException as e:          # a rewrite produced invalid IR: report as a schedule error
            raise ScheduleError(f"{name}: produced invalid IR: {e}") from e
        self.kernel = work
        self.trace.append([name, list(args), dict(kw)])
        return self

    def __getattr__(self, name):
        if name in PRIMITIVES:
            return lambda *a, **kw: self.apply(name, *a, **kw)
        raise AttributeError(name)

    # ------------------------------------------------------------------------------------- inspection
    def loops(self) -> list:
        """[(tag, extent, bind, kind, unroll)] in program order."""
        return [(u.tag_of(L), u.extent(L), L.bind.data, self.kernel.kinds.get(u.tag_of(L), "spatial"),
                 int(L.unroll.value.data)) for L in u.loops(self.kernel.func)]

    def tags(self) -> list:
        return [x[0] for x in self.loops()]

    def run(self, buffers, **kw):
        return self.kernel.run(buffers, **kw)

    def validate(self):
        return validate(self.kernel, self.target)

    def finalize(self):
        """The canonical, validated kernel SG3b compiles (raises ScheduleError if the schedule breaks a rule)."""
        return canonicalize(self.kernel, self.target)

    # ---------------------------------------------------------------------------------------- replay
    def to_json(self) -> str:
        return json.dumps(self.trace)

    @classmethod
    def replay(cls, kernel, trace, target: Target = Target()) -> "Schedule":
        if isinstance(trace, str):
            trace = json.loads(trace)
        s = cls(kernel, target)
        for name, args, kw in trace:
            s.apply(name, *args, **kw)
        return s
