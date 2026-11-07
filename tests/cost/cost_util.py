"""Shared helpers for tests/cost: kernels of the dataset suite as tuner Tasks, built candidates, tiny programs."""
import functools
import random

from anvil.runtime import GpuParams, Job, Launch
from anvil.tune import Candidate, Invalid, Task, build, sketches_for


@functools.cache
def specs():
    from anvil.cost.dataset import suite
    return {s.kid: s for s in suite(("xf", "syn"))}


def task(kid, params=GpuParams(), **kw):
    s = specs()[kid]
    return Task.from_program(s.program, s.name, s.values, params, **kw)


def builts(kid, params=GpuParams(), n=3, seed=0, **kw):
    """The template seeds then random valid candidates of a kernel (Built objects)."""
    t = task(kid, params, **kw)
    sks = sketches_for(t.kernel, t.target, t.params)
    rng = random.Random(f"{kid}/{seed}")
    out, cands, tries = [], list(sks[0].seeds()), 0
    while len(out) < n and tries < 400:
        tries += 1
        if cands:
            c = cands.pop(0)
        else:
            sk = rng.choice(sks)
            c = Candidate.make(sk.name, **{p: rng.choice(v) for p, v in sk.domain.items()})
        b = build(t, c)
        if not isinstance(b, Invalid):
            out.append(b)
    return t, out


ALU = ("ADD", "SUB", "MUL", "MAX", "SRA")


def straight_line(rng, n, *, smem=True, glob=True, bar=True):
    """A random straight-line program (no branches) over R0-R7 with in-range global/smem addresses: assembly text."""
    lines = ["CONST R0, #0", "ADD R1, R0, %threadIdx", "CONST R2, #2"]
    ops = ["alu", "const", "cmp"] + (["ldr", "str", "ldr_same"] if glob else []) + (["lds", "sts"] if smem else []) \
        + (["bar"] if bar else [])
    for _ in range(n):
        k = rng.choice(ops)
        r = lambda: f"R{rng.randrange(3, 8)}"     # noqa: E731
        if k == "alu":
            lines.append(f"{rng.choice(ALU)} {r()}, {r()}, R2")
        elif k == "const":
            lines.append(f"CONST {r()}, #{rng.randrange(256)}")
        elif k == "cmp":
            lines.append(f"CMP {r()}, {r()}")
        elif k == "ldr":
            lines.append(f"LDR {r()}, R1")
        elif k == "ldr_same":
            lines.append(f"LDR {r()}, R0")
        elif k == "str":
            lines.append(f"STR R1, {r()}")
        elif k == "lds":
            lines.append(f"LDS {r()}, %threadIdx")
        elif k == "sts":
            lines.append(f"STS %threadIdx, {r()}")
        elif k == "bar":
            lines.append("BAR")
    lines.append("RET")
    return "\n".join(lines) + "\n"


def asm_job(src, threads, params=GpuParams(), mem_latency=0, words=256):
    from xf.asm import assemble
    return Job(list(range(words)), [Launch(assemble(src), threads)], params, mem_latency)
