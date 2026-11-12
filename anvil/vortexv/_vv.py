"""Where Vortex-V lives and the pieces Anvil borrows from it (all read-only).

Vortex-V is a pinned submodule at ``vortexv/`` (override with ``VORTEXV_DIR``). Anvil uses its RV64IM + Xmlacc
assembler (``verif/core/rvasm.py``), its golden instruction-set simulator (``model/iss``) and, for RTL runs, its
cocotb harness (``verif/core``). Nothing in Vortex-V is modified.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VV = Path(os.environ.get("VORTEXV_DIR") or ROOT / "vortexv")
VV_CORE = VV / "verif" / "core"


def available() -> bool:
    return (VV_CORE / "rvasm.py").exists() and (VV / "model" / "iss").is_dir()


def _path():
    if not available():
        raise ImportError(f"Vortex-V not found at {VV}: run `git submodule update --init vortexv` "
                          "or set VORTEXV_DIR")
    for p in (str(VV_CORE), str(VV)):
        if p not in sys.path:
            sys.path.insert(0, p)


def rvasm():
    _path()
    import rvasm as m
    return m


def iss():
    _path()
    from model.iss import rv64i
    return rv64i


def xmlacc_model():
    """Vortex-V's numpy golden model of the Xmlacc instructions (dot8, macc, rqs, ...)."""
    _path()
    from model import xmlacc_model as m
    return m
