"""frontend: PyTorch -> anvil.graph (SG4a) and the hand-built fixture graphs (from_image, SG1).

    q = from_torch(module, (x_example,), calib, headroom=1.5, names={...})   # capture + quantize
    fg = capture(module, (x_example,)); q = quantize(fg, calib)               # the two steps

Both submodules import torch lazily, so this package (and from_image) imports without torch.
"""
from . import capture as _capture_mod  # noqa: F401  (bind the submodules before the same-named functions)
from . import quantize as _quantize_mod  # noqa: F401
from .capture import FGraph, FNode, UnsupportedOpError, capture
from .quantize import Quantized, fixed_point, prove_no_overflow, quantize

__all__ = ["FGraph", "FNode", "Quantized", "UnsupportedOpError", "capture", "fixed_point", "from_torch",
           "prove_no_overflow", "quantize"]


def from_torch(module, example_inputs, calib_inputs, headroom=1.5, **kw) -> Quantized:
    """capture(module, example_inputs) then quantize(..., calib_inputs, headroom, **kw)."""
    return quantize(capture(module, example_inputs), calib_inputs, headroom, **kw)
