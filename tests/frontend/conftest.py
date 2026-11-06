"""Shared torch fixtures: the xf block built exactly as xf/export.py::build_block builds it, quantized by
anvil.frontend.from_torch on xf's own calibration data. Torch tests fail (not skip) without NANOGPT_DIR."""
import os

import numpy as np
import pytest

from xf import attention, export as xe

SDPA = "scaled_dot_product_attention"
# FGraph node names (from the exported graph) -> the xf image segment names
NAMES = {
    "mlp_block": {"b.ln_2": "h", "b.ff.gate_proj": "gate", "b.ff.up_proj": "up", "mul_2": "act",
                  "b.ff.down_proj": "down", "add_1": "out"},
    "attention": {"b.ln_1": "h", "b.attn.c_attn.0": "q", "b.attn.c_attn.1": "k", "b.attn.c_attn.2": "v",
                  f"{SDPA}.kT": "kT", f"{SDPA}.scores": "scores", f"{SDPA}.p": "p", SDPA: "o",
                  "b.attn.c_proj": "y"},
}
OUT = {"mlp_block": "out", "attention": "y"}
GOLDEN = {"mlp_block": xe.mlp_block_int, "attention": attention.attention_int}


def nanogpt_dir():
    d = os.environ.get("NANOGPT_DIR")
    if not d:
        pytest.fail("torch tests need NANOGPT_DIR (nanoGPT Extended @ 03dbab8)")
    return d


def wrappers():
    import torch

    class MlpBlock(torch.nn.Module):          # x + ff(ln_2(x)), xf/export.py float_acts
        def __init__(self, b):
            super().__init__()
            self.b = b

        def forward(self, x):
            return x + self.b.ff(self.b.ln_2(x))

    class AttentionHead(torch.nn.Module):     # attn(ln_1(x)), xf/export.py float_attention
        def __init__(self, b):
            super().__init__()
            self.b = b

        def forward(self, x):
            return self.b.attn(self.b.ln_1(x))

    return {"mlp_block": MlpBlock, "attention": AttentionHead}


def calib(block):
    n = xe.N_CALIB if block == "mlp_block" else xe.ATTN_N_CALIB
    return np.random.default_rng(xe.SEED).standard_normal((n, xe.T, xe.N_EMBD))


@pytest.fixture(scope="session")
def nanogpt_block():
    pytest.importorskip("torch")
    return xe.build_block(nanogpt_dir())


@pytest.fixture(scope="session")
def modules(nanogpt_block):
    return {k: cls(nanogpt_block) for k, cls in wrappers().items()}


@pytest.fixture(scope="session")
def quantized(modules):
    import torch

    from anvil.frontend import from_torch

    example = torch.zeros(1, xe.T, xe.N_EMBD, dtype=torch.float64)
    return {k: from_torch(m, (example,), calib(k), headroom=xe.HEADROOM, names=NAMES[k], sym_name=k)
            for k, m in modules.items()}
