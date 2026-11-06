"""SG4a capture: whitelist errors by name, pattern recognition (decomposed RMSNorm and softmax, causal masks,
SDPA, split linears, heads), determinism, and the shapes SG5 needs (residual Block, two layers, multi-head)."""
import math
import sys

import numpy as np
import pytest

from anvil.interp import graph as gi
from anvil.ir import print_ir

from conftest import NAMES, calib, nanogpt_dir

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch")
F = torch.nn.functional
T, D = 4, 8


def _x(b=1, d=D):
    return torch.zeros(b, T, d, dtype=torch.float64)


def _ops(fg):
    return [n.op for n in fg.nodes]


def _check_float(fg, module, n=64, d=D):
    xs = torch.as_tensor(np.random.default_rng(5).standard_normal((n, T, d)))
    with torch.no_grad():
        got, want = fg.evaluate([xs])[fg.outputs[0].name], module(xs)
    assert torch.allclose(got, want, rtol=0, atol=1e-12), float((got - want).abs().max())


def _int_error_lsb(q, module, n=200, d=D, seed=1):
    """Max |dequant(graph) - module(dequant(x_q))| in output LSB over seeded Gaussian inputs."""
    sx, out = q.scales["x"], q.graph.outputs[0].name
    s_out = q.scales[q.syms[out]]
    xq = np.clip(np.rint(np.random.default_rng(seed).standard_normal((n, T, d)) / sx), -128, 127).astype(np.int64)
    with torch.no_grad():
        ref = module(torch.as_tensor(xq * sx)).numpy()
    got = np.stack([gi.run(q.module, {"x": x}, q.consts).outputs[0] for x in xq]) * s_out
    return float(np.abs(got - ref).max()) / s_out


# ------------------------------------------------------------------------------------------------ errors

class _Gelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(D, D, bias=False)

    def forward(self, x):
        return torch.tanh(F.gelu(self.fc(x)))


def test_unsupported_ops_rejected_by_name():
    from anvil.frontend import UnsupportedOpError, capture

    with pytest.raises(UnsupportedOpError) as e:
        capture(_Gelu().double(), (_x(),))
    msg = str(e.value)
    assert "aten.gelu.default" in msg and "aten.tanh.default" in msg, msg     # every offender, by name


def test_linear_bias_rejected():
    from anvil.frontend import UnsupportedOpError, capture

    with pytest.raises(UnsupportedOpError, match="aten.linear.default.*bias"):
        capture(torch.nn.Linear(D, D).double(), (_x(),))


def test_non_causal_mask_rejected():
    from anvil.frontend import UnsupportedOpError, capture

    class M(torch.nn.Module):
        def forward(self, x):
            s = x @ x.transpose(-2, -1)
            mask = torch.ones(T, T, dtype=torch.bool).triu()          # anti-causal
            return F.softmax(s.masked_fill(~mask, float("-inf")), dim=-1) @ x

    with pytest.raises(UnsupportedOpError, match="causal"):
        capture(M(), (_x(),))


def test_stray_primitive_outside_pattern_rejected():
    from anvil.frontend import UnsupportedOpError, capture

    class M(torch.nn.Module):
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + 1)    # sum, not mean: not an RMSNorm

    with pytest.raises(UnsupportedOpError, match="aten.rsqrt.default.*outside a recognised pattern"):
        capture(M(), (_x(),))


def test_multihead_not_lowered_yet():
    from anvil.frontend import capture, quantize

    sys.path.insert(0, nanogpt_dir())
    from model import GPTConfig
    from models.gpt import Block

    torch.manual_seed(0)
    cfg = GPTConfig(block_size=T, n_layer=1, n_head=2, n_embd=D, dropout=0.0, bias=False, pos_embedding="none",
                    norm_type="rmsnorm", activation="swiglu")
    blk = Block(cfg).double().eval()

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.b = blk

        def forward(self, x):
            return self.b.attn(self.b.ln_1(x))

    fg = capture(Head(), (_x(),))
    assert _ops(fg).count("split_heads") == 3 and _ops(fg).count("merge_heads") == 1
    assert fg.node("scaled_dot_product_attention.scores").attrs["factor"] == ("inv_sqrt", 4)
    _check_float(fg, Head())
    with pytest.raises(NotImplementedError, match="SG5"):
        quantize(fg, calib("attention")[:256])


# ------------------------------------------------------------------------------------------------ patterns

def test_mlp_and_attention_graphs(modules):
    from anvil.frontend import capture

    mlp = capture(modules["mlp_block"], (_x(),))
    assert _ops(mlp) == ["input", "rmsnorm", "linear", "silu", "linear", "mul", "linear", "add"]
    rms = mlp.node("b.ln_2")
    assert rms.attrs["eps"] == 1e-6 and rms.attrs["compute_dtype"] == torch.float32
    att = capture(modules["attention"], (_x(),))
    assert _ops(att) == ["input", "rmsnorm", "linear", "linear", "linear", "transpose", "matmul", "softmax",
                         "matmul", "linear"]
    assert [att.node(f"b.attn.c_attn.{i}").attrs["rows"] for i in range(3)] == [(0, 8), (8, 16), (16, 24)]
    assert att.node("scaled_dot_product_attention.p").attrs["causal"] is True
    for k, fg in (("mlp_block", mlp), ("attention", att)):
        assert set(NAMES[k]) <= {n.name for n in fg.nodes}
        _check_float(fg, modules[k])


class _Manual(torch.nn.Module):
    """Hand-written RMSNorm, causal attention with a buffer mask, F.softmax and an exp/amax/sum softmax."""

    def __init__(self, decomposed_softmax):
        super().__init__()
        torch.manual_seed(1)
        self.w = torch.nn.Parameter(torch.rand(D) + 0.5)
        self.qkv = torch.nn.Linear(D, 3 * D, bias=False)
        self.proj = torch.nn.Linear(D, D, bias=False)
        self.register_buffer("bias", torch.tril(torch.ones(8, 8)).view(1, 8, 8))
        self.decomposed = decomposed_softmax

    def forward(self, x):
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.w
        q, k, v = self.qkv(h).split(D, dim=-1)
        t = x.shape[-2]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))
        att = att.masked_fill(self.bias[:, :t, :t] == 0, float("-inf"))
        if self.decomposed:
            e = torch.exp(att - att.amax(-1, keepdim=True))
            att = e / e.sum(-1, keepdim=True)
        else:
            att = F.softmax(att, dim=-1)
        return self.proj(att @ v)


@pytest.mark.parametrize("decomposed", [False, True])
def test_manual_attention_recognised(decomposed):
    from anvil.frontend import capture, quantize

    m = _Manual(decomposed).double().eval()
    fg = capture(m, (_x(),))
    assert _ops(fg) == ["input", "rmsnorm", "linear", "linear", "linear", "transpose", "matmul", "softmax",
                        "matmul", "linear"]
    rms = fg.node("w") if any(n.name == "w" for n in fg.nodes) else fg.nodes[1]
    assert rms.attrs["eps"] == 1e-5 and rms.attrs["weight"] is not None
    scores = [n for n in fg.nodes if n.op == "matmul"][0]
    assert scores.attrs["factor"] == ("mul", 1.0 / math.sqrt(D))
    assert [n for n in fg.nodes if n.op == "softmax"][0].attrs["causal"] is True
    _check_float(fg, m)
    q = quantize(fg, np.random.default_rng(0).standard_normal((4096, T, D)))
    assert _int_error_lsb(q, m) <= 6


def _nanogpt_blocks(n, nh=1, seed=0):
    sys.path.insert(0, nanogpt_dir())
    from model import GPTConfig
    from models.gpt import Block

    cfg = GPTConfig(block_size=T, n_layer=n, n_head=nh, n_embd=D, dropout=0.0, bias=False, pos_embedding="none",
                    norm_type="rmsnorm", activation="swiglu")
    blocks = []
    for i in range(n):
        torch.manual_seed(seed + i)
        blocks.append(Block(cfg).double().eval())

    class Stack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.b = torch.nn.ModuleList(blocks)

        def forward(self, x):
            for b in self.b:
                x = b(x)[0]
            return x

    return Stack().eval()


@pytest.mark.parametrize("layers", [1, 2])
def test_full_blocks_with_residuals(layers):
    """SG5's shape at d = 8: complete nanoGPT Blocks (attention + MLP, both residuals, hidden 256), stacked."""
    from anvil.frontend import from_torch

    m = _nanogpt_blocks(layers)
    q = from_torch(m, (_x(),), np.random.default_rng(0).standard_normal((4096, T, D)))
    ops = _ops(q.graph)
    assert ops.count("add") == 2 * layers and ops.count("softmax") == layers and ops.count("rmsnorm") == 2 * layers
    _check_float(q.graph, m)
    assert len(q.bounds) == 15 * layers                                   # every requant op proven
    assert _int_error_lsb(q, m) <= 6


# ------------------------------------------------------------------------------------------------ determinism

@pytest.mark.parametrize("block", ["mlp_block", "attention"])
def test_capture_deterministic(block, modules):
    from anvil.frontend import capture, quantize

    a, b = capture(modules[block], (_x(),)), capture(modules[block], (_x(),))
    assert a.text() == b.text()
    assert capture(modules[block], (_x(2),)).text() == a.text()          # example batch size does not matter
    c = calib(block)[:1024]
    qa, qb = quantize(a, c, names=NAMES[block]), quantize(b, c, names=NAMES[block])
    assert print_ir(qa.module) == print_ir(qb.module)
    assert qa.consts.keys() == qb.consts.keys() and all(np.array_equal(qa.consts[k], qb.consts[k]) for k in qa.consts)
