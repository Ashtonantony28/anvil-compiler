"""Graph IR for the xf workloads, built by hand from an xf fixture image (SG1; the torch.export path is SG4).

    module, consts = mlp_block(img)          # x + down(swiglu(gate(h), up(h))), h = rmsnorm(x)
    module, consts = attention_head(img)     # y = (softmax(q kT) v) Wo, q/k/v = rmsnorm(x) Wq/Wk/Wv

Requant constants become op properties; weights and LUTs are `const` ops bound by segment name to `consts`
(or inlined as dense attributes with inline=True). Every op's sym is the image segment it produces, so the graph
interpreter's values line up with xf.export.mlp_block_int / xf.attention.attention_int key by key.
"""
from ..ir.graph import GraphBuilder


def _c(img, name):
    return int(img.read(name)[0])


class _FromImage:
    def __init__(self, img, sym_name, inline):
        self.img, self.inline = img, inline
        self.g = GraphBuilder(sym_name)
        self.consts = {}

    def seg(self, name):
        s = self.img.segments[name]
        return s["shape"], s["dtype"], s["scale"]

    def input(self, name):
        shape, dtype, scale = self.seg(name)
        return self.g.input(name, shape, dtype, scale)

    def const(self, name):
        shape, dtype, scale = self.seg(name)
        data = self.img.read(name)
        self.consts[name] = data
        return self.g.const(name, shape, dtype, scale, value=data if self.inline else None)

    def scale(self, name):
        return self.img.segments[name]["scale"]


def mlp_block(img, inline=False, sym_name="mlp_block"):
    """The xf MLP block (xf/export.py mlp_block_int) as anvil.graph -> (module, consts)."""
    b = _FromImage(img, sym_name, inline)
    g, c = b.g, lambda n: _c(img, n)
    x = b.input("x")
    h = g.rmsnorm_int("h", x, b.const("norm_w"), b.const("rsqrt_lut"), c("norm_M"), c("norm_s"), b.scale("h"))
    gate = g.qlinear("gate", h, b.const("w_gate"), c("gate_M"), c("gate_s"), b.scale("gate"))
    up = g.qlinear("up", h, b.const("w_up"), c("up_M"), c("up_s"), b.scale("up"))
    act = g.lut_mul("act", gate, up, b.const("silu_lut"), c("act_M"), c("act_s"), b.scale("act"))
    down = g.qlinear("down", act, b.const("w_down"), c("down_M"), c("down_s"), b.scale("down"))
    out = g.add_requant("out", x, down, c("res_Ma"), c("res_Mb"), c("res_s"), b.scale("out"))
    g.output(out)
    return g.module(), b.consts


def attention_head(img, inline=False, sym_name="attention_head"):
    """The xf causal attention head (xf/attention.py attention_int) as anvil.graph -> (module, consts)."""
    b = _FromImage(img, sym_name, inline)
    g, c = b.g, lambda n: _c(img, n)
    x = b.input("x")
    h = g.rmsnorm_int("h", x, b.const("norm_w"), b.const("rsqrt_lut"), c("norm_M"), c("norm_s"), b.scale("h"))
    q, k, v = (g.qlinear(p, h, b.const(f"w_{p}"), c(f"{p}_M"), c(f"{p}_s"), b.scale(p)) for p in ("q", "k", "v"))
    kT = g.transpose("kT", k, [1, 0], b.scale("kT"))
    scores = g.qmatmul("scores", q, kT, c("scores_M"), c("scores_s"), b.scale("scores"))
    p = g.softmax_lut("p", scores, b.const("exp_lut"), c("one"), True, b.scale("p"))
    o = g.qmatmul("o", p, v, c("o_M"), c("o_s"), b.scale("o"))
    y = g.qlinear("y", o, b.const("w_o"), c("y_M"), c("y_s"), b.scale("y"))
    g.output(y)
    return g.module(), b.consts


BLOCKS = {"mlp_block": mlp_block, "attention": attention_head}
