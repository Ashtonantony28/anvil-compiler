"""SG4a capture: torch.export -> a small float op graph (FGraph) that quantize.py lowers to anvil.graph.

    fg = capture(module, (x,))           # x: (B, T, C) example input; the leading dim is the batch
    print(fg.text())                     # deterministic textual form
    acts = fg.evaluate({"x": xs})        # float activations of every node on a batch (torch, float64)

Pipeline. `torch.export.export` (eval mode) gives an ATen graph; `run_decompositions` lowers it to core ATen
except for the ops in KEEP (linear, matmul, silu, split, softmax, scaled_dot_product_attention), which stay
whole so their weights and structure survive. Every node that depends only on parameters, buffers and literals
is constant-folded (causal mask construction, buffer slicing). Every other node must be in SUPPORTED, or capture
raises UnsupportedOpError naming each offending op. The remaining ATen nodes are then pattern-matched, from the
outputs back, into the frontend op set:

    input                         graph input (per-sample shape, the batch dim dropped)
    rmsnorm (x) {weight, eps}     x * rsqrt(mean(x^2, -1) + eps) * w, as RMSNorm decomposes (with its dtype casts)
    linear  (x) {weight, rows}    x @ W[rows].T, W a parameter in torch (out, in) layout; `rows` from a split
    matmul  (a, b) {factor}       (a @ b) * factor (None, ("inv_sqrt", n) or ("mul", c))
    softmax (x) {causal}          softmax over the last axis, optionally under a lower-triangular mask
    silu (x), mul (a, b), add (a, b)
    transpose (x) {perm}, reshape (x) {shape}
    split_heads (x) {nh}, merge_heads (x) {nh}     (B,T,C) <-> (B,nh,T,hd); identities when nh == 1

Recognised patterns: RMSNorm (pow/mean/add/rsqrt/mul with optional casts and weight), SiLU as x*sigmoid(x),
softmax as aten softmax or exp(z - amax z) / sum, causal masks as masked_fill/where with a constant
upper-triangular mask, scaled_dot_product_attention (expanded to matmul/softmax/matmul), a scalar multiply of a
matmul (folded into its factor), split of a linear output (one linear per chunk), and the nanoGPT head
reshapes. torch is imported lazily so the module imports without it.
"""
from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field


class UnsupportedOpError(NotImplementedError):
    """An op (or a use of an op) outside the frontend's whitelist; the message names the ATen op(s)."""


# ------------------------------------------------------------------------------------------------ FGraph

@dataclass(eq=False)
class FNode:
    op: str
    name: str
    inputs: list
    attrs: dict
    shape: tuple                      # per-sample shape (no batch dim)
    order: tuple = (0,)               # position in the exported graph (sort key)

    def __repr__(self):
        return f"FNode({self.op} {self.name} {list(self.shape)})"


def _fmt_attr(v):
    import numpy as np

    if isinstance(v, np.ndarray):
        return f"array{list(v.shape)}"
    return repr(v)


@dataclass
class FGraph:
    nodes: list                       # topological, in exported-graph order
    inputs: list
    outputs: list
    batch: int = 1                    # batch size of the example inputs

    def node(self, name) -> FNode:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(name)

    def users(self, node) -> list:
        return [n for n in self.nodes if any(i is node for i in n.inputs)]

    def text(self) -> str:
        lines = []
        for n in self.nodes:
            ins = ", ".join(i.name for i in n.inputs)
            attrs = ", ".join(f"{k}={_fmt_attr(v)}" for k, v in sorted(n.attrs.items()) if k != "weight")
            lines.append(f"{n.name} = {n.op}({ins}) {{{attrs}}} : {list(n.shape)}")
        lines.append("return " + ", ".join(o.name for o in self.outputs))
        return "\n".join(lines) + "\n"

    def evaluate(self, inputs) -> dict:
        """Float activations {node name: torch tensor} on batched inputs (dict by input name, or a sequence)."""
        import torch
        import torch.nn.functional as F

        if not isinstance(inputs, dict):
            inputs = {n.name: x for n, x in zip(self.inputs, inputs, strict=True)}
        env, memo = {}, {}
        for n in self.nodes:
            a = [env[i.name] for i in n.inputs]
            at = n.attrs
            if n.op == "input":
                v = torch.as_tensor(inputs[n.name], dtype=at["dtype"])
            elif n.op == "rmsnorm":
                x = a[0].to(at["compute_dtype"])
                v = (x * x.pow(2).mean(dim=-1, keepdim=True).add(at["eps"]).rsqrt()).to(at["dtype"])
                if at["weight"] is not None:
                    v = v * torch.as_tensor(at["weight"], dtype=v.dtype)
            elif n.op == "linear":
                key = (n.inputs[0].name, at["weight_name"])
                if key not in memo:
                    memo[key] = F.linear(a[0], torch.as_tensor(at["weight"], dtype=a[0].dtype))
                v = memo[key]
                if at["rows"] is not None:
                    v = v[..., at["rows"][0]:at["rows"][1]]
            elif n.op == "matmul":
                v = torch.matmul(a[0], a[1])
                f = at["factor"]
                if f is not None:
                    v = v / math.sqrt(f[1]) if f[0] == "inv_sqrt" else v * f[1]
            elif n.op == "softmax":
                x = a[0]
                if at["causal"]:
                    keep = torch.ones(x.shape[-2], x.shape[-1], dtype=torch.bool).tril()
                    x = x.masked_fill(~keep, float("-inf"))
                v = F.softmax(x, dim=-1)
            elif n.op == "silu":
                v = F.silu(a[0])
            elif n.op == "mul":
                v = a[0] * a[1]
            elif n.op == "add":
                v = a[0] + a[1]
            elif n.op == "transpose":
                nb = a[0].dim() - len(at["perm"])
                v = a[0].permute(*range(nb), *(nb + p for p in at["perm"]))
            elif n.op == "reshape":
                v = a[0].reshape(*a[0].shape[:a[0].dim() - len(n.inputs[0].shape)], *n.shape)
            elif n.op == "split_heads":
                x = a[0]
                v = x.reshape(*x.shape[:-1], at["nh"], x.shape[-1] // at["nh"]).transpose(-3, -2)
            elif n.op == "merge_heads":
                x = a[0].transpose(-3, -2)
                v = x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
            else:
                raise NotImplementedError(n.op)
            env[n.name] = v
        return env


# ------------------------------------------------------------------------------------------- ATen level

def _aten():
    import torch

    return torch.ops.aten


def keep_ops():
    """ATen ops kept whole through run_decompositions (their structure carries weights and semantics)."""
    a = _aten()
    return (a.linear.default, a.matmul.default, a.silu.default, a.split.Tensor, a.softmax.int,
            a.scaled_dot_product_attention.default)


def supported_ops():
    """Whitelist of ATen ops allowed on input-dependent values (constant subgraphs are folded instead)."""
    a = _aten()
    return {
        a._to_copy.default, a.to.dtype, a.pow.Tensor_Scalar, a.mean.dim, a.add.Tensor, a.sub.Tensor,
        a.mul.Tensor, a.mul.Scalar, a.div.Tensor, a.div.Scalar, a.rsqrt.default, a.exp.default, a.amax.default,
        a.sum.dim_IntList, a.linear.default, a.matmul.default, a.silu.default, a.sigmoid.default,
        a.split.Tensor, a.split_with_sizes.default, operator.getitem, a.view.default, a.reshape.default,
        a._unsafe_view.default, a.permute.default, a.transpose.int, a.clone.default, a.alias.default,
        a.expand.default, a.scaled_dot_product_attention.default, a._softmax.default, a.softmax.int,
        a.where.self, a.masked_fill.Scalar, a.dropout.default,
    }


def _ignored_ops():
    a = _aten()
    return {a._assert_tensor_metadata.default}


def op_name(target) -> str:
    """'aten.gelu.default' for an OpOverload, the qualified name otherwise."""
    if hasattr(target, "_schema") and hasattr(target, "_overloadname"):
        ns = target.namespace if hasattr(target, "namespace") else "aten"
        return f"{ns}.{target._schema.name.split('::')[-1]}.{target._overloadname}"
    return getattr(target, "__qualname__", None) or getattr(target, "__name__", None) or str(target)


@dataclass(eq=False)
class _P:
    """An exported-graph value: a constant (torch tensor), a batched activation, or a tuple (split)."""
    kind: str                                   # "const" | "act" | "tuple"
    target: object = None
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    name: str = ""
    index: int = 0
    shape: tuple = ()                           # per-sample shape for acts
    dtype: object = None
    value: object = None                        # torch tensor (const) / python scalar
    param: str | None = None                    # state_dict target for parameter / buffer placeholders
    users: list = field(default_factory=list)

    @property
    def op(self):
        return op_name(self.target) if self.target is not None else self.kind


def _norm_dim(d, rank):
    """An ATen dim of a batched tensor whose per-sample rank is `rank`, as a negative per-sample index."""
    d, full = int(d), rank + 1
    if not -rank <= d < full or d == 0:
        raise UnsupportedOpError(f"dim {d} of a rank-{full} batched tensor (the batch dim or out of range)")
    return d - full if d > 0 else d


class _Capture:
    def __init__(self, ep, batch):
        self.ep, self.batch = ep, batch
        self.vals = {}                          # fx node -> _P (or literal)
        self.memo = {}                          # _P -> FNode (or a tuple of FNodes for split)
        self.names = set()

    # ---- exported graph -> _P
    def build(self):
        import torch

        ep = self.ep
        sig = ep.graph_signature
        consts = {}
        consts.update({k: ep.state_dict[v] for k, v in sig.inputs_to_parameters.items()})
        consts.update({k: (ep.state_dict[v] if v in ep.state_dict else ep.constants[v])
                       for k, v in sig.inputs_to_buffers.items()})
        consts.update({k: ep.constants[v] for k, v in sig.inputs_to_lifted_tensor_constants.items()})
        owner = {**sig.inputs_to_parameters, **sig.inputs_to_buffers, **sig.inputs_to_lifted_tensor_constants}
        user_inputs = list(sig.user_inputs)
        bad, inputs, outputs = {}, [], []
        nodes = list(ep.graph.nodes)
        for idx, n in enumerate(nodes):
            if n.op == "placeholder":
                if n.name in consts:
                    self.vals[n] = _P("const", name=n.name, index=idx, value=consts[n.name].detach(),
                                      param=owner[n.name])
                elif n.name in user_inputs:
                    v = n.meta["val"]
                    if v.dim() < 1 or v.shape[0] != self.batch:
                        raise UnsupportedOpError(f"input {n.name}: leading (batch) dim must be {self.batch}")
                    p = _P("act", name=n.name, index=idx, shape=tuple(v.shape[1:]), dtype=v.dtype)
                    self.vals[n] = p
                    inputs.append(p)
                else:
                    raise UnsupportedOpError(f"placeholder {n.name} is neither a user input nor a constant")
                continue
            if n.op == "output":
                outputs = [self.vals[a] for a in n.args[0]]
                continue
            if n.op != "call_function":
                raise UnsupportedOpError(f"{n.op} node {n.name} ({n.target})")
            if n.target in _ignored_ops():
                continue
            args = torch.fx.node.map_arg(n.args, lambda a: self.vals[a])
            kwargs = torch.fx.node.map_arg(n.kwargs, lambda a: self.vals[a])
            flat = [x for x in _flatten((args, kwargs)) if isinstance(x, _P)]
            if all(x.kind == "const" for x in flat) and n.target is not operator.getitem:
                val = n.target(*_unwrap(args), **_unwrap(kwargs))      # constant-fold
                self.vals[n] = _P("const", target=n.target, name=n.name, index=idx, value=val)
                continue
            if n.target not in supported_ops():
                bad.setdefault(op_name(n.target), []).append(n.name)
                self.vals[n] = _P("act", target=n.target, name=n.name, index=idx, shape=(), dtype=None)
                continue
            v = n.meta.get("val")
            if n.target is operator.getitem:
                src = args[0]
                if src.kind == "const":
                    self.vals[n] = _P("const", name=n.name, index=idx, value=src.value[args[1]])
                    continue
            if isinstance(v, (tuple, list)):
                p = _P("tuple", target=n.target, args=args, kwargs=kwargs, name=n.name, index=idx)
            else:
                shape = tuple(v.shape[1:]) if v.dim() >= 1 and v.shape[0] == self.batch else None
                if shape is None:
                    raise UnsupportedOpError(f"{op_name(n.target)} (node {n.name}) drops the batch dimension")
                p = _P("act", target=n.target, args=args, kwargs=kwargs, name=n.name, index=idx,
                       shape=shape, dtype=v.dtype)
            for x in flat:
                x.users.append(p)
            self.vals[n] = p
        if bad:
            raise UnsupportedOpError("unsupported op(s): " + ", ".join(
                f"{k} (node{'s' if len(v) > 1 else ''} {', '.join(v)})" for k, v in sorted(bad.items())))
        for o in outputs:
            if not isinstance(o, _P) or o.kind != "act":
                raise UnsupportedOpError("every graph output must depend on a user input")
        return inputs, outputs

    # ---- _P -> FNode
    def fresh(self, name):
        base, i = name, 1
        while name in self.names:
            i += 1
            name = f"{base}_{i}"
        self.names.add(name)
        return name

    def mk(self, op, name, inputs, shape, order, **attrs):
        return FNode(op, self.fresh(name), list(inputs), attrs, tuple(shape), order)

    def lower(self, p):
        if p in self.memo:
            return self.memo[p]
        if not isinstance(p, _P) or p.kind != "act":
            raise UnsupportedOpError(f"constant {getattr(p, 'name', p)!r} used as an activation")
        r = self._lower(p)
        self.memo[p] = r
        return r

    def _lower(self, p):
        a, op = _aten(), p.target
        order = (p.index,)
        if op is None:                                            # user input
            return self.mk("input", p.name, [], p.shape, order, dtype=p.dtype)
        for m in (self._rmsnorm, self._softmax, self._silu_mul):
            r = m(p)
            if r is not None:
                return r
        if op in (a._to_copy.default, a.to.dtype, a.clone.default, a.alias.default):
            return self.lower(p.args[0])
        if op is a.dropout.default:
            if p.args[1] != 0 and p.args[2]:
                raise UnsupportedOpError(f"aten.dropout.default (node {p.name}) with p > 0 in training mode")
            return self.lower(p.args[0])
        if op is a.expand.default:
            if tuple(p.args[0].shape) != p.shape:
                raise UnsupportedOpError(f"aten.expand.default (node {p.name}) that broadcasts")
            return self.lower(p.args[0])
        if op is a.linear.default:
            return self._linear(p, None, p.name)
        if op is operator.getitem:
            return self._getitem(p)
        if op is a.matmul.default:
            x, y = self.lower(p.args[0]), self.lower(p.args[1])
            return self.mk("matmul", p.name, [x, y], p.shape, order, factor=None)
        if op in (a.mul.Tensor, a.mul.Scalar, a.div.Tensor, a.div.Scalar):
            return self._mul(p)
        if op is a.add.Tensor:
            x, y = p.args[0], p.args[1]
            if not (isinstance(x, _P) and isinstance(y, _P) and x.kind == y.kind == "act"):
                raise UnsupportedOpError(f"aten.add.Tensor (node {p.name}) with a constant outside an RMSNorm")
            if p.kwargs.get("alpha", 1) != 1 or (len(p.args) > 2 and p.args[2] != 1):
                raise UnsupportedOpError(f"aten.add.Tensor (node {p.name}) with alpha != 1")
            return self.mk("add", p.name, [self.lower(x), self.lower(y)], p.shape, order)
        if op is a.silu.default:
            return self.mk("silu", p.name, [self.lower(p.args[0])], p.shape, order)
        if op is a.scaled_dot_product_attention.default:
            return self._sdpa(p)
        if op in (a.permute.default, a.transpose.int):
            return self._permute(p)
        if op in (a.view.default, a.reshape.default, a._unsafe_view.default):
            return self._view(p)
        raise UnsupportedOpError(f"{p.op} (node {p.name}) outside a recognised pattern")

    # ---- patterns
    def _rmsnorm(self, p):
        """mul(core, w) or core, core = cast?(mul(a, rsqrt(add(mean(pow(a, 2), [-1], True), eps))))."""
        a = _aten()
        if p.target not in (a.mul.Tensor, a._to_copy.default, a.to.dtype):
            return None
        weight = None
        core = p
        if p.target is a.mul.Tensor:
            x, y = p.args[0], p.args[1]
            for t, w in ((x, y), (y, x)):
                if isinstance(w, _P) and w.kind == "const" and w.value.dim() == 1 and isinstance(t, _P) \
                        and t.kind == "act" and _norm_core(t) is not None:
                    core, weight = t, w
                    break
        m = _norm_core(core)
        if m is None:
            return None
        src, eps = m
        x = _strip_casts(src)
        if weight is not None and weight.value.shape[0] != x.shape[-1]:
            return None
        name = weight.param.rsplit(".", 1)[0] if weight is not None and weight.param else p.name
        import numpy as np

        return self.mk("rmsnorm", name, [self.lower(x)], p.shape, (p.index,),
                       weight=None if weight is None else weight.value.detach().double().numpy().astype(np.float64),
                       weight_name=None if weight is None else weight.param, eps=float(eps),
                       compute_dtype=src.dtype, dtype=core.dtype)

    def _softmax(self, p):
        a = _aten()
        z = None
        if p.target in (a.softmax.int, a._softmax.default):
            if _norm_dim(p.args[1], len(p.shape)) != -1:
                raise UnsupportedOpError(f"{p.op} (node {p.name}) over a dim other than the last")
            z = p.args[0]
        elif p.target in (a.div.Tensor,) and isinstance(p.args[1], _P) and p.args[1].kind == "act":
            e, s = p.args
            if s.target is a.sum.dim_IntList and s.args[0] is e and _last_keepdim(s) and e.target is a.exp.default:
                sub = e.args[0]
                if sub.target is a.sub.Tensor and isinstance(sub.args[1], _P) and sub.args[1].target is a.amax.default \
                        and sub.args[1].args[0] is sub.args[0] and _last_keepdim(sub.args[1]):
                    z = sub.args[0]
        if z is None:
            return None
        causal, inner = self._causal_mask(z)
        return self.mk("softmax", p.name, [self.lower(inner)], p.shape, (p.index,), causal=causal)

    def _causal_mask(self, z):
        """(causal, scores) for z = scores, masked_fill(scores, ~tril, -inf) or where(...)."""
        import torch

        a = _aten()
        if z.target is a.masked_fill.Scalar and isinstance(z.args[1], _P) and z.args[1].kind == "const":
            fill_where, val, inner = z.args[1].value, z.args[2], z.args[0]
        elif z.target is a.where.self and isinstance(z.args[0], _P) and z.args[0].kind == "const":
            c, x, y = z.args
            if isinstance(x, _P) and x.kind == "const" and isinstance(y, _P) and y.kind == "act":
                fill_where, val, inner = c.value, x.value, y
            elif isinstance(y, _P) and y.kind == "const" and isinstance(x, _P) and x.kind == "act":
                fill_where, val, inner = ~c.value.bool(), y.value, x
            else:
                return False, z
        else:
            return False, z
        val = float(val)
        rows, cols = z.shape[-2], z.shape[-1]
        mask = torch.as_tensor(fill_where).bool()
        mask = mask.reshape(mask.shape[-2:]) if mask.dim() >= 2 and all(d == 1 for d in mask.shape[:-2]) else None
        want = ~torch.ones(rows, cols, dtype=torch.bool).tril()
        if val != float("-inf") or mask is None or not torch.equal(mask, want):
            raise UnsupportedOpError(f"{op_name(z.target)} (node {z.name}): only a causal (lower-triangular) "
                                     "-inf mask is supported before softmax")
        return True, inner

    def _silu_mul(self, p):
        """mul(x, sigmoid(x)) -> silu(x)."""
        a = _aten()
        if p.target is not a.mul.Tensor:
            return None
        x, y = p.args[0], p.args[1]
        for u, s in ((x, y), (y, x)):
            if isinstance(s, _P) and s.target is a.sigmoid.default and s.args[0] is u:
                return self.mk("silu", p.name, [self.lower(u)], p.shape, (p.index,))
        return None

    def _mul(self, p):
        a = _aten()
        x, y = p.args[0], p.args[1]
        div = p.target in (a.div.Tensor, a.div.Scalar)
        if isinstance(y, _P) and y.kind == "act" and isinstance(x, _P) and x.kind == "act":
            if div:
                raise UnsupportedOpError(f"{p.op} (node {p.name}) between activations")
            return self.mk("mul", p.name, [self.lower(x), self.lower(y)], p.shape, (p.index,))
        t, c = (x, y) if isinstance(x, _P) and x.kind == "act" else (y, x)
        if div and t is not x:
            raise UnsupportedOpError(f"{p.op} (node {p.name}): constant divided by an activation")
        c = c.value if isinstance(c, _P) else c
        if hasattr(c, "numel"):
            if c.numel() != 1:
                raise UnsupportedOpError(f"{p.op} (node {p.name}) by a non-scalar constant")
            c = c.item()
        c = 1.0 / float(c) if div else float(c)
        inner = self.lower(t)
        if inner.op == "matmul" and inner.attrs["factor"] is None and len(t.users) == 1:
            return self.mk("matmul", p.name, inner.inputs, p.shape, inner.order, factor=("mul", c))
        raise UnsupportedOpError(f"{p.op} (node {p.name}): scaling by a constant is supported only on a "
                                 "matmul result")

    def _linear(self, p, rows, name):
        a = _aten()
        x, w = p.args[0], p.args[1]
        bias = p.args[2] if len(p.args) > 2 else p.kwargs.get("bias")
        if bias is not None:
            raise UnsupportedOpError(f"aten.linear.default (node {p.name}) with a bias")
        if not (isinstance(w, _P) and w.kind == "const"):
            raise UnsupportedOpError(f"aten.linear.default (node {p.name}) with a non-constant weight")
        wname = w.param or w.name
        base = wname.rsplit(".", 1)[0] if w.param and wname.endswith(".weight") else p.name
        assert p.target is a.linear.default
        out = (rows[1] - rows[0]) if rows is not None else w.value.shape[0]
        shape = (*x.shape[:-1], out)
        return self.mk("linear", base if rows is None else f"{base}.{name}", [self.lower(x)], shape, (p.index,),
                       weight=w.value.detach().double().numpy(), weight_name=wname, rows=rows)

    def _getitem(self, p):
        a = _aten()
        src, i = p.args
        if src.kind != "tuple" or src.target not in (a.split.Tensor, a.split_with_sizes.default):
            raise UnsupportedOpError(f"getitem (node {p.name}) of {src.op}")
        lin = src.args[0]
        rank = len(lin.shape)
        dim = _norm_dim(src.args[2] if len(src.args) > 2 else src.kwargs.get("dim", 0), rank)
        if dim != -1:
            raise UnsupportedOpError(f"{src.op} (node {src.name}) along a dim other than the last")
        n = lin.shape[-1]
        sizes = src.args[1]
        if isinstance(sizes, int):
            sizes = [sizes] * (n // sizes) + ([n % sizes] if n % sizes else [])
        starts = [sum(sizes[:j]) for j in range(len(sizes))]
        if lin.target is not a.linear.default or len(lin.users) != 1:
            raise UnsupportedOpError(f"{src.op} (node {src.name}) of {lin.op}: only the output of a linear "
                                     "can be split")
        f = self._linear(lin, (starts[i], starts[i] + sizes[i]), str(i))
        f.order = (p.index,)
        return f

    def _permute(self, p):
        a = _aten()
        x = p.args[0]
        rank = len(p.shape)
        perm = self._perm_of(p)
        if perm is None:
            raise UnsupportedOpError(f"{p.op} (node {p.name}) moves the batch dim")
        # split_heads: permute(view(x, [..., T, nh, hd]), swap the T and nh axes)
        if rank >= 3 and perm == list(range(rank - 3)) + [rank - 2, rank - 3, rank - 1] and \
                x.target in (a.view.default, a.reshape.default, a._unsafe_view.default) and \
                len(x.args[0].shape) == rank - 1 and x.shape[-2] * x.shape[-1] == x.args[0].shape[-1] and \
                tuple(x.shape[:-2]) == tuple(x.args[0].shape[:-1]):
            src, nh = self.lower(x.args[0]), x.shape[-2]
            if nh == 1:
                return src
            return self.mk("split_heads", p.name, [src], p.shape, (p.index,), nh=nh)
        inner = self.lower(x)
        if perm == list(range(rank)):
            return inner
        if inner.shape != tuple(x.shape):        # x was a collapsed (nh == 1) head layout
            raise UnsupportedOpError(f"{p.op} (node {p.name}) of a head-split value")
        return self.mk("transpose", p.name, [inner], p.shape, (p.index,), perm=perm)

    def _view(self, p):
        a = _aten()
        x = p.args[0]
        # merge_heads: view(permute(z, swap T and nh), [..., T, nh * hd])
        if x.target in (a.permute.default, a.transpose.int) and len(x.shape) == len(p.shape) + 1 and \
                x.shape[-2] * x.shape[-1] == p.shape[-1] and tuple(x.shape[:-2]) == tuple(p.shape[:-1]):
            z = x.args[0]
            r = len(z.shape)
            perm = self._perm_of(x)
            if perm is not None and perm == list(range(r - 3)) + [r - 2, r - 3, r - 1]:
                src, nh = self.lower(z), z.shape[-3]
                if nh == 1:
                    return src
                return self.mk("merge_heads", p.name, [src], p.shape, (p.index,), nh=nh)
        inner = self.lower(x)
        if inner.shape == p.shape:
            return inner
        return self.mk("reshape", p.name, [inner], p.shape, (p.index,))

    @staticmethod
    def _perm_of(x):
        """Per-sample permutation of a permute/transpose node, None if it moves the batch dim."""
        a = _aten()
        full_rank = len(x.shape) + 1
        if x.target is a.transpose.int:
            d0, d1 = (int(d) % full_rank for d in x.args[1:3])
            full = list(range(full_rank))
            full[d0], full[d1] = full[d1], full[d0]
        else:
            full = [int(d) % full_rank for d in x.args[1]]
        if full[0] != 0:
            return None
        return [d - 1 for d in full[1:]]

    def _sdpa(self, p):
        args = list(p.args) + [None] * (8 - len(p.args))
        q, k, v, mask, drop, causal = args[:6]
        kw = p.kwargs
        mask, drop = kw.get("attn_mask", mask), kw.get("dropout_p", drop or 0.0)
        causal = kw.get("is_causal", causal or False)
        scale = kw.get("scale", args[6])
        if mask is not None:
            raise UnsupportedOpError(f"{p.op} (node {p.name}) with an explicit attn_mask")
        if drop:
            raise UnsupportedOpError(f"{p.op} (node {p.name}) with dropout")
        if kw.get("enable_gqa", args[7]):
            raise UnsupportedOpError(f"{p.op} (node {p.name}) with enable_gqa")
        fq, fk, fv = self.lower(q), self.lower(k), self.lower(v)
        r, i = len(fk.shape), p.index
        kT = self.mk("transpose", f"{p.name}.kT", [fk], (*fk.shape[:-2], fk.shape[-1], fk.shape[-2]), (i, 0),
                     perm=list(range(r - 2)) + [r - 1, r - 2])
        factor = ("inv_sqrt", int(fq.shape[-1])) if scale is None else ("mul", float(scale))
        sshape = (*fq.shape[:-1], fk.shape[-2])
        scores = self.mk("matmul", f"{p.name}.scores", [fq, kT], sshape, (i, 1), factor=factor)
        prob = self.mk("softmax", f"{p.name}.p", [scores], sshape, (i, 2), causal=bool(causal))
        return self.mk("matmul", p.name, [prob, fv], (*fq.shape[:-1], fv.shape[-1]), (i, 3), factor=None)


def _flatten(x):
    if isinstance(x, (list, tuple)):
        for y in x:
            yield from _flatten(y)
    elif isinstance(x, dict):
        for y in x.values():
            yield from _flatten(y)
    else:
        yield x


def _unwrap(x):
    if isinstance(x, _P):
        return x.value
    if isinstance(x, (list, tuple)):
        return type(x)(_unwrap(y) for y in x)
    if isinstance(x, dict):
        return {k: _unwrap(v) for k, v in x.items()}
    return x


def _strip_casts(p):
    a = _aten()
    while isinstance(p, _P) and p.target in (a._to_copy.default, a.to.dtype):
        p = p.args[0]
    return p


def _last_keepdim(p):
    dims = p.args[1] if len(p.args) > 1 else p.kwargs.get("dim")
    keep = p.args[2] if len(p.args) > 2 else p.kwargs.get("keepdim", False)
    rank = len(p.args[0].shape)
    return dims is not None and [_norm_dim(d, rank) for d in dims] == [-1] and bool(keep)


def _norm_core(t):
    """(a, eps) if t = cast?(mul(a, rsqrt(add(mean(pow(a, 2), [-1], keepdim), eps)))), else None."""
    a = _aten()
    if not isinstance(t, _P) or t.kind != "act":
        return None
    while t.target in (a._to_copy.default, a.to.dtype):
        t = t.args[0]
    if t.target is not a.mul.Tensor:
        return None
    x, y = t.args[0], t.args[1]
    for src, r in ((x, y), (y, x)):
        if not (isinstance(r, _P) and r.target is a.rsqrt.default):
            continue
        add = r.args[0]
        if not (isinstance(add, _P) and add.target is a.add.Tensor):
            continue
        mean, eps = add.args[0], add.args[1]
        if isinstance(eps, _P):
            mean, eps = (eps, mean) if eps.kind == "act" else (mean, eps.value)
        if hasattr(eps, "item"):
            eps = eps.item()
        if not (isinstance(mean, _P) and mean.target is a.mean.dim and _last_keepdim(mean)):
            continue
        pw = mean.args[0]
        if isinstance(pw, _P) and pw.target is a.pow.Tensor_Scalar and pw.args[0] is src and pw.args[1] == 2:
            return src, eps
        if isinstance(pw, _P) and pw.target is a.mul.Tensor and pw.args[0] is src and pw.args[1] is src:
            return src, eps
    return None


# ----------------------------------------------------------------------------------------------- entry

def export_program(module, example_inputs):
    """torch.export + run_decompositions keeping keep_ops() whole -> ExportedProgram."""
    import torch

    module = module.eval()
    ep = torch.export.export(module, tuple(example_inputs))
    table = torch.export.default_decompositions()
    for op in keep_ops():
        table.pop(op, None)
    return ep.run_decompositions(table)


def capture(module, example_inputs) -> FGraph:
    """torch.export `module` on `example_inputs` (each with a leading batch dim) -> FGraph.
    Raises UnsupportedOpError naming every ATen op outside the whitelist or a recognised pattern."""
    example_inputs = tuple(example_inputs)
    if not example_inputs:
        raise ValueError("capture needs at least one example input")
    batch = int(example_inputs[0].shape[0])
    ep = export_program(module, example_inputs)
    c = _Capture(ep, batch)
    inputs, outputs = c.build()
    fin = [c.lower(i) for i in inputs]
    fout = [c.lower(o) for o in outputs]
    nodes, seen = [], set()

    def visit(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        for i in n.inputs:
            visit(i)
        nodes.append(n)

    for n in fin + fout:
        visit(n)
    nodes.sort(key=lambda n: n.order)
    return FGraph(nodes, fin, fout, batch)
