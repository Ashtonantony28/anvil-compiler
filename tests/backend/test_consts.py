"""const32 materialization: every 32-bit word (signed or unsigned spelling) is built exactly, in few
instructions, and constants are CSE'd across a function."""
from hypothesis import given, settings, strategies as st

from anvil import semantics as sem
from anvil.backend import compile_mir, consts
from anvil.backend.lir import Fresh, Ins
from anvil.ir import mir
from xf import isa

from backend_util import interp, sim

EDGES = [0, 1, 255, 256, 257, 511, 4096, 65535, 65536, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF,
         0xFFFFFF80, 0xFFFF0000, 0x01000000, 0x00FF00FF, 0x12345678, 0xDEADBEEF]


def eval_tree(code, root):
    regs = {}
    for i in code:
        if i.op == "const":
            regs[i.d] = i.imm
        else:
            regs[i.d] = {"mul": sem.w_mul, "add": sem.w_add, "sub": sem.w_sub}[i.op](regs[i.a], regs[i.b])
    return regs[root]


@settings(max_examples=300, deadline=None)
@given(st.one_of(st.integers(0, (1 << 32) - 1), st.sampled_from(EDGES), st.integers(0, 1 << 17)))
def test_plan_builds_the_word(v):
    code, root, regs = consts.emit_tree(v, Fresh([]))
    assert eval_tree(code, root) == v
    assert all(i.op != "const" or 0 <= i.imm <= 255 for i in code)
    assert len(code) == consts.cost(v) == len(set(regs))
    assert code[-1].d == root


def test_plan_costs():
    assert consts.cost(200) == 1
    assert consts.cost(256) == 2                         # CONST 16; MUL
    assert consts.cost(1000) == 3                        # 8 * 125
    assert consts.cost(0xFFFFFFFF) == 3                  # 0 - 1
    assert consts.cost(0xFFFFFF80) == 3                  # 0 - 128
    assert max(consts.cost(v) for v in range(1 << 16)) <= 6         # CONST hi; CONST 16; MUL; MUL; CONST lo; ADD
    assert max(consts.cost(v) for v in EDGES) <= 12        # 4-byte Horner: 1 + 2 + 3 * 3


@settings(max_examples=40, deadline=None)
@given(st.lists(st.one_of(st.integers(-(1 << 31), (1 << 32) - 1), st.sampled_from(EDGES)), min_size=1,
                max_size=10))
def test_const32_compiled_on_isasim(values):
    ops = []
    for k, v in enumerate(values):
        ops += [mir.Const32Op(f"v{2 * k}", v), mir.Const32Op(f"v{2 * k + 1}", 100 + k),
                mir.StrOp(f"v{2 * k + 1}", f"v{2 * k}")]
    f = mir.func("c", ops + [mir.RetOp()])
    p = compile_mir(f, args_base=0)
    got = sim(p.words, [], 1).memory
    assert got[100:100 + len(values)] == [sem.wrap(v) for v in values] == interp(f, [], 1)[100:100 + len(values)]


def test_cse_one_materialization_per_value():
    """The same wide value in five registers and both spellings of -1: built once each, at entry."""
    ops = []
    for k in range(5):
        ops += [mir.Const32Op(f"v{k}", 0x12345678), mir.Const32Op(f"v{10 + k}", 200 + k),
                mir.StrOp(f"v{10 + k}", f"v{k}")]
    ops += [mir.Const32Op("v20", -1), mir.Const32Op("v21", 0xFFFFFFFF), mir.AddOp("v22", "v20", "v21"),
            mir.Const32Op("v23", 210), mir.StrOp("v23", "v22"), mir.RetOp()]
    f = mir.func("cse", ops)
    p = compile_mir(f, args_base=0)
    body = [isa.decode(w) for w in p.words[6:]]
    assert p.stats["const_values"] <= consts.cost(0x12345678) + 5 + 1 + 3 + 1
    assert sum(m == "CONST" and fl["imm"] == 0x12 for m, fl in body) <= 1
    assert sim(p.words, [], 1).memory[200:211] == [0x12345678] * 5 + [0] * 5 + [sem.wrap(-2)]


def test_multi_def_register_keeps_its_constant_definition():
    """A loop counter initialized by const32 and then incremented is not a constant: lowered in place."""
    ops = [mir.Const32Op("v0", 70000), mir.Const32Op("v1", 1), mir.Const32Op("v2", 70003),
           mir.LabelOp("top"), mir.AddOp("v0", "v0", "v1"), mir.CmpOp("v0", "v2"), mir.BrOp(0b100, "top"),
           mir.Const32Op("v3", 300), mir.StrOp("v3", "v0"), mir.RetOp()]
    f = mir.func("ctr", ops)
    assert sim(compile_mir(f, args_base=0).words, [], 1).memory[300] == 70003


def test_canonicalize_hoists_to_entry():
    code = [Ins("label", sym="top"), Ins("const32", "v0", imm=4096), Ins("str", None, "v0", "v0"),
            Ins("const", "v1", imm=7), Ins("str", None, "v1", "v1")]
    out, vals = consts.canonicalize(code, Fresh(code))
    first_label = next(k for k, i in enumerate(out) if i.op == "label")
    assert all(i.op in ("const", "mul", "add", "sub") for i in out[:first_label])
    assert sorted(vals.values()) == sorted([*consts.plan(4096), 7])
