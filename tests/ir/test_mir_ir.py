"""anvil.mir: one op per ISA opcode, verifier, textual round trip, encode/decode against xf.isa and the xf kernels."""
import pytest
from hypothesis import given, settings, strategies as st
from xdsl.utils.exceptions import VerifyException

from anvil.ir import module, parse_ir, print_ir
from anvil.ir import mir
from xf import isa
from xf.asm import assemble
from xf.kernels import KERNEL_DIR


def test_every_opcode_has_an_op():
    have = {c.MNEMONIC for c in mir.OPS if getattr(c, "MNEMONIC", None)}
    assert have == set(isa.OPCODES) and len(isa.OPCODES) == 16


regs = st.one_of(st.integers(0, 40).map(lambda i: f"v{i}"), st.integers(0, 12).map(lambda i: f"R{i}"))
src = st.one_of(regs, st.sampled_from(["R13", "R14", "R15"]))
labels = st.sampled_from(["L0", "L1", "loop", "exit"])

instr = st.one_of(
    st.builds(lambda c, d, s, t: mir.ALU[c](d, s, t), st.sampled_from(sorted(mir.ALU)), regs, src, src),
    st.builds(mir.CmpOp, src, src),
    st.builds(mir.LdrOp, regs, src), st.builds(mir.LdsOp, regs, src),
    st.builds(mir.StrOp, src, src), st.builds(mir.StsOp, src, src),
    st.builds(mir.ConstOp, regs, st.integers(0, 255)),
    st.builds(mir.Const32Op, regs, st.integers(-(1 << 31), (1 << 32) - 1)),
    st.builds(mir.SregOp, regs, st.sampled_from(sorted(mir.SPECIAL))),
    st.builds(mir.BrOp, st.integers(1, 7), labels), st.builds(mir.SyncOp, labels),
    st.builds(mir.NopOp), st.builds(mir.BarOp), st.builds(mir.RetOp),
)


@settings(max_examples=100, deadline=None)
@given(st.lists(instr, max_size=30))
def test_round_trip_random_mir(ops):
    body = [mir.LabelOp(n) for n in ("L0", "L1", "loop", "exit")] + ops
    m = module(mir.func("rand", body))
    text = print_ir(m)
    again = parse_ir(text)
    assert print_ir(again) == text and again.is_structurally_equivalent(m)


def test_verifier():
    with pytest.raises(VerifyException, match="read-only"):
        mir.func("f", [mir.AddOp("R15", "R0", "R1")])
    with pytest.raises(VerifyException, match="not a register"):
        mir.func("f", [mir.AddOp("x1", "R0", "R1")])
    with pytest.raises(VerifyException, match="0..255"):
        mir.func("f", [mir.ConstOp("v0", 256)])
    with pytest.raises(VerifyException, match="nzp"):
        mir.func("f", [mir.BrOp(0, "L")])
    with pytest.raises(VerifyException, match="which"):
        mir.func("f", [mir.SregOp("v0", "laneId")])
    with pytest.raises(VerifyException, match="duplicate"):
        mir.labels(mir.func("f", [mir.LabelOp("a"), mir.LabelOp("a")]))


@pytest.mark.parametrize("path", sorted(KERNEL_DIR.glob("*.asm")), ids=lambda p: p.stem)
def test_decode_encode_identity_on_xf_kernels(path):
    words = assemble(path.read_text())
    f = mir.decode(words, path.stem)
    f.verify()
    assert mir.encode(f) == words
    assert mir.encode(parse_ir(print_ir(module(f))).body.block.first_op) == words


def test_encode_rejects_unallocated():
    with pytest.raises(ValueError):
        mir.encode(mir.func("f", [mir.AddOp("v0", "R0", "R1")]))
    with pytest.raises(ValueError, match="lowered"):
        mir.encode(mir.func("f", [mir.Const32Op("R0", 1000)]))
