"""Emission: size limit, ARGS prologue compatible with xf.kernels.program patching, labels, asm text, peephole,
input validation, sreg lowering, DCE."""
import pytest

from anvil.backend import BackendError, ProgramTooLarge, compile_mir
from anvil.backend.emit import peephole
from anvil.backend.idioms import Builder
from anvil.backend.lir import Ins
from anvil.ir import mir
from xf import isa
from xf.asm import assemble

from backend_util import interp, sim


def _store_sum(n_adds):
    b = Builder()
    x = b.ldr("R12")
    for _ in range(n_adds):
        x = b.add(x, "R15")
    b.str_(b.add("R15", 100), x)
    b.emit(mir.RetOp())
    return mir.func("long", b.ops)


def test_oversize_program_names_the_overflow():
    compile_mir(_store_sum(240), args_base=0)                     # fits
    with pytest.raises(ProgramTooLarge, match=r"'long' needs 2\d\d words, \d+ over the 256-word"):
        compile_mir(_store_sum(260), args_base=0)


@pytest.mark.parametrize("base", [0, 7, 255, 256, 0x0ABC, 0xFFFF])
def test_prologue_patching_is_xf_compatible(base):
    """Patching words 0 and 4 the way xf.kernels.program does == compiling at that base."""
    f = _store_sum(3)
    p0 = compile_mir(f, args_base=0x0102)
    patched = list(p0.words)
    for i, imm in ((0, base >> 8), (4, base & 0xFF)):
        m, fl = isa.decode(patched[i])
        assert m == "CONST"
        patched[i] = isa.encode("CONST", rd=fl["rd"], imm=imm)
    assert patched == compile_mir(f, args_base=base).words
    assert isa.decode(p0.words[0]) == ("CONST", {"rd": 12, "imm": 1})


def test_prologue_runs_and_reads_args():
    f = _store_sum(2)
    data = [0] * 300 + [41]
    p = compile_mir(f, args_base=300)
    assert sim(p.words, data, 4).memory[100:104] == [41 + 2 * t for t in range(4)] == interp(f, data, 4, 300)[100:104]
    with pytest.raises(BackendError, match="args_base"):
        compile_mir(f, args_base=1 << 16)


def test_asm_text_reassembles_with_labels_and_branches():
    ops = [mir.Const32Op("v0", 0), mir.Const32Op("v1", 3), mir.LabelOp("loop"), mir.SyncOp("loop"),
           mir.AddOp("v0", "v0", "R15"), mir.CmpOp("v0", "v1"), mir.BrOp(0b101, "loop"), mir.BarOp(),
           mir.Const32Op("v2", 900), mir.StrOp("v2", "v0"), mir.RetOp()]
    p = compile_mir(mir.func("lbl", ops), args_base=5)
    assert assemble(p.asm_text) == p.words
    assert "loop:" in p.asm_text and "BRnp loop" in p.asm_text and "SYNC loop" in p.asm_text


def test_label_past_the_end_is_rejected():
    ops = [mir.Const32Op("v0", 1), mir.CmpOp("v0", "v0"), mir.BrOp(0b010, "end"), mir.RetOp(), mir.LabelOp("end")]
    with pytest.raises(BackendError, match="past the last instruction"):
        compile_mir(mir.func("e", ops), args_base=0)


@pytest.mark.parametrize("ops,match", [
    ([mir.AddOp("v0", "R3", "R15")], "physical register R3"),
    ([mir.AddOp("R12", "v0", "R15")], "writes physical register R12"),
])
def test_input_validation(ops, match):
    with pytest.raises(BackendError, match=match):
        compile_mir(mir.func("bad", ops + [mir.RetOp()]), args_base=0)


def test_sreg_single_and_multi_def():
    """Single-def sreg folds into R13-R15 (no instructions); a reassigned one becomes a copy."""
    ops = [mir.SregOp("v0", "threadIdx"), mir.SregOp("v1", "blockIdx"), mir.SregOp("v2", "blockDim"),
           mir.MulOp("v3", "v1", "v2"), mir.AddOp("v3", "v3", "v0"),
           mir.SregOp("v4", "threadIdx"), mir.AddOp("v4", "v4", "v4"),                 # v4 reassigned
           mir.Const32Op("v5", 100), mir.AddOp("v6", "v5", "v3"), mir.StrOp("v6", "v4"), mir.RetOp()]
    f = mir.func("sreg", ops)
    p = compile_mir(f, args_base=0)
    assert sim(p.words, [], 8).memory[100:108] == [2 * (t % 4) for t in range(8)] == interp(f, [], 8)[100:108]
    assert p.stats["body_words"] <= 9


def test_dead_code_is_removed():
    ops = [mir.Const32Op("v0", 70000), mir.AddOp("v1", "v0", "R15"), mir.LdrOp("v2", "v1"),
           mir.Const32Op("v3", 5), mir.StrOp("v3", "R15"), mir.RetOp()]
    p = compile_mir(mir.func("dce", ops), args_base=0)
    assert p.stats["dead_removed"] >= 3 and p.stats["body_words"] == 3


def test_peephole_rules():
    code = [Ins("const", "R0", imm=0), Ins("const", "R1", imm=1), Ins("nop"),
            Ins("add", "R2", "R2", "R0"), Ins("mul", "R2", "R1", "R2"), Ins("sub", "R2", "R2", "R0"),
            Ins("max", "R3", "R3", "R3"), Ins("const", "R0", imm=0),                    # all no-ops
            Ins("const", "R4", imm=16), Ins("mul", "R5", "R4", "R4"), Ins("mul", "R5", "R4", "R4"),   # remat dup
            Ins("br", imm=0b010, sym="next"), Ins("label", sym="next"),
            Ins("const", "R0", imm=0),                                                   # after a label: kept
            Ins("ldr", "R1", "R0"), Ins("const", "R1", imm=1),                           # R1 clobbered: kept
            Ins("ret")]
    out, removed = peephole(code)
    assert [str(i) for i in out] == ["const R0, 0", "const R1, 1", "const R4, 16", "mul R5, R4, R4", "label next",
                                     "const R0, 0", "ldr R1, R0", "const R1, 1", "ret"]
    assert removed == 8
