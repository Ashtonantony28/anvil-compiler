"""anvil.tensor: verifier and textual round trip of random loop nests (hypothesis), semantics preserved."""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from xdsl.ir.affine import AffineMap
from xdsl.utils.exceptions import VerifyException

from anvil import semantics as sem
from anvil.interp import tensor as ti
from anvil.ir import parse_ir, print_ir
from anvil.ir import tensor as T

OPS = {"add": sem.add_i32, "sub": sem.sub_i32, "mul": sem.mul_i32, "max": sem.max_i32, "min": sem.min_i32}


@st.composite
def elementwise(draw):
    """out = chain(x or x.T via the load map, small consts), over a random loop structure and binding."""
    r, c = draw(st.integers(1, 5)), draw(st.integers(1, 5))
    transposed = draw(st.booleans())
    split = draw(st.sampled_from([1, 2, 3]))          # split the column loop: j = jo * split + ji (padded + masked)
    binds = draw(st.tuples(st.sampled_from(T.BINDS), st.sampled_from(T.BINDS)))
    chain = draw(st.lists(st.tuples(st.sampled_from(sorted(OPS)), st.integers(-8, 8)), max_size=4))
    b = T.TensorBuilder("ew")
    x = b.alloc("x", [c, r] if transposed else [r, c], "i8")
    out = b.alloc("out", [r, c])
    with b.loop(0, r, bind=binds[0], tag="i") as i:
        with b.loop(0, -(-c // split), bind=binds[1], tag="jo") as jo:
            with b.loop(0, split, unroll=split) as ji:
                # clamp the padded tail into range: column min(jo*split+ji, c-1) — writes the last column again
                col = lambda i, jo, ji: (jo * split + ji)  # noqa: E731
                if transposed:
                    m = AffineMap.from_callable(lambda i, jo, ji: (col(i, jo, ji) % c, i))
                    om = AffineMap.from_callable(lambda i, jo, ji: (i, col(i, jo, ji) % c))
                else:
                    m = om = AffineMap.from_callable(lambda i, jo, ji: (i, col(i, jo, ji) % c))
                v = b.load(x, [i, jo, ji], m)
                for op, k in chain:
                    v = b.binary(op, v, b.const(k))
                b.store(v, out, [i, jo, ji], om)
    xs = np.random.default_rng(draw(st.integers(0, 1 << 30))).integers(-128, 128, (r, c))
    ref = xs
    for op, k in chain:
        ref = OPS[op](ref, k)
    return b.module(), (xs.T if transposed else xs), ref


@settings(max_examples=60, deadline=None)
@given(elementwise())
def test_round_trip_random_loop_nests(case):
    module, x, ref = case
    text = print_ir(module)
    again = parse_ir(text)
    assert print_ir(again) == text and again.is_structurally_equivalent(module)
    assert np.array_equal(ti.run(module, {"x": x})["out"], ref)
    assert np.array_equal(ti.run(again, {"x": x})["out"], ref)


def test_verifier():
    b = T.TensorBuilder("bad")
    x = b.alloc("x", [4, 4], "i8")
    with pytest.raises(VerifyException, match="map"):
        with b.loop(0, 4) as i:
            b.load(x, [i], lambda i: (i,))
        b.module()
    with pytest.raises(VerifyException, match="bounds"):
        T.ForOp(4, 0).verify()
    with pytest.raises(VerifyException, match="bind"):
        T.ForOp(0, 4, bind="thread").verify()
    with pytest.raises(VerifyException, match="sym"):
        T.AllocOp(result_types=[T.buffer_type([4], "i8", "global")]).verify()
    with pytest.raises(ValueError):
        T.buffer_type([4], "i8", "l2")


def test_buffer_types_round_trip():
    for space in T.SPACES:
        for dtype in ("i8", "i32"):
            ty = T.buffer_type([2, 3], dtype, space)
            assert T.space_of(ty) == space and T.dtype_of(ty) == dtype
