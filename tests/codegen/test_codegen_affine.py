"""Index arithmetic: affine maps as linear forms over ranged atoms (anvil.codegen.affine). Trivial maps simplify
without materializing anything; every result equals direct evaluation of the map."""
from hypothesis import given, settings, strategies as st
from xdsl.ir.affine import AffineExpr, AffineMap

from anvil.codegen.affine import Affine, Atom, Lin


def mk():
    made = []

    def div_atom(r, d):
        lo, hi = r.range()
        a = Atom("div", lo // d, hi // d, r.depth, r.varying)
        a.src = (r, d)
        made.append(a)
        return Lin.of(a)
    return Affine(div_atom), made


def value(x: Lin, env):
    out = x.const
    for a, c in x.terms:
        if a in env:
            v = env[a]
        else:
            r, d = a.src
            v = value(r, env) // d
        out += c * v
    return out


D = [AffineExpr.dimension(i) for i in range(3)]


def test_trivial_maps_simplify():
    aff, made = mk()
    o, i = Atom("o", 0, 7, 1), Atom("i", 0, 3, 1)
    iv = Lin.of(o).scale(4) + Lin.of(i)                 # unrolled-by-4 loop iv
    assert aff.eval(D[0] // 1, [iv]) == iv
    assert aff.eval(D[0] // 4, [iv]) == Lin.of(o)
    assert aff.eval(D[0] % 4, [iv]) == Lin.of(i)
    assert aff.eval(D[0] % 32, [iv]) == iv              # iv < 32
    assert aff.eval((D[0] * 4 + 3) // 4, [Lin.of(i)]) == Lin.of(i)
    assert aff.eval(D[0] // 4 * 4 + D[0] % 4, [iv]) == iv
    assert aff.eval(D[0] + D[1], [Lin.of(5), Lin.of(-2)]) == Lin.of(3)
    assert not made


def test_floordiv_materializes_one_atom_when_needed():
    aff, made = mk()
    t = Atom("t", 0, 15, 0, varying=True)
    q = aff.eval(D[0] // 4, [Lin.of(t)])
    r = aff.eval(D[0] % 4, [Lin.of(t)])
    assert len(made) == 2 and q.varying and r.varying     # (memoization is the code generator's cache)
    for v in range(16):
        env = {t: v}
        assert value(q, env) == v // 4 and value(r, env) == v % 4


EXPR = st.deferred(lambda: st.one_of(
    st.sampled_from(D),
    st.integers(-5, 9).map(AffineExpr.constant),
    st.tuples(EXPR, EXPR).map(lambda p: p[0] + p[1]),
    st.tuples(EXPR, st.integers(-3, 5)).map(lambda p: p[0] * p[1]),
    st.tuples(EXPR, st.integers(1, 6)).map(lambda p: p[0] // p[1]),
    st.tuples(EXPR, st.integers(1, 6)).map(lambda p: p[0] % p[1]),
    st.tuples(EXPR, st.integers(1, 6)).map(lambda p: p[0].ceil_div(p[1])),
))


@settings(max_examples=300, deadline=None)
@given(e=EXPR, ext=st.lists(st.integers(1, 9), min_size=3, max_size=3), pick=st.data())
def test_eval_matches_affine_semantics(e, ext, pick):
    aff, _ = mk()
    atoms = [Atom(f"d{k}", 0, n - 1, k) for k, n in enumerate(ext)]
    x = aff.eval(e, [Lin.of(a) for a in atoms])
    vals = [pick.draw(st.integers(0, n - 1)) for n in ext]
    want = AffineMap(3, 0, (e,)).eval(vals, [])[0]
    assert value(x, dict(zip(atoms, vals))) == want
    lo, hi = x.range()
    assert lo <= want <= hi
