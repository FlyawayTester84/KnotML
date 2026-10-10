"""Cross-check legality and results of every move against Regina (skipped if not installed)."""

import random

import pytest
from _util import random_diagram

from knotml import ISOTOPY_KINDS, TRIVIAL, MoveKind
from knotml.interop import from_regina, to_regina

regina = pytest.importorskip("regina")


def arc(R, d, p):
    """Regina's (arc, side) for our side-port ``p``: arcs start at their tail."""
    t = d.edge_tail(p)
    c = R.crossing(t >> 2)
    return (c.upper() if t & 1 else c.lower()), (0 if p == t else 1)


def sig(R):
    return R.sig(False, False, False)  # strict: no reflection, reversal or rotation


def regina_result(R, d, mv):
    if mv.kind == MoveKind.R1_REMOVE:
        return R.withR1(R.crossing(mv.a >> 2))
    if mv.kind == MoveKind.R2_REMOVE:
        return R.withR2(R.crossing(d.edge_tail(mv.a) >> 2))
    if mv.kind == MoveKind.R3:
        t = d.edge_tail(mv.a)
        return R.withR3(R.crossing(t >> 2), 0 if t == mv.a else 1)
    if mv.kind == MoveKind.R1_ADD:
        u, s = arc(R, d, mv.a)
        return R.withR1(u, s, mv.sign)
    (ux, sx), (uy, sy) = arc(R, d, mv.a), arc(R, d, mv.b)
    return R.withR2(ux, sx, uy, sy)


@pytest.mark.parametrize("seed", range(2))
def test_legal_sets_and_results_match_regina(seed):
    rng = random.Random(seed)
    for _ in range(25):
        d = random_diagram(rng, rng.randint(0, 10))
        R = to_regina(d)
        assert from_regina(R) == d
        n, m = d.n_crossings, len(d.nbr)
        ours = d.legal_moves()
        by_kind = {k: [mv for mv in ours if mv.kind == k] for k in MoveKind}.__getitem__
        assert {mv.a >> 2 for mv in by_kind(MoveKind.R1_REMOVE)} == {
            c for c in range(n) if R.hasR1(R.crossing(c))
        }
        assert {d.edge_tail(mv.a) >> 2 for mv in by_kind(MoveKind.R2_REMOVE)} == {
            c for c in range(n) if R.hasR2(R.crossing(c))
        }
        assert {(d.edge_tail(mv.a) >> 2, int(d.edge_tail(mv.a) != mv.a)) for mv in by_kind(MoveKind.R3)} == {
            (c, s) for c in range(n) for s in (0, 1) if R.hasR3(R.crossing(c), s)
        }
        theirs = set()
        for x in range(m):
            for y in range(m):
                if x != y and R.hasR2(*arc(R, d, x), *arc(R, d, y)):
                    theirs.add((x, y))
        assert {(mv.a, mv.b) for mv in by_kind(MoveKind.R2_ADD) if TRIVIAL not in (mv.a, mv.b)} == theirs
        for mv in ours:
            if TRIVIAL in (mv.a, mv.b) and mv.kind in (MoveKind.R1_ADD, MoveKind.R2_ADD):
                continue
            assert sig(regina_result(R, d, mv)) == sig(to_regina(d.apply(mv))), mv


def test_canonical_key_agrees_with_regina_signature():
    rng = random.Random(7)
    diagrams = []
    for _ in range(40):
        d = random_diagram(rng, 3, ISOTOPY_KINDS)
        diagrams += [d, d.apply(d.random_move(rng, [MoveKind.R3, MoveKind.R1_ADD]))]
    ours = [d.canonical_key() for d in diagrams]
    theirs = [sig(to_regina(d)) for d in diagrams]
    for i in range(len(diagrams)):
        for j in range(len(diagrams)):
            assert (ours[i] == ours[j]) == (theirs[i] == theirs[j])


def test_jones_polynomial_is_invariant_along_random_walks():
    rng = random.Random(11)
    for _ in range(10):
        d = random_diagram(rng, 0)
        jones = to_regina(d).jones()
        for _ in range(15):
            d.apply_inplace(d.random_move(rng, ISOTOPY_KINDS))
        assert to_regina(d).jones() == jones
