import random

import numpy as np
import pytest
from _util import KNOTS, random_diagram

from knotml import (
    ALL_KINDS,
    ISOTOPY_KINDS,
    TRIVIAL,
    trivial_anchor,
    IllegalMoveError,
    LinkDiagram,
    Move,
    MoveKind,
    determinant,
)


def kinds_count(d, kind):
    return sum(m.kind == kind for m in d.legal_moves(ALL_KINDS))


@pytest.mark.parametrize("name, r2_add", [("3_1", 18), ("4_1", 28)])
def test_move_counts_on_minimal_diagrams(name, r2_add):
    # Same counts as Regina's hasR* on these diagrams (planning/research-summary.md).
    d = LinkDiagram.from_pd(KNOTS[name][0])
    n = d.n_crossings
    assert kinds_count(d, MoveKind.R1_ADD) == 8 * n
    assert kinds_count(d, MoveKind.R2_ADD) == r2_add
    for kind in (MoveKind.R1_REMOVE, MoveKind.R2_REMOVE, MoveKind.R3):
        assert kinds_count(d, kind) == 0


def test_r2_add_count_is_sum_over_faces():
    d = random_diagram(random.Random(3), 12, ISOTOPY_KINDS)
    if d.n_unknots == 0 and len(set(d.diagram_component_of_crossing())) == 1:
        expected = sum(len(f) * (len(f) - 1) for f in d.faces())
        assert kinds_count(d, MoveKind.R2_ADD) == expected


def test_kink_removal_reaches_crossingless_unknot():
    d = LinkDiagram.from_braid([1])  # one-crossing unknot: both loops are monogons
    moves = d.legal_moves([MoveKind.R1_REMOVE])
    assert len(moves) == 2
    d.apply_inplace(moves[0])
    assert d.n_crossings == 0 and d.n_unknots == 1


def test_bigon_removal():
    d = LinkDiagram.from_braid([1, -1], n_strands=2)  # two parallel-ish strands crossing twice
    moves = d.legal_moves([MoveKind.R2_REMOVE])
    assert moves
    d.apply_inplace(moves[0])
    assert d.is_trivial() and d.n_unknots == 2


def test_trivial_component_moves():
    d = LinkDiagram.unknot(2)
    assert not d.is_legal(Move(MoveKind.R2_ADD, TRIVIAL, TRIVIAL))  # a circle with itself
    assert d.is_legal(Move(MoveKind.R2_ADD, TRIVIAL, trivial_anchor(1)))
    inv = d.apply_inplace(Move(MoveKind.R2_ADD, TRIVIAL, trivial_anchor(1)))
    assert d.n_crossings == 2 and d.n_unknots == 0
    d.apply_inplace(inv)
    assert d.is_trivial() and d.n_unknots == 2
    d.apply_inplace(Move(MoveKind.R1_ADD, TRIVIAL, sign=-1))
    assert d.n_crossings == 1 and d.writhe() == -1


def test_illegal_moves_raise():
    d = LinkDiagram.from_pd(KNOTS["3_1"][0])
    with pytest.raises(IllegalMoveError):
        d.apply_inplace(Move(MoveKind.R1_REMOVE, 0))
    with pytest.raises(IllegalMoveError):
        d.apply_inplace(Move(MoveKind.R2_ADD, 0, 0))
    assert not d.is_legal(Move(MoveKind.R1_ADD, TRIVIAL, sign=1))  # no crossingless component
    assert not d.is_legal(Move(MoveKind.R1_ADD, 0, sign=0))


def test_apply_does_not_mutate():
    d = LinkDiagram.from_pd(KNOTS["5_1"][0])
    before = d.copy()
    for m in d.legal_moves(ALL_KINDS):
        d.apply(m)
    assert d == before


@pytest.mark.parametrize("seed", range(4))
def test_random_walks_preserve_validity_invariants_and_invert(seed):
    rng = random.Random(seed)
    for _ in range(25):
        d = random_diagram(rng, 0)
        det = determinant(d)
        for _ in range(12):
            m = d.random_move(rng, ISOTOPY_KINDS)
            before = d.copy()
            inverse = d.apply_inplace(m)
            d.check()
            assert determinant(d) == det
            if inverse is not None:
                assert d.is_legal(inverse)
                assert d.apply(inverse).canonical_key() == before.canonical_key()


def test_crossing_change_is_an_involution():
    d = LinkDiagram.from_pd(KNOTS["4_1"][0])
    e = d.copy()
    e.apply_inplace(Move(MoveKind.CROSSING_CHANGE, 2))
    e.check()
    assert e.writhe() == d.writhe() - 2 * d.signs[2]
    e.apply_inplace(Move(MoveKind.CROSSING_CHANGE, 2))
    assert e == d


@pytest.mark.parametrize("seed", range(3))
def test_masks_agree_with_legal_moves(seed):
    rng = random.Random(seed)
    for _ in range(30):
        d = random_diagram(rng, rng.randint(0, 10))
        masks = d.move_masks()
        moves = set(d.legal_moves(ALL_KINDS))
        for name, kind in (("r1_remove", MoveKind.R1_REMOVE), ("r2_remove", MoveKind.R2_REMOVE), ("r3", MoveKind.R3)):
            assert set(np.flatnonzero(masks[name])) == {m.a for m in moves if m.kind == kind}
        face, piece = masks["side_face"], masks["side_piece"]
        nbr = np.asarray(d.nbr)
        m = len(nbr)
        pair = (face[:, None] == face[None, :]) | (piece[:, None] != piece[None, :])
        pair &= ~np.eye(m, dtype=bool) & (nbr[:, None] != np.arange(m)[None, :])
        expected = {(int(x), int(y)) for x, y in zip(*np.nonzero(pair), strict=True)}
        got = {(mv.a, mv.b) for mv in moves if mv.kind == MoveKind.R2_ADD and mv.a >= 0 and mv.b >= 0}
        assert got == expected
        for mv in moves:
            assert d.is_legal(mv)
