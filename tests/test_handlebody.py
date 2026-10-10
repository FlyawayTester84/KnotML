import random

import pytest
from _util import LINKS

from knotml import HANDLEBODY_KINDS, ISOTOPY_KINDS, Handlebody, IllegalMoveError, KirbyKind, KirbyMove, LinkDiagram

L7A6 = [[8, 1, 9, 2], [10, 4, 11, 3], [14, 10, 7, 9], [12, 6, 13, 5], [2, 7, 3, 8], [4, 12, 5, 11], [6, 14, 1, 13]]  # L7a6{0}
BORROMEAN = [[6, 1, 7, 2], [12, 8, 9, 7], [4, 12, 1, 11], [10, 5, 11, 6], [8, 4, 5, 3], [2, 9, 3, 10]]
SLIDE = KirbyKind.HANDLE_SLIDE


def chain():
    """Dotted circle K1 - 2-handle K0 (framing 2) - 2-handle K2 (framing -3): K1 is a meridian of K0."""
    return Handlebody(LinkDiagram.from_braid([1, 1, 2, 2]), {1: None, 0: 2, 2: -3})


def slides(H, dotted_slide=None):
    out = H.legal_moves([SLIDE])
    if dotted_slide is not None:
        out = [m for m in out if H.is_dotted(H.diagram.label_of_port(m.a)) == dotted_slide]
    return out


def test_basic_invariants():
    S1B3 = Handlebody(LinkDiagram.unknot(1), [None])
    assert (S1B3.euler_characteristic(), S1B3.homology(), S1B3.intersection_form()) == (0, [0], [])
    assert S1B3.boundary().homology() == [0]  # S¹×S²
    disk = Handlebody(LinkDiagram.unknot(1), [3])  # D²-bundle over S² with Euler number 3
    assert (disk.euler_characteristic(), disk.homology(), disk.intersection_form(), disk.signature()) == (2, [], [[3]], 1)
    ball = Handlebody.from_pd(LINKS["L2a1{1}"][0], [None, 5])  # cancelling pair: B⁴
    assert (ball.euler_characteristic(), ball.homology(), ball.intersection_form()) == (1, [], [])
    assert ball.boundary().homology() == []
    mazur = Handlebody.from_pd(L7A6, [None, -2])  # contractible, hyperbolic boundary
    assert (mazur.euler_characteristic(), mazur.homology(), mazur.intersection_form()) == (1, [], [])


def test_dotted_circles_must_be_unlinked():
    with pytest.raises(ValueError, match="unlink"):
        Handlebody.from_pd(LINKS["L2a1{1}"][0], [None, None])
    with pytest.raises(ValueError):
        Handlebody.from_pd(LINKS["L2a1{1}"][0], [None])


def test_cancel_12_recognises_meridians_only():
    H = chain()
    assert [m for m in H.legal_moves() if m.kind == KirbyKind.CANCEL_12] == [KirbyMove(KirbyKind.CANCEL_12, a=1)]
    inv = H.invariants()
    H.apply_inplace(KirbyMove(KirbyKind.CANCEL_12, a=1))
    assert H.components == {2: -3} and H.invariants()["H1"] == inv["H1"]
    assert H.intersection_form() == [[-3]] and H.euler_characteristic() == inv["euler"]
    mazur = Handlebody.from_pd(L7A6, [None, -2])  # the 2-handle runs through the dotted circle more than once
    assert not any(m.kind == KirbyKind.CANCEL_12 for m in mazur.legal_moves())


def test_create_and_cancel_pairs_round_trip():
    H = Handlebody.from_pd(L7A6, [None, -2])
    key, inv = H.canonical_key(), H.invariants()
    undo = H.apply_inplace(KirbyMove(KirbyKind.CREATE_12, sign=1))
    assert H.n_1handles == 2 and H.n_2handles == 2 and H.invariants() == inv
    assert H.is_legal(undo)
    assert H.apply_inplace(undo) == KirbyMove(KirbyKind.CREATE_12, sign=1)
    assert H.canonical_key() == key


def test_slide_rules():
    H = Handlebody.from_pd(L7A6, [None, -2])
    assert slides(H, dotted_slide=False) and not slides(H, dotted_slide=True)  # no dot over a 2-handle
    dot_port = next(p for p in range(len(H.diagram.nbr)) if H.is_dotted(H.diagram.label_of_port(p)))
    framed_side = next(m.a for m in slides(H, dotted_slide=False) if m.b == dot_port)
    assert not H.is_legal(KirbyMove(SLIDE, dot_port, framed_side))
    tangled = Handlebody.from_pd(BORROMEAN, [None, None, 1])  # the two dotted circles cross each other
    assert not tangled.dotted_circles_flat() and not slides(tangled, dotted_slide=True)
    flat = Handlebody(LinkDiagram.from_braid([1, 1, 2, 2]), {0: 3, 1: None, 2: None})
    assert flat.dotted_circles_flat() and slides(flat, dotted_slide=True)
    assert not H.is_legal(KirbyMove(KirbyKind.BLOW_UP, sign=1))  # no blow-ups in the 4-manifold calculus


@pytest.mark.parametrize(
    "make",
    [
        lambda: Handlebody.from_pd(L7A6, [None, -2]),
        lambda: Handlebody(LinkDiagram.from_braid([1, 1, 2, 2]), {0: 3, 1: None, 2: None}),
        lambda: Handlebody(LinkDiagram.from_braid([1, 1, 1, 1, 2, 2], n_strands=3), {0: 1, 1: None, 2: None}),
        lambda: Handlebody.from_pd(BORROMEAN, [None, 2, -1]),
    ],
)
def test_every_slide_preserves_the_4_manifold_invariants(make):
    H = make()
    inv = H.invariants()
    for mv in slides(H):
        G = H.apply(mv)
        G.diagram.check()
        assert G.invariants() == inv, mv
        if H.is_dotted(H.diagram.label_of_port(mv.a)):
            assert G.dotted_circles_flat()  # a dotted slide keeps the dotted circles a planar unlink


def test_2_3_moves_need_a_closed_manifold():
    H = Handlebody(LinkDiagram.unknot(1), [0])
    assert not H.is_legal(KirbyMove(KirbyKind.CANCEL_23, a=0)) and not H.is_legal(KirbyMove(KirbyKind.CREATE_23))
    C = Handlebody(LinkDiagram.unknot(1), [0], closed=True)
    inv = C.invariants()
    undo = C.apply_inplace(KirbyMove(KirbyKind.CANCEL_23, a=0))
    assert C.is_empty() and C.invariants() == inv
    C.apply_inplace(undo)
    assert C.components == {0: 0}


@pytest.mark.parametrize("closed", [False, True])
def test_random_walks_preserve_invariants(closed):
    rng = random.Random(3)
    shrink = {KirbyKind.CANCEL_12, KirbyKind.CANCEL_23}
    for make in (chain, lambda: Handlebody.from_pd(BORROMEAN, [None, 2, -1])):
        H = make()
        H.closed = closed
        inv = H.invariants()
        for _ in range(12):
            kinds = ISOTOPY_KINDS | HANDLEBODY_KINDS if H.diagram.n_crossings < 40 else shrink | {KirbyKind.CREATE_12}
            H.apply_inplace(H.random_move(rng, kinds))
            H.diagram.check()
            Handlebody(H.diagram, H.components)  # dotted circles still unlinked
            assert H.invariants() == inv


def test_illegal_moves_raise():
    H = chain()
    with pytest.raises(IllegalMoveError):
        H.apply_inplace(KirbyMove(KirbyKind.CANCEL_12, a=0))  # K0 is a 2-handle, not a dotted circle


def test_boundary_matches_snappy():
    pytest.importorskip("snappy")
    pytest.importorskip("spherogram")
    H = Handlebody.from_pd(L7A6, [None, -2])
    M = H.to_snappy_boundary()
    assert M.solution_type() == "all tetrahedra positively oriented"
    for mv in slides(H)[::4]:
        assert M.is_isometric_to(H.apply(mv).to_snappy_boundary())
    paired = H.apply(KirbyMove(KirbyKind.CREATE_12, sign=0))  # split cancelling pair: an S³ summand
    assert M.is_isometric_to(paired.to_snappy_boundary())


def _reference_kirby_moves(H):
    """Slow, independent reference: try every candidate and apply the rules from scratch."""
    d = H.diagram
    lab = lambda p: d.label_of_port(p)  # noqa: E731
    crossing_labels = [(d.label_of_port(4 * c), d.label_of_port(4 * c + 1)) for c in range(d.n_crossings)]
    flat = not any(H.is_dotted(a) and H.is_dotted(b) for a, b in crossing_labels)
    face, piece = d.face_of_port(), d.diagram_component_of_crossing()
    out = set()
    for x in range(len(d.nbr)):
        for y in range(len(d.nbr)):
            if lab(x) == lab(y) or (face[x] != face[y] and piece[x >> 2] == piece[y >> 2]):
                continue
            if H.is_dotted(lab(x)) and not (H.is_dotted(lab(y)) and flat):
                continue
            out.add(KirbyMove(SLIDE, x, y))
    for dot in H.dotted_labels():
        cs = [c for c, (a, b) in enumerate(crossing_labels) if dot in (a, b)]
        if len(cs) != 2 or any(a == b for a, b in (crossing_labels[c] for c in cs)):
            continue
        ks = {a if b == dot else b for a, b in (crossing_labels[c] for c in cs)}
        if len(ks) != 1:
            continue
        (k,) = ks
        overs = [crossing_labels[c][1] == dot for c in cs]
        joined = any(
            d.edge_tail(p) >> 2 == ca and d.nbr[d.edge_tail(p)] >> 2 == cb
            for ca, cb in (cs, cs[::-1])
            for p in range(4 * ca, 4 * ca + 4)
            if d.label_of_port(p) == k
        )
        if not H.is_dotted(k) and overs[0] != overs[1] and joined:
            out.add(KirbyMove(KirbyKind.CANCEL_12, a=dot))
    return out


def test_enumeration_matches_reference_and_counts_and_masks():
    import numpy as np

    rng = random.Random(5)
    starts = [chain, lambda: Handlebody.from_pd(L7A6, [None, -2]),
              lambda: Handlebody(LinkDiagram.from_braid([1, 1, 2, 2]), {0: 3, 1: None, 2: None})]
    for make in starts:
        H = make()
        for _ in range(8):
            H.apply_inplace(H.random_move(rng, ISOTOPY_KINDS | HANDLEBODY_KINDS) if H.diagram.n_crossings < 30
                            else KirbyMove(KirbyKind.CREATE_12, sign=0))
            moves = H.legal_moves()
            kirby = {m for m in moves if m.kind in (SLIDE, KirbyKind.CANCEL_12)}
            assert kirby == _reference_kirby_moves(H)
            assert all(H.is_legal(m) for m in moves)
            assert list(H.iter_legal_moves()) == moves
            counts = {}
            for m in moves:
                counts[m.kind] = counts.get(m.kind, 0) + 1
            assert {k: v for k, v in H.count_legal_moves().items() if v} == counts
            mk = H.move_masks()
            lab, dot, face, piece = mk["side_label"], mk["side_dotted"], mk["side_face"], mk["side_piece"]
            pair = (lab[:, None] != lab[None, :]) & ((face[:, None] == face[None, :]) | (piece[:, None] != piece[None, :]))
            pair &= ~dot[:, None] | (dot[None, :] & bool(mk["dotted_flat"]))
            assert {(int(x), int(y)) for x, y in zip(*np.nonzero(pair), strict=True)} == {
                (m.a, m.b) for m in moves if m.kind == SLIDE
            }
            assert set(mk["cancel_12"].tolist()) == {m.a for m in moves if m.kind == KirbyKind.CANCEL_12}
