import random

import pytest
from _util import KNOTS, LINKS

from knotml import ISOTOPY_KINDS, IllegalMoveError, LinkDiagram, Move, MoveKind
from knotml.kirby import SURGERY_KINDS, FramedLink, KirbyKind, KirbyMove, pushoff

WHITEHEAD = [[6, 1, 7, 2], [10, 7, 5, 8], [4, 5, 1, 6], [2, 10, 3, 9], [8, 4, 9, 3]]  # LinkInfo L5a1{0}
BORROMEAN = [  # LinkInfo L6a4{0,0}
    [6, 1, 7, 2], [12, 8, 9, 7], [4, 12, 1, 11], [10, 5, 11, 6], [8, 4, 5, 3], [2, 9, 3, 10],
]


def slides(F):
    return F.legal_moves([KirbyKind.HANDLE_SLIDE])


def test_linking_matrix_and_homology():
    assert FramedLink.unknot(5).homology() == [5]
    assert FramedLink.unknot(0).homology() == [0]
    assert FramedLink.unknot(1).homology() == []
    trefoil = FramedLink.from_pd(KNOTS["3_1"][0], [1])  # Poincaré homology sphere
    assert trefoil.linking_matrix() == [[1]] and trefoil.homology() == []
    hopf = FramedLink.from_pd(LINKS["L2a1{1}"][0], [0, 7])
    assert hopf.linking_matrix() == [[0, 1], [1, 7]] and hopf.homology() == []
    L4 = FramedLink.from_pd(LINKS["L4a1{0}"][0], [2, 3])
    assert L4.linking_matrix() == [[2, -2], [-2, 3]] and L4.homology() == [2]
    assert FramedLink.from_pd(BORROMEAN, [0, 0, 0]).homology() == [0, 0, 0]
    assert FramedLink.from_pd(LINKS["L2a1{1}"][0], [2, 2]).signature() == 2


def test_blow_up_and_down():
    F = FramedLink.from_pd(WHITEHEAD, [1, 2])
    key, hom, sig = F.canonical_key(), F.homology(), F.signature()
    inverse = F.apply_inplace(KirbyMove(KirbyKind.BLOW_UP, sign=-1))
    assert F.n_components == 3 and F.homology() == hom and F.signature() == sig - 1
    assert inverse in F.legal_moves() and F.is_legal(inverse)
    F.apply_inplace(inverse)
    assert F.canonical_key() == key
    assert not F.is_legal(KirbyMove(KirbyKind.CANCEL_12, a=0))  # 4-manifold moves live in Handlebody
    assert not F.is_legal(KirbyMove(KirbyKind.BLOW_DOWN, a=0))  # not a split crossingless unknot


def test_crossingless_components_need_unit_framing_to_blow_down():
    F = FramedLink(LinkDiagram.unknot(2), [1, 2])
    blow_downs = [m for m in F.legal_moves(SURGERY_KINDS) if m.kind == KirbyKind.BLOW_DOWN]
    assert blow_downs == [KirbyMove(KirbyKind.BLOW_DOWN, a=0)]


@pytest.mark.parametrize("pd, framings", [(WHITEHEAD, [1, 2]), (BORROMEAN, [-3, 5, 2]), (LINKS["L4a1{0}"][0], [2, 3])])
def test_handle_slides_are_valid_and_act_on_the_linking_matrix(pd, framings):
    F = FramedLink.from_pd(pd, framings)
    Q, labels = F.linking_matrix(), F.labels()
    r = len(Q)
    assert slides(F)
    for mv in slides(F):
        G = F.apply(mv)
        G.diagram.check()
        assert G.labels() == labels and G.homology() == F.homology()
        i, j = labels.index(F.diagram.label_of_port(mv.a)), labels.index(F.diagram.label_of_port(mv.b))
        expected = []
        for eps in (1, -1):  # Q' = E Q E^T for the elementary matrix adding eps * row j to row i
            E = [[int(a == b) + (eps if (a, b) == (i, j) else 0) for b in range(r)] for a in range(r)]
            expected.append([[sum(E[a][k] * Q[k][m] * E[b][m] for k in range(r) for m in range(r)) for b in range(r)]
                             for a in range(r)])
        assert G.linking_matrix() in expected


def test_illegal_slides():
    F = FramedLink.from_pd(WHITEHEAD, [1, 2])
    nbr = F.diagram.nbr
    same = next(p for p in range(len(nbr)) if F.diagram.label_of_port(p) == F.diagram.label_of_port(0) and p != 0)
    assert not F.is_legal(KirbyMove(KirbyKind.HANDLE_SLIDE, 0, same))
    with pytest.raises(IllegalMoveError):
        F.apply_inplace(Move(MoveKind.CROSSING_CHANGE, 0))  # not a Kirby move


def test_framings_survive_isotopy_and_enter_the_key():
    rng = random.Random(4)
    F = FramedLink.from_pd(BORROMEAN, [1, 2, 3])
    Q = F.linking_matrix()
    for _ in range(30):
        F.apply_inplace(F.random_move(rng, ISOTOPY_KINDS))
        F.diagram.check()
    assert F.linking_matrix() == Q
    a, b = FramedLink.from_pd(WHITEHEAD, [1, 2]), FramedLink.from_pd(WHITEHEAD, [2, 1])
    assert a.canonical_key() != b.canonical_key()
    assert a.canonical_key() == FramedLink.from_pd(WHITEHEAD, [1, 2]).canonical_key()


def test_random_kirby_walks_keep_homology():
    rng = random.Random(9)
    for start in ([1, 2], [-2, 3]):
        F = FramedLink.from_pd(WHITEHEAD, start)
        hom = F.homology()
        for _ in range(6):
            kinds = ISOTOPY_KINDS | SURGERY_KINDS if F.diagram.n_crossings < 60 else {MoveKind.R1_REMOVE, MoveKind.R3}
            moves = F.legal_moves(kinds)
            if not moves:
                break
            F.apply_inplace(rng.choice(moves))
            F.diagram.check()
            assert F.homology() == hom


def test_reverse_component():
    d = LinkDiagram.from_pd(LINKS["L2a1{1}"][0])
    e = d.copy()
    e.reverse_component(e.labels()[0])
    e.check()
    assert e.linking_matrix() == [[0, -1], [-1, 0]]
    e.reverse_component(e.labels()[0])
    assert e == d


def test_pushoff_links_with_writhe():
    d = LinkDiagram.from_pd(KNOTS["5_1"][0])
    for left in (True, False):
        D, outer, push_outer = pushoff(d, 0, left, 7)
        D.check()
        assert D.n_crossings == 4 * d.n_crossings  # every crossing is a self-crossing
        assert D.linking_matrix()[0][1] == d.writhe()
        assert len(push_outer) == len(outer) == 4 * d.n_crossings


def test_pushoff_matches_regina_blackboard_cable():
    regina = pytest.importorskip("regina")
    from knotml.interop import to_regina

    rng = random.Random(1)
    for _ in range(15):
        d = LinkDiagram.from_braid([rng.choice((1, -1)) * rng.randint(1, 2) for _ in range(5)], n_strands=3)
        if d.n_components != 1:
            continue
        for _ in range(5):
            d.apply_inplace(d.random_move(rng, ISOTOPY_KINDS))
        ref = to_regina(d).parallel(2, regina.Framing.Blackboard).sig(False, False, False)
        for left in (True, False):
            assert to_regina(pushoff(d, d.labels()[0], left, 9)[0]).sig(False, False, False) == ref


def test_homology_and_slides_agree_with_snappy():
    pytest.importorskip("snappy")
    pytest.importorskip("spherogram")
    for pd, framings in ((WHITEHEAD, [5, 2]), (BORROMEAN, [-3, 5, 5])):
        F = FramedLink.from_pd(pd, framings)
        M = F.to_snappy()
        assert M.homology().elementary_divisors() == F.homology()
        assert M.solution_type() == "all tetrahedra positively oriented"
        for mv in slides(F)[::5]:
            N = F.apply(mv).to_snappy()
            assert N.homology() == M.homology() and M.is_isometric_to(N)
    assert FramedLink.from_pd(KNOTS["3_1"][0], [1]).to_snappy().homology().elementary_divisors() == []
