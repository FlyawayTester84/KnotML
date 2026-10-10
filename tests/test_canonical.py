import random

from _util import KNOTS, permuted, random_diagram

from knotml import LinkDiagram, Move, MoveKind


def test_key_ignores_crossing_numbering():
    rng = random.Random(0)
    for _ in range(100):
        d = random_diagram(rng, rng.randint(0, 10))
        perm = list(range(d.n_crossings))
        rng.shuffle(perm)
        assert permuted(d, perm).canonical_key() == d.canonical_key()


def test_key_ignores_pd_labels_and_crossing_order():
    pd = KNOTS["7_4"][0]
    shuffled = [[label * 7 + 3 for label in quad] for quad in reversed(pd)]
    assert LinkDiagram.from_pd(shuffled).canonical_key() == LinkDiagram.from_pd(pd).canonical_key()


def test_key_separates_different_diagrams():
    keys = {name: LinkDiagram.from_pd(pd).canonical_key() for name, (pd, _, _) in KNOTS.items()}
    assert len(set(keys.values())) == len(keys)
    trefoil = LinkDiagram.from_pd(KNOTS["3_1"][0])
    mirror = trefoil.copy()
    for c in range(3):
        mirror.apply_inplace(Move(MoveKind.CROSSING_CHANGE, c))
    assert mirror.canonical_key() != trefoil.canonical_key()


def test_key_counts_crossingless_components():
    assert LinkDiagram.unknot(1).canonical_key() != LinkDiagram.unknot(2).canonical_key()
    hopf_plus_circle = LinkDiagram.from_pd([[4, 1, 3, 2], [2, 3, 1, 4]], n_unknots=1)
    hopf = LinkDiagram.from_pd([[4, 1, 3, 2], [2, 3, 1, 4]])
    assert hopf_plus_circle.canonical_key() != hopf.canonical_key()
