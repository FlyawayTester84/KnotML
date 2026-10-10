import random

import pytest
from _util import KNOTS, LINKS, random_diagram

from knotml import LinkDiagram, PDError, determinant


@pytest.mark.parametrize("name", sorted(KNOTS))
def test_knotinfo_pd_signs_faces_determinant(name):
    pd, writhe, det = KNOTS[name]
    d = LinkDiagram.from_pd(pd)
    assert d.n_crossings == len(pd)
    assert d.n_components == 1
    assert d.writhe() == writhe
    assert d.n_faces == d.n_crossings + 2
    assert determinant(d) == det
    assert LinkDiagram.from_pd(d.to_pd()) == d


@pytest.mark.parametrize("name", sorted(LINKS))
def test_links_linking_number_and_determinant(name):
    pd, lk, det = LINKS[name]
    d = LinkDiagram.from_pd(pd)
    assert d.n_components == 2
    assert d.linking_matrix() == [[0, lk], [lk, 0]]
    assert determinant(d) == det


def test_pd_labels_need_not_be_consecutive_or_one_based():
    pd = [[a * 10 for a in quad] for quad in KNOTS["3_1"][0]]
    assert LinkDiagram.from_pd(pd) == LinkDiagram.from_pd(KNOTS["3_1"][0])
    zero_based = [[a - 1 for a in quad] for quad in KNOTS["4_1"][0]]
    assert LinkDiagram.from_pd(zero_based) == LinkDiagram.from_pd(KNOTS["4_1"][0])


def test_pd_round_trip_on_random_diagrams():
    rng = random.Random(0)
    for _ in range(200):
        d = random_diagram(rng, rng.randint(0, 10))
        d.check()
        if not d.is_pd_ambiguous():
            assert LinkDiagram.from_pd(d.to_pd(), d.n_unknots) == d


@pytest.mark.parametrize(
    "pd, message",
    [
        ([[1, 2, 3, 4], [1, 2, 3, 3]], "appears"),
        ([[1, 2, 3]], "expected 4"),
        ([[1, 3, 2, 4], [4, 2, 1, 3]], "not planar"),
        ([[1, 3, 2, 4], [1, 4, 2, 3]], "orientation"),
    ],
)
def test_invalid_pd_codes_are_rejected(pd, message):
    with pytest.raises(PDError, match=message):
        LinkDiagram.from_pd(pd)


def test_braid_closures():
    trefoil = LinkDiagram.from_braid([1, 1, 1])
    assert trefoil.writhe() == 3 and determinant(trefoil) == 3
    assert trefoil.is_isomorphic(LinkDiagram.from_pd(KNOTS["3_1"][0]))
    assert LinkDiagram.from_braid([1, -1]).n_components == 2
    unlink = LinkDiagram.from_braid([], n_strands=3)
    assert unlink.n_crossings == 0 and unlink.n_unknots == 3
    assert LinkDiagram.from_braid([2]).n_unknots == 1  # strand 1 never crosses


def test_gauss_code_and_components():
    d = LinkDiagram.from_pd(KNOTS["3_1"][0])
    (code,) = d.to_gauss()
    assert sorted(abs(g) for g in code) == [1, 1, 2, 2, 3, 3]
    assert sum(g > 0 for g in code) == 3
    assert d.component_writhes() == [3]
    assert LinkDiagram.unknot(2).n_components == 2


def test_copy_is_independent():
    d = LinkDiagram.from_pd(KNOTS["4_1"][0])
    e = d.copy()
    e.apply_inplace(e.legal_moves()[0])
    assert d == LinkDiagram.from_pd(KNOTS["4_1"][0])
    assert d != e
