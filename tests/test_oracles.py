import random

import pytest
from _util import KNOTS

from knotml import ISOTOPY_KINDS, LinkDiagram
from knotml.interop import from_knotinfo, from_spherogram, to_spherogram

pytest.importorskip("spherogram")
pytest.importorskip("knot_floer_homology")
from knotml.oracles import is_unknot, seifert_genus  # noqa: E402


def test_unknot_recognition():
    rng = random.Random(5)
    d = LinkDiagram.from_braid([1])
    for _ in range(25):
        d.apply_inplace(d.random_move(rng, ISOTOPY_KINDS))
    assert d.n_crossings > 0 and is_unknot(d)
    assert not is_unknot(LinkDiagram.from_pd(KNOTS["3_1"][0]))
    assert seifert_genus(LinkDiagram.from_pd(KNOTS["7_4"][0])) == 1


def test_spherogram_round_trip():
    d = LinkDiagram.from_pd(KNOTS["4_1"][0])
    assert from_spherogram(to_spherogram(d)).is_isomorphic(d)


def test_knotinfo_lookup():
    pytest.importorskip("database_knotinfo")
    assert from_knotinfo("3_1") == LinkDiagram.from_pd(KNOTS["3_1"][0])
    assert from_knotinfo("L2a1{1}").linking_matrix() == [[0, 1], [1, 0]]
    assert from_knotinfo("0_1").is_trivial()
