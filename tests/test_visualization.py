import pytest
from _util import KNOTS, LINKS

from knotml import LinkDiagram

pytest.importorskip("plink")
pytest.importorskip("spherogram")
from knotml.visualization import layout, to_svg  # noqa: E402


def test_layout_matches_diagram():
    d = LinkDiagram.from_pd(KNOTS["3_1"][0])
    components, crossings = layout(d)
    assert len(components) == 1
    assert len(components[0]) == 3  # one strand between consecutive undercrossings
    assert sorted(label for _, _, label in crossings) == [0, 1, 2]


def test_svg_colours_components_and_highlights_crossings():
    hopf = LinkDiagram.from_pd(LINKS["L2a1{1}"][0])
    svg = to_svg(hopf, labels=True, highlight=[1])
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert svg.count("<polygon") == 2  # one orientation arrow per component
    assert svg.count('stroke="#f39c12"') == 1
    assert "<circle" in to_svg(LinkDiagram.unknot())
