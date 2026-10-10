"""Conversion to and from SnapPy/spherogram, Regina and KnotInfo.

All three use the same PD convention as :mod:`knotml.core` (verified in
``planning/research-summary.md``), so conversion is a PD round trip.  The
packages are optional: ``pip install -e ".[oracles]"``.
"""

from __future__ import annotations

import re

from .core import LinkDiagram

__all__ = ["to_spherogram", "from_spherogram", "to_regina", "from_regina", "from_knotinfo"]


def to_spherogram(d: LinkDiagram):
    import spherogram

    link = spherogram.Link(d.to_pd())
    link.unlinked_unknot_components = d.n_unknots
    return link


def from_spherogram(link) -> LinkDiagram:
    return LinkDiagram.from_pd(link.PD_code(), n_unknots=link.unlinked_unknot_components)


def to_regina(d: LinkDiagram):
    """An identical ``regina.Link``: crossing ``i`` is crossing ``i`` of ``d``.

    Built from signs and signed Gauss sequences rather than PD, so components
    that pass over at every crossing keep their orientation.
    """
    import regina

    comps = d.to_gauss() + [[]] * d.n_unknots
    return regina.Link.fromData(list(d.signs), comps) if comps else regina.Link()


def from_regina(link) -> LinkDiagram:
    """An identical :class:`LinkDiagram` (exact, unlike a PD round trip)."""
    n = link.size()
    sign = [link.crossing(i).sign() for i in range(n)]

    def out_port(c: int, upper: int) -> int:
        return 4 * c + (2 if not upper else (1 if sign[c] > 0 else 3))

    def in_port(c: int, upper: int) -> int:
        return 4 * c + (0 if not upper else (3 if sign[c] > 0 else 1))

    nbr = [-1] * (4 * n)
    for c in range(n):
        for upper in (0, 1):
            nxt = link.crossing(c).next(upper)
            a, b = out_port(c, upper), in_port(nxt.crossing().index(), nxt.strand())
            nbr[a], nbr[b] = b, a
    return LinkDiagram(nbr, sign, link.countTrivialComponents())


def from_knotinfo(name: str) -> LinkDiagram:
    """A knot (e.g. ``"8_20"``) or link (e.g. ``"L4a1{0}"``) from the KnotInfo/LinkInfo tables."""
    from database_knotinfo import link_list

    is_link = name.startswith("L")
    for row in link_list(proper_links=is_link):
        if row["name"] == name:
            text = row["pd_notation_vector" if is_link else "pd_notation"]
            labels = [int(v) for v in re.findall(r"-?\d+", text)]
            pd = [labels[i : i + 4] for i in range(0, len(labels), 4)]
            return LinkDiagram.from_pd(pd, n_unknots=0 if pd else 1)
    raise KeyError(f"{name!r} is not in KnotInfo/LinkInfo")
