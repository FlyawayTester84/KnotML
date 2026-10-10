"""Canonical form of an oriented link diagram up to relabelling.

From a starting edge, walk the diagram deterministically: follow the component
through each crossing, number crossings in order of first arrival, and when a
component closes, continue with the first numbered crossing that still has an
unwalked strand.  Each arrival emits ``8 * label + 2 * slot (+ sign bit on
first arrival)``; that sequence determines the diagram.  The canonical code of
a connected projection is the smallest sequence over all ``2n`` starting
edges.  Comparison aborts at the first token larger than the best so far, so
the O(n²) worst case (very symmetric diagrams) is rarely reached.

Reflection, reversal and mirroring are *not* quotiented out: the key describes
the oriented diagram on the oriented sphere, which is what moves act on.
"""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from .diagram import LinkDiagram

__all__ = ["canonical_key"]


def canonical_key(d: LinkDiagram) -> bytes:
    nbr, sign = d._nbr, d._sign
    pieces: dict[int, list[int]] = {}
    for c, k in enumerate(d._ensure_dcomp()):
        pieces.setdefault(k, []).append(c)
    codes = sorted(_min_code(nbr, sign, crossings) for crossings in pieces.values())
    out = array("i", (d._n_unknots, len(codes)))
    for code in codes:
        out.append(len(code))
        out.extend(code)
    return out.tobytes()


def _min_code(nbr: Sequence[int], sign: Sequence[int], crossings: list[int]) -> list[int]:
    best: Optional[list[int]] = None
    for c in crossings:
        for start in (4 * c + 2, 4 * c + (1 if sign[c] > 0 else 3)):
            code = _code(nbr, sign, start, best)
            if code is not None:
                best = code
    assert best is not None
    return best


def _code(nbr: Sequence[int], sign: Sequence[int], start: int, best: Optional[list[int]]) -> Optional[list[int]]:
    """The walk code from ``start``, or ``None`` as soon as it cannot beat ``best``."""
    label: dict[int, int] = {}
    order: list[int] = []
    walked: set[int] = set()
    out: list[int] = []
    smaller = best is None
    pos = 0
    p = start
    while True:
        while p not in walked:
            walked.add(p)
            q = nbr[p]
            c = q >> 2
            lab = label.get(c)
            if lab is None:
                lab = label[c] = len(order)
                order.append(c)
                tok = 8 * lab + 2 * (q & 3) + (sign[c] > 0)
            else:
                tok = 8 * lab + 2 * (q & 3)
            if not smaller:
                ref = best[len(out)]  # type: ignore[index]
                if tok > ref:
                    return None
                smaller = tok < ref
            out.append(tok)
            p = q ^ 2
        while pos < len(order):  # next component: first numbered crossing with an unwalked strand
            c = order[pos]
            if 4 * c + 2 not in walked:
                p = 4 * c + 2
                break
            over_out = 4 * c + (1 if sign[c] > 0 else 3)
            if over_out not in walked:
                p = over_out
                break
            pos += 1
        else:
            return out if smaller else None
        if not smaller:
            if best[len(out)] < -1:  # type: ignore[index]
                return None
            smaller = best[len(out)] > -1  # type: ignore[index]
        out.append(-1)
