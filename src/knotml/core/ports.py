"""Port arithmetic shared by the diagram and the moves.

Crossing ``c`` owns ports ``4c .. 4c + 3`` in counterclockwise order, starting at
the incoming under-strand.  See :mod:`knotml.core.diagram` for the conventions.
"""

from __future__ import annotations

from typing import Sequence

TRIVIAL = -1
"""Anchor standing for a crossingless unknotted component (PD codes cannot show these)."""


def next_ccw(p: int) -> int:
    """The port after ``p`` counterclockwise around the same crossing."""
    return p + 1 if p & 3 != 3 else p - 3


def prev_ccw(p: int) -> int:
    """The port before ``p`` counterclockwise around the same crossing."""
    return p - 1 if p & 3 else p + 3


def sigma(nbr: Sequence[int], p: int) -> int:
    """The corner after corner ``p`` when walking around its face."""
    return nbr[p + 1 if p & 3 != 3 else p - 3]
