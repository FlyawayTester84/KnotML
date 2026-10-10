"""Invariants and certificates that need the optional packages (``pip install -e ".[oracles]"``)."""

from __future__ import annotations

from .core import LinkDiagram, determinant
from .interop import to_regina, to_spherogram

__all__ = ["is_unknot", "seifert_genus", "jones_polynomial", "alexander_polynomial", "signature"]


def seifert_genus(d: LinkDiagram) -> int:
    """Seifert genus of a knot, certified by knot Floer homology."""
    if d.n_components != 1:
        raise ValueError("seifert_genus expects a knot")
    link = to_spherogram(d)
    link.simplify("basic")  # the HFK program rejects diagrams with removable kinks
    if not link.crossings:
        return 0
    return link.knot_floer_homology()["seifert_genus"]


def is_unknot(d: LinkDiagram) -> bool:
    """Certified unknot recognition: genus 0 in knot Floer homology.

    A determinant other than 1 proves knottedness first, without the HFK call.
    """
    if d.n_components != 1:
        raise ValueError("is_unknot expects a knot")
    if d.n_crossings == 0:
        return True
    if determinant(d) != 1:
        return False
    return seifert_genus(d) == 0


def jones_polynomial(d: LinkDiagram):
    """Jones polynomial from Regina, as a Laurent polynomial in ``x = sqrt(t)``."""
    return to_regina(d).jones()


def alexander_polynomial(d: LinkDiagram):
    return to_spherogram(d).alexander_polynomial()


def signature(d: LinkDiagram) -> int:
    return to_spherogram(d).signature()
