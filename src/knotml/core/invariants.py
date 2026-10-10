"""Invariants computed directly from the PD structure (no optional dependencies).

Writhe and linking numbers live on :class:`~knotml.core.diagram.LinkDiagram`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .diagram import LinkDiagram

__all__ = ["goeritz_matrix", "determinant"]


def goeritz_matrix(d: LinkDiagram) -> list[list[int]]:
    """Goeritz matrix of a connected projection, with rows indexed by the shaded faces.

    Faces are checkerboard-shaded so that face 0 is unshaded.  A crossing is
    of type +1 if its shaded corners follow the over-strand counterclockwise,
    else -1; ``G[i][j] = -sum(type)`` over crossings where shaded faces ``i != j``
    meet, and rows sum to zero.
    """
    if d._ensure_dcomp() and max(d._dcomp) > 0:  # type: ignore[arg-type]
        raise ValueError("the Goeritz matrix needs a connected projection")
    nbr = d._nbr
    face = d._ensure_faces()
    nf = d.n_faces
    colour = [-1] * nf
    adjacent: list[list[int]] = [[] for _ in range(nf)]
    for p in range(len(nbr)):  # the two sides of the edge at p
        adjacent[face[p]].append(face[nbr[p]])
    if nf:
        colour[0] = 0
        stack = [0]
        while stack:
            f = stack.pop()
            for g in adjacent[f]:
                if colour[g] < 0:
                    colour[g] = 1 - colour[f]
                    stack.append(g)
    index = {}
    for f in range(nf):
        if colour[f] == 1:
            index[f] = len(index)
    G = [[0] * len(index) for _ in index]
    for c in range(len(d._sign)):
        k = 1 if colour[face[4 * c + 1]] == 1 else 0  # shaded corners are k and k + 2
        i, j = index[face[4 * c + k]], index[face[4 * c + k + 2]]
        if i == j:
            continue
        t = 1 if k == 1 else -1
        G[i][j] -= t
        G[j][i] -= t
        G[i][i] += t
        G[j][j] += t
    return G


def determinant(d: LinkDiagram) -> int:
    """Knot/link determinant |Δ(-1)|, via the Goeritz matrix in O(F³) exact integer arithmetic."""
    if d.n_crossings == 0:
        return 1 if d.n_unknots <= 1 else 0
    if d.n_unknots or max(d._ensure_dcomp()) > 0:
        return 0  # split
    G = goeritz_matrix(d)
    minor = [row[:-1] for row in G[:-1]]
    return abs(_bareiss_det(minor))


def _bareiss_det(M: list[list[int]]) -> int:
    n = len(M)
    if n == 0:
        return 1
    A = [row[:] for row in M]
    sign, prev = 1, 1
    for k in range(n - 1):
        if A[k][k] == 0:
            for i in range(k + 1, n):
                if A[i][k]:
                    A[k], A[i] = A[i], A[k]
                    sign = -sign
                    break
            else:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]) // prev
        prev = A[k][k]
    return sign * A[n - 1][n - 1]
