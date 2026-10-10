"""Reidemeister moves and crossing changes on a :class:`~knotml.core.diagram.LinkDiagram`.

Anchors
-------
A move is the tuple ``Move(kind, a, b=-1, sign=0)`` whose anchors are ports.
Port ``p`` doubles as the side of the edge at ``p`` lying counterclockwise
after ``p`` (an edge's tail names its left side, its head its right side), so
"a side of an edge" and "a corner of a face" are both just a port.

===================  =========================================  ==========================================
kind                 anchors                                    legal when
===================  =========================================  ==========================================
``R1_REMOVE``        ``a``: the corner inside a monogon         face of ``a`` is a monogon
``R1_ADD``           ``a``: side receiving the kink;            always; ``a = TRIVIAL`` needs a
                     ``sign``: sign of the new crossing         crossingless component
``R2_REMOVE``        ``a``: bigon side of its upper edge        face of ``a`` is a bigon on two crossings
                                                                and the edge at ``a`` is over at both ends
``R2_ADD``           ``a``: side of the edge pushed over,       the edges differ and the sides share a
                     ``b``: side of the edge pushed under       face, or lie in different pieces of the
                                                                projection; ``TRIVIAL`` allowed
``R3``               ``a``: triangle side of its top edge       face of ``a`` is a triangle on three
                                                                crossings and the edge at ``a`` is over
                                                                at both ends
``CROSSING_CHANGE``  ``a``: crossing                            always (not an isotopy)
===================  =========================================  ==========================================

Each legal move has exactly one anchor, so enumerations have no duplicates.

Complexity (``n`` crossings)
----------------------------
* ``is_legal``: O(1).  Only R2_ADD reads a cache (face ids, built once per
  diagram state in O(n)); every other check walks at most three corners.
* ``legal_moves``: O(n) per kind except R2_ADD, which is O(n + output); its
  output is the sum over faces of ``|f|(|f| - 1)``, up to Θ(n²).
* ``move_masks``: O(n) time and space for all kinds.  R2_ADD is returned
  factored (face and projection piece of each side) because the full pairwise
  mask has Θ(n²) entries.
* ``apply_inplace``: O(1).  Deleted crossings are filled by moving the last
  crossing into their slot, and the inverse move is computed alongside.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Iterable, NamedTuple, Optional

import numpy as np

from .ports import TRIVIAL, next_ccw, prev_ccw, sigma

if TYPE_CHECKING:
    from .diagram import LinkDiagram

__all__ = [
    "Move",
    "MoveKind",
    "IllegalMoveError",
    "TRIVIAL",
    "ISOTOPY_KINDS",
    "NON_INCREASING_KINDS",
    "ALL_KINDS",
    "is_legal",
    "legal_moves",
    "move_masks",
    "apply_inplace",
]


class MoveKind(IntEnum):
    R1_REMOVE = 0
    R1_ADD = 1
    R2_REMOVE = 2
    R2_ADD = 3
    R3 = 4
    CROSSING_CHANGE = 5


ALL_KINDS = frozenset(MoveKind)
ISOTOPY_KINDS = ALL_KINDS - {MoveKind.CROSSING_CHANGE}
NON_INCREASING_KINDS = frozenset({MoveKind.R1_REMOVE, MoveKind.R2_REMOVE, MoveKind.R3})


class Move(NamedTuple):
    """A move anchored at ports; see the module docstring for the meaning of each field."""

    kind: MoveKind
    a: int
    b: int = TRIVIAL
    sign: int = 0

    def __repr__(self) -> str:
        if self.kind == MoveKind.R1_ADD:
            return f"R1_ADD(a={self.a}, sign={self.sign:+d})"
        if self.kind == MoveKind.R2_ADD:
            return f"R2_ADD(over={self.a}, under={self.b})"
        return f"{self.kind.name}(a={self.a})"


class IllegalMoveError(ValueError):
    pass


# ----------------------------------------------------------------------
# Legality: O(1) per move
# ----------------------------------------------------------------------


def _bigon(nbr: list[int], x: int) -> bool:
    if not (x & 1 and nbr[x] & 1):
        return False
    z = sigma(nbr, x)
    return z != x and sigma(nbr, z) == x and z >> 2 != x >> 2


def _triangle(nbr: list[int], x: int) -> bool:
    if not (x & 1 and nbr[x] & 1):
        return False
    y = sigma(nbr, x)
    if y == x:
        return False
    z = sigma(nbr, y)
    if z == x or sigma(nbr, z) != x:
        return False
    a, b, c = x >> 2, y >> 2, z >> 2
    return a != b and b != c and a != c


def _r2_add_ok(d: LinkDiagram, x: int, y: int) -> bool:
    m = len(d._nbr)
    if x == TRIVIAL or y == TRIVIAL:
        other = y if x == TRIVIAL else x
        need = 2 if other == TRIVIAL else 1
        return d._n_unknots >= need and (other == TRIVIAL or 0 <= other < m)
    if not (0 <= x < m and 0 <= y < m) or x == y or d._nbr[x] == y:
        return False
    face = d._ensure_faces()
    if face[x] == face[y]:
        return True
    dcomp = d._ensure_dcomp()
    return dcomp[x >> 2] != dcomp[y >> 2]


def is_legal(d: LinkDiagram, move: Move) -> bool:
    kind, a, b, s = move
    nbr = d._nbr
    if kind == MoveKind.R2_ADD:
        return _r2_add_ok(d, a, b)
    if kind == MoveKind.R1_ADD:
        if s != 1 and s != -1:
            return False
        return d._n_unknots > 0 if a == TRIVIAL else 0 <= a < len(nbr)
    if kind == MoveKind.CROSSING_CHANGE:
        return 0 <= a < len(d._sign)
    if not 0 <= a < len(nbr):
        return False
    if kind == MoveKind.R1_REMOVE:
        return sigma(nbr, a) == a
    if kind == MoveKind.R2_REMOVE:
        return _bigon(nbr, a)
    if kind == MoveKind.R3:
        return _triangle(nbr, a)
    return False


# ----------------------------------------------------------------------
# Enumeration
# ----------------------------------------------------------------------


def legal_moves(d: LinkDiagram, kinds: Iterable[MoveKind] = ISOTOPY_KINDS) -> list[Move]:
    kinds = frozenset(kinds)
    nbr = d._nbr
    m = len(nbr)
    out: list[Move] = []
    if MoveKind.R1_REMOVE in kinds:
        out += [Move(MoveKind.R1_REMOVE, p) for p in range(m) if sigma(nbr, p) == p]
    if MoveKind.R2_REMOVE in kinds:
        out += [Move(MoveKind.R2_REMOVE, p) for p in range(1, m, 2) if _bigon(nbr, p)]
    if MoveKind.R3 in kinds:
        out += [Move(MoveKind.R3, p) for p in range(1, m, 2) if _triangle(nbr, p)]
    if MoveKind.R1_ADD in kinds:
        anchors = list(range(m)) + ([TRIVIAL] if d._n_unknots else [])
        out += [Move(MoveKind.R1_ADD, p, TRIVIAL, s) for p in anchors for s in (1, -1)]
    if MoveKind.R2_ADD in kinds:
        out += _r2_add_moves(d)
    if MoveKind.CROSSING_CHANGE in kinds:
        out += [Move(MoveKind.CROSSING_CHANGE, c) for c in range(len(d._sign))]
    return out


def _r2_add_moves(d: LinkDiagram) -> list[Move]:
    R2 = MoveKind.R2_ADD
    out = []
    for corners in d.faces():
        out += [Move(R2, x, y) for x in corners for y in corners if x != y]
    dcomp = d._ensure_dcomp()
    pieces: dict[int, list[int]] = {}
    for p in range(len(d._nbr)):
        pieces.setdefault(dcomp[p >> 2], []).append(p)
    groups = list(pieces.values())
    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            if i != j:
                out += [Move(R2, x, y) for x in gi for y in gj]
    if d._n_unknots:
        ports = range(len(d._nbr))
        out += [Move(R2, x, TRIVIAL) for x in ports] + [Move(R2, TRIVIAL, y) for y in ports]
        if d._n_unknots >= 2:
            out.append(Move(R2, TRIVIAL, TRIVIAL))
    return out


def move_masks(d: LinkDiagram) -> dict[str, np.ndarray]:
    """Legality of every move in O(n) space, indexed by port (crossing for changes).

    * ``r1_remove``, ``r2_remove``, ``r3``: bool ``(4n,)``; entry ``p`` is the move anchored at ``p``.
    * ``r1_add``: bool ``(4n, 2)``; column 0 is sign +1, column 1 sign -1.
    * ``side_face``, ``side_piece``: int ``(4n,)``.  ``R2_ADD(x, y)`` is legal iff
      ``x != y``, ``nbr[x] != y``, and ``side_face[x] == side_face[y]`` or
      ``side_piece[x] != side_piece[y]``.
    * ``crossing_change``: bool ``(n,)``.

    ``TRIVIAL`` anchors are not covered; they are legal whenever ``d.n_unknots``
    is large enough.
    """
    nbr = np.asarray(d._nbr, dtype=np.int64)
    m = nbr.size
    p = np.arange(m, dtype=np.int64)
    s1 = nbr[(p & ~3) | ((p + 1) & 3)]  # next corner around the face
    s2 = s1[s1]
    s3 = s1[s2]
    upper = ((p & 1) & (nbr & 1)).astype(bool)  # edge at p is over at both ends
    c0, c1, c2 = p >> 2, s1 >> 2, s2 >> 2
    r1 = s1 == p
    r2 = upper & ~r1 & (s2 == p) & (c0 != c1)
    r3 = upper & ~r1 & (s2 != p) & (s3 == p) & (c0 != c1) & (c1 != c2) & (c0 != c2)
    face = np.asarray(d._ensure_faces(), dtype=np.int64)
    piece = np.asarray(d._ensure_dcomp(), dtype=np.int64)[p >> 2]
    return {
        "r1_remove": r1,
        "r2_remove": r2,
        "r3": r3,
        "r1_add": np.ones((m, 2), dtype=bool),
        "side_face": face,
        "side_piece": piece,
        "crossing_change": np.ones(len(d._sign), dtype=bool),
    }


# ----------------------------------------------------------------------
# Application: O(1) rewiring, returns the inverse move
# ----------------------------------------------------------------------


def apply_inplace(d: LinkDiagram, move: Move, check: bool = True) -> Optional[Move]:
    if check and not is_legal(d, move):
        raise IllegalMoveError(f"{move!r} is not legal on {d!r}")
    inverse = _APPLY[move.kind](d, move)
    d._invalidate()
    return inverse


def _link(nbr: list[int], a: int, b: int) -> None:
    nbr[a] = b
    nbr[b] = a


# R1_ADD wiring keyed by (kink on the left?, sign): slots of the new crossing that
# meet the old tail and head, the two slots joined by the loop, and the slot whose
# corner lies inside the new monogon.  Derived from the four local pictures.
_R1_WIRING = {
    (True, 1): (0, 1, (2, 3), 2),
    (True, -1): (1, 2, (3, 0), 3),
    (False, -1): (0, 3, (2, 1), 1),
    (False, 1): (3, 2, (1, 0), 0),
}


def _r1_add(d: LinkDiagram, move: Move) -> Move:
    nbr, sign = d._nbr, d._sign
    x, s = move.a, move.sign
    left = x == TRIVIAL or d.is_out(x)
    from_tail, to_head, (la, lb), mono = _R1_WIRING[(left, s)]
    X = len(nbr)
    if x == TRIVIAL:
        d._n_unknots -= 1
    else:
        tail, head = (x, nbr[x]) if left else (nbr[x], x)
    sign.append(s)
    nbr.extend((-1, -1, -1, -1))
    if x == TRIVIAL:
        _link(nbr, X + from_tail, X + to_head)
    else:
        _link(nbr, tail, X + from_tail)
        _link(nbr, X + to_head, head)
    _link(nbr, X + la, X + lb)
    return Move(MoveKind.R1_REMOVE, X + mono)


def _r1_remove(d: LinkDiagram, move: Move) -> Move:
    nbr = d._nbr
    x = move.a
    k = x & 3
    s = d._sign[x >> 2]
    left = k >= 2  # loop on slots {2,3} or {3,0}; see _R1_WIRING
    ends = _splice(d, [(x - k + ((k + 2) & 3), x - k + ((k + 3) & 3))])
    anchor = TRIVIAL
    a = next(iter(ends.values()))
    if a is not None:
        tail = a if d.is_out(a) else nbr[a]
        anchor = tail if left else nbr[tail]
    anchor = _delete_crossings(d, [x >> 2]).get(anchor, anchor)
    return Move(MoveKind.R1_ADD, anchor, TRIVIAL, s)


# R2_ADD local picture: the edge E1 at side x runs horizontally above the shared
# face F, the edge E2 at side y below it, and a finger of E1 is pushed down over
# E2, creating crossings A (west) and B (east).  The port of a new crossing at
# compass point W/S/E/N depends only on the direction of E2 (under, so slot 0 is
# where E2 enters):
_R2_COMPASS = {1: (0, 1, 2, 3), -1: (2, 3, 0, 1)}  # E2 eastward / westward -> slots of W, S, E, N


def _r2_add(d: LinkDiagram, move: Move) -> Move:
    nbr, sign = d._nbr, d._sign
    x, y = move.a, move.b
    # x is the east end of E1 and y the west end of E2 (that is where their sides face F).
    e1 = 1 if x == TRIVIAL or not d.is_out(x) else -1  # E1 runs eastward
    e2 = 1 if y == TRIVIAL or d.is_out(y) else -1  # E2 runs eastward
    w, s_, e, n_ = _R2_COMPASS[e2]
    west1 = nbr[x] if x != TRIVIAL else TRIVIAL
    east2 = nbr[y] if y != TRIVIAL else TRIVIAL
    A = len(nbr)
    B = A + 4
    sign += (e1 * e2, -e1 * e2)
    nbr.extend((-1,) * 8)
    if x == TRIVIAL:
        d._n_unknots -= 1
        _link(nbr, A + n_, B + n_)
    else:
        _link(nbr, west1, A + n_)
        _link(nbr, B + n_, x)
    _link(nbr, A + s_, B + s_)
    if y == TRIVIAL:
        d._n_unknots -= 1
        _link(nbr, B + e, A + w)
    else:
        _link(nbr, y, A + w)
        _link(nbr, B + e, east2)
    _link(nbr, A + e, B + w)
    return Move(MoveKind.R2_REMOVE, A + s_)


def _r2_remove(d: LinkDiagram, move: Move) -> Optional[Move]:
    nbr = d._nbr
    aS = move.a  # compass names as in _r2_add: the bigon lies south-east of A
    aN, aW = aS ^ 2, prev_ccw(aS)
    bS = nbr[aS]
    bN, bE = bS ^ 2, next_ccw(bS)
    ends = _splice(d, [(aN, bN), (aW, bE)])
    east1, west2 = ends[bN], ends[aW]
    reloc = _delete_crossings(d, [aS >> 2, bS >> 2])
    x = TRIVIAL if east1 is None else reloc.get(east1, east1)
    y = TRIVIAL if west2 is None else reloc.get(west2, west2)
    if x != TRIVIAL and y != TRIVIAL and (x == y or nbr[x] == y):
        return None  # E1 and E2 merged into one edge: undoing needs an edge pushed over itself
    return Move(MoveKind.R2_ADD, x, y)


def _r3(d: LinkDiagram, move: Move) -> Move:
    """Swap the order of the two triangle crossings along each of the three strands.

    Crossings keep their index, sign and slot roles; only the twelve ports
    around the triangle are rewired.
    """
    nbr = d._nbr
    x = move.a
    y = sigma(nbr, x)
    z = sigma(nbr, y)
    moved = {}  # old boundary port -> port playing its role afterwards
    inner = []
    for u, v in ((next_ccw(x), y), (next_ccw(y), z), (next_ccw(z), x)):
        tail, head = (u, v) if d.is_out(u) else (v, u)
        moved[tail ^ 2] = head  # the strand now enters the triangle at the other crossing
        moved[head ^ 2] = tail  # ... and leaves it at the first one
        inner.append((head ^ 2, tail ^ 2))
    outside = {p: nbr[p] for p in moved}
    for p, q in outside.items():
        _link(nbr, moved[p], moved.get(q, q))
    for a, b in inner:
        _link(nbr, a, b)
    return Move(MoveKind.R3, x ^ 2)  # the triangle flips to the other side of every strand


def _crossing_change(d: LinkDiagram, move: Move) -> Move:
    nbr, sign = d._nbr, d._sign
    c = move.a
    base = 4 * c
    r = 3 if sign[c] > 0 else 1  # new slot k is old slot k + r: slot 0 must be the new under-in
    old = nbr[base : base + 4]
    for k in range(4):
        q = old[(k + r) & 3]
        if q >> 2 == c:
            q = base + (((q & 3) - r) & 3)
        _link(nbr, base + k, q)
    sign[c] = -sign[c]
    return move


_APPLY = {
    MoveKind.R1_ADD: _r1_add,
    MoveKind.R1_REMOVE: _r1_remove,
    MoveKind.R2_ADD: _r2_add,
    MoveKind.R2_REMOVE: _r2_remove,
    MoveKind.R3: _r3,
    MoveKind.CROSSING_CHANGE: _crossing_change,
}


# ----------------------------------------------------------------------
# Removing crossings
# ----------------------------------------------------------------------


def _splice(d: LinkDiagram, pairs: list[tuple[int, int]]) -> dict[int, Optional[int]]:
    """Reconnect the strands through crossings that are about to be deleted.

    ``pairs`` lists ports ``(a, b)`` of the doomed crossings that one strand
    joins inside the deleted region.  Each strand is reconnected to itself
    outside the region, or becomes a crossingless component if it closes up.
    Returns, for every port in ``pairs``, the surviving port reached by leaving
    along its edge (``None`` on a closed loop).
    """
    nbr = d._nbr
    through = {}
    for a, b in pairs:
        through[a] = b
        through[b] = a
    traced = {p: _trace(nbr, through, p) for p in through}
    done: set[int] = set()
    for a, b in pairs:
        if a in done:
            continue
        (ea, va), (eb, vb) = traced[a], traced[b]
        done.update(va)
        done.update(vb)
        if ea is None:
            d._n_unknots += 1
        else:
            _link(nbr, ea, eb)
    return {p: end for p, (end, _) in traced.items()}


def _trace(nbr: list[int], through: dict[int, int], start: int) -> tuple[Optional[int], list[int]]:
    target = through[start]
    visited = [start, target]
    q = nbr[start]
    while q in through:
        if q == target:
            return None, visited
        visited += (q, through[q])
        q = nbr[through[q]]
    return q, visited


def _delete_crossings(d: LinkDiagram, crossings: Iterable[int]) -> dict[int, int]:
    """Delete detached crossings by moving the last crossing into each freed slot.

    Returns ``{old port: new port}`` for every port that moved.
    """
    nbr, sign = d._nbr, d._sign
    reloc: dict[int, int] = {}
    origin: dict[int, int] = {}
    for c in sorted(crossings, reverse=True):
        last = len(sign) - 1
        if c != last:
            src, dst = 4 * last, 4 * c
            for k in range(4):
                q = nbr[src + k]
                if q >> 2 == last:
                    q = dst + (q & 3)
                _link(nbr, dst + k, q)
                o = origin.pop(src + k, src + k)
                reloc[o] = dst + k
                origin[dst + k] = o
            sign[c] = sign[last]
        del nbr[-4:]
        sign.pop()
    return reloc
