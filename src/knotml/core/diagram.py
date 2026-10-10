"""Oriented link diagrams in planar-diagram (PD) form.

Representation
--------------
A diagram with ``n`` crossings is two flat integer lists plus one counter:

* ``nbr`` (length ``4n``).  Crossing ``c`` owns the *ports* (edge ends)
  ``4c + k``, ``k = 0..3``, listed counterclockwise starting at the incoming
  under-strand; ``nbr[p]`` is the port at the other end of the edge at ``p``.
* ``sign`` (length ``n``): crossing signs, ``+1`` or ``-1``.
* ``n_unknots``: crossingless unknotted components, which PD codes cannot show.

This is the PD convention of KnotInfo, KnotAtlas, SnapPy/spherogram, Regina and
Sage: a PD tuple ``(a, b, c, d)`` lists the edge labels at ports ``0..3``.
Consequences used throughout:

* Even slots carry the under-strand, which enters at slot 0 and leaves at
  slot 2.  Odd slots carry the over-strand, which enters at slot 3 if the
  crossing is positive and at slot 1 if it is negative.
* ``p ^ 2`` is the port where the strand through ``p`` continues.

Sides and faces
---------------
Port ``p`` also names one *side* of the edge at ``p``: the side lying
counterclockwise-after ``p`` at its crossing, i.e. the corner between slots
``k`` and ``k + 1``.  For an edge from tail ``t`` to head ``h``, ``t`` names its
left side and ``h`` its right side, so ports and edge-sides are in bijection.
Faces are the orbits of the corner map ``p -> nbr[next_ccw(p)]``, and
:meth:`LinkDiagram.face_of_port` gives the face on each side.  Move anchors
(:mod:`knotml.core.moves`) are ports, so they need no face bookkeeping.

Faces, diagram components and link components are cached and rebuilt lazily in
O(n) after a mutation; nothing else is stored.
"""

from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence

from . import moves as _moves
from .moves import Move, MoveKind
from .ports import next_ccw

__all__ = ["LinkDiagram", "PDError"]


class PDError(ValueError):
    """Raised when data does not describe a classical oriented link diagram."""


class LinkDiagram:
    """An oriented classical link diagram on the 2-sphere.

    Build one with :meth:`from_pd`, :meth:`from_braid` or :meth:`unknot`.  The
    constructor takes the raw port arrays described in the module docstring.

    Diagrams are mutable (:meth:`apply_inplace`), so they are not hashable; use
    :meth:`canonical_key` as a dictionary key for the diagram up to relabelling.
    """

    __slots__ = ("_nbr", "_sign", "_n_unknots", "_face", "_face_start", "_dcomp", "_comps")

    def __init__(
        self,
        nbr: Sequence[int] = (),
        sign: Sequence[int] = (),
        n_unknots: int = 0,
        *,
        check: bool = True,
    ) -> None:
        self._nbr = list(nbr)
        self._sign = list(sign)
        self._n_unknots = int(n_unknots)
        self._invalidate()
        if check:
            self.check()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pd(cls, pd: Iterable[Sequence[int]], n_unknots: int = 0) -> LinkDiagram:
        """Build a diagram from a PD code (any hashable edge labels, each used twice).

        Crossing ``i`` of the result is the ``i``-th tuple of ``pd``.  A component
        that passes over at every crossing cannot be oriented from the slots
        alone; it is oriented so that its labels increase, as KnotTheory does.
        """
        quads = [tuple(q) for q in pd]
        n = len(quads)
        label: list = [None] * (4 * n)
        where: dict = {}
        for c, quad in enumerate(quads):
            if len(quad) != 4:
                raise PDError(f"crossing {c} has {len(quad)} entries, expected 4")
            for k, lab in enumerate(quad):
                label[4 * c + k] = lab
                where.setdefault(lab, []).append(4 * c + k)
        nbr = [-1] * (4 * n)
        for lab, ports in where.items():
            if len(ports) != 2:
                raise PDError(f"edge label {lab!r} appears {len(ports)} times, expected 2")
            a, b = ports
            nbr[a] = b
            nbr[b] = a
        return cls(nbr, _orient(nbr, label), n_unknots)

    @classmethod
    def from_braid(cls, word: Iterable[int], n_strands: Optional[int] = None) -> LinkDiagram:
        """Closure of a braid word.

        Generator ``i`` (``-i``) crosses strands ``i`` and ``i + 1`` (1-based) with a
        positive (negative) crossing.  Strands that never cross become
        crossingless components.
        """
        word = [int(g) for g in word]
        k = max([abs(g) + 1 for g in word] + [n_strands or 1])
        nbr: list[int] = []
        sign: list[int] = []
        top = [-1] * (k + 2)  # current outgoing port of each strand position
        bottom = [-1] * (k + 2)  # first incoming port of each strand position
        for g in word:
            i = abs(g)
            if i == 0:
                raise PDError("braid generators are nonzero integers")
            b = len(nbr)
            nbr.extend((-1, -1, -1, -1))
            if g > 0:  # strand i passes over: in at slot 3, out at slot 1
                sign.append(1)
                in_i, in_j, out_j, out_i = b + 3, b, b + 1, b + 2
            else:  # strand i + 1 passes over: in at slot 1, out at slot 3
                sign.append(-1)
                in_i, in_j, out_j, out_i = b, b + 1, b + 2, b + 3
            for pos, port in ((i, in_i), (i + 1, in_j)):
                if top[pos] < 0:
                    bottom[pos] = port
                else:
                    nbr[top[pos]] = port
                    nbr[port] = top[pos]
            top[i], top[i + 1] = out_i, out_j
        n_unknots = 0
        for pos in range(1, k + 1):
            if top[pos] < 0:
                n_unknots += 1
            else:
                nbr[top[pos]] = bottom[pos]
                nbr[bottom[pos]] = top[pos]
        return cls(nbr, sign, n_unknots)

    @classmethod
    def unknot(cls, n_components: int = 1) -> LinkDiagram:
        """The crossingless diagram of the ``n_components``-component unlink."""
        return cls((), (), n_components)

    def copy(self) -> LinkDiagram:
        """An independent copy (caches are not copied)."""
        return LinkDiagram(self._nbr, self._sign, self._n_unknots, check=False)

    # ------------------------------------------------------------------
    # Basic data
    # ------------------------------------------------------------------

    @property
    def n_crossings(self) -> int:
        return len(self._sign)

    @property
    def n_unknots(self) -> int:
        """Number of crossingless unknotted components."""
        return self._n_unknots

    @property
    def n_components(self) -> int:
        """Number of link components, including crossingless ones."""
        return len(self._ensure_components()) + self._n_unknots

    @property
    def nbr(self) -> Sequence[int]:
        """Read-only view of the neighbour map (do not mutate)."""
        return self._nbr

    @property
    def signs(self) -> Sequence[int]:
        """Read-only view of the crossing signs (do not mutate)."""
        return self._sign

    def is_out(self, p: int) -> bool:
        """Whether the strand leaves the crossing at port ``p``."""
        k = p & 3
        if k == 2:
            return True
        if k == 0:
            return False
        return (k == 1) == (self._sign[p >> 2] > 0)

    def edge_tail(self, p: int) -> int:
        """The outgoing end of the edge at port ``p`` (edges are named by their tails)."""
        return p if self.is_out(p) else self._nbr[p]

    def writhe(self) -> int:
        return sum(self._sign)

    def is_trivial(self) -> bool:
        """Whether the diagram has no crossings (a diagram of an unlink)."""
        return not self._sign

    def is_pd_ambiguous(self) -> bool:
        """Whether some component passes over at every crossing.

        A PD code cannot record the orientation of such a component, so
        :meth:`from_pd` of :meth:`to_pd` may reverse it (flipping its signs).
        """
        nbr = self._nbr
        return any(all(nbr[p] & 1 for p in seq) for seq in self._ensure_components())

    # ------------------------------------------------------------------
    # Derived structure (cached)
    # ------------------------------------------------------------------

    def face_of_port(self) -> Sequence[int]:
        """Face id of the side named by each port (read-only view)."""
        return self._ensure_faces()

    @property
    def n_faces(self) -> int:
        self._ensure_faces()
        return len(self._face_start)

    def faces(self) -> list[list[int]]:
        """Each face as the list of its corners (ports), in walking order."""
        self._ensure_faces()
        nbr = self._nbr
        out = []
        for start in self._face_start:
            corners = [start]
            p = nbr[next_ccw(start)]
            while p != start:
                corners.append(p)
                p = nbr[next_ccw(p)]
            out.append(corners)
        return out

    def diagram_component_of_crossing(self) -> Sequence[int]:
        """Connected component of the projection containing each crossing (read-only view)."""
        return self._ensure_dcomp()

    def components(self) -> list[list[int]]:
        """Link components with crossings, each as the tails of its edges in order."""
        return [list(seq) for seq in self._ensure_components()]

    def component_writhes(self) -> list[int]:
        """Sum of signs of each component's self-crossings (its blackboard framing)."""
        comp_of = self._component_of_port()
        out = [0] * len(self._ensure_components())
        for c, s in enumerate(self._sign):
            k = comp_of[4 * c]
            if comp_of[4 * c + 1] == k:
                out[k] += s
        return out

    def linking_matrix(self) -> list[list[int]]:
        """Pairwise linking numbers of the components with crossings."""
        comp_of = self._component_of_port()
        r = len(self._ensure_components())
        twice = [[0] * r for _ in range(r)]
        for c, s in enumerate(self._sign):
            i, j = comp_of[4 * c], comp_of[4 * c + 1]
            if i != j:
                twice[i][j] += s
                twice[j][i] += s
        return [[v // 2 for v in row] for row in twice]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_pd(self) -> list[tuple[int, int, int, int]]:
        """PD code with edge labels ``1..2n`` running consecutively along each component.

        Crossingless components are not representable and are dropped; see
        :attr:`n_unknots`.
        """
        nbr = self._nbr
        label = [0] * len(nbr)
        nxt = 1
        for seq in self._ensure_components():
            for p in seq:
                label[p] = label[nbr[p]] = nxt
                nxt += 1
        return [tuple(label[4 * c : 4 * c + 4]) for c in range(len(self._sign))]  # type: ignore[misc]

    def to_gauss(self) -> list[list[int]]:
        """Gauss code per component: ``+(c+1)`` passing over crossing ``c``, ``-(c+1)`` under."""
        nbr = self._nbr
        out = []
        for seq in self._ensure_components():
            out.append([((nbr[p] >> 2) + 1) * (1 if nbr[p] & 1 else -1) for p in seq])
        return out

    # ------------------------------------------------------------------
    # Moves (implemented in knotml.core.moves)
    # ------------------------------------------------------------------

    def is_legal(self, move: Move) -> bool:
        """O(1) legality test (see :mod:`knotml.core.moves`)."""
        return _moves.is_legal(self, move)

    def legal_moves(self, kinds: Iterable[MoveKind] = _moves.ISOTOPY_KINDS) -> list[Move]:
        """All legal moves of the given kinds, each listed once."""
        return _moves.legal_moves(self, kinds)

    def move_masks(self) -> dict:
        """Legality of every move as O(n)-size numpy arrays (see :func:`moves.move_masks`)."""
        return _moves.move_masks(self)

    def apply_inplace(self, move: Move, check: bool = True) -> Optional[Move]:
        """Apply ``move`` in O(1); return a move undoing it (``None`` if no single move does)."""
        return _moves.apply_inplace(self, move, check)

    def apply(self, move: Move, check: bool = True) -> LinkDiagram:
        """A new diagram with ``move`` applied; ``self`` is unchanged."""
        out = self.copy()
        out.apply_inplace(move, check)
        return out

    def inverse(self, move: Move) -> Optional[Move]:
        """The move undoing ``move``, valid on ``self.apply(move)``."""
        return self.copy().apply_inplace(move)

    def random_move(self, rng: random.Random, kinds: Iterable[MoveKind] = _moves.ISOTOPY_KINDS) -> Move:
        """A uniformly random legal move (raises ``IndexError`` if there is none)."""
        return rng.choice(self.legal_moves(kinds))

    # ------------------------------------------------------------------
    # Canonical form
    # ------------------------------------------------------------------

    def canonical_key(self) -> bytes:
        """Bytes identifying the oriented diagram up to relabelling (see :mod:`canonical`)."""
        from .canonical import canonical_key

        return canonical_key(self)

    def is_isomorphic(self, other: LinkDiagram) -> bool:
        return self.canonical_key() == other.canonical_key()

    # ------------------------------------------------------------------
    # Validation, equality, repr
    # ------------------------------------------------------------------

    def check(self) -> None:
        """Raise :class:`PDError` unless this is a valid classical oriented diagram.

        Checks the neighbour map is a fixed-point-free involution, every edge runs
        from an outgoing to an incoming port, and every connected piece of the
        projection is planar: by Euler's formula it must have ``n_k + 2`` faces.
        """
        nbr, sign = self._nbr, self._sign
        n, m = len(sign), len(nbr)
        if m != 4 * n:
            raise PDError(f"{m} ports for {n} crossings")
        if self._n_unknots < 0:
            raise PDError("negative number of crossingless components")
        if any(s != 1 and s != -1 for s in sign):
            raise PDError("crossing signs must be +1 or -1")
        for p in range(m):
            q = nbr[p]
            if not 0 <= q < m or q == p or nbr[q] != p:
                raise PDError(f"port {p}: neighbour map is not a fixed-point-free involution")
            if self.is_out(p) == self.is_out(q):
                raise PDError(f"edge {p}-{q} does not run from an outgoing to an incoming port")
        self._invalidate()
        dcomp = self._ensure_dcomp()
        self._ensure_faces()
        n_pieces = max(dcomp, default=-1) + 1
        excess = [-2] * n_pieces  # faces - crossings - 2 for each piece
        for c in range(n):
            excess[dcomp[c]] -= 1
        for start in self._face_start:
            excess[dcomp[start >> 2]] += 1
        if any(excess):
            raise PDError("not planar: the code describes a virtual (non-classical) diagram")

    def __eq__(self, other: object) -> bool:
        """Exact equality of the stored data (not up to relabelling)."""
        if not isinstance(other, LinkDiagram):
            return NotImplemented
        return (
            self._sign == other._sign
            and self._nbr == other._nbr
            and self._n_unknots == other._n_unknots
        )

    __hash__ = None  # type: ignore[assignment]  # mutable

    def __repr__(self) -> str:
        return f"LinkDiagram(crossings={self.n_crossings}, components={self.n_components})"

    # ------------------------------------------------------------------
    # Caches
    # ------------------------------------------------------------------

    def _invalidate(self) -> None:
        self._face: Optional[list[int]] = None
        self._face_start: list[int] = []
        self._dcomp: Optional[list[int]] = None
        self._comps: Optional[list[list[int]]] = None

    def _ensure_faces(self) -> list[int]:
        if self._face is None:
            nbr = self._nbr
            face = [-1] * len(nbr)
            starts = []
            for p0 in range(len(nbr)):
                if face[p0] >= 0:
                    continue
                f = len(starts)
                starts.append(p0)
                p = p0
                while face[p] < 0:
                    face[p] = f
                    p = nbr[p + 1 if p & 3 != 3 else p - 3]
            self._face, self._face_start = face, starts
        return self._face

    def _ensure_dcomp(self) -> list[int]:
        if self._dcomp is None:
            nbr = self._nbr
            dcomp = [-1] * len(self._sign)
            k = 0
            for c0 in range(len(dcomp)):
                if dcomp[c0] >= 0:
                    continue
                dcomp[c0] = k
                stack = [c0]
                while stack:
                    c = stack.pop()
                    for p in range(4 * c, 4 * c + 4):
                        d = nbr[p] >> 2
                        if dcomp[d] < 0:
                            dcomp[d] = k
                            stack.append(d)
                k += 1
            self._dcomp = dcomp
        return self._dcomp

    def _ensure_components(self) -> list[list[int]]:
        if self._comps is None:
            nbr, sign = self._nbr, self._sign
            seen = bytearray(len(nbr))
            comps = []
            for c in range(len(sign)):
                for p in (4 * c + 2, 4 * c + (1 if sign[c] > 0 else 3)):
                    if seen[p]:
                        continue
                    seq = []
                    while not seen[p]:
                        seen[p] = 1
                        seq.append(p)
                        p = nbr[p] ^ 2
                    comps.append(seq)
            self._comps = comps
        return self._comps

    def _component_of_port(self) -> list[int]:
        nbr = self._nbr
        comp_of = [0] * len(nbr)
        for k, seq in enumerate(self._ensure_components()):
            for p in seq:
                comp_of[p] = comp_of[nbr[p]] = k
        return comp_of


def _orient(nbr: list[int], label: list) -> list[int]:
    """Crossing signs of a PD code given as a neighbour map plus edge labels.

    Strands are oriented by the under-slots (in at 0, out at 2); walking each
    component then reveals at which odd slot the over-strand enters, i.e. the
    sign.  Components that never pass under fall back to label order.
    """
    n = len(nbr) // 4
    sign = [0] * n
    seen = bytearray(len(nbr))
    for c in range(n):
        start = 4 * c + 2
        if seen[start]:
            continue
        p = start
        while True:
            q = nbr[p]
            if seen[p] or seen[q] or q & 3 == 2:
                raise PDError("inconsistent orientation: a strand runs into an outgoing under-slot")
            seen[p] = seen[q] = 1
            if q & 1:
                sign[q >> 2] = 1 if q & 3 == 3 else -1
            p = q ^ 2
            if p == start:
                break
    for p0 in range(1, len(nbr), 2):
        if seen[p0]:
            continue
        c0 = p0 >> 2
        start = 4 * c0 + (3 if _labels_increase_through_slot3(nbr, label, c0) else 1)
        q = start
        while True:
            if seen[q] or not q & 1:
                raise PDError("inconsistent orientation in an over-only component")
            sign[q >> 2] = 1 if q & 3 == 3 else -1
            seen[q] = seen[q ^ 2] = 1
            q = nbr[q ^ 2]
            if q == start:
                break
    return sign


def _labels_increase_through_slot3(nbr: list[int], label: list, c: int) -> bool:
    """For an over-only component: do its labels increase entering crossing ``c`` at slot 3?"""
    labels = []
    q = start = 4 * c + 3
    while True:
        labels.append(label[q])
        q = nbr[q ^ 2]
        if q == start or len(labels) > len(nbr):
            break
    a, b = label[4 * c + 3], label[4 * c + 1]
    try:
        return b == a + 1 or (a == max(labels) and b == min(labels))
    except TypeError:  # non-numeric labels carry no order; either orientation is valid
        return True
