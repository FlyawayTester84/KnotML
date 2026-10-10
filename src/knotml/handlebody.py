"""4-manifold Kirby calculus: dotted circles (1-handles) and framed 2-handles.

A :class:`Handlebody` is a :class:`~knotml.core.LinkDiagram` whose components are
either *dotted circles*, 1-handles in Akbulut's dotted-circle notation, or framed
knots, the attaching circles of 2-handles.  It describes the 4-manifold
X = B⁴ ∪ 1-handles ∪ 2-handles.  With ``closed=True``, 3- and 4-handles are added
implicitly to close X up, which allows 2/3-handle pairs to be created and cancelled.
The dotted circles must form an unlink; the constructor checks that their pairwise
linking numbers vanish, and every move keeps them an unlink.

Moves (``HANDLEBODY_KINDS`` plus Reidemeister moves; no blow-ups)
------------------------------------------------------------------
* Reidemeister moves: isotopy of the whole diagram (crossing changes excluded).
* ``HANDLE_SLIDE(a, b)``: slide the component through side ``a`` over the one
  through side ``b``, with the same band rule as :class:`~knotml.kirby.FramedLink`.
  Allowed: a 2-handle over a 2-handle; a 2-handle over a dotted circle (treated
  as a 0-framed unknot); a dotted circle over a dotted circle (0-framed push-off),
  but only while no two dotted strands cross anywhere in the diagram.  Then the
  dotted circles bound disjoint flat disks, the push-off and the band are planar,
  and the result is again an unlink; an arbitrary band could thread another
  dotted circle's disk and link the circles.  Reidemeister moves untangle the
  dotted circles first if needed.  Not allowed: a dotted circle over a 2-handle.
* ``CANCEL_12(a)``: erase the dotted circle ``a`` together with the 2-handle ``K``
  passing once through it.  Recognised locally: ``a`` has exactly two crossings,
  both with ``K``, over at one and under at the other, and an edge of ``K`` joins
  them, so ``a`` is a meridian of ``K`` and nothing else passes through its disk.
  Other strands through a dotted circle are slid off over ``K`` first.
* ``CREATE_12(sign=n)``: add a split cancelling pair, a dotted circle with an
  ``n``-framed unknot through it, ``n ∈ {-1, 0, 1}``.
* ``CANCEL_23(a)`` / ``CREATE_23()``: remove / add a split crossingless 0-framed
  2-handle together with a 3-handle (``closed=True`` only).

Invariants of X: :meth:`Handlebody.euler_characteristic`, :meth:`homology`
(H₁(X) = coker ∂, where ∂ records how often each 2-handle runs over each
1-handle), :meth:`intersection_form` (on H₂(X) = ker ∂), :meth:`signature`, and
:meth:`boundary`, the 3-manifold ∂X as a :class:`~knotml.kirby.FramedLink`
(dotted circles become 0-framed unknots).
"""

from __future__ import annotations

from typing import Iterable, Iterator, Mapping, NamedTuple, Optional, Sequence, Union

import numpy as np

from .core import ISOTOPY_KINDS, IllegalMoveError, LinkDiagram
from .kirby import (
    HANDLEBODY_KINDS,
    AnyMove,
    FramedLink,
    KirbyKind,
    KirbyMove,
    _smith_divisors,
    abelian_group,
    band_slide,
    count_slide_locations,
    integer_kernel,
    iter_slide_locations,
    linking_matrix,
    signature,
    slide_location_ok,
)

__all__ = ["Handlebody"]

_HOPF = [[4, 2, 3, 1], [2, 4, 1, 3]]  # LinkInfo L2a1{1}: a cancelling 1/2 pair when one component is dotted


class _State(NamedTuple):
    """Per-diagram-state data behind the O(1) legality checks, built in one O(n) pass."""

    stamp: int
    dotted_port: bytearray  # 1 where the strand through the port is a dotted circle
    flat: bool  # no crossing has two dotted strands
    partner: dict  # dotted label -> 2-handle it is a meridian of (CANCEL_12 is legal)


class Handlebody:
    """A Kirby diagram of a 4-manifold: dotted circles (1-handles) and framed 2-handles."""

    __slots__ = ("diagram", "_framing", "_dotted", "closed", "_cache")

    def __init__(
        self,
        diagram: LinkDiagram,
        components: Union[Mapping[int, Optional[int]], Sequence[Optional[int]]],
        *,
        closed: bool = False,
    ) -> None:
        """``components`` gives each label's framing, or ``None`` for a dotted circle,
        as a mapping or in label order."""
        labels = diagram.labels()
        if isinstance(components, Mapping):
            items = dict(components)
        else:
            if len(components) != len(labels):
                raise ValueError(f"{len(components)} entries for {len(labels)} components")
            items = dict(zip(labels, components, strict=True))
        if sorted(items) != labels:
            raise ValueError(f"entries given for labels {sorted(items)}, diagram has {labels}")
        self.diagram = diagram
        self._cache: Optional[_State] = None
        self._dotted = {lab for lab, f in items.items() if f is None}
        self._framing = {lab: int(f) for lab, f in items.items() if f is not None}
        self.closed = closed
        dotted = sorted(self._dotted)
        lk = linking_matrix(diagram, dotted, {lab: 0 for lab in dotted})
        if any(any(row) for row in lk):
            raise ValueError("dotted circles must form an unlink (they have nonzero linking numbers)")

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pd(cls, pd, components: Sequence[Optional[int]], unknots: Sequence[Optional[int]] = (), *,
                closed: bool = False) -> Handlebody:
        """``components`` follow the PD's components in order of their smallest edge label;
        ``unknots`` add crossingless components (``None`` = dotted)."""
        d = LinkDiagram.from_pd(pd, n_unknots=len(unknots))
        return cls(d, list(components) + list(unknots), closed=closed)

    @classmethod
    def from_braid(cls, word, components: Sequence[Optional[int]], n_strands: Optional[int] = None, *,
                   closed: bool = False) -> Handlebody:
        return cls(LinkDiagram.from_braid(word, n_strands), components, closed=closed)

    @classmethod
    def empty(cls, closed: bool = False) -> Handlebody:
        """B⁴ (S⁴ if closed)."""
        return cls(LinkDiagram.unknot(0), {}, closed=closed)

    def copy(self) -> Handlebody:
        return Handlebody(self.diagram.copy(), self.components, closed=self.closed)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def labels(self) -> list[int]:
        return sorted(self._dotted | set(self._framing))

    @property
    def components(self) -> dict[int, Optional[int]]:
        """Each label's framing, ``None`` for dotted circles."""
        return {lab: None if lab in self._dotted else self._framing[lab] for lab in self.labels()}

    def dotted_labels(self) -> list[int]:
        return sorted(self._dotted)

    def two_handle_labels(self) -> list[int]:
        return sorted(self._framing)

    def is_dotted(self, label: int) -> bool:
        return label in self._dotted

    def framing(self, label: int) -> int:
        return self._framing[label]

    @property
    def n_1handles(self) -> int:
        return len(self._dotted)

    @property
    def n_2handles(self) -> int:
        return len(self._framing)

    def is_empty(self) -> bool:
        return not self._dotted and not self._framing

    def dotted_circles_flat(self) -> bool:
        """Whether no crossing has two dotted strands: the dotted circles are then disjoint
        planar curves, a certified unlink, and may be slid over each other."""
        return self._state().flat

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------

    def euler_characteristic(self) -> int:
        """χ of B⁴ ∪ 1-handles ∪ 2-handles (closing up adds the implicit 3- and 4-handles)."""
        return 1 - self.n_1handles + self.n_2handles

    def boundary_map(self) -> list[list[int]]:
        """∂: C₂ → C₁, rows = dotted circles, columns = 2-handles: algebraic passes through each disk."""
        dotted, framed = self.dotted_labels(), self.two_handle_labels()
        lk = linking_matrix(self.diagram, dotted + framed, {lab: 0 for lab in dotted + framed})
        h1 = len(dotted)
        return [[lk[i][h1 + j] for j in range(len(framed))] for i in range(h1)]

    def homology(self) -> list[int]:
        """H₁(X) = coker ∂ as elementary divisors (``0`` = a ``Z`` summand)."""
        return abelian_group(self.boundary_map())

    def intersection_form(self) -> list[list[int]]:
        """The intersection form on H₂(X) = ker ∂, in a Z-basis of ker ∂."""
        framed = self.two_handle_labels()
        L = linking_matrix(self.diagram, framed, self._framing)
        basis = integer_kernel(self.boundary_map(), len(framed)) if self._dotted else [
            [int(i == j) for i in range(len(framed))] for j in range(len(framed))
        ]
        return [[sum(u[a] * L[a][b] * v[b] for a in range(len(framed)) for b in range(len(framed))) for v in basis]
                for u in basis]

    def signature(self) -> int:
        return signature(self.intersection_form())

    def form_invariants(self) -> dict:
        """Basis-independent invariants of the intersection form: rank, signature, nullity,
        |det| of its nondegenerate part, and whether it is even."""
        Q = self.intersection_form()
        divisors = _smith_divisors(Q)
        nonzero = [v for v in divisors if v]
        det = 1
        for v in nonzero:
            det *= v
        return {
            "rank": len(Q),
            "signature": signature(Q),
            "nullity": len(Q) - len(nonzero),
            "det": det,
            "even": all(Q[i][i] % 2 == 0 for i in range(len(Q))),
        }

    def boundary(self) -> FramedLink:
        """∂X as integral surgery: dotted circles become 0-framed unknots."""
        return FramedLink(self.diagram.copy(), {lab: 0 if f is None else f for lab, f in self.components.items()})

    def to_snappy_boundary(self):
        """∂X as a SnapPy ``Manifold``.  Split cancelling pairs (a dotted circle with a
        2-handle through it and nothing else) are S³ summands and are cancelled first."""
        h = self.copy()
        lab = h.diagram._label
        for dot in h.dotted_labels():
            k = h._cancelling_partner(dot)
            if k is not None and sum(k in (lab[2 * c], lab[2 * c + 1]) for c in range(h.diagram.n_crossings)) == 2:
                h.apply_inplace(KirbyMove(KirbyKind.CANCEL_12, a=dot))
                lab = h.diagram._label
        return h.boundary().to_snappy()

    def invariants(self) -> dict:
        """Invariants preserved by every legal move (used to check move sequences)."""
        out = {"H1": self.homology(), **self.form_invariants()}
        if not self.closed:  # 2/3 moves change χ and ∂X, so these are only invariant without them
            out["euler"] = self.euler_characteristic()
            out["boundary_H1"] = self.boundary().homology()
        else:
            out.pop("nullity")  # a 0-framed split unknot adds to the radical only
            out.pop("rank")
        return out

    def canonical_key(self) -> bytes:
        """Key of the diagram up to relabelling, with dots and framings."""
        return self.diagram.canonical_key(attr=lambda lab: 1 if lab in self._dotted else 2 * self._framing[lab])

    # ------------------------------------------------------------------
    # Moves
    # ------------------------------------------------------------------

    def is_legal(self, move: AnyMove) -> bool:
        """O(1) legality test (after an O(n) pass once per diagram state)."""
        d = self.diagram
        if not isinstance(move.kind, KirbyKind):
            return move.kind in ISOTOPY_KINDS and d.is_legal(move)
        kind = move.kind
        if kind == KirbyKind.HANDLE_SLIDE:
            if not slide_location_ok(d, move.a, move.b):
                return False
            st = self._state()
            return not st.dotted_port[move.a] or (bool(st.dotted_port[move.b]) and st.flat)
        if kind == KirbyKind.CANCEL_12:
            return move.a in self._state().partner
        if kind == KirbyKind.CREATE_12:
            return move.sign in (-1, 0, 1)
        if kind == KirbyKind.CANCEL_23:
            return (
                self.closed
                and move.a in d.trivial_labels
                and move.a not in self._dotted
                and self._framing.get(move.a) == 0
            )
        if kind == KirbyKind.CREATE_23:
            return self.closed
        return False  # blow-ups and blow-downs are not 4-manifold moves

    def legal_moves(self, kinds: Iterable = ISOTOPY_KINDS | HANDLEBODY_KINDS) -> list:
        """Every legal move of the given kinds (Reidemeister and Kirby), each listed once.

        O(n + output) time.  The output can be Θ(n²) (R2_ADD and slides pair up sides);
        use :meth:`iter_legal_moves`, :meth:`count_legal_moves` or :meth:`move_masks`
        to avoid materialising it.
        """
        return list(self.iter_legal_moves(kinds))

    def iter_legal_moves(self, kinds: Iterable = ISOTOPY_KINDS | HANDLEBODY_KINDS) -> Iterator:
        """The moves of :meth:`legal_moves`, generated lazily: O(n) extra memory."""
        kinds = frozenset(kinds)
        d = self.diagram
        yield from d.iter_legal_moves(kinds & ISOTOPY_KINDS)
        if KirbyKind.HANDLE_SLIDE in kinds:
            st = self._state()
            yield from (
                KirbyMove(KirbyKind.HANDLE_SLIDE, x, y) for x, y in iter_slide_locations(d, st.dotted_port, st.flat)
            )
        if KirbyKind.CANCEL_12 in kinds:
            yield from (KirbyMove(KirbyKind.CANCEL_12, a=lab) for lab in sorted(self._state().partner))
        if KirbyKind.CREATE_12 in kinds:
            yield from (KirbyMove(KirbyKind.CREATE_12, sign=n) for n in (-1, 0, 1))
        if self.closed and KirbyKind.CANCEL_23 in kinds:
            yield from (KirbyMove(KirbyKind.CANCEL_23, a=lab) for lab in self._cancellable_23())
        if self.closed and KirbyKind.CREATE_23 in kinds:
            yield KirbyMove(KirbyKind.CREATE_23)

    def count_legal_moves(self, kinds: Iterable = ISOTOPY_KINDS | HANDLEBODY_KINDS) -> dict:
        """Number of legal moves of each kind, in O(n) time and memory without listing them."""
        kinds = frozenset(kinds)
        d = self.diagram
        out: dict = dict(d.count_legal_moves(kinds & ISOTOPY_KINDS))
        st = self._state()
        if KirbyKind.HANDLE_SLIDE in kinds:
            out[KirbyKind.HANDLE_SLIDE] = count_slide_locations(d, st.dotted_port, st.flat)
        if KirbyKind.CANCEL_12 in kinds:
            out[KirbyKind.CANCEL_12] = len(st.partner)
        if KirbyKind.CREATE_12 in kinds:
            out[KirbyKind.CREATE_12] = 3
        if KirbyKind.CANCEL_23 in kinds:
            out[KirbyKind.CANCEL_23] = len(self._cancellable_23()) if self.closed else 0
        if KirbyKind.CREATE_23 in kinds:
            out[KirbyKind.CREATE_23] = int(self.closed)
        return out

    def move_masks(self) -> dict:
        """Legality of every move as O(n)-size numpy arrays (the diagram's masks, plus):

        * ``side_label``, ``side_dotted``: component and dottedness of each side (port).
          ``HANDLE_SLIDE(x, y)`` is legal iff ``side_label[x] != side_label[y]``,
          (``side_face[x] == side_face[y]`` or ``side_piece[x] != side_piece[y]``), and
          (not ``side_dotted[x]`` or (``side_dotted[y]`` and ``dotted_flat``)).
        * ``dotted_flat``: whether dotted circles may slide over each other.
        * ``cancel_12``: dotted labels that can be cancelled; ``create_12``: framings -1, 0, 1.
        * ``cancel_23``: labels that can be cancelled with a 3-handle; ``create_23``: bool.

        Crossing changes are not 4-manifold moves, so ``crossing_change`` is all False.
        """
        masks = self.diagram.move_masks()
        st = self._state()
        masks["crossing_change"][:] = False
        masks["side_label"] = np.asarray(self.diagram._label, dtype=np.int64)[
            ((np.arange(len(self.diagram.nbr)) >> 2) << 1) | (np.arange(len(self.diagram.nbr)) & 1)
        ]
        masks["side_dotted"] = np.frombuffer(bytes(st.dotted_port), dtype=np.uint8).astype(bool)
        masks["dotted_flat"] = np.bool_(st.flat)
        masks["cancel_12"] = np.array(sorted(st.partner), dtype=np.int64)
        masks["create_12"] = np.ones(3, dtype=bool)
        masks["cancel_23"] = np.array(self._cancellable_23() if self.closed else [], dtype=np.int64)
        masks["create_23"] = np.bool_(self.closed)
        return masks

    def apply_inplace(self, move: AnyMove, check: bool = True) -> Optional[AnyMove]:
        """Apply a move; return a move undoing it, or ``None`` when no single move does
        (handle slides, and cancelling a pair that is not a split Hopf link)."""
        if check and not self.is_legal(move):
            raise IllegalMoveError(f"{move!r} is not legal on {self!r}")
        d = self.diagram
        if not isinstance(move.kind, KirbyKind):
            return d.apply_inplace(move, check=False)  # the diagram's new stamp invalidates the cache
        self._cache = None  # dotted circles and framings may change
        kind = move.kind
        if kind == KirbyKind.HANDLE_SLIDE:
            ki, kj = d.label_of_port(move.a), d.label_of_port(move.b)
            framing_j = 0 if kj in self._dotted else self._framing[kj]
            self.diagram, eps, twice_lk = band_slide(d, move.a, move.b, framing_j)
            if ki not in self._dotted:
                self._framing[ki] += framing_j + eps * twice_lk
            return None
        if kind == KirbyKind.CANCEL_12:
            k = self._cancelling_partner(move.a)
            # K crossing only the dotted circle (twice) means the pair is a split Hopf link
            hopf = sum(k in (d._label[2 * c], d._label[2 * c + 1]) for c in range(d.n_crossings)) == 2
            framing = self._framing.pop(k)
            self._dotted.discard(move.a)
            d.remove_components([move.a, k])
            return KirbyMove(KirbyKind.CREATE_12, sign=framing) if hopf and framing in (-1, 0, 1) else None
        if kind == KirbyKind.CREATE_12:
            dot, knot = self._new_labels(2)
            pair = LinkDiagram.from_pd(_HOPF)
            pair.relabel({0: dot, 1: knot})
            d.insert(pair)
            self._dotted.add(dot)
            self._framing[knot] = move.sign
            return KirbyMove(KirbyKind.CANCEL_12, a=dot)
        if kind == KirbyKind.CANCEL_23:
            d._trivial.remove(move.a)
            d._invalidate()
            del self._framing[move.a]
            return KirbyMove(KirbyKind.CREATE_23)
        (label,) = self._new_labels(1)
        d._trivial.append(label)
        d._invalidate()
        self._framing[label] = 0
        return KirbyMove(KirbyKind.CANCEL_23, a=label)

    def apply(self, move: AnyMove, check: bool = True) -> Handlebody:
        out = self.copy()
        out.apply_inplace(move, check)
        return out

    def random_move(self, rng, kinds: Iterable = ISOTOPY_KINDS | HANDLEBODY_KINDS):
        return rng.choice(self.legal_moves(kinds))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cancelling_partner(self, dot: int) -> Optional[int]:
        """The 2-handle that dotted circle ``dot`` is a meridian of, if any (see module docstring)."""
        return self._state().partner.get(dot)

    def _cancellable_23(self) -> list[int]:
        return [lab for lab in self.diagram.trivial_labels if lab not in self._dotted and self._framing[lab] == 0]

    def _state(self) -> _State:
        """Per-state data, rebuilt in O(n) when the diagram or the handle data changed."""
        d = self.diagram
        if self._cache is not None and self._cache.stamp == d.stamp:
            return self._cache
        lab, sign, nbr = d._label, d._sign, d.nbr
        dotted = self._dotted
        n = len(sign)
        dotted_port = bytearray(4 * n)
        flat = True
        crossings: dict[int, list[int]] = {}  # dotted label -> its crossings (one entry per strand passage)
        for c in range(n):
            under, over = lab[2 * c] in dotted, lab[2 * c + 1] in dotted
            if under:
                dotted_port[4 * c] = dotted_port[4 * c + 2] = 1
                crossings.setdefault(lab[2 * c], []).append(c)
            if over:
                dotted_port[4 * c + 1] = dotted_port[4 * c + 3] = 1
                crossings.setdefault(lab[2 * c + 1], []).append(c)
            if under and over:
                flat = False
        partner = {}
        for dot, cs in crossings.items():
            if len(cs) != 2 or cs[0] == cs[1]:
                continue
            c1, c2 = cs
            k = lab[2 * c1 + 1] if lab[2 * c1] == dot else lab[2 * c1]
            k2 = lab[2 * c2 + 1] if lab[2 * c2] == dot else lab[2 * c2]
            if k != k2 or k in dotted or (lab[2 * c1 + 1] == dot) == (lab[2 * c2 + 1] == dot):
                continue  # not one 2-handle, or the circle is over (or under) at both: no linking

            def k_out(c: int, k: int = k) -> int:  # the port where K leaves crossing c
                return 4 * c + 2 if lab[2 * c] == k else 4 * c + (1 if sign[c] > 0 else 3)

            if nbr[k_out(c1)] >> 2 == c2 or nbr[k_out(c2)] >> 2 == c1:  # an edge of K joins them
                partner[dot] = k
        self._cache = _State(d.stamp, dotted_port, flat, partner)
        return self._cache

    def _new_labels(self, count: int) -> list[int]:
        start = max(self.labels(), default=-1) + 1
        return list(range(start, start + count))

    def __repr__(self) -> str:
        parts = ", ".join(f"K{lab}: {'dot' if f is None else f'{f:+d}'}" for lab, f in self.components.items())
        closed = ", closed" if self.closed else ""
        return f"Handlebody(crossings={self.diagram.n_crossings}, {{{parts}}}{closed})"
