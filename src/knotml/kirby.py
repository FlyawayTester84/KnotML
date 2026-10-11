"""Framed links and 3-manifold (surgery) Kirby calculus on PD diagrams.

A :class:`FramedLink` is a :class:`~knotml.core.LinkDiagram` whose components,
identified by their stable labels, carry integer framing coefficients.  The
framing of a component is the linking number with its framing push-off,
measured against the Seifert framing, so Reidemeister moves never change it;
the blackboard framing (self-writhe) is a separate, diagram-dependent number.
A framed link describes the 3-manifold obtained by integral surgery on S³.

Moves (``SURGERY_KINDS`` plus Reidemeister moves)
-------------------------------------------------
* Reidemeister moves: the same :class:`~knotml.core.Move` objects as for diagrams
  (crossing changes excluded).
* ``BLOW_UP(sign)``: add a split, crossingless unknot framed ``sign = ±1``.
* ``BLOW_DOWN(a)``: remove the split, crossingless ``±1``-framed unknot labelled ``a``.
* ``HANDLE_SLIDE(a, b)``: slide the component through side ``a`` over the
  component through side ``b`` along a band in their common face (the same
  side/face rule as ``R2_ADD``).  The band is untwisted, so the slide adds the
  handle when the sides are both left or both right of their edges, and
  subtracts it otherwise; a kink (R1) first gives the other choice.

By Kirby's theorem two framed links give homeomorphic 3-manifolds iff they
are related by these moves.  For 4-manifolds (dotted circles, handle
cancellations, no blow-ups) see :mod:`knotml.handlebody`.

A handle slide of ``K_i`` over ``K_j`` (:func:`band_slide`) is built in
O(n + |f_j - w_j|): a blackboard push-off of ``K_j`` on the band side (each
crossing of ``K_j`` with another strand gains one crossing, each self-crossing
three), ``f_j - w_j`` full twists between ``K_j`` and the push-off so their
linking number is the framing ``f_j``, and a band joining the push-off to
``K_i``.  The new framing is ``f_i + f_j + 2·ε·lk(K_i, K_j)`` with ``ε = ±1``
for addition/subtraction.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Iterable, Iterator, Mapping, NamedTuple, Optional, Sequence, Union

import numpy as np

from .core import ISOTOPY_KINDS, IllegalMoveError, LinkDiagram, Move
from .core.moves import _ports_by_piece
from .core.ports import TRIVIAL, next_ccw, prev_ccw, strand_index

__all__ = [
    "FramedLink",
    "KirbyKind",
    "KirbyMove",
    "SURGERY_KINDS",
    "HANDLEBODY_KINDS",
    "band_slide",
    "iter_slide_locations",
    "count_slide_locations",
    "pushoff",
]


class KirbyKind(IntEnum):
    BLOW_UP = 10
    BLOW_DOWN = 11
    HANDLE_SLIDE = 12
    CANCEL_12 = 13
    CREATE_12 = 14
    CANCEL_23 = 15
    CREATE_23 = 16


SURGERY_KINDS = frozenset({KirbyKind.BLOW_UP, KirbyKind.BLOW_DOWN, KirbyKind.HANDLE_SLIDE})
"""Kirby moves of the 3-manifold (surgery) calculus, used by :class:`FramedLink`."""
HANDLEBODY_KINDS = frozenset(
    {KirbyKind.HANDLE_SLIDE, KirbyKind.CANCEL_12, KirbyKind.CREATE_12, KirbyKind.CANCEL_23, KirbyKind.CREATE_23}
)
"""Kirby moves of the 4-manifold calculus, used by :class:`knotml.handlebody.Handlebody`."""


class KirbyMove(NamedTuple):
    """A Kirby move; ``a``, ``b`` and ``sign`` are interpreted per kind (see ``__repr__``)."""

    kind: KirbyKind
    a: int = TRIVIAL
    b: int = TRIVIAL
    sign: int = 0

    def __repr__(self) -> str:
        k = self.kind
        if k == KirbyKind.BLOW_UP:
            return f"BLOW_UP(sign={self.sign:+d})"
        if k == KirbyKind.BLOW_DOWN:
            return f"BLOW_DOWN(label={self.a})"
        if k == KirbyKind.HANDLE_SLIDE:
            return f"HANDLE_SLIDE(slide={self.a}, over={self.b})"
        if k == KirbyKind.CANCEL_12:
            return f"CANCEL_12(dotted={self.a})"
        if k == KirbyKind.CREATE_12:
            return f"CREATE_12(framing={self.sign:+d})"
        if k == KirbyKind.CANCEL_23:
            return f"CANCEL_23(label={self.a})"
        return "CREATE_23()"


AnyMove = Union[Move, KirbyMove]


class FramedLink:
    """A link diagram with an integer framing on each component: integral surgery on S³."""

    __slots__ = ("diagram", "_framing")

    def __init__(self, diagram: LinkDiagram, framings: Union[Mapping[int, int], Sequence[int]]) -> None:
        """``framings`` maps component labels to framings, or lists them in label order."""
        labels = diagram.labels()
        if isinstance(framings, Mapping):
            framing = {int(k): int(v) for k, v in framings.items()}
        else:
            if len(framings) != len(labels):
                raise ValueError(f"{len(framings)} framings for {len(labels)} components")
            framing = {lab: int(f) for lab, f in zip(labels, framings, strict=True)}
        if sorted(framing) != labels:
            raise ValueError(f"framings given for labels {sorted(framing)}, diagram has {labels}")
        self.diagram = diagram
        self._framing = framing

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pd(cls, pd, framings: Sequence[int], unknot_framings: Sequence[int] = ()) -> FramedLink:
        """``framings`` follow the PD's components in order of their smallest edge label
        (KnotInfo/LinkInfo order); ``unknot_framings`` add crossingless components."""
        d = LinkDiagram.from_pd(pd, n_unknots=len(unknot_framings))
        return cls(d, list(framings) + list(unknot_framings))

    @classmethod
    def from_braid(cls, word, framings: Sequence[int], n_strands: Optional[int] = None) -> FramedLink:
        """``framings`` follow the closure's components in label order (see ``LinkDiagram``)."""
        return cls(LinkDiagram.from_braid(word, n_strands), framings)

    @classmethod
    def unknot(cls, framing: int) -> FramedLink:
        return cls(LinkDiagram.unknot(), [framing])

    def copy(self) -> FramedLink:
        return FramedLink(self.diagram.copy(), dict(self._framing))

    # ------------------------------------------------------------------
    # Data and invariants
    # ------------------------------------------------------------------

    @property
    def n_components(self) -> int:
        return len(self._framing)

    def labels(self) -> list[int]:
        return sorted(self._framing)

    def framing(self, label: int) -> int:
        return self._framing[label]

    def is_empty(self) -> bool:
        """Whether no components are left (the empty diagram: S³)."""
        return not self._framing

    @property
    def framings(self) -> dict[int, int]:
        return {lab: self._framing[lab] for lab in self.labels()}

    def linking_matrix(self) -> list[list[int]]:
        """Linking numbers off the diagonal, framings on it, in label order.

        It presents H₁ of the surgered 3-manifold, and is the intersection form of
        the 4-manifold B⁴ ∪ 2-handles that it bounds.
        """
        return linking_matrix(self.diagram, self.labels(), self._framing)

    def homology(self) -> list[int]:
        """H₁ of the surgered 3-manifold as elementary divisors (``0`` = a ``Z`` summand), like SnapPy."""
        return abelian_group(self.linking_matrix())

    def signature(self) -> int:
        """Signature of the linking matrix (of the 2-handlebody it bounds)."""
        return signature(self.linking_matrix())

    def canonical_key(self) -> bytes:
        """Key of the framed diagram up to relabelling (diagram and framings)."""
        return self.diagram.canonical_key(attr=self._framing.__getitem__)

    def to_snappy(self):
        """The surgered 3-manifold as a SnapPy ``Manifold``.

        Split crossingless ±1-framed unknots (blow-ups) are S³ summands and are
        dropped; any other split diagram raises ``ValueError``, as SnapPy needs a
        connected link projection.
        """
        import snappy  # noqa: F401  (spherogram needs SnapPy loaded to build exteriors)

        from .interop import to_spherogram

        d = self.diagram.copy()
        for label in list(d.trivial_labels):
            if self._framing[label] in (1, -1):
                d._trivial.remove(label)
        if d.n_unknots or len(set(d.diagram_component_of_crossing())) > 1:
            raise ValueError("SnapPy needs a connected diagram; this framed link is split")
        if d.n_crossings == 0:
            raise ValueError("the surgery is S³, which SnapPy cannot represent as a link exterior")
        link = to_spherogram(d)
        manifold = link.exterior()
        fillings = []
        for comp in link.link_components:
            entry = comp[0]
            label = d.label_of_port(4 * entry.crossing.label + entry.strand_index)
            fillings.append((self._framing[label], 1))  # SnapPy longitudes are the Seifert ones
        manifold.dehn_fill(fillings)
        return manifold

    # ------------------------------------------------------------------
    # Moves
    # ------------------------------------------------------------------

    def is_legal(self, move: AnyMove) -> bool:
        d = self.diagram
        if isinstance(move.kind, KirbyKind):
            if move.kind == KirbyKind.BLOW_UP:
                return move.sign in (1, -1)
            if move.kind == KirbyKind.BLOW_DOWN:
                return move.a in d.trivial_labels and self._framing.get(move.a) in (1, -1)
            if move.kind == KirbyKind.HANDLE_SLIDE:
                return slide_location_ok(d, move.a, move.b)
            return False  # handle cancellations belong to the 4-manifold calculus
        return move.kind in ISOTOPY_KINDS and d.is_legal(move)

    def legal_moves(self, kinds: Iterable = ISOTOPY_KINDS | SURGERY_KINDS) -> list:
        """Legal isotopy and surgery-calculus moves of the given kinds."""
        return list(self.iter_legal_moves(kinds))

    def iter_legal_moves(self, kinds: Iterable = ISOTOPY_KINDS | SURGERY_KINDS) -> Iterator:
        """The moves of :meth:`legal_moves`, generated lazily in O(n) extra memory."""
        kinds = frozenset(kinds)
        d = self.diagram
        yield from d.iter_legal_moves(kinds & ISOTOPY_KINDS)
        if KirbyKind.BLOW_UP in kinds:
            yield from (KirbyMove(KirbyKind.BLOW_UP, sign=s) for s in (1, -1))
        if KirbyKind.BLOW_DOWN in kinds:
            blowable = [lab for lab in d.trivial_labels if self._framing[lab] in (1, -1)]
            yield from (KirbyMove(KirbyKind.BLOW_DOWN, a=lab) for lab in blowable)
        if KirbyKind.HANDLE_SLIDE in kinds:
            yield from (KirbyMove(KirbyKind.HANDLE_SLIDE, x, y) for x, y in iter_slide_locations(d))

    def count_legal_moves(self, kinds: Iterable = ISOTOPY_KINDS | SURGERY_KINDS) -> dict:
        """Number of legal moves of each kind, in O(n) without listing them."""
        kinds = frozenset(kinds)
        d = self.diagram
        out: dict = dict(d.count_legal_moves(kinds & ISOTOPY_KINDS))
        if KirbyKind.BLOW_UP in kinds:
            out[KirbyKind.BLOW_UP] = 2
        if KirbyKind.BLOW_DOWN in kinds:
            out[KirbyKind.BLOW_DOWN] = sum(self._framing[lab] in (1, -1) for lab in d.trivial_labels)
        if KirbyKind.HANDLE_SLIDE in kinds:
            out[KirbyKind.HANDLE_SLIDE] = count_slide_locations(d)
        return out

    def apply_inplace(self, move: AnyMove, check: bool = True) -> Optional[AnyMove]:
        """Apply a move; return a move undoing it (``None`` for handle slides)."""
        if check and not self.is_legal(move):
            raise IllegalMoveError(f"{move!r} is not legal on {self!r}")
        d = self.diagram
        if not isinstance(move.kind, KirbyKind):
            return d.apply_inplace(move, check=False)
        if move.kind == KirbyKind.BLOW_UP:
            label = max(self._framing, default=-1) + 1
            d._trivial.append(label)
            d._invalidate()  # new stamp: the components changed
            self._framing[label] = move.sign
            return KirbyMove(KirbyKind.BLOW_DOWN, a=label)
        if move.kind == KirbyKind.BLOW_DOWN:
            d._trivial.remove(move.a)
            d._invalidate()
            return KirbyMove(KirbyKind.BLOW_UP, sign=self._framing.pop(move.a))
        ki, kj = d.label_of_port(move.a), d.label_of_port(move.b)
        self.diagram, eps, twice_lk = band_slide(d, move.a, move.b, self._framing[kj])
        self._framing[ki] += self._framing[kj] + eps * twice_lk
        return None

    def apply(self, move: AnyMove, check: bool = True) -> FramedLink:
        out = self.copy()
        out.apply_inplace(move, check)
        return out

    def random_move(self, rng, kinds: Iterable = ISOTOPY_KINDS | SURGERY_KINDS):
        return rng.choice(self.legal_moves(kinds))

    def __repr__(self) -> str:
        return f"FramedLink(crossings={self.diagram.n_crossings}, framings={self.framings})"


# ----------------------------------------------------------------------
# Shared by the 3- and 4-manifold calculi
# ----------------------------------------------------------------------


def slide_location_ok(d: LinkDiagram, x: int, y: int) -> bool:
    """Whether sides ``x`` and ``y`` lie on different components and can be joined by a band."""
    m = len(d._nbr)
    if not (0 <= x < m and 0 <= y < m) or d.label_of_port(x) == d.label_of_port(y):
        return False
    face = d._ensure_faces()
    return face[x] == face[y] or d._ensure_dcomp()[x >> 2] != d._ensure_dcomp()[y >> 2]


def slide_locations(d: LinkDiagram) -> list[tuple[int, int]]:
    """All ``(slide, over)`` side pairs satisfying :func:`slide_location_ok`."""
    return list(iter_slide_locations(d))


def iter_slide_locations(d: LinkDiagram, dotted=None, flat: bool = True) -> Iterator[tuple[int, int]]:
    """Generate ``(slide, over)`` side pairs on different components that share a face or
    lie in different pieces of the projection.

    ``dotted`` (per-port flags, for handlebodies) restricts a dotted slid component to dotted
    targets, and only when ``flat``.  Each face's corners are grouped by component, so pairs
    on one component are skipped wholesale: O(n + output + Σ_f |f|·(components meeting f))
    time and O(n) extra memory.
    """
    label = _port_labels(d)
    for corners in d.faces():
        buckets: dict[int, list[int]] = {}
        for p in corners:
            buckets.setdefault(label[p], []).append(p)
        if len(buckets) < 2:
            continue
        groups = list(buckets.items())
        dotted_groups = [(lab, ps) for lab, ps in groups if dotted[ps[0]]] if dotted is not None else []
        for x in corners:
            if dotted is not None and dotted[x]:
                if not flat:
                    continue
                pool = dotted_groups
            else:
                pool = groups
            lx = label[x]
            for ly, ys in pool:
                if ly != lx:
                    for y in ys:
                        yield x, y
    pieces = _ports_by_piece(d)
    if len(pieces) > 1:
        dotted_pieces = [[p for p in g if dotted[p]] for g in pieces] if dotted is not None else pieces
        for i, gi in enumerate(pieces):
            for x in gi:
                if dotted is not None and dotted[x]:
                    if not flat:
                        continue
                    pools = dotted_pieces
                else:
                    pools = pieces
                for j, g in enumerate(pools):
                    if j != i:
                        for y in g:
                            yield x, y


def count_slide_locations(d: LinkDiagram, dotted=None, flat: bool = True) -> int:
    """``len(list(iter_slide_locations(d, dotted, flat)))`` in O(n) time and memory."""
    label = _port_labels(d)
    face = d._ensure_faces()
    nf = len(d._face_start)
    size = [0] * nf
    dotted_size = [0] * nf
    per_label: dict[tuple[int, int], int] = {}
    for p, f in enumerate(face):
        size[f] += 1
        per_label[(f, label[p])] = per_label.get((f, label[p]), 0) + 1
        if dotted is not None and dotted[p]:
            dotted_size[f] += 1
    total = 0
    for p, f in enumerate(face):
        own = per_label[(f, label[p])]
        if dotted is not None and dotted[p]:
            total += dotted_size[f] - own if flat else 0
        else:
            total += size[f] - own
    pieces = _ports_by_piece(d)
    if len(pieces) > 1:
        n_all = sum(len(g) for g in pieces)
        n_dot = sum(1 for g in pieces for p in g if dotted is not None and dotted[p])
        for g in pieces:
            g_dot = sum(1 for p in g if dotted is not None and dotted[p])
            total += (len(g) - g_dot) * (n_all - len(g))
            if flat:
                total += g_dot * (n_dot - g_dot)
    return total


def _port_labels(d: LinkDiagram) -> list[int]:
    lab = d._label
    return [lab[((p >> 2) << 1) | (p & 1)] for p in range(len(d._nbr))]


def band_slide(d: LinkDiagram, x: int, y: int, framing_j: int) -> tuple[LinkDiagram, int, int]:
    """Slide the component through side ``x`` over the one through side ``y``.

    ``framing_j`` is the framing used for the push-off of the component at ``y``
    (0 for a dotted circle).  Returns the new diagram, ``ε = ±1`` (handle
    addition/subtraction) and ``2·lk`` of the two components before the slide.
    The slid component keeps its label.
    """
    ki, kj = d.label_of_port(x), d.label_of_port(y)
    writhe_j = twice_lk = 0
    for c, s in enumerate(d._sign):
        pair = {d._label[2 * c], d._label[2 * c + 1]}
        if pair == {kj}:
            writhe_j += s
        elif pair == {ki, kj}:
            twice_lk += s
    left = d.is_out(y)  # the push-off runs on the side of K_j that faces the band
    tmp = max(d.labels()) + 1
    D, outer, push_outer = pushoff(d, kj, left, tmp)

    # Full twists so that lk(K_j, push-off) equals the framing of K_j.  They go on the
    # edge after y's edge, leaving the push-off edge beside y free for the band.
    twists = framing_j - writhe_j
    if twists:
        tail_y = y if d.is_out(y) else d._nbr[y]
        nxt = d._nbr[tail_y] ^ 2
        _insert_twists(D, outer[nxt], push_outer[nxt], twists, kj_left=not left)

    # Band: K_i's edge at x is reconnected to the push-off's edge beside y.
    X, B = outer[x], push_outer[y]
    eps = 1
    if D.is_out(D._nbr[X]) == D.is_out(B):  # untwisted band needs the push-off reversed
        eps = -1
        rotated = set(D.reverse_component(tmp))
        X = X ^ 2 if X >> 2 in rotated else X
        B = B ^ 2 if B >> 2 in rotated else B
    nbr = D._nbr
    WA, BE = nbr[X], nbr[B]
    nbr[WA], nbr[B] = B, WA
    nbr[X], nbr[BE] = BE, X
    D._label = [ki if v == tmp else v for v in D._label]
    D._invalidate()
    return D, eps, twice_lk


def linking_matrix(d: LinkDiagram, labels: Sequence[int], diagonal: Mapping[int, int]) -> list[list[int]]:
    """Linking numbers between ``labels`` (in that order), with ``diagonal[label]`` on the diagonal."""
    index = {lab: k for k, lab in enumerate(labels)}
    twice = [[0] * len(labels) for _ in labels]
    lab = d._label
    for c, s in enumerate(d._sign):
        i, j = index.get(lab[2 * c]), index.get(lab[2 * c + 1])
        if i is not None and j is not None and i != j:
            twice[i][j] += s
            twice[j][i] += s
    Q = [[v // 2 for v in row] for row in twice]
    for k, label in enumerate(labels):
        Q[k][k] = diagonal[label]
    return Q


def abelian_group(M: list[list[int]]) -> list[int]:
    """Cokernel of an integer matrix as elementary divisors (``0`` = a ``Z`` summand), like SnapPy."""
    divisors = _smith_divisors(M)
    if M and len(M) > len(M[0]):  # more rows than columns: the extra rows are free summands
        divisors += [0] * (len(M) - len(M[0]))
    return sorted(v for v in divisors if v > 1) + [0] * divisors.count(0)


def signature(Q: list[list[int]]) -> int:
    if not Q:
        return 0
    eig = np.linalg.eigvalsh(np.array(Q, dtype=float))
    return int((eig > 1e-9).sum() - (eig < -1e-9).sum())


def integer_kernel(A: list[list[int]], n_cols: int) -> list[list[int]]:
    """A Z-basis of ``{v in Z^n_cols : A v = 0}`` (unimodular column reduction)."""
    M = [row[:] for row in A]
    U = [[int(i == j) for j in range(n_cols)] for i in range(n_cols)]  # columns are tracked

    def col_op(dst: int, src: int, q: int) -> None:  # column dst -= q * column src
        for row in M:
            row[dst] -= q * row[src]
        for row in U:
            row[dst] -= q * row[src]

    def swap(i: int, j: int) -> None:
        for row in M + U:
            row[i], row[j] = row[j], row[i]

    col = 0
    for r in range(len(M)):
        while col < n_cols:
            nonzero = [j for j in range(col, n_cols) if M[r][j]]
            if not nonzero:
                break
            swap(col, min(nonzero, key=lambda j: abs(M[r][j])))
            for j in range(col + 1, n_cols):
                if M[r][j]:
                    col_op(j, col, M[r][j] // M[r][col])
            if all(M[r][j] == 0 for j in range(col + 1, n_cols)):
                col += 1
                break
    return [[U[i][j] for i in range(n_cols)] for j in range(col, n_cols)]


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------


def pushoff(d: LinkDiagram, label: int, left: bool, new_label: int):
    """A copy of ``d`` plus a blackboard-parallel copy of component ``label``.

    The copy runs on the left (``left=True``) or right of the component, with
    the same orientation, and gets ``new_label``.  Original crossings keep their
    indices; each crossing of the component with another strand gains one
    crossing, each self-crossing three, all with the original's sign and slot
    roles.  Returns ``(diagram, outer, push_outer)``: ``outer[p]`` is the port
    that now carries the edge of original port ``p``, and ``push_outer[p]`` the
    corresponding push-off port for ports ``p`` on the component.  O(n).
    """
    nbr, sign, lab = d._nbr, d._sign, d._label
    n = len(sign)
    new_sign, new_lab = list(sign), list(lab)

    def add(under: int, over: int, like: int) -> int:
        new_sign.append(sign[like])
        new_lab.extend((under, over))
        return len(new_sign) - 1

    def in_port(c: int, par: int) -> int:
        return 4 * c if par == 0 else 4 * c + (3 if sign[c] > 0 else 1)

    def side(p: int) -> int:  # port of the other strand on the push-off side of the strand entering at p
        return prev_ccw(p) if left else next_ccw(p)

    seq: dict[tuple[int, int], list[int]] = {}  # crossings an original strand passes, in order
    push: dict[tuple[int, int], list[int]] = {}  # ... and its push-off
    for c in range(n):
        ku, ko = lab[2 * c] == label, lab[2 * c + 1] == label
        if ku and ko:  # self-crossing: a 2x2 grid
            p0, p1 = in_port(c, 0), in_port(c, 1)
            a = add(new_label, label, c)  # push-off of the under-strand x over-strand
            b = add(label, new_label, c)  # under-strand x push-off of the over-strand
            e = add(new_label, new_label, c)
            first0 = side(p1) == p0  # the over-strand's push-off lies where the under-strand enters
            first1 = side(p0) == p1
            seq[(c, 0)] = [b, c] if first0 else [c, b]
            seq[(c, 1)] = [a, c] if first1 else [c, a]
            push[(c, 0)] = [e, a] if first0 else [a, e]
            push[(c, 1)] = [e, b] if first1 else [b, e]
        elif ku or ko:
            s = 0 if ku else 1
            p = in_port(c, s)
            c1 = add(new_label, lab[2 * c + 1], c) if s == 0 else add(lab[2 * c], new_label, c)
            seq[(c, s)], push[(c, s)] = [c], [c1]
            seq[(c, 1 - s)] = [c1, c] if side(p) == in_port(c, 1 - s) else [c, c1]
        else:
            seq[(c, 0)] = seq[(c, 1)] = [c]

    new_nbr = [-1] * (4 * len(new_sign))

    def link(u: int, v: int) -> None:
        new_nbr[u], new_nbr[v] = v, u

    for table in (seq, push):
        for (c, par), crossings in table.items():
            i = in_port(c, par) & 3
            for u, v in zip(crossings, crossings[1:]):  # noqa: B905
                link(4 * u + (i ^ 2), 4 * v + i)
    outer: dict[int, int] = {}
    push_outer: dict[int, int] = {}
    for t in range(4 * n):
        if not d.is_out(t):
            continue
        h = nbr[t]
        st, sh = (t >> 2, t & 1), (h >> 2, h & 1)
        outer[t], outer[h] = 4 * seq[st][-1] + (t & 3), 4 * seq[sh][0] + (h & 3)
        link(outer[t], outer[h])
        if lab[strand_index(t)] == label:
            push_outer[t], push_outer[h] = 4 * push[st][-1] + (t & 3), 4 * push[sh][0] + (h & 3)
            link(push_outer[t], push_outer[h])
    out = LinkDiagram(new_nbr, new_sign, labels=new_lab, trivial=d.trivial_labels, check=False)
    return out, outer, push_outer


def _insert_twists(D: LinkDiagram, ta: int, tb: int, twists: int, kj_left: bool) -> None:
    """Insert ``twists`` full twists between parallel edges with tails ``ta`` (K_j) and ``tb``."""
    nbr, sign, lab = D._nbr, D._sign, D._label
    L, R = (ta, tb) if kj_left else (tb, ta)
    Lt, Rt = nbr[L], nbr[R]
    lab_l, lab_r = lab[strand_index(L)], lab[strand_index(R)]
    g = 1 if twists > 0 else -1
    in_l, in_r, out_r, out_l = (3, 0, 1, 2) if g > 0 else (0, 1, 2, 3)  # braid generator sigma^g
    for _ in range(2 * abs(twists)):
        X = len(nbr)
        sign.append(g)
        nbr.extend((-1, -1, -1, -1))
        lab.extend((lab_r, lab_l) if g > 0 else (lab_l, lab_r))  # (under, over)
        nbr[L], nbr[X + in_l] = X + in_l, L
        nbr[R], nbr[X + in_r] = X + in_r, R
        L, R = X + out_l, X + out_r
        lab_l, lab_r = lab_r, lab_l
    nbr[L], nbr[Lt] = Lt, L
    nbr[R], nbr[Rt] = Rt, R
    D._invalidate()


def _smith_divisors(M: list[list[int]]) -> list[int]:
    """Diagonal of the Smith normal form of an integer matrix (zeros for the free part)."""
    A = [row[:] for row in M]
    rows = len(A)
    cols = len(A[0]) if A else 0
    out = []
    for t in range(min(rows, cols)):
        while True:
            entries = [(abs(A[i][j]), i, j) for i in range(t, rows) for j in range(t, cols) if A[i][j]]
            if not entries:
                return out + [0] * (min(rows, cols) - t)
            _, i, j = min(entries)
            A[t], A[i] = A[i], A[t]
            for row in A:
                row[t], row[j] = row[j], row[t]
            p = A[t][t]
            clean = True
            for i in range(t + 1, rows):
                q = A[i][t] // p
                if q:
                    A[i] = [a - q * b for a, b in zip(A[i], A[t])]  # noqa: B905
                clean &= A[i][t] == 0
            for j in range(t + 1, cols):
                q = A[t][j] // p
                if q:
                    for row in A:
                        row[j] -= q * row[t]
                clean &= A[t][j] == 0
            if not clean:
                continue
            bad = [(i, j) for i in range(t + 1, rows) for j in range(t + 1, cols) if A[i][j] % p]
            if not bad:
                out.append(abs(p))
                break
            i, _ = bad[0]
            A[t] = [a + b for a, b in zip(A[t], A[i])]  # noqa: B905  (fold a non-multiple into row t)
    return out
