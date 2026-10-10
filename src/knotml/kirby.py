"""Framed links and Kirby calculus on PD diagrams.

A :class:`FramedLink` is a :class:`~knotml.core.LinkDiagram` whose components,
identified by their stable labels, carry integer framing coefficients.  The
framing of a component is the linking number with its framing push-off,
measured against the Seifert framing, so Reidemeister moves never change it;
the blackboard framing (self-writhe) is a separate, diagram-dependent number.

A framed link describes the 3-manifold obtained by integral surgery on S³,
and the 4-manifold obtained from B⁴ by attaching 2-handles along it.

Moves
-----
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
are related by these moves; without blow-ups and blow-downs they give
diffeomorphic 4-manifolds.

A handle slide of ``K_i`` over ``K_j`` is built in O(n + |f_j - w_j|): a
blackboard push-off of ``K_j`` on the band side (each crossing of ``K_j`` with
another strand gains one crossing, each self-crossing three), ``f_j - w_j``
full twists between ``K_j`` and the push-off so their linking number is the
framing ``f_j``, and a band joining the push-off to ``K_i``.  The new framing
is ``f_i + f_j + 2·ε·lk(K_i, K_j)`` with ``ε = ±1`` for addition/subtraction.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Iterable, Mapping, NamedTuple, Optional, Sequence, Union

import numpy as np

from .core import ISOTOPY_KINDS, IllegalMoveError, LinkDiagram, Move
from .core.ports import TRIVIAL, next_ccw, prev_ccw, strand_index

__all__ = [
    "FramedLink",
    "KirbyKind",
    "KirbyMove",
    "KIRBY_KINDS",
    "pushoff",
]


class KirbyKind(IntEnum):
    BLOW_UP = 10
    BLOW_DOWN = 11
    HANDLE_SLIDE = 12


KIRBY_KINDS = frozenset(KirbyKind)


class KirbyMove(NamedTuple):
    """``BLOW_UP(sign)``, ``BLOW_DOWN(a=label)`` or ``HANDLE_SLIDE(a=side, b=side)``."""

    kind: KirbyKind
    a: int = TRIVIAL
    b: int = TRIVIAL
    sign: int = 0

    def __repr__(self) -> str:
        if self.kind == KirbyKind.BLOW_UP:
            return f"BLOW_UP(sign={self.sign:+d})"
        if self.kind == KirbyKind.BLOW_DOWN:
            return f"BLOW_DOWN(label={self.a})"
        return f"HANDLE_SLIDE(slide={self.a}, over={self.b})"


AnyMove = Union[Move, KirbyMove]


class FramedLink:
    """A link diagram with an integer framing on each component (a Kirby diagram)."""

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

    @property
    def framings(self) -> dict[int, int]:
        return {lab: self._framing[lab] for lab in self.labels()}

    def linking_matrix(self) -> list[list[int]]:
        """Linking numbers off the diagonal, framings on it, in label order.

        This is the intersection form of the 4-manifold and a presentation
        matrix for H₁ of the surgered 3-manifold.
        """
        labels = self.labels()
        index = {lab: k for k, lab in enumerate(labels)}
        twice = [[0] * len(labels) for _ in labels]
        lab = self.diagram._label
        for c, s in enumerate(self.diagram._sign):
            i, j = index[lab[2 * c]], index[lab[2 * c + 1]]
            if i != j:
                twice[i][j] += s
                twice[j][i] += s
        Q = [[v // 2 for v in row] for row in twice]
        for k, label in enumerate(labels):
            Q[k][k] = self._framing[label]
        return Q

    def homology(self) -> list[int]:
        """H₁ of the surgered 3-manifold as elementary divisors (``0`` = a ``Z`` summand), like SnapPy."""
        divisors = _smith_divisors(self.linking_matrix())
        return sorted(d for d in divisors if d > 1) + [0] * divisors.count(0)

    def signature(self) -> int:
        """Signature of the linking matrix (the 4-manifold's intersection form)."""
        Q = self.linking_matrix()
        if not Q:
            return 0
        eig = np.linalg.eigvalsh(np.array(Q, dtype=float))
        return int((eig > 1e-9).sum() - (eig < -1e-9).sum())

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

    def is_legal(self, move: AnyMove, calculus: str = "3-manifold") -> bool:
        d = self.diagram
        if isinstance(move.kind, KirbyKind):
            if move.kind == KirbyKind.BLOW_UP:
                return calculus == "3-manifold" and move.sign in (1, -1)
            if move.kind == KirbyKind.BLOW_DOWN:
                return (
                    calculus == "3-manifold"
                    and move.a in d.trivial_labels
                    and self._framing.get(move.a) in (1, -1)
                )
            return self._slide_ok(move.a, move.b)
        return move.kind in ISOTOPY_KINDS and d.is_legal(move)

    def legal_moves(self, kinds: Iterable = ISOTOPY_KINDS | KIRBY_KINDS, calculus: str = "3-manifold") -> list:
        """Legal isotopy and Kirby moves; ``calculus="4-manifold"`` excludes blow-ups and blow-downs."""
        kinds = frozenset(kinds)
        d = self.diagram
        out: list = d.legal_moves(kinds & ISOTOPY_KINDS)
        if calculus == "3-manifold":
            if KirbyKind.BLOW_UP in kinds:
                out += [KirbyMove(KirbyKind.BLOW_UP, sign=s) for s in (1, -1)]
            if KirbyKind.BLOW_DOWN in kinds:
                out += [
                    KirbyMove(KirbyKind.BLOW_DOWN, a=lab)
                    for lab in d.trivial_labels
                    if self._framing[lab] in (1, -1)
                ]
        if KirbyKind.HANDLE_SLIDE in kinds:
            out += self._slides()
        return out

    def apply_inplace(self, move: AnyMove, check: bool = True, calculus: str = "3-manifold") -> Optional[AnyMove]:
        """Apply a move; return a move undoing it (``None`` for handle slides)."""
        if check and not self.is_legal(move, calculus):
            raise IllegalMoveError(f"{move!r} is not legal on {self!r}")
        d = self.diagram
        if not isinstance(move.kind, KirbyKind):
            return d.apply_inplace(move, check=False)
        if move.kind == KirbyKind.BLOW_UP:
            label = max(self._framing, default=-1) + 1
            d._trivial.append(label)
            self._framing[label] = move.sign
            return KirbyMove(KirbyKind.BLOW_DOWN, a=label)
        if move.kind == KirbyKind.BLOW_DOWN:
            d._trivial.remove(move.a)
            return KirbyMove(KirbyKind.BLOW_UP, sign=self._framing.pop(move.a))
        self._handle_slide(move.a, move.b)
        return None

    def apply(self, move: AnyMove, check: bool = True, calculus: str = "3-manifold") -> FramedLink:
        out = self.copy()
        out.apply_inplace(move, check, calculus)
        return out

    def random_move(self, rng, kinds: Iterable = ISOTOPY_KINDS | KIRBY_KINDS, calculus: str = "3-manifold"):
        return rng.choice(self.legal_moves(kinds, calculus))

    # ------------------------------------------------------------------
    # Handle slides
    # ------------------------------------------------------------------

    def _slide_ok(self, x: int, y: int) -> bool:
        d = self.diagram
        m = len(d._nbr)
        if not (0 <= x < m and 0 <= y < m):
            return False
        if d.label_of_port(x) == d.label_of_port(y):
            return False
        face = d._ensure_faces()
        return face[x] == face[y] or d._ensure_dcomp()[x >> 2] != d._ensure_dcomp()[y >> 2]

    def _slides(self) -> list[KirbyMove]:
        d = self.diagram
        lab = [d.label_of_port(p) for p in range(len(d._nbr))]
        HS = KirbyKind.HANDLE_SLIDE
        out = []
        for corners in d.faces():
            out += [KirbyMove(HS, x, y) for x in corners for y in corners if lab[x] != lab[y]]
        dcomp = d._ensure_dcomp()
        pieces: dict[int, list[int]] = {}
        for p in range(len(d._nbr)):
            pieces.setdefault(dcomp[p >> 2], []).append(p)
        groups = list(pieces.values())
        for i, gi in enumerate(groups):
            for j, gj in enumerate(groups):
                if i != j:
                    out += [KirbyMove(HS, x, y) for x in gi for y in gj]
        return out

    def _handle_slide(self, x: int, y: int) -> None:
        d = self.diagram
        ki, kj = d.label_of_port(x), d.label_of_port(y)
        writhe_j = twice_lk = 0
        for c, s in enumerate(d._sign):
            pair = {d._label[2 * c], d._label[2 * c + 1]}
            if pair == {kj}:
                writhe_j += s
            elif pair == {ki, kj}:
                twice_lk += s
        left = d.is_out(y)  # the push-off runs on the side of K_j that faces the band
        tmp = max(self._framing) + 1
        D, outer, push_outer = pushoff(d, kj, left, tmp)

        # Full twists so that lk(K_j, push-off) equals the framing of K_j.  They go on the
        # edge after y's edge, leaving the push-off edge beside y free for the band.
        twists = self._framing[kj] - writhe_j
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
        self.diagram = D
        self._framing[ki] += self._framing[kj] + eps * twice_lk

    def __repr__(self) -> str:
        return f"FramedLink(crossings={self.diagram.n_crossings}, framings={self.framings})"


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
