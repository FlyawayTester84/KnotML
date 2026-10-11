"""Text notation for moves: ``"KIND a b"`` specs, and what a move does in words.

A spec is the kind's name followed by its integer arguments; anchors are ports
(``4c + k``), negative anchors name crossingless components, and labels name
components:

====================  ===============  ==================================
kind                  arguments        example
====================  ===============  ==================================
R1_REMOVE, R3         port             ``R3 9``
R2_REMOVE             port             ``R2_REMOVE 4``
R1_ADD                side, sign       ``R1_ADD 8 -1``
R2_ADD                over, under      ``R2_ADD 3 10``
CROSSING_CHANGE       crossing         ``CROSSING_CHANGE 2``
HANDLE_SLIDE          slide, over      ``HANDLE_SLIDE 14 26``
BLOW_UP, CREATE_12    sign / framing   ``BLOW_UP -1``, ``CREATE_12 0``
BLOW_DOWN, CANCEL_12  label            ``CANCEL_12 1``
CANCEL_23             label            ``CANCEL_23 2``
CREATE_23             (none)           ``CREATE_23``
====================  ===============  ==================================

:func:`parse_move` and :func:`move_spec` are inverse to each other.
"""

from __future__ import annotations

from typing import Mapping, Optional, Union

from .core import LinkDiagram, Move, MoveKind
from .core.ports import sigma
from .handlebody import Handlebody
from .kirby import AnyMove, FramedLink, KirbyKind, KirbyMove

__all__ = [
    "parse_move",
    "move_spec",
    "describe_move",
    "move_crossings",
    "move_labels",
    "touched_crossings",
    "format_components",
    "signed",
]

Obj = Union[LinkDiagram, FramedLink, Handlebody]

_ARGS = {
    "R1_ADD": "side sign",
    "R2_ADD": "over under",
    "CROSSING_CHANGE": "crossing",
    "HANDLE_SLIDE": "slide over",
    "BLOW_UP": "sign",
    "BLOW_DOWN": "label",
    "CANCEL_12": "dotted",
    "CREATE_12": "framing",
    "CANCEL_23": "label",
    "CREATE_23": "",
}


def parse_move(spec: str) -> AnyMove:
    """``"KIND a [b]"`` -> a :class:`Move` or :class:`KirbyMove` (see the module docstring).

    Raises ``ValueError`` for an unknown kind or the wrong number of integers.
    """
    name, *rest = spec.replace(",", " ").split() or [""]
    name = name.upper()
    if name not in MoveKind.__members__ and name not in KirbyKind.__members__:
        kinds = ", ".join([*MoveKind.__members__, *KirbyKind.__members__])
        raise ValueError(f"unknown move kind {name!r}; choose from {kinds}")
    try:
        vals = [int(v) for v in rest]
    except ValueError:
        raise ValueError(f"{name} takes integers, got {' '.join(rest)!r}") from None
    expected = _ARGS.get(name, "port").split()
    if len(vals) != len(expected):
        raise ValueError(f"{name} takes {len(expected)} integer(s): {' '.join(expected) or 'none'}")
    if name in MoveKind.__members__:
        kind = MoveKind[name]
        if kind == MoveKind.R1_ADD:
            return Move(kind, vals[0], sign=vals[1])
        return Move(kind, vals[0], vals[1]) if kind == MoveKind.R2_ADD else Move(kind, vals[0])
    kind = KirbyKind[name]
    if kind in (KirbyKind.BLOW_UP, KirbyKind.CREATE_12):
        return KirbyMove(kind, sign=vals[0])
    if kind == KirbyKind.CREATE_23:
        return KirbyMove(kind)
    if kind == KirbyKind.HANDLE_SLIDE:
        return KirbyMove(kind, vals[0], vals[1])
    return KirbyMove(kind, a=vals[0])


def move_spec(move: AnyMove) -> str:
    """The spec :func:`parse_move` reads back as ``move``."""
    kind = move.kind
    if kind in (KirbyKind.BLOW_UP, KirbyKind.CREATE_12):
        args = [move.sign]
    elif kind == KirbyKind.CREATE_23:
        args = []
    elif kind == MoveKind.R1_ADD:
        args = [move.a, move.sign]
    elif kind in (MoveKind.R2_ADD, KirbyKind.HANDLE_SLIDE):
        args = [move.a, move.b]
    else:
        args = [move.a]
    return " ".join([kind.name, *map(str, args)])


def _parts(obj: Obj):
    """``(diagram, kirby object or None)``."""
    return (obj, None) if isinstance(obj, LinkDiagram) else (obj.diagram, obj)


def describe_move(obj: Obj, move: AnyMove) -> str:
    """What a (legal) move does, in words, with its anchors located on the diagram of ``obj``."""
    d, kirby = _parts(obj)
    kind = move.kind
    dotted = set(kirby.dotted_labels()) if isinstance(kirby, Handlebody) else set()
    face = d.face_of_port() if d.n_crossings else []

    def where(p: int) -> str:
        if p < 0:
            return f"crossingless component K{d.trivial_labels[-1 - p]} (anchor {p})"
        tail = d.edge_tail(p)
        side = "left" if p == tail else "right"
        lab = d.label_of_port(p)
        dot = " (dotted)" if lab in dotted else ""
        return f"K{lab}{dot} crossing {p >> 2} slot {p & 3}: {side} side of edge {tail}->{d.nbr[tail]}, face {face[p]}"

    if kind == KirbyKind.BLOW_UP:
        return f"add a split {move.sign:+d}-framed unknot"
    if kind == KirbyKind.BLOW_DOWN:
        return f"remove K{move.a} (split unknot, framing {kirby.framing(move.a):+d})"
    if kind == KirbyKind.CANCEL_12:
        k = kirby._cancelling_partner(move.a)
        return f"erase dotted K{move.a} with the 2-handle K{k} (framing {kirby.framing(k):+d}) through it"
    if kind == KirbyKind.CREATE_12:
        return f"add a split dotted circle with a {move.sign:+d}-framed unknot through it"
    if kind == KirbyKind.CANCEL_23:
        return f"remove the split 0-framed unknot K{move.a} with a 3-handle"
    if kind == KirbyKind.CREATE_23:
        return "add a split 0-framed unknot with a 3-handle"
    if kind == KirbyKind.HANDLE_SLIDE:
        adds = d.is_out(move.a) == d.is_out(move.b)  # same side of both edges: the untwisted band adds
        return f"slide [{where(move.a)}] over [{where(move.b)}] -> {'adds' if adds else 'subtracts'} the handle"
    if kind == MoveKind.CROSSING_CHANGE:
        return f"crossing {move.a} (K{d.label_of_port(4 * move.a)} under K{d.label_of_port(4 * move.a + 1)})"
    if kind == MoveKind.R2_ADD:
        return f"push [{where(move.a)}] over [{where(move.b)}]"
    if kind == MoveKind.R1_ADD:
        return f"kink of sign {move.sign:+d} on [{where(move.a)}]"
    return where(move.a)


def touched_crossings(d: LinkDiagram, move: AnyMove) -> set:
    """Crossings an R3 move rearranges (they keep their indices); other moves add or delete crossings."""
    if move.kind != MoveKind.R3:
        return set()
    x = move.a
    y = sigma(d.nbr, x)
    return {x >> 2, y >> 2, sigma(d.nbr, y) >> 2}


def move_crossings(d: LinkDiagram, move: AnyMove) -> set:
    """The crossings a move is anchored at or acts on, for pointing at it on a drawing."""
    kind = move.kind
    if kind == MoveKind.R3:
        return touched_crossings(d, move)
    if kind == MoveKind.R2_REMOVE:
        return {move.a >> 2, sigma(d.nbr, move.a) >> 2}
    if kind == MoveKind.CROSSING_CHANGE:
        return {move.a}
    if kind in (MoveKind.R1_REMOVE, MoveKind.R1_ADD, MoveKind.R2_ADD, KirbyKind.HANDLE_SLIDE):
        return {p >> 2 for p in _anchors(move) if p >= 0}
    return set()


def _anchors(move: AnyMove) -> tuple:
    """The anchor fields a move uses (``b`` defaults to ``TRIVIAL`` when unused)."""
    return (move.a, move.b) if move.kind in (MoveKind.R2_ADD, KirbyKind.HANDLE_SLIDE) else (move.a,)


def move_labels(obj: Obj, move: AnyMove) -> set:
    """The components a move removes, slides or is anchored on (crossingless ones included)."""
    d, kirby = _parts(obj)
    kind = move.kind
    if kind in (KirbyKind.BLOW_DOWN, KirbyKind.CANCEL_23):
        return {move.a}
    if kind == KirbyKind.CANCEL_12:
        return {move.a, kirby._cancelling_partner(move.a)}
    if kind == KirbyKind.HANDLE_SLIDE:
        return {d.label_of_port(move.a), d.label_of_port(move.b)}
    if kind in (MoveKind.R1_ADD, MoveKind.R2_ADD):
        return {d.trivial_labels[-1 - p] for p in _anchors(move) if p < 0 and -1 - p < d.n_unknots}
    return set()


def format_components(comps: Optional[Mapping[int, Optional[int]]]) -> str:
    """``{K0: dot, K1: -2}`` for a ``{label: framing or None (dotted)}`` map."""
    if comps is None:
        return "{}"
    return "{" + ", ".join(f"K{lab}: {'dot' if f is None else f'{f:+d}'}" for lab, f in comps.items()) + "}"


def signed(n: int) -> str:
    """``+2`` / ``−2`` with a typographic minus, for display."""
    return f"{n:+d}".replace("-", "\N{MINUS SIGN}")
