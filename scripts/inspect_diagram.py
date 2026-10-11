"""Inspect a knot diagram: PD code, faces, legal moves, and optionally timings.

Usage:
    python scripts/inspect_diagram.py --braid 1 1 1
    python scripts/inspect_diagram.py --pd "[[1,5,2,4],[3,1,4,6],[5,3,6,2]]"
    python scripts/inspect_diagram.py --knotinfo 8_20          # needs database_knotinfo
    python scripts/inspect_diagram.py --snappy K12n123         # SnapPy table name or "DT: ..."; needs snappy
    python scripts/inspect_diagram.py --random 100 --bench     # 100 random moves, then time it
    python scripts/inspect_diagram.py --random 30 --trace      # print every move and PD code on the way
    python scripts/inspect_diagram.py --random 30 --html walk.html   # step through drawings in a browser
    python scripts/inspect_diagram.py --view                   # open the final diagram in PLink
    python scripts/inspect_diagram.py --knotinfo "L5a1{0}" --framings 5 2 --random 5 --trace   # Kirby moves
    python scripts/inspect_diagram.py --knotinfo "L5a1{0}" --framings 5 2 --random 8 --verify --html kirby.html
    python scripts/inspect_diagram.py --knotinfo "L5a1{0}" --framings 5 2 --list HANDLE_SLIDE   # see the options
    python scripts/inspect_diagram.py --knotinfo "L5a1{0}" --framings 5 2 --move "HANDLE_SLIDE 0 10"   # apply one
    python scripts/inspect_diagram.py --knotinfo "L7a6{0}" --framings dot -2 --list CANCEL_12   # 4-manifold
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from knotml import (  # noqa: E402
    ALL_KINDS,
    ISOTOPY_KINDS,
    HANDLEBODY_KINDS,
    SURGERY_KINDS,
    FramedLink,
    Handlebody,
    KirbyKind,
    KirbyMove,
    LinkDiagram,
    MoveKind,
    determinant,
)
from knotml.cli import add_diagram_arguments, diagram_from_args, kirby_from_args  # noqa: E402
from knotml.notation import describe_move, format_components, move_spec, parse_move, touched_crossings  # noqa: E402


def timed(fn, repeat: int = 5) -> float:
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def bench(d: LinkDiagram) -> None:
    def fresh():
        return d.copy()  # drop caches so face/component rebuilds are included

    print(f"legal_moves (all kinds) : {1e3 * timed(lambda: fresh().legal_moves(ALL_KINDS)):8.2f} ms")
    nonr2 = ISOTOPY_KINDS - {MoveKind.R2_ADD}
    print(f"legal_moves (no R2_ADD) : {1e3 * timed(lambda: fresh().legal_moves(nonr2)):8.2f} ms")
    print(f"move_masks              : {1e3 * timed(lambda: fresh().move_masks()):8.2f} ms")
    print(f"canonical_key           : {1e3 * timed(lambda: d.canonical_key()):8.2f} ms")
    move = d.legal_moves([MoveKind.R1_ADD])[0]
    print(f"apply_inplace + undo    : {1e6 * timed(lambda: d.apply_inplace(d.apply_inplace(move))):8.2f} us")
    from knotml.interop import to_regina

    try:
        R = to_regina(d)
    except ImportError:  # Regina is optional
        return
    arcs = [R.crossing(c).lower() for c in range(R.size())] + [R.crossing(c).upper() for c in range(R.size())]

    def regina_r2_add():
        return sum(R.hasR2(a, s, b, t) for a in arcs for b in arcs if a != b for s in (0, 1) for t in (0, 1))

    print(f"Regina hasR2 over all arc pairs (R2_ADD enumeration): {1e3 * timed(regina_r2_add, 1):8.2f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_diagram_arguments(parser)
    parser.add_argument(
        "--move",
        action="append",
        default=[],
        metavar="SPEC",
        help='apply a specific move, e.g. "R1_ADD 8 -1", "R2_ADD 3 10", "R3 9", "BLOW_UP -1", '
        '"BLOW_DOWN 2", "HANDLE_SLIDE 8 6"; repeatable, applied in order before --random',
    )
    parser.add_argument(
        "--list",
        metavar="KIND",
        help="after all moves, list the legal moves of this kind (e.g. HANDLE_SLIDE, R2_ADD, BLOW_UP) "
        "with what each anchor refers to",
    )
    parser.add_argument("--random", type=int, default=0, help="apply this many random moves (after --move)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-crossings",
        type=int,
        help="once the diagram has this many crossings, random moves only remove or keep crossings",
    )
    parser.add_argument("--trace", action="store_true", help="print each random move and the PD code after it")
    parser.add_argument("--html", help="write a page stepping through drawings of every step (needs snappy)")
    parser.add_argument("--view", action="store_true", help="open the final diagram in PLink (needs snappy)")
    parser.add_argument("--bench", action="store_true", help="time the core operations")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="after each random move, check the diagram and its invariants "
        "(determinant; for framed links H1 and, with snappy, the surgered manifold)",
    )
    args = parser.parse_args()

    d = diagram_from_args(args)
    try:
        kirby = kirby_from_args(args, d)
    except ValueError as err:
        parser.error(f"--framings: {err}")
    obj = kirby or d
    calculus = "4" if isinstance(kirby, Handlebody) else "3" if kirby else None
    kinds = ISOTOPY_KINDS | {"4": HANDLEBODY_KINDS, "3": SURGERY_KINDS, None: frozenset()}[calculus]
    rng = random.Random(args.seed)

    def components():
        if isinstance(kirby, Handlebody):
            return kirby.components
        return kirby.framings if kirby else None

    def describe(step_label: str) -> str:
        extra = f"  components {format_components(components())}" if kirby else ""
        return f"{step_label}{d.n_crossings} crossings{extra}  PD {d.to_pd()}"

    if args.trace:
        print(describe("step 0: "))
    verify = make_verifier(obj) if args.verify else None
    failures = unknown = steps = 0
    history = [("start", d.copy(), set(), components(), "")]

    def apply(move) -> None:
        nonlocal d, failures, unknown, steps
        touched = touched_crossings(d, move)
        obj.apply_inplace(move)  # raises IllegalMoveError for an illegal move
        d = kirby.diagram if kirby else d  # a handle slide builds a new diagram
        steps += 1
        if args.trace:
            print(describe(f"step {steps}: {move!r} -> "))
        status = ""
        if verify:
            ok, status = verify(obj)
            failures += ok is False
            unknown += ok is None
            print(f"verify step {steps}: {'ok' if ok else 'inconclusive' if ok is None else 'FAILED'} ({status})")
        if args.html:
            new = set(range(history[-1][1].n_crossings, d.n_crossings))
            history.append((repr(move), d.copy(), touched | new, components(), status))

    for spec in args.move:
        try:
            move = parse_move(spec)
        except ValueError as err:
            parser.error(f"--move {spec!r}: {err}")
        if isinstance(move, KirbyMove) and kirby is None:
            parser.error(f"--move {spec!r}: Kirby moves need --framings")
        if not obj.is_legal(move):
            parser.error(f"--move {spec!r}: {move!r} is not legal here (see --list {move.kind.name})")
        apply(move)
    shrinking = {MoveKind.R1_REMOVE, MoveKind.R2_REMOVE, MoveKind.R3, KirbyKind.BLOW_DOWN}
    for _ in range(args.random):
        if args.max_crossings is not None and d.n_crossings >= args.max_crossings:
            options = obj.legal_moves(kinds & shrinking)
            if not options:
                print(f"stopping: {d.n_crossings} crossings and no move that keeps the size")
                break
            apply(rng.choice(options))
        else:
            apply(obj.random_move(rng, kinds))
    if verify:
        print(f"verified {steps} steps: {failures} failures, {unknown} inconclusive")

    print(d)
    print("PD:", d.to_pd() if d.n_crossings <= 30 else f"<{d.n_crossings} crossings>")
    print("faces:", d.n_faces, "| writhe:", d.writhe(), "| determinant:", determinant(d))
    counts = Counter(m.kind.name for m in d.legal_moves(ALL_KINDS))
    print("legal moves:", dict(counts))
    if isinstance(kirby, FramedLink):
        print("framings:", kirby.framings, "| linking matrix:", kirby.linking_matrix())
        print("H1 of surgery:", kirby.homology(), "| signature:", kirby.signature())
        print("legal Kirby moves:", dict(Counter(m.kind.name for m in kirby.legal_moves(SURGERY_KINDS))))
    elif isinstance(kirby, Handlebody):
        print("components:", format_components(kirby.components), "| closed" if kirby.closed else "")
        print(f"1-handles: {kirby.n_1handles} | 2-handles: {kirby.n_2handles} | euler characteristic: "
              f"{kirby.euler_characteristic()} | H1(X): {kirby.homology()}")
        print("intersection form:", kirby.intersection_form(), "| signature:", kirby.signature(),
              "| H1 of boundary:", kirby.boundary().homology())
        print("legal Kirby moves:", dict(Counter(m.kind.name for m in kirby.legal_moves(HANDLEBODY_KINDS))))
    if args.list:
        list_moves(obj, args.list.upper())
    if args.bench:
        bench(d)
    if args.html:
        from knotml.visualization import write_steps_page

        write_steps_page(args.html, history, calculus)
        print("wrote", args.html)
    if args.view:
        from knotml.visualization import view

        view(d)
    return 0


def list_moves(obj, kind_name: str) -> None:
    """Print every legal move of one kind with a description of its anchors."""
    kirby = None if isinstance(obj, LinkDiagram) else obj
    kinds = {**MoveKind.__members__, **KirbyKind.__members__}
    if kind_name not in kinds:
        raise SystemExit(f"unknown move kind {kind_name}; choose from {', '.join(kinds)}")
    kind = kinds[kind_name]
    moves = obj.legal_moves([kind]) if kind in ALL_KINDS or kirby else []
    print(f"{len(moves)} legal {kind_name} moves:")
    for m in moves:
        print(f'  --move "{move_spec(m)}"  # {describe_move(obj, m)}')


def make_verifier(start):
    """A check comparing a diagram, framed link or handlebody with ``start``: returns ``(ok, description)``."""
    if isinstance(start, Handlebody):
        invariants = start.invariants()
        manifold = None
        if not start.closed:  # 2/3 moves change the boundary, so it is only compared without them
            try:
                manifold = start.to_snappy_boundary()
                hyperbolic = manifold.solution_type() == "all tetrahedra positively oriented"
            except Exception:  # SnapPy missing, split, or S³ boundary: invariants only
                manifold = None

        def verify_handlebody(h):
            h.diagram.check()
            Handlebody(h.diagram, h.components, closed=h.closed)  # dotted circles still have linking number 0
            now = h.invariants()
            if now != invariants:
                return False, f"invariants changed: {now}"
            notes = [f"H1(X) {now['H1']}, signature {now['signature']}"]
            if manifold is not None:
                try:
                    other = h.to_snappy_boundary()
                except ValueError:
                    return None, notes[0] + ", boundary split (not checked in SnapPy)"
                if other.homology() != manifold.homology():
                    return False, notes[0] + ", SnapPy boundary homology differs"
                if hyperbolic:
                    verdict, note = _same_hyperbolic(manifold, other)
                    notes.append("boundary " + note)
                    return (False if verdict is False else True if verdict else None), ", ".join(notes)
                notes.append("boundary homology agrees in SnapPy")
            return True, ", ".join(notes)

        return verify_handlebody
    if isinstance(start, FramedLink):
        hom = start.homology()
        try:
            manifold = start.to_snappy()
            hyperbolic = manifold.solution_type() == "all tetrahedra positively oriented"
        except Exception:  # SnapPy missing or the start is split: homology only
            manifold = None

        def verify(framed):
            framed.diagram.check()
            ok = framed.homology() == hom
            notes = [f"H1 {framed.homology()}"]
            if manifold is not None:
                other = framed.to_snappy()
                if other.homology() != manifold.homology():
                    return False, f"H1 {framed.homology()}, SnapPy homology {other.homology()} differs"
                if not hyperbolic:
                    notes.append("SnapPy homology agrees")
                else:
                    verdict, note = _same_hyperbolic(manifold, other)
                    notes.append(note)
                    if verdict is False:
                        ok = False
                    elif verdict is None:
                        ok = ok and None  # inconclusive unless something else failed
            return ok, ", ".join(notes)

        return verify
    det = determinant(start)

    def verify_diagram(d):
        d.check()
        return determinant(d) == det, f"determinant {determinant(d)}"

    return verify_diagram


def _same_hyperbolic(manifold, other):
    """``(True | False | None, note)``: isometric, provably not, or SnapPy cannot tell.

    Large triangulations can defeat SnapPy's solver (degenerate tetrahedra), so
    retry on re-randomized triangulations, and trust "not isometric" only when
    both hyperbolic structures were found.
    """
    geometric = "all tetrahedra positively oriented"
    for _ in range(4):
        try:
            if manifold.is_isometric_to(other):
                return True, "isometric in SnapPy"
            if other.solution_type() == geometric:
                return False, "SnapPy: not isometric"
        except RuntimeError:
            pass
        other.randomize()
    return None, f"inconclusive: SnapPy found no hyperbolic structure ({other.solution_type()})"


if __name__ == "__main__":
    raise SystemExit(main())
