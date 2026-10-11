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
import ast
import html
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
    Move,
    MoveKind,
    determinant,
)
from knotml.core.ports import sigma  # noqa: E402


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
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--braid", type=int, nargs="+", help="braid word, e.g. 1 1 1")
    source.add_argument("--pd", help='PD code, e.g. "[[1,5,2,4],[3,1,4,6],[5,3,6,2]]"')
    source.add_argument("--knotinfo", help="KnotInfo/LinkInfo name, e.g. 8_20")
    source.add_argument("--snappy", help='SnapPy name or code, e.g. K12n123 or "DT: [(4,6,2)]"')
    parser.add_argument(
        "--framings",
        nargs="+",
        metavar="F",
        help="one per component in label order: an integer framing, or 'dot' for a dotted circle (1-handle). "
        "Integers only: 3-manifold surgery calculus (FramedLink); any 'dot' or --four: 4-manifold calculus "
        "(Handlebody)",
    )
    parser.add_argument("--four", action="store_true", help="use the 4-manifold calculus even without dotted circles")
    parser.add_argument("--closed", action="store_true", help="4-manifold closed up by 3- and 4-handles (2/3 moves)")
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

    if args.pd:
        d = LinkDiagram.from_pd(ast.literal_eval(args.pd))
    elif args.knotinfo:
        from knotml.interop import from_knotinfo

        d = from_knotinfo(args.knotinfo)
    elif args.snappy:
        import spherogram

        from knotml.interop import from_spherogram

        d = from_spherogram(spherogram.Link(args.snappy))
    else:
        d = LinkDiagram.from_braid(args.braid or [1, 1, 1])
    kirby = None
    if args.framings:
        try:
            comps = [None if v.lower() in ("dot", "d", ".") else int(v) for v in args.framings]
            if None in comps or args.four or args.closed:
                kirby = Handlebody(d, comps, closed=args.closed)
            else:
                kirby = FramedLink(d, comps)
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
        extra = f"  components {_format_components(components())}" if kirby else ""
        return f"{step_label}{d.n_crossings} crossings{extra}  PD {d.to_pd()}"

    if args.trace:
        print(describe("step 0: "))
    verify = make_verifier(obj) if args.verify else None
    failures = unknown = steps = 0
    history = [("start", d.copy(), set(), components(), "")]

    def apply(move) -> None:
        nonlocal d, failures, unknown, steps
        touched = _touched_crossings(d, move)
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
        except (ValueError, KeyError) as err:
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
        print("components:", _format_components(kirby.components), "| closed" if kirby.closed else "")
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
        write_html(args.html, history, calculus)
        print("wrote", args.html)
    if args.view:
        from knotml.visualization import view

        view(d)
    return 0


def parse_move(spec: str):
    """``"KIND a [b]"`` -> a Move or KirbyMove (anchors are ports; see --list)."""
    name, *rest = spec.replace(",", " ").split()
    name, vals = name.upper(), [int(v) for v in rest]
    expected = {"R1_ADD": "a sign", "R2_ADD": "over under", "HANDLE_SLIDE": "slide over",
                "BLOW_UP": "sign", "BLOW_DOWN": "label", "CANCEL_12": "dotted", "CREATE_12": "framing",
                "CANCEL_23": "label", "CREATE_23": ""}.get(name, "a")
    if len(vals) != len(expected.split()):
        raise ValueError(f"{name} takes {len(expected.split())} integer(s): {expected}")
    if name in MoveKind.__members__:
        kind = MoveKind[name]
        if kind == MoveKind.R1_ADD:
            return Move(kind, vals[0], sign=vals[1])
        return Move(kind, vals[0], vals[1]) if kind == MoveKind.R2_ADD else Move(kind, vals[0])
    kind = KirbyKind[name]
    if kind == KirbyKind.BLOW_UP:
        return KirbyMove(kind, sign=vals[0])
    if kind in (KirbyKind.BLOW_DOWN, KirbyKind.CANCEL_12, KirbyKind.CANCEL_23):
        return KirbyMove(kind, a=vals[0])
    if kind == KirbyKind.CREATE_12:
        return KirbyMove(kind, sign=vals[0])
    if kind == KirbyKind.CREATE_23:
        return KirbyMove(kind)
    return KirbyMove(kind, vals[0], vals[1])


def list_moves(obj, kind_name: str) -> None:
    """Print every legal move of one kind with a description of its anchors."""
    kirby = None if isinstance(obj, LinkDiagram) else obj
    d = kirby.diagram if kirby else obj
    framed = kirby  # framing() works on both FramedLink and Handlebody
    kinds = {**MoveKind.__members__, **KirbyKind.__members__}
    if kind_name not in kinds:
        raise SystemExit(f"unknown move kind {kind_name}; choose from {', '.join(kinds)}")
    kind = kinds[kind_name]
    moves = (kirby.legal_moves([kind]) if kirby else d.legal_moves([kind])) if kind in ALL_KINDS or kirby else []
    dotted = set(kirby.dotted_labels()) if isinstance(kirby, Handlebody) else set()
    face = d.face_of_port()

    def where(p: int) -> str:
        if p < 0:
            return f"crossingless component K{d.trivial_labels[-1 - p]} (anchor {p})"
        tail = d.edge_tail(p)
        side = "left" if p == tail else "right"
        lab = d.label_of_port(p)
        dot = " (dotted)" if lab in dotted else ""
        return f"K{lab}{dot} crossing {p >> 2} slot {p & 3}: {side} side of edge {tail}->{d.nbr[tail]}, face {face[p]}"

    print(f"{len(moves)} legal {kind_name} moves:")
    for m in moves:
        if kind == KirbyKind.BLOW_UP:
            text = f"add a split {m.sign:+d}-framed unknot"
        elif kind == KirbyKind.BLOW_DOWN:
            text = f"remove K{m.a} (split unknot, framing {framed.framing(m.a):+d})"
        elif kind == KirbyKind.CANCEL_12:
            k = kirby._cancelling_partner(m.a)
            text = f"erase dotted K{m.a} with the 2-handle K{k} (framing {kirby.framing(k):+d}) through it"
        elif kind == KirbyKind.CREATE_12:
            text = f"add a split dotted circle with a {m.sign:+d}-framed unknot through it"
        elif kind == KirbyKind.CANCEL_23:
            text = f"remove the split 0-framed unknot K{m.a} with a 3-handle"
        elif kind == KirbyKind.CREATE_23:
            text = "add a split 0-framed unknot with a 3-handle"
        elif kind == KirbyKind.HANDLE_SLIDE:
            adds = d.is_out(m.a) == d.is_out(m.b)  # same side of both edges: untwisted band adds
            text = f"slide [{where(m.a)}] over [{where(m.b)}] -> {'adds' if adds else 'subtracts'} the handle"
        elif kind == MoveKind.CROSSING_CHANGE:
            text = f"crossing {m.a} (K{d.label_of_port(4 * m.a)} under K{d.label_of_port(4 * m.a + 1)})"
        elif kind == MoveKind.R2_ADD:
            text = f"push [{where(m.a)}] over [{where(m.b)}]"
        elif kind == MoveKind.R1_ADD:
            text = f"kink of sign {m.sign:+d} on [{where(m.a)}]"
        else:
            text = where(m.a)
        args = {KirbyKind.BLOW_UP: f"{m.sign}", KirbyKind.BLOW_DOWN: f"{m.a}", MoveKind.R1_ADD: f"{m.a} {m.sign}",
                KirbyKind.CANCEL_12: f"{m.a}", KirbyKind.CREATE_12: f"{m.sign}", KirbyKind.CANCEL_23: f"{m.a}",
                KirbyKind.CREATE_23: ""}
        spec = f"{kind_name} {args.get(kind, f'{m.a} {m.b}' if kind in (MoveKind.R2_ADD, KirbyKind.HANDLE_SLIDE) else m.a)}".strip()
        print(f'  --move "{spec}"  # {text}')


def _touched_crossings(d: LinkDiagram, move) -> set:
    """Crossings an R3 move rearranges (they keep their indices); other moves add or delete crossings."""
    if move.kind != MoveKind.R3:
        return set()
    x = move.a
    y = sigma(d.nbr, x)
    return {x >> 2, y >> 2, sigma(d.nbr, y) >> 2}


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


def _format_components(comps) -> str:
    return "{" + ", ".join(f"K{lab}: {'dot' if f is None else f'{f:+d}'}" for lab, f in comps.items()) + "}"


def _signed(n: int) -> str:
    return f"{n:+d}".replace("-", "\N{MINUS SIGN}")


_PAGE_TOKENS = {
    "light": "--page:#f9f9f7;--ink:#0b0b0b;--ink-2:#52514e;--border:rgba(11,11,11,0.10);color-scheme:light",
    "dark": "--page:#0d0d0d;--ink:#f0efec;--ink-2:#c3c2b7;--border:rgba(255,255,255,0.10);color-scheme:dark",
}


def write_html(path: str, history: list, calculus=None) -> None:
    """A self-contained page with one drawing per step, a slider and arrow-key navigation."""
    from knotml.visualization import svg_css, to_svg

    frames = []
    for step, (label, diagram, marked, comps, status) in enumerate(history):
        caption = f"step {step}: {html.escape(label)} &rarr; {diagram.n_crossings} crossings"
        dotted, notes = [], {}
        if comps is None:
            caption += f", determinant {determinant(diagram)}"
        else:
            dotted = [lab for lab, f in comps.items() if f is None]
            notes = {lab: "1-handle" if f is None else f"framing {_signed(f)}" for lab, f in comps.items()}
            if calculus == "4":
                h = Handlebody(diagram, comps)
                caption += (
                    f"<br>&chi; = {h.euler_characteristic()} &middot; H<sub>1</sub>(X) = {h.homology()}"
                    f" &middot; &sigma; = {h.signature()} &middot; H<sub>1</sub>(&part;X) = {h.boundary().homology()}"
                )
            else:
                caption += f"<br>H<sub>1</sub> = {FramedLink(diagram, comps).homology()}"
        if status:
            caption += f"<br><small>verified: {html.escape(status)}</small>"
        svg = to_svg(diagram, size=520, labels=True, highlight=marked, dotted=dotted, notes=notes, style=False)
        frames.append(
            f'<figure class="frame" hidden><figcaption>{caption}</figcaption>{svg}'
            f"<details><summary>PD code</summary><code>{html.escape(str(diagram.to_pd()))}</code></details></figure>"
        )
    light, dark = _PAGE_TOKENS["light"], _PAGE_TOKENS["dark"]
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Knot move sequence</title>
<style>
  :root {{ {light} }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme=light]) {{ {dark} }} }}
  :root[data-theme=dark] {{ {dark} }}
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 24px auto; max-width: 600px;
         padding: 0 16px; background: var(--page); color: var(--ink); }}
  .controls {{ display: flex; gap: 8px; align-items: center; margin: 12px 0; }}
  input[type=range] {{ flex: 1; }}
  figure {{ margin: 0; }} figcaption {{ font-weight: 600; margin-bottom: 8px; }}
  svg {{ width: 100%; height: auto; border: 1px solid var(--border); border-radius: 6px; }}
  code {{ font-size: 12px; word-break: break-all; }} .hint, small, summary {{ color: var(--ink-2); }}
  .hint {{ font-size: 13px; }}
{svg_css()}
</style></head><body>
<h1>Knot move sequence</h1>
<p class="hint">Drawn with spherogram's orthogonal layout. Each component keeps its colour and line pattern
across steps (legend under the drawing); circled crossings were created or rearranged by the move.
Each step is laid out from scratch, so pictures can jump between steps. Use &larr; / &rarr; to step.</p>
<div class="controls"><button id="prev" aria-label="Previous step">&larr;</button>
<input id="slider" type="range" min="0" max="{len(frames) - 1}" value="0" aria-label="Step">
<button id="next" aria-label="Next step">&rarr;</button><button id="theme">Light / dark</button></div>
{"".join(frames)}
<script>
  const frames = document.querySelectorAll('.frame'), slider = document.getElementById('slider');
  function show(i) {{ i = Math.max(0, Math.min(frames.length - 1, i)); frames.forEach((f, k) => f.hidden = k !== i); slider.value = i; }}
  slider.oninput = () => show(+slider.value);
  document.getElementById('prev').onclick = () => show(+slider.value - 1);
  document.getElementById('next').onclick = () => show(+slider.value + 1);
  document.onkeydown = e => {{ if (e.key === 'ArrowLeft') show(+slider.value - 1); if (e.key === 'ArrowRight') show(+slider.value + 1); }};
  document.getElementById('theme').onclick = () => {{
    const root = document.documentElement, t = root.dataset.theme;
    const dark = t ? t === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
  }};
  show(0);
</script></body></html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(page)


if __name__ == "__main__":
    raise SystemExit(main())
