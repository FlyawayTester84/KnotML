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

from knotml import ALL_KINDS, ISOTOPY_KINDS, LinkDiagram, MoveKind, determinant  # noqa: E402
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
    parser.add_argument("--random", type=int, default=0, help="apply this many random isotopy moves first")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace", action="store_true", help="print each random move and the PD code after it")
    parser.add_argument("--html", help="write a page stepping through drawings of every step (needs snappy)")
    parser.add_argument("--view", action="store_true", help="open the final diagram in PLink (needs snappy)")
    parser.add_argument("--bench", action="store_true", help="time the core operations")
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
    rng = random.Random(args.seed)
    if args.trace:
        print(f"step 0: {d.n_crossings} crossings  PD {d.to_pd()}")
    history = [("start", d.copy(), set())]
    for step in range(1, args.random + 1):
        move = d.random_move(rng, ISOTOPY_KINDS)
        touched = _touched_crossings(d, move)
        d.apply_inplace(move)
        if args.trace:
            print(f"step {step}: {move!r} -> {d.n_crossings} crossings  PD {d.to_pd()}")
        if args.html:
            history.append((repr(move), d.copy(), touched | set(range(history[-1][1].n_crossings, d.n_crossings))))

    print(d)
    print("PD:", d.to_pd() if d.n_crossings <= 30 else f"<{d.n_crossings} crossings>")
    print("faces:", d.n_faces, "| writhe:", d.writhe(), "| determinant:", determinant(d))
    counts = Counter(m.kind.name for m in d.legal_moves(ALL_KINDS))
    print("legal moves:", dict(counts))
    if args.bench:
        bench(d)
    if args.html:
        write_html(args.html, history)
        print("wrote", args.html)
    if args.view:
        from knotml.visualization import view

        view(d)
    return 0


def _touched_crossings(d: LinkDiagram, move) -> set:
    """Crossings an R3 move rearranges (they keep their indices); other moves add or delete crossings."""
    if move.kind != MoveKind.R3:
        return set()
    x = move.a
    y = sigma(d.nbr, x)
    return {x >> 2, y >> 2, sigma(d.nbr, y) >> 2}


def write_html(path: str, history: list) -> None:
    """A self-contained page with one drawing per step, a slider and arrow-key navigation."""
    from knotml.visualization import to_svg

    frames = []
    for step, (label, diagram, marked) in enumerate(history):
        caption = (
            f"step {step}: {html.escape(label)} &rarr; {diagram.n_crossings} crossings, "
            f"determinant {determinant(diagram)}"
        )
        frames.append(
            f'<figure class="frame" hidden><figcaption>{caption}</figcaption>'
            f"{to_svg(diagram, size=520, labels=True, highlight=marked)}"
            f"<details><summary>PD code</summary><code>{html.escape(str(diagram.to_pd()))}</code></details></figure>"
        )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Knot move sequence</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px auto; max-width: 600px; padding: 0 16px; color: #222; }}
  .controls {{ display: flex; gap: 8px; align-items: center; margin: 12px 0; }}
  input[type=range] {{ flex: 1; }}
  figure {{ margin: 0; }} figcaption {{ font-weight: 600; margin-bottom: 8px; }}
  svg {{ width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px; }}
  code {{ font-size: 12px; word-break: break-all; }} .hint {{ color: #777; font-size: 13px; }}
</style></head><body>
<h1>Knot move sequence</h1>
<p class="hint">Drawn with spherogram's orthogonal layout; circled crossings were created or rearranged by the move.
Each step is laid out from scratch, so pictures can jump between steps. Use &larr; / &rarr; to step.</p>
<div class="controls"><button id="prev">&larr;</button><input id="slider" type="range" min="0" max="{len(frames) - 1}" value="0">
<button id="next">&rarr;</button></div>
{"".join(frames)}
<script>
  const frames = document.querySelectorAll('.frame'), slider = document.getElementById('slider');
  function show(i) {{ i = Math.max(0, Math.min(frames.length - 1, i)); frames.forEach((f, k) => f.hidden = k !== i); slider.value = i; }}
  slider.oninput = () => show(+slider.value);
  document.getElementById('prev').onclick = () => show(+slider.value - 1);
  document.getElementById('next').onclick = () => show(+slider.value + 1);
  document.onkeydown = e => {{ if (e.key === 'ArrowLeft') show(+slider.value - 1); if (e.key === 'ArrowRight') show(+slider.value + 1); }};
  show(0);
</script></body></html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(page)


if __name__ == "__main__":
    raise SystemExit(main())
