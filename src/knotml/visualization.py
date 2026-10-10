"""Drawing link diagrams with spherogram's orthogonal layout (needs the ``oracles`` extra).

* :func:`to_svg` renders a diagram to an SVG string without opening a window.
* :func:`view` opens SnapPy's interactive PLink viewer.

Crossing ``c`` of the picture is crossing ``c`` of the :class:`LinkDiagram`, so
move anchors (port ``4c + k``) can be located on it.  Each call lays the
diagram out from scratch, so consecutive diagrams in a move sequence are not
drawn continuously.
"""

from __future__ import annotations

import math
from html import escape
from itertools import pairwise
from typing import Iterable

from .core import LinkDiagram
from .interop import to_spherogram

__all__ = ["layout", "to_svg", "view", "visualize_knot_diagram"]

_COLORS = ["#1f4e9c", "#c0392b", "#1e8449", "#8e44ad", "#d35400", "#117a8b", "#7d6608"]


def layout(d: LinkDiagram):
    """Spherogram's orthogonal layout of ``d``.

    Returns ``(components, crossings)``: per component, its strands as point
    lists with gaps at undercrossings; and ``(x, y, crossing index)`` triples.
    """
    import plink

    manager = to_spherogram(d).view(viewer=plink.LinkManager())
    components = [polylines for polylines, _ in manager.polylines(break_at_overcrossings=False)]
    crossings = [(c.x, c.y, int(c.label)) for c in manager.Crossings]  # PLink labels are strings
    return components, crossings


def to_svg(
    d: LinkDiagram,
    size: int = 420,
    labels: bool = False,
    title: str = "",
    highlight: Iterable[int] = (),
) -> str:
    """An SVG drawing of ``d``: one colour per component, an arrow showing each orientation.

    Crossings listed in ``highlight`` are circled.
    """
    components, crossings = layout(d) if d.n_crossings else ([], [])
    points = [p for comp in components for line in comp for p in line]
    if points:
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        x0, y0, span = min(xs), min(ys), max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    else:
        x0, y0, span = 0.0, 0.0, 1.0
    pad = 24
    scale = (size - 2 * pad) / span

    def xy(p):
        return pad + (p[0] - x0) * scale, pad + (p[1] - y0) * scale

    width = max(1.5, min(4.0, 0.25 * scale))
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}">',
        f'<rect width="{size}" height="{size}" fill="white"/>',
    ]
    if title:
        out.append(f"<title>{escape(title)}</title>")
    for i, comp in enumerate(components):
        color = _COLORS[i % len(_COLORS)]
        for line in comp:
            pts = " ".join("%.1f,%.1f" % xy(p) for p in line)
            out.append(
                f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width:.1f}" '
                'stroke-linejoin="round" stroke-linecap="round"/>'
            )
        out.append(_arrow(comp, xy, color, 3 * width))
    for k in range(d.n_unknots):  # crossingless components, drawn as separate circles
        r = size / (4 + 2 * d.n_unknots)
        cx = size - pad - r - k * 2.5 * r if points else size / 2 + (k - (d.n_unknots - 1) / 2) * 2.5 * r
        cy = size - pad - r if points else size / 2
        color = _COLORS[(len(components) + k) % len(_COLORS)]
        out.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none" stroke="{color}" stroke-width="{width:.1f}"/>'
        )
    marked = set(highlight)
    for x, y, label in crossings:
        if label in marked:
            px, py = xy((x, y))
            out.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{max(7.0, 0.6 * scale):.1f}" fill="none" '
                f'stroke="#f39c12" stroke-width="{0.8 * width:.1f}"/>'
            )
    if labels:
        font = max(8.0, min(14.0, 0.9 * scale))
        for x, y, label in crossings:
            px, py = xy((x, y))
            out.append(
                f'<text x="{px + 0.35 * font:.1f}" y="{py - 0.35 * font:.1f}" font-size="{font:.1f}" '
                f'font-family="sans-serif" fill="#555">{escape(str(label))}</text>'
            )
    out.append("</svg>")
    return "\n".join(out)


def _arrow(comp, xy, color: str, length: float) -> str:
    """An arrowhead at the middle of the component's longest straight segment."""
    best = None
    for line in comp:
        for a, b in pairwise(line):
            seg = math.dist(a, b)
            if best is None or seg > best[0]:
                best = (seg, a, b)
    if best is None:
        return ""
    _, a, b = best
    (ax, ay), (bx, by) = xy(a), xy(b)
    mx, my = (ax + bx) / 2, (ay + by) / 2
    dx, dy = bx - ax, by - ay
    norm = math.hypot(dx, dy) or 1.0
    dx, dy = dx / norm, dy / norm
    tip = (mx + dx * length / 2, my + dy * length / 2)
    left = (mx - dx * length / 2 - dy * length / 2, my - dy * length / 2 + dx * length / 2)
    right = (mx - dx * length / 2 + dy * length / 2, my - dy * length / 2 - dx * length / 2)
    pts = " ".join("%.1f,%.1f" % p for p in (tip, left, right))
    return f'<polygon points="{pts}" fill="{color}"/>'


def view(d: LinkDiagram, labels: bool = True, block: bool = True):
    """Open the diagram in SnapPy's PLink viewer window (blocks until it is closed)."""
    viewer = to_spherogram(d).view(show_crossing_labels=labels)
    if block and viewer is not None:
        viewer.window.mainloop()
    return viewer


visualize_knot_diagram = view
