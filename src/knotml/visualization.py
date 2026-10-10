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

__all__ = ["layout", "to_svg", "view", "visualize_knot_diagram", "component_color"]

_COLORS = ["#1f4e9c", "#c0392b", "#1e8449", "#8e44ad", "#d35400", "#117a8b", "#7d6608"]


def component_color(label: int) -> str:
    """Drawing colour of the component with this label (stable across moves)."""
    return _COLORS[label % len(_COLORS)]


def layout(d: LinkDiagram):
    """Spherogram's orthogonal layout of ``d``.

    Returns ``(components, crossings)``: per component, ``(label, strands)`` with
    the strands as point lists with gaps at undercrossings; and
    ``(x, y, crossing index)`` triples.
    """
    import plink

    manager = to_spherogram(d).view(viewer=plink.LinkManager())
    crossings = [(c.x, c.y, int(c.label)) for c in manager.Crossings]  # PLink labels are strings
    components = []
    strands = manager.polylines(break_at_overcrossings=False)  # same order as arrow_components()
    for arrows, (polylines, _) in zip(manager.arrow_components(), strands, strict=True):
        arrows = set(arrows)
        label = next(  # the component's label, read off any crossing it passes
            d.label_of_port(4 * int(c.label) + (0 if c.under in arrows else 1))
            for c in manager.Crossings
            if c.under in arrows or c.over in arrows
        )
        components.append((label, polylines))
    return components, crossings


def to_svg(
    d: LinkDiagram,
    size: int = 420,
    labels: bool = False,
    title: str = "",
    highlight: Iterable[int] = (),
) -> str:
    """An SVG drawing of ``d``: one colour per component, an arrow showing each orientation.

    Crossings listed in ``highlight`` are circled.  Crossingless components are
    drawn as labelled circles in a strip below the rest of the diagram.
    """
    components, crossings = layout(d) if d.n_crossings else ([], [])
    points = [p for _, comp in components for line in comp for p in line]
    pad = 24
    n_circles = d.n_unknots
    strip = 0.22 * size if (n_circles and points) else 0.0  # height reserved for crossingless circles
    gap = 0.05 * size if strip else 0.0  # space between the link and the strip
    if points:
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        x0, y0 = min(xs), min(ys)
        span_x, span_y = max(max(xs) - x0, 1.0), max(max(ys) - y0, 1.0)
    else:
        x0 = y0 = 0.0
        span_x = span_y = 1.0
    box_w, box_h = size - 2 * pad, size - 2 * pad - strip - gap
    scale = min(box_w / span_x, box_h / span_y)
    off_x = pad + (box_w - span_x * scale) / 2  # centre the link in the space above the strip
    off_y = pad + (box_h - span_y * scale) / 2

    def xy(p):
        return off_x + (p[0] - x0) * scale, off_y + (p[1] - y0) * scale

    width = max(1.5, min(4.0, 0.25 * scale))
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}">',
        f'<rect width="{size}" height="{size}" fill="white"/>',
    ]
    if title:
        out.append(f"<title>{escape(title)}</title>")
    for label, comp in components:
        color = component_color(label)
        for line in comp:
            pts = " ".join("%.1f,%.1f" % xy(p) for p in line)
            out.append(
                f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width:.1f}" '
                'stroke-linejoin="round" stroke-linecap="round"/>'
            )
        out.append(_arrow(comp, xy, color, 3 * width))
    if n_circles:  # crossingless components: separate circles, evenly spaced in their own area
        top = size - pad - strip if points else pad
        height = strip if points else size - 2 * pad
        slot = (size - 2 * pad) / n_circles
        r = max(4.0, min(0.32 * height, 0.32 * slot))
        if points:
            line_y = top - gap / 2
            out.append(
                f'<line x1="{pad}" y1="{line_y:.1f}" x2="{size - pad}" y2="{line_y:.1f}" '
                'stroke="#ddd" stroke-dasharray="4 4"/>'
            )
        for k, label in enumerate(d.trivial_labels):
            cx, cy = pad + (k + 0.5) * slot, top + height / 2
            color = component_color(label)
            out.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none" stroke="{color}" '
                f'stroke-width="{width:.1f}"/>'
            )
            out.append(
                f'<text x="{cx:.1f}" y="{cy + 0.3 * min(r, 14):.1f}" font-size="{min(r, 14):.1f}" '
                f'text-anchor="middle" font-family="sans-serif" fill="{color}">K{label}</text>'
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
