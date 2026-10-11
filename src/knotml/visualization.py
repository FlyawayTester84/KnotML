"""Drawing link diagrams with spherogram's orthogonal layout (needs the ``oracles`` extra).

* :func:`to_svg` renders a diagram to an SVG string without opening a window.
* :func:`view` opens SnapPy's interactive PLink viewer.

Crossing ``c`` of the picture is crossing ``c`` of the :class:`LinkDiagram`, so
move anchors (port ``4c + k``) can be located on it.  Each call lays the
diagram out from scratch, so consecutive diagrams in a move sequence are not
drawn continuously.

Component styles follow the label, so a component looks the same at every step
of a move sequence.  Any two components can meet at a crossing, so every pair
of styles must be told apart, with or without colour-vision deficiency.  Only
three hues pass that test (all pairs, checked with the dataviz validator in
light and dark mode), so:

* labels 0-2 get the three hues;
* labels 3-11 reuse them with a dashed, dash-dot or dotted line;
* labels from 12 on are drawn in neutral ink.

Colours are never cycled.  Every drawing with two or more components has a
legend and a ``K<label>`` tag on each component, so identity never rests on
colour alone.  The SVG's colours are CSS custom properties with light and dark
values: it follows the system theme or a ``data-theme`` attribute on an
ancestor.
"""

from __future__ import annotations

import math
from html import escape
from itertools import pairwise
from typing import Iterable, Mapping

import numpy as np

from .core import LinkDiagram
from .interop import to_spherogram

__all__ = [
    "layout",
    "to_svg",
    "svg_css",
    "view",
    "visualize_knot_diagram",
    "component_color",
    "component_style",
    "PATTERNS",
]

# Reference palette of the dataviz skill.  Series: its first three categorical
# slots (blue, orange, aqua), the largest prefix that validates on all pairs.
# Aqua is below 3:1 on the light surface, which the direct labels make up for.
_TOKENS = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink-2": "#52514e", "muted": "#898781", "rule": "#c3c2b7",
        "k0": "#2a78d6", "k1": "#eb6834", "k2": "#1baf7a",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink-2": "#c3c2b7", "muted": "#898781", "rule": "#383835",
        "k0": "#3987e5", "k1": "#d95926", "k2": "#199e70",
    },
}  # fmt: skip
_N_HUES = 3
PATTERNS = ("solid", "dashed", "dash-dot", "dotted")
# Dash arrays in stroke widths.  With round caps a dash looks one width longer and a gap one width shorter.
_DASH = {"solid": (), "dashed": (2.5, 2.5), "dash-dot": (2.5, 2.2, 0, 2.2), "dotted": (0, 2.2)}
_FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def component_style(label: int) -> tuple[int | None, str]:
    """``(hue slot, line pattern)`` of the component with this label; slot ``None`` is neutral ink.

    The style depends on the label alone, so a component keeps it across moves.
    The 12 styles of labels 0-11 are all different.
    """
    if 0 <= label < _N_HUES * len(PATTERNS):
        return label % _N_HUES, PATTERNS[label // _N_HUES]
    return None, "solid"


def component_color(label: int, mode: str = "light") -> str:
    """Stroke colour of the component with this label in ``mode`` (``"light"`` or ``"dark"``)."""
    slot, _ = component_style(label)
    return _TOKENS[mode]["muted" if slot is None else f"k{slot}"]


def svg_css() -> str:
    """The stylesheet of :func:`to_svg` drawings: light tokens, plus dark tokens chosen by
    ``prefers-color-scheme`` or a ``data-theme="dark"`` ancestor (``data-theme="light"`` forces light)."""

    def tokens(mode):
        return "".join(f"--{k}:{v};" for k, v in _TOKENS[mode].items()) + f"color-scheme:{mode}"

    rules = [f".knotml{{{tokens('light')}}}"]
    rules.append(
        "@media (prefers-color-scheme: dark){"
        f".knotml:not([data-theme=light],[data-theme=light] *){{{tokens('dark')}}}}}"
    )
    rules.append(f".knotml:is([data-theme=dark],[data-theme=dark] *){{{tokens('dark')}}}")
    rules += [f".knotml .k{s}{{stroke:var(--k{s});fill:var(--k{s})}}" for s in range(_N_HUES)]
    rules += [
        ".knotml .kx{stroke:var(--muted);fill:var(--muted)}",
        ".knotml .bg{fill:var(--surface)}",
        ".knotml .rule{stroke:var(--rule)}",
        ".knotml .mark{stroke:var(--ink)}",
        ".knotml .t{fill:var(--ink);stroke:var(--surface)}",
        ".knotml .t2{fill:var(--ink-2);stroke:var(--surface)}",
    ]
    return "\n".join(rules)


def layout(d: LinkDiagram):
    """Spherogram's orthogonal layout of ``d``.

    Returns ``(components, crossings)``: per component, ``(label, strands)`` with
    the strands as point lists with gaps at undercrossings; and
    ``(x, y, crossing index)`` triples.  The layout needs a connected projection,
    so a split diagram is laid out piece by piece and the pieces placed side by side.
    """
    pieces: dict[int, list[int]] = {}
    for c, k in enumerate(d.diagram_component_of_crossing()):
        pieces.setdefault(k, []).append(c)
    if len(pieces) <= 1:
        return _layout_connected(d)
    components: list = []
    crossings: list = []
    x_end = 0.0
    for members in pieces.values():
        sub, original = _piece(d, members)
        comps, xs = _layout_connected(sub)
        points = [pt for _, comp in comps for line in comp for pt in line]
        lo_x, lo_y = min(pt[0] for pt in points), min(pt[1] for pt in points)
        hi_x = max(pt[0] for pt in points)
        shift = x_end - lo_x + (0.2 * (hi_x - lo_x) if components else 0.0)  # leave a gap between pieces
        components += [
            (label, [[(x + shift, y - lo_y) for x, y in line] for line in comp]) for label, comp in comps
        ]
        crossings += [(x + shift, y - lo_y, original[c]) for x, y, c in xs]
        x_end = hi_x + shift
    return components, crossings


def _piece(d: LinkDiagram, members: list[int]):
    """The sub-diagram on the crossings ``members`` (one connected piece), and its crossing map back."""
    new = {c: i for i, c in enumerate(members)}
    nbr = [4 * new[d.nbr[4 * c + k] >> 2] + (d.nbr[4 * c + k] & 3) for c in members for k in range(4)]
    labels = [d.label_of_port(4 * c + k) for c in members for k in (0, 1)]
    sub = LinkDiagram(nbr, [d.signs[c] for c in members], labels=labels, trivial=[], check=False)
    return sub, members


def _layout_connected(d: LinkDiagram):
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


def _group(label: int, width: float) -> str:
    """Opening tag of a component's group: its colour (CSS class plus light fallback), width and pattern."""
    slot, pattern = component_style(label)
    color = component_color(label)
    dash = " ".join(f"{u * width:.1f}" for u in _DASH[pattern])
    return (
        f'<g class="{"kx" if slot is None else f"k{slot}"}" stroke="{color}" fill="{color}" '
        f'stroke-width="{width:.1f}" stroke-linecap="round" stroke-linejoin="round"'
        + (f' stroke-dasharray="{dash}">' if dash else ">")
    )


def _text(x: float, y: float, text: str, size: float, cls: str = "t", anchor: str = "middle") -> str:
    """Text in ink (``t``) or secondary ink (``t2``), with a surface-coloured halo so it reads over strands."""
    fill = _TOKENS["light"]["ink" if cls == "t" else "ink-2"]
    return (
        f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" font-size="{size:.1f}" text-anchor="{anchor}" '
        f'fill="{fill}" stroke="{_TOKENS["light"]["surface"]}" stroke-width="3" stroke-linejoin="round" '
        f'paint-order="stroke">{escape(text)}</text>'
    )


def to_svg(
    d: LinkDiagram,
    size: int = 420,
    labels: bool = False,
    title: str = "",
    highlight: Iterable[int] = (),
    dotted: Iterable[int] = (),
    notes: Mapping[int, str] | None = None,
    legend: bool = True,
    style: bool = True,
) -> str:
    """An SVG drawing of ``d``, ``size`` pixels wide.

    Each component is drawn in its :func:`component_style`, with an arrow
    showing its orientation.

    * ``labels``: number the crossings.
    * ``highlight``: crossings to circle.
    * ``dotted``: labels of components (1-handles of a Kirby diagram) that get
      a solid dot instead of an arrow.
    * ``notes``: text for a component's legend entry, e.g. ``{0: "framing -2"}``.

    Crossingless components are drawn as circles in a strip below the rest of
    the diagram.  With two or more components, each component is tagged
    ``K<label>`` and a legend sits below the drawing (``legend=False`` drops
    it).  ``style=False`` leaves out the ``<style>`` block, for pages that
    include :func:`svg_css` once.
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

    width = max(2.0, min(3.0, 0.2 * scale))
    marker = max(10.0, 4 * width)  # arrow length and dot diameter
    dots = set(dotted)
    all_labels = sorted({label for label, _ in components} | set(d.trivial_labels))
    tagged = len(all_labels) >= 2
    entries = _legend_rows(all_labels, notes or {}, size - 2 * pad) if (legend and tagged) else []
    height = size + (22 * entries[-1][0] + 26 if entries else 0)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" class="knotml" viewBox="0 0 {size} {height}" '
        f'width="{size}" height="{height}" font-family="{_FONT}">'
    ]
    if title:
        out.append(f"<title>{escape(title)}</title>")
    if style:
        out.append(f"<style>\n{svg_css()}\n</style>")
    out.append(f'<rect class="bg" width="{size}" height="{height}" fill="{_TOKENS["light"]["surface"]}"/>')
    sites = []  # per component: (label, its segments in picture coordinates, its marker)
    for label, comp in components:
        out.append(_group(label, width))
        out += [f'<polyline points="{" ".join("%.1f,%.1f" % xy(p) for p in line)}" fill="none"/>' for line in comp]
        site = _marker_site(comp, xy)
        if site is not None:
            mx, my, dx, dy = site
            if label in dots:
                out.append(f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="{marker / 2:.1f}" stroke="none"/>')
            else:
                out.append(_arrow(mx, my, dx, dy, marker))
            sites.append((label, [(xy(a), xy(b)) for line in comp for a, b in pairwise(line)], (mx, my)))
        out.append("</g>")
    tags = []
    if tagged and sites:
        busy = [xy((x, y)) for x, y, _ in crossings]  # crossings and their numbers
        bounds = (6.0, 6.0, size - 6.0, size - pad - strip - gap / 2 if strip else size - pad / 2)
        for label, (x, y) in _place_tags(sites, busy, bounds, marker):
            tags.append(_text(x, y + 4.5, f"K{label}", 13))
    if n_circles:  # crossingless components: separate circles, evenly spaced in their own area
        top = size - pad - strip if points else pad
        band = strip if points else size - 2 * pad
        slot = (size - 2 * pad) / n_circles
        r = max(6.0, min(0.32 * band, 0.32 * slot))
        if points:
            line_y = top - gap / 2
            out.append(
                f'<line class="rule" x1="{pad}" y1="{line_y:.1f}" x2="{size - pad}" y2="{line_y:.1f}" '
                f'stroke="{_TOKENS["light"]["rule"]}" stroke-width="1"/>'
            )
        for k, label in enumerate(d.trivial_labels):
            cx, cy = pad + (k + 0.5) * slot, top + band / 2
            out.append(_group(label, width))
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none"/>')
            if label in dots:
                out.append(f'<circle cx="{cx:.1f}" cy="{cy - r:.1f}" r="{marker / 2:.1f}" stroke="none"/>')
            out.append("</g>")
            font = min(13.0, max(9.0, 0.75 * r))
            tags.append(_text(cx, cy + 0.35 * font, f"K{label}", font))
    marked = set(highlight)
    for x, y, label in crossings:
        if label in marked:
            px, py = xy((x, y))
            out.append(
                f'<circle class="mark" cx="{px:.1f}" cy="{py:.1f}" r="{max(11.0, 0.6 * scale):.1f}" fill="none" '
                f'stroke="{_TOKENS["light"]["ink"]}" stroke-width="1.5"/>'
            )
    if labels:
        for x, y, label in crossings:
            px, py = xy((x, y))
            out.append(_text(px + 9, py - 9, str(label), 11, "t2", "start"))
    out += tags
    for row, x, label, text in entries:
        y = size + 4 + 22 * row
        out.append(_group(label, width))
        out.append(f'<line x1="{pad + x:.1f}" y1="{y:.1f}" x2="{pad + x + 24:.1f}" y2="{y:.1f}"/>')
        if label in dots:
            out.append(f'<circle cx="{pad + x + 12:.1f}" cy="{y:.1f}" r="{marker / 2:.1f}" stroke="none"/>')
        out.append("</g>")
        out.append(_text(pad + x + 30, y + 4.5, text, 13, anchor="start"))
    out.append("</svg>")
    return "\n".join(out)


def _legend_rows(labels: list[int], notes: Mapping[int, str], width: float):
    """``(row, x, label, text)`` per legend entry, flowing left to right and wrapping at ``width``."""
    entries, row, x = [], 0, 0.0
    for label in labels:
        text = f"K{label}" + (f" ({notes[label]})" if notes.get(label) else "")
        w = 30 + 7.2 * len(text)  # swatch + gap + rough text width at 13px
        if x and x + w > width:
            row, x = row + 1, 0.0
        entries.append((row, x, label, text))
        x += w + 18
    return entries


def _marker_site(comp, xy):
    """Middle of the component's longest straight segment and its unit direction, in picture coordinates."""
    segs = [(a, b) for line in comp for a, b in pairwise(line)]
    if not segs:
        return None
    a, b = max(segs, key=lambda ab: math.dist(*ab))
    (ax, ay), (bx, by) = xy(a), xy(b)
    norm = math.hypot(bx - ax, by - ay) or 1.0
    return (ax + bx) / 2, (ay + by) / 2, (bx - ax) / norm, (by - ay) / norm


def _arrow(mx: float, my: float, dx: float, dy: float, length: float) -> str:
    """An arrowhead of the given length centred at ``(mx, my)``, pointing along ``(dx, dy)``."""
    h = length / 2
    tip = (mx + dx * h, my + dy * h)
    left = (mx - dx * h - dy * h, my - dy * h + dx * h)
    right = (mx - dx * h + dy * h, my - dy * h - dx * h)
    return f'<polygon points="{" ".join("%.1f,%.1f" % p for p in (tip, left, right))}" stroke="none"/>'


def _place_tags(sites, busy, bounds, marker):
    """Centre of each component's ``K<label>`` tag: beside one of its long segments, where there is most room.

    ``sites`` lists ``(label, segments, marker centre)`` per component; ``busy``
    the points to keep clear of (crossings).  Candidates sit 11 px off the
    middle part of a component's eight longest segments, on either side; the
    winner maximises the distance to other strands, crossings, markers, tags
    already placed and the edges of ``bounds``.  Vectorised: O(candidates x segments).
    """
    seg = np.array([ab for _, segs, _ in sites for ab in segs], dtype=float).reshape(-1, 2, 2)
    a, v = seg[:, 0], seg[:, 1] - seg[:, 0]
    vv = np.maximum((v * v).sum(1), 1e-9)
    points = [np.asarray(busy, dtype=float).reshape(-1, 2)]
    points.append(np.array([m for _, _, m in sites], dtype=float))
    radius = [np.full(len(busy), 14.0), np.full(len(sites), marker / 2 + 6)]
    x0, y0, x1, y1 = bounds
    placed, start = [], 0
    for label, segs, _ in sites:
        hosts = sorted(range(len(segs)), key=lambda i: -math.dist(*segs[i]))[:8]
        cands, own = [], []
        half = 3.6 * len(f"K{label}") + 6  # half the tag's width at 13px, plus a margin
        for i in hosts:
            (ax, ay), (bx, by) = segs[i]
            vertical = abs(by - ay) > abs(bx - ax)
            for t in (0.3, 0.5, 0.7):
                px, py = ax + t * (bx - ax), ay + t * (by - ay)
                for side in (1, -1):
                    cands.append((px + side * (11 + half), py) if vertical else (px, py + side * 15))
                    own.append(start + i)
        c = np.array(cands)
        rel = c[:, None, :] - a[None]
        t = np.clip((rel * v[None]).sum(2) / vv[None], 0.0, 1.0)
        d = np.hypot(*(rel - t[..., None] * v[None]).transpose(2, 0, 1))
        d[np.arange(len(c)), own] = np.inf  # the host segment is 11-15 px away by construction
        room = d.min(1) - half
        p, r = np.concatenate(points), np.concatenate(radius)
        if len(p):
            room = np.minimum(room, (np.hypot(*(c[:, None, :] - p[None]).transpose(2, 0, 1)) - r[None]).min(1))
        edge = np.minimum.reduce([c[:, 0] - half - x0, x1 - c[:, 0] - half, c[:, 1] - 8 - y0, y1 - c[:, 1] - 8])
        best = c[int(np.argmax(np.minimum(room, edge)))]
        placed.append((label, (float(best[0]), float(best[1]))))
        points.append(best[None])
        radius.append(np.array([2 * half]))
        start += len(segs)
    return placed


def view(d: LinkDiagram, labels: bool = True, block: bool = True):
    """Open the diagram in SnapPy's PLink viewer window (blocks until it is closed)."""
    viewer = to_spherogram(d).view(show_crossing_labels=labels)
    if block and viewer is not None:
        viewer.window.mainloop()
    return viewer


visualize_knot_diagram = view
