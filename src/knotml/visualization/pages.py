"""Self-contained HTML pages built from :func:`~knotml.visualization.svg.to_svg` drawings.

* :func:`steps_page` / :func:`write_steps_page`: one frame per step of a move
  sequence, with a slider and arrow-key navigation (``inspect_diagram.py --html``).
* :func:`page_head`, :func:`invariants_html`, :func:`write_text`: building blocks
  shared with other pages (``knotml.play``).

Pages use the same light/dark tokens as the drawings.  The "Light / dark"
button overrides the system theme and is remembered in ``localStorage``.
"""

from __future__ import annotations

import html
import os
from typing import Mapping, Optional, Sequence

from ..core import LinkDiagram, determinant
from ..handlebody import Handlebody
from ..kirby import FramedLink
from ..notation import signed
from .svg import svg_css, to_svg

__all__ = ["steps_page", "write_steps_page", "page_head", "invariants_html", "component_notes", "write_text"]

_PAGE_TOKENS = {
    "light": "--page:#f9f9f7;--ink:#0b0b0b;--ink-2:#52514e;--rule:#e1e0d9;--border:rgba(11,11,11,0.10);"
    "color-scheme:light",
    "dark": "--page:#0d0d0d;--ink:#f0efec;--ink-2:#c3c2b7;--rule:#2c2c2a;--border:rgba(255,255,255,0.10);"
    "color-scheme:dark",
}

_BASE_CSS = """
  body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 24px auto; padding: 0 16px;
         background: var(--page); color: var(--ink); }
  figure { margin: 0; } figcaption { font-weight: 600; margin-bottom: 8px; }
  svg.knotml { width: 100%; height: auto; border: 1px solid var(--border); border-radius: 6px; }
  code { font-size: 12px; word-break: break-all; }
  .hint, small, summary { color: var(--ink-2); } .hint { font-size: 13px; }
"""

_THEME_EARLY = """<script>
  try {
    const t = localStorage.getItem('knotml-theme');
    if (t) document.documentElement.dataset.theme = t;
  } catch (e) {}
</script>"""

THEME_BUTTON = '<button id="theme" type="button">Light / dark</button>'
THEME_JS = """
  document.getElementById('theme').onclick = () => {
    const root = document.documentElement, t = root.dataset.theme;
    const dark = t ? t === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('knotml-theme', root.dataset.theme); } catch (e) {}
  };
"""


def page_head(title: str, css: str = "", extra: str = "") -> str:
    """``<!doctype html>`` through ``<body>``: tokens for both themes, base styles, drawing styles and ``css``."""
    light, dark = _PAGE_TOKENS["light"], _PAGE_TOKENS["dark"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
{_THEME_EARLY}
<style>
  :root {{ {light} }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme=light]) {{ {dark} }} }}
  :root[data-theme=dark] {{ {dark} }}
{_BASE_CSS}{css}
{svg_css()}
</style>{extra}</head><body>
"""


def component_notes(comps: Optional[Mapping[int, Optional[int]]]):
    """``(dotted labels, legend notes)`` for a ``{label: framing or None (dotted)}`` map."""
    if comps is None:
        return [], {}
    dotted = [lab for lab, f in comps.items() if f is None]
    notes = {lab: "1-handle" if f is None else f"framing {signed(f)}" for lab, f in comps.items()}
    return dotted, notes


def invariants_html(diagram: LinkDiagram, comps=None, calculus: Optional[str] = None, closed: bool = False) -> str:
    """The invariants a caption shows: the determinant (plain diagram), H₁ (``calculus="3"``),
    or χ, H₁(X), σ and H₁(∂X) (``"4"``; χ and H₁(∂X) only when not ``closed``)."""
    if comps is None:
        return f"determinant {determinant(diagram)}"
    if calculus == "4":
        h = Handlebody(diagram, comps, closed=closed)
        text = (f"&chi; = {h.euler_characteristic()} &middot; H<sub>1</sub>(X) = {h.homology()}"
                f" &middot; &sigma; = {h.signature()}")
        return text if closed else text + f" &middot; H<sub>1</sub>(&part;X) = {h.boundary().homology()}"
    return f"H<sub>1</sub> = {FramedLink(diagram, comps).homology()}"


def steps_page(history: Sequence[tuple], calculus: Optional[str] = None) -> str:
    """A page stepping through ``history``: ``(label, diagram, highlighted crossings, components, note)``
    per step, ``components`` being ``{label: framing or None}`` (``None`` for a plain diagram) and
    ``note`` a line of small print under the caption (may be empty)."""
    frames = []
    for step, (label, diagram, marked, comps, status) in enumerate(history):
        caption = f"step {step}: {html.escape(label)} &rarr; {diagram.n_crossings} crossings"
        caption += (", " if comps is None else "<br>") + invariants_html(diagram, comps, calculus)
        if status:
            caption += f"<br><small>{html.escape(status)}</small>"
        dotted, notes = component_notes(comps)
        svg = to_svg(diagram, size=520, labels=True, highlight=marked, dotted=dotted, notes=notes, style=False)
        frames.append(
            f'<figure class="frame" hidden><figcaption>{caption}</figcaption>{svg}'
            f"<details><summary>PD code</summary><code>{html.escape(str(diagram.to_pd()))}</code></details></figure>"
        )
    css = """  body { max-width: 600px; }
  .controls { display: flex; gap: 8px; align-items: center; margin: 12px 0; }
  input[type=range] { flex: 1; }
"""
    return f"""{page_head("Knot move sequence", css)}<h1>Knot move sequence</h1>
<p class="hint">Drawn with spherogram's orthogonal layout. Each component keeps its colour and line pattern
across steps (legend under the drawing); circled crossings were created or rearranged by the move.
Each step is laid out from scratch, so pictures can jump between steps. Use &larr; / &rarr; to step.</p>
<div class="controls"><button id="prev" aria-label="Previous step">&larr;</button>
<input id="slider" type="range" min="0" max="{len(frames) - 1}" value="0" aria-label="Step">
<button id="next" aria-label="Next step">&rarr;</button>{THEME_BUTTON}</div>
{"".join(frames)}
<script>
  const frames = document.querySelectorAll('.frame'), slider = document.getElementById('slider');
  function show(i) {{
    i = Math.max(0, Math.min(frames.length - 1, i));
    frames.forEach((f, k) => f.hidden = k !== i);
    slider.value = i;
  }}
  slider.oninput = () => show(+slider.value);
  document.getElementById('prev').onclick = () => show(+slider.value - 1);
  document.getElementById('next').onclick = () => show(+slider.value + 1);
  document.onkeydown = e => {{
    if (e.key === 'ArrowLeft') show(+slider.value - 1);
    if (e.key === 'ArrowRight') show(+slider.value + 1);
  }};{THEME_JS}  show(0);
</script></body></html>
"""


def write_steps_page(path: str, history: Sequence[tuple], calculus: Optional[str] = None) -> None:
    """Write :func:`steps_page` to ``path`` (directories are created)."""
    write_text(path, steps_page(history, calculus))


def write_text(path: str, text: str) -> None:
    """Write ``text`` to ``path`` atomically (a page being reloaded never sees half a file)."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
