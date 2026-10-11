"""Drawings of link diagrams (``svg``) and HTML pages built from them (``pages``).

Drawing needs the ``oracles`` extra (spherogram and plink); importing does not.
"""

from .pages import steps_page, write_steps_page
from .svg import PATTERNS, component_color, component_style, layout, svg_css, to_svg, view, visualize_knot_diagram

__all__ = [
    "layout",
    "to_svg",
    "svg_css",
    "view",
    "visualize_knot_diagram",
    "component_color",
    "component_style",
    "PATTERNS",
    "steps_page",
    "write_steps_page",
]
