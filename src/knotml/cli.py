"""Command-line options shared by the scripts: which diagram to load, and its framings."""

from __future__ import annotations

import argparse
import ast
from typing import Optional, Sequence, Union

from .core import LinkDiagram
from .handlebody import Handlebody
from .kirby import FramedLink

__all__ = ["add_diagram_arguments", "diagram_from_args", "kirby_from_args", "parse_framings", "source_args"]


def add_diagram_arguments(parser: argparse.ArgumentParser) -> None:
    """``--braid/--pd/--knotinfo/--snappy`` (one of them) and ``--framings/--four/--closed``."""
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--braid", type=int, nargs="+", help="braid word, e.g. 1 1 1")
    source.add_argument("--pd", help='PD code, e.g. "[[1,5,2,4],[3,1,4,6],[5,3,6,2]]"')
    source.add_argument("--knotinfo", help="KnotInfo/LinkInfo name, e.g. 8_20 or \"L2a1{1}\"")
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


def diagram_from_args(args: argparse.Namespace, default: Optional[Sequence[int]] = (1, 1, 1)) -> Optional[LinkDiagram]:
    """The diagram the options name; with none given, the closure of the braid ``default`` (or ``None``)."""
    if args.pd:
        return LinkDiagram.from_pd(ast.literal_eval(args.pd))
    if args.knotinfo:
        from .interop import from_knotinfo

        return from_knotinfo(args.knotinfo)
    if args.snappy:
        import spherogram

        from .interop import from_spherogram

        return from_spherogram(spherogram.Link(args.snappy))
    if args.braid or default is not None:
        return LinkDiagram.from_braid(list(args.braid or default))
    return None


def parse_framings(values: Sequence[str]) -> list:
    """``["dot", "-2"]`` -> ``[None, -2]``: an integer framing per component, ``None`` for a dotted circle."""
    return [None if v.lower() in ("dot", "d", ".") else int(v) for v in values]


def source_args(args: argparse.Namespace) -> str:
    """The diagram options as they would be typed again (for printing replay commands)."""
    import shlex

    if args.pd:
        return f"--pd {shlex.quote(args.pd)}"
    if args.knotinfo:
        return f"--knotinfo {shlex.quote(args.knotinfo)}"
    if args.snappy:
        return f"--snappy {shlex.quote(args.snappy)}"
    return "--braid " + " ".join(map(str, args.braid or [1, 1, 1]))


def kirby_from_args(args: argparse.Namespace, d: LinkDiagram) -> Union[FramedLink, Handlebody, None]:
    """``d`` with the ``--framings`` given: a FramedLink, a Handlebody (any ``dot``, ``--four`` or
    ``--closed``), or ``None`` without ``--framings``.  Raises ``ValueError`` for bad framings."""
    if not args.framings:
        return None
    comps = parse_framings(args.framings)
    if None in comps or args.four or args.closed:
        return Handlebody(d, comps, closed=args.closed)
    return FramedLink(d, comps)
