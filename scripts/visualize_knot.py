"""Run a quick knot visualization from the repository without installing the package.

Usage:
    python scripts/visualize_knot.py --crossings 10 --seed 42

This script ensures the local `src/` directory is on the import path so that the
library modules can be imported directly.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure we can import the library from the local src/ directory.
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from src.utils import generate_random_knot
from src.utils.visualization import draw_knot_diagram, graph_to_dot


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize a random knot diagram.")
    parser.add_argument("--crossings", type=int, default=8, help="Number of crossings")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not show the plot (prints DOT instead).",
    )

    args = parser.parse_args()

    knot = generate_random_knot(crossings=args.crossings)

    if args.no_show:
        dot = graph_to_dot(knot.to_graph(args.crossings))
        print(dot)
        return 0

    try:
        draw_knot_diagram(knot, show=True)
        return 0
    except Exception:
        print("Unable to render diagram (matplotlib may be missing). Falling back to DOT output.")
        print(graph_to_dot(knot.to_graph(args.crossings)))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
