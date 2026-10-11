"""Let a trained agent from models/ reduce a Kirby diagram to the empty diagram, at most --max-steps moves.

Usage:
    python scripts/solve_link.py --list-agents
    python scripts/solve_link.py --knotinfo "L2a1{1}" --framings dot 3 --agent ppo-small-cpu --max-steps 20
    python scripts/solve_link.py --pd "[[4,1,3,2],[2,3,1,4]]" --framings dot 0 --attempts 10 --open

Prints the moves, a command that replays them in inspect_diagram.py, and writes a step-through page
(outputs/solve.html).  Needs the rl extra.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from knotml.rl.cli import solve_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(solve_main())
