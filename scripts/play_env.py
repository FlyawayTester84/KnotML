"""Play the Kirby environment from the terminal and watch it on an HTML page.

Usage:
    python scripts/play_env.py                                   # 4-manifold, scrambled start
    python scripts/play_env.py --calculus 3 --seed 4 --reward example
    python scripts/play_env.py --knotinfo "L2a1{1}" --framings dot 0   # start from a given diagram
    python scripts/play_env.py --help

Then type commands (help lists them): ``step 12``, ``step HANDLE_SLIDE 14 26``,
``preview 3``, ``random 5``, ``undo``, ``reset 7``, ``save``, ``quit``.  The page
(outputs/play.html by default) reloads itself after every command.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from knotml.play import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
