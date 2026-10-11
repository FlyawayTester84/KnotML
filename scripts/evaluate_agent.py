"""Evaluate a trained agent; optionally record its episodes as HTML step pages.

Run with --help for options.  Needs the rl extra: pip install -e ".[rl]".
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from knotml.rl.cli import evaluate_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(evaluate_main())
