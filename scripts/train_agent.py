"""Train an agent to reduce Kirby diagrams to the empty diagram (hyperparameters: configs/ppo.yaml).

Run with --help for options.  Needs the rl extra: pip install -e ".[rl]".
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from knotml.rl.cli import train_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(train_main())
