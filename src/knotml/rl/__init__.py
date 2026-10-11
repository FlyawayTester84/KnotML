"""Reinforcement learning on Kirby diagrams: reduce a diagram to the empty diagram.

Needs the ``rl`` extra (PyTorch, PyYAML).  Pieces, each replaceable:

* :mod:`~knotml.rl.config`: the YAML configuration (``configs/ppo.yaml``).
* :mod:`~knotml.rl.envs`: :class:`~knotml.envs.KirbyEnv` + candidate actions, vectorised.
* :mod:`~knotml.rl.policies`: policy-and-value networks (registry: ``gnn``).
* :mod:`~knotml.rl.algorithms`: training algorithms (registry: ``ppo``).
* :mod:`~knotml.rl.curriculum`: scramble-length curriculum.
* :mod:`~knotml.rl.trainer`: the training loop, logging and checkpoints.
* :mod:`~knotml.rl.evaluate`: success rates and recorded episodes.
* :mod:`~knotml.rl.solve`: a trained agent (from ``models/``) on a given diagram.

Scripts: ``scripts/train_agent.py``, ``scripts/evaluate_agent.py`` and ``scripts/solve_link.py``.
"""

from .algorithms import ALGORITHMS, Algorithm, make_algorithm, register_algorithm
from .config import Config, load_config
from .evaluate import evaluate, record_episodes
from .policies import POLICIES, GraphPolicy, Policy, make_policy, register_policy
from .trainer import Trainer, load_policy, resolve_device

__all__ = [
    "Config",
    "load_config",
    "Trainer",
    "load_policy",
    "resolve_device",
    "evaluate",
    "record_episodes",
    "Policy",
    "GraphPolicy",
    "POLICIES",
    "register_policy",
    "make_policy",
    "Algorithm",
    "ALGORITHMS",
    "register_algorithm",
    "make_algorithm",
]
