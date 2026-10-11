"""Training algorithms, registered by name (``algorithm.name`` in the config).

To add one, subclass :class:`Algorithm` in a new module, decorate it with
``@register_algorithm("name")`` and import the module here.
"""

from . import ppo  # noqa: F401  (registers "ppo")
from .base import ALGORITHMS, Algorithm, make_algorithm, register_algorithm
from .ppo import PPO

__all__ = ["Algorithm", "ALGORITHMS", "register_algorithm", "make_algorithm", "PPO"]
