"""Reward functions for :class:`~knotml.envs.KirbyEnv`, which leaves the reward open.

Each reward is a picklable callable ``reward(transition) -> float``, so it also
works in subprocess vector environments.  :func:`make_reward` builds one from a
name and keyword arguments (as in a training config):

* ``zero``: always 0.
* ``sparse``: ``solved`` for reaching the empty diagram, ``step`` for every
  other legal step, ``illegal`` for an action that changes nothing.
* ``potential``: ``sparse`` plus potential-based shaping
  ``F = gamma * phi(after) - phi(before)`` with
  ``phi = -(component_weight * #components + crossing_weight * #crossings)``.
  Shaping of this form leaves the optimal policies unchanged (Ng, Harada and
  Russell, 1999), and ``phi`` of the empty diagram is 0.
* ``example``: :func:`example_reward`, an illustration for the play console.
* ``my_reward``: :func:`knotml.envs.my_rewards.my_reward`, the one you edit;
  keyword arguments are passed to it on every call.
* ``module:function``: your own; with keyword arguments it is called with them
  to build the reward, otherwise it is the reward.
"""

from __future__ import annotations

import importlib
import os
import sys
from functools import partial
from typing import Any, Callable

from .kirby_env import Transition

__all__ = ["Zero", "Sparse", "Potential", "REWARDS", "example_reward", "make_reward"]


class Zero:
    """Always 0 (the environment's default)."""

    def __call__(self, t: Transition) -> float:
        return 0.0

    def __repr__(self) -> str:
        return "Zero()"


class Sparse:
    """``solved`` on reaching the empty diagram, ``step`` per other legal step, ``illegal`` for a no-op."""

    def __init__(self, solved: float = 1.0, step: float = -0.01, illegal: float = -0.1) -> None:
        self.solved, self.step, self.illegal = solved, step, illegal

    def __call__(self, t: Transition) -> float:
        if not t.legal:
            return self.illegal
        return self.solved if t.terminated else self.step

    def __repr__(self) -> str:
        return f"Sparse(solved={self.solved}, step={self.step}, illegal={self.illegal})"


class Potential(Sparse):
    """:class:`Sparse` plus the potential-based shaping described in the module docstring."""

    def __init__(self, gamma: float = 0.99, component_weight: float = 0.1, crossing_weight: float = 0.01,
                 **sparse: float) -> None:
        super().__init__(**sparse)
        self.gamma, self.component_weight, self.crossing_weight = gamma, component_weight, crossing_weight

    def phi(self, state) -> float:
        return -(self.component_weight * len(state.labels()) + self.crossing_weight * state.diagram.n_crossings)

    def __call__(self, t: Transition) -> float:
        return super().__call__(t) + self.gamma * self.phi(t.after) - self.phi(t.before)

    def __repr__(self) -> str:
        return (f"Potential(gamma={self.gamma}, component_weight={self.component_weight}, "
                f"crossing_weight={self.crossing_weight}, solved={self.solved}, step={self.step}, "
                f"illegal={self.illegal})")


def example_reward(t: Transition) -> float:
    """An illustrative shaping for trying the play console, not a recommendation.

    -1 for an illegal action, +10 for reaching the empty diagram, otherwise
    0.5 per component removed and 0.05 per crossing removed (negative when
    added), minus 0.01 per step.
    """
    if not t.legal:
        return -1.0
    if t.terminated:
        return 10.0
    d_components = len(t.before.labels()) - len(t.after.labels())
    d_crossings = t.before.diagram.n_crossings - t.after.diagram.n_crossings
    return 0.5 * d_components + 0.05 * d_crossings - 0.01


def _my_reward(**params: Any) -> Callable[[Transition], float]:
    """``my_reward`` from ``my_rewards.py``, with ``params`` bound (a ``partial``, so it still pickles)."""
    from . import my_rewards  # imported only when chosen, so a slip in that file cannot break the others

    return partial(my_rewards.my_reward, **params) if params else my_rewards.my_reward


REWARDS: dict[str, Callable[..., Callable[[Transition], float]]] = {
    "zero": Zero,
    "sparse": Sparse,
    "potential": Potential,
    "my_reward": _my_reward,
}


def make_reward(name: str, **kwargs: Any) -> Callable[[Transition], float]:
    """The reward called ``name`` (see the module docstring), built with ``kwargs``."""
    if name in REWARDS:
        try:
            return REWARDS[name](**kwargs)
        except TypeError as err:
            raise ValueError(f"reward {name!r}: {err}") from None
    if name == "example":
        if kwargs:
            raise ValueError("reward 'example' takes no parameters")
        return example_reward
    module, sep, attr = name.partition(":")
    if not sep:
        raise ValueError(f"unknown reward {name!r}: use {', '.join(REWARDS)}, example or module:function")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    obj = getattr(importlib.import_module(module), attr)
    return obj(**kwargs) if kwargs else obj
