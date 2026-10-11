"""The interface a training algorithm implements, and its registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch

__all__ = ["Algorithm", "ALGORITHMS", "register_algorithm", "make_algorithm"]

ALGORITHMS: dict[str, type] = {}


def register_algorithm(name: str) -> Callable[[type], type]:
    """Class decorator: make an :class:`Algorithm` available as ``algorithm.name`` in the config."""

    def deco(cls: type) -> type:
        ALGORITHMS[name] = cls
        return cls

    return deco


class Algorithm(ABC):
    """One training iteration at a time; the :class:`~knotml.rl.trainer.Trainer` does the rest
    (environments, logging, evaluation, curriculum, checkpoints).

    A subclass declares its hyperparameters as a nested ``Hyperparameters``
    dataclass (the ``algorithm`` section of the config, minus ``name``) and
    implements :meth:`iteration`.  It receives the policy and a
    :class:`~knotml.rl.rollout.Collector` that steps the vector environment and
    tracks episodes, so on-policy and off-policy methods alike can collect
    experience their own way.
    """

    @dataclass
    class Hyperparameters:
        gamma: float = 0.99

    def __init__(self, hp, policy: torch.nn.Module, collector, device: torch.device, total_timesteps: int,
                 seed: int = 0) -> None:
        self.hp, self.policy, self.collector, self.device = hp, policy, collector, device
        self.total_timesteps = total_timesteps
        self.generator = torch.Generator().manual_seed(seed)

    @abstractmethod
    def iteration(self, global_step: int) -> dict:
        """Collect experience and update the policy once.  Returns metrics; ``env_steps`` (required)
        is the number of environment steps taken."""

    def state_dict(self) -> dict:
        """Algorithm state to checkpoint besides the policy weights (optimizer, schedules)."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


def make_algorithm(name: str, params: dict, **kwargs) -> Algorithm:
    if name not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {name!r}; registered: {', '.join(ALGORITHMS)}")
    cls = ALGORITHMS[name]
    try:
        hp = cls.Hyperparameters(**params)
    except TypeError as err:
        raise ValueError(f"algorithm {name!r}: {err}") from None
    return cls(hp, **kwargs)
