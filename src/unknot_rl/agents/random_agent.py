"""A minimal agent implementation for baseline experiments."""

from __future__ import annotations

import random

from ..envs.unknot_env import UnknotEnv


class RandomAgent:
    """A random policy agent for UnknotEnv."""

    def __init__(self, action_space_size: int = 4, seed: int | None = None):
        self._rng = random.Random(seed)
        self.action_space_size = action_space_size

    def select_action(self, obs: dict) -> int:
        """Return a random action index."""
        return self._rng.randrange(self.action_space_size)

    def train_step(self, *args, **kwargs):
        """Optional training step placeholder."""
        pass
