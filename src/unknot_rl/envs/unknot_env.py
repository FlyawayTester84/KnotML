"""Environment abstractions for unknotting tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


class UnknotEnv:
    """A minimal placeholder for a knot unknotting environment.

    This is intended to be a starting point. In future iterations, it should
    implement a full Gym interface (reset, step, render, etc.) and define a
    suitable observation/action space.
    """

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self.state: Dict[str, Any] = {}

    def reset(self) -> Dict[str, Any]:
        """Reset the environment and return the initial observation."""
        self.state = {"knot": None, "steps": 0}
        return self.state

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Apply an action and return (obs, reward, done, info)."""
        self.state["steps"] += 1
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        return self.state, reward, done, info

    def render(self) -> None:
        """Render the current state (placeholder)."""
        print(self.state)
