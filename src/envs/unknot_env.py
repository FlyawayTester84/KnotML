"""Environment abstractions for unknotting tasks."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from utils.knot import Knot, generate_random_knot, is_unknot


Observation = Dict[str, np.ndarray]  # graph-style: {"adj": ..., "node": ...}


class UnknotEnv(gym.Env):
    """A Gymnasium environment for unknotting knots.

    This environment uses a graph-style observation (adjacency + node features)
    and keeps a lightweight knot model in `self.state["knot"]`.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        max_crossings: int = 16,
        max_steps: int = 200,
        seed: Optional[int] = None,
    ):
        self.max_crossings = max_crossings
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)

        self.state: Dict[str, Any] = {}

        self.observation_space = self._make_observation_space()
        self.action_space = gym.spaces.Discrete(self._default_action_space_size())

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Observation, dict]:
        """Reset the environment and return (observation, info)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.state = {
            "knot": generate_random_knot(self.max_crossings),
            "steps": 0,
            "is_unknot": False,
        }
        return self._encode_observation(), {}

    def step(
        self, action: int
    ) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        self.state["steps"] += 1

        self._apply_action(action)

        obs = self._encode_observation()
        reward = self._compute_reward()
        terminated = self._is_done()
        truncated = self.state["steps"] >= self.max_steps
        info: Dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render the current state (human-readable).

        This attempts to visualize the current knot state using the graph-style
        observation produced by :meth:`_encode_observation`.

        If ``matplotlib`` is unavailable, this will fall back to printing a
        GraphViz DOT description.
        """

        knot = self.state.get("knot")
        if knot is None:
            print("UnknotEnv(state=", self.state, ")")
            return

        try:
            from utils.visualization import draw_knot_diagram

            # We ignore the returned figure; the function will show it if possible.
            draw_knot_diagram(knot, show=True)
        except Exception:
            # Fallback: print a lightweight description (e.g., DOT) when rendering
            # is not supported in the current environment.
            from utils.visualization import graph_to_dot

            print(graph_to_dot(self._encode_observation()))

    def _make_observation_space(self) -> gym.Space:
        """Construct the Gym observation space for the graph representation."""
        N = self.max_crossings
        return gym.spaces.Dict(
            {
                "adj": gym.spaces.Box(0.0, 1.0, shape=(N, N), dtype=np.float32),
                "node": gym.spaces.Box(-1.0, 1.0, shape=(N, 4), dtype=np.float32),
            }
        )

    def _default_action_space_size(self) -> int:
        """Return a default discrete action space size (placeholder)."""
        # Example: allow 3 possible move types per crossing.
        return self.max_crossings * 3

    def _encode_observation(self) -> Observation:
        """Encode the current canonical knot into the graph observation."""
        knot: Optional[Knot] = self.state.get("knot")
        if knot is None:
            N = self.max_crossings
            return {"adj": np.zeros((N, N), dtype=np.float32), "node": np.zeros((N, 4), dtype=np.float32)}

        return knot.to_graph(self.max_crossings)

    def _graph_observation(self) -> Dict[str, np.ndarray]:
        """Return a graph-style observation (adjacency + node features)."""
        N = self.max_crossings
        adj = np.zeros((N, N), dtype=np.float32)
        node = np.zeros((N, 4), dtype=np.float32)
        return {"adj": adj, "node": node}

    def _sequence_observation(self) -> np.ndarray:
        """Return a sequence-style observation (Gauss code)."""
        L = self.max_crossings * 2
        return np.zeros((L,), dtype=np.int32)

    def _image_observation(self) -> np.ndarray:
        """Return an image-style observation (rasterized diagram)."""
        H, W = self.image_size
        return np.zeros((H, W, 1), dtype=np.float32)

    def _coords_observation(self) -> np.ndarray:
        """Return a coordinate-style observation (3D curve)."""
        N = self.max_crossings * 2
        return np.zeros((N, 3), dtype=np.float32)

    def _apply_action(self, action: int) -> None:
        """Apply an action to the canonical knot state.

        Actions are encoded as: (move_type, target_index).

        - move_type 0: Reidemeister I at target_index
        - move_type 1: Reidemeister II at target_index (paired with next crossing)
        - move_type 2: Reidemeister III at target_index (affects triple)
        """
        knot: Optional[Knot] = self.state.get("knot")
        if knot is None:
            return

        move_type = action // self.max_crossings
        target = action % self.max_crossings

        if move_type == 0:
            knot.reidemeister_I(target)
        elif move_type == 1:
            knot.reidemeister_II(target, (target + 1) % self.max_crossings)
        elif move_type == 2:
            knot.reidemeister_III(target, (target + 1) % self.max_crossings, (target + 2) % self.max_crossings)
        else:
            # Unknown action; no-op.
            return

        self.state["is_unknot"] = is_unknot(knot)

    def _compute_reward(self) -> float:
        """Compute a reward based on whether the knot is reduced to an unknot."""
        return 1.0 if self.state.get("is_unknot") else 0.0

    def _is_done(self) -> bool:
        """Return True when the episode should terminate."""
        return bool(self.state.get("is_unknot"))
