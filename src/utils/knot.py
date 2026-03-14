"""Utilities for representing and manipulating knots."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Knot:
    """A simple knot representation.

    This approximates a planar knot diagram using a list of crossings.
    Each crossing is represented by a 4-tuple of strand indices.

    The structure below is intentionally lightweight for early prototyping.
    """

    crossings: List[Tuple[int, int, int, int]]
    signs: List[int]

    def to_graph(self, max_crossings: int) -> dict:
        """Return a graph-style observation (adjacency + node features)."""
        N = max_crossings
        adj = [[0.0] * N for _ in range(N)]
        node = [[0.0] * 4 for _ in range(N)]

        for i, (a, b, c, d) in enumerate(self.crossings):
            if i >= N:
                break
            # Simple structural topology: connect this crossing to the next/previous.
            adj[i][(i + 1) % N] = 1.0
            adj[i][(i - 1) % N] = 1.0

            # Node features: (sign, has_crossing, dummy, dummy)
            node[i][0] = float(self.signs[i])
            node[i][1] = 1.0

        return {"adj": np.array(adj, dtype=np.float32), "node": np.array(node, dtype=np.float32)}

    def reidemeister_I(self, crossing_idx: int) -> None:
        """Perform a (simplified) Reidemeister I move by removing a crossing."""
        if 0 <= crossing_idx < len(self.crossings):
            self.crossings.pop(crossing_idx)
            self.signs.pop(crossing_idx)

    def reidemeister_II(self, idx_a: int, idx_b: int) -> None:
        """Perform a (simplified) Reidemeister II move by removing two crossings."""
        to_remove = sorted({idx_a, idx_b}, reverse=True)
        for idx in to_remove:
            if 0 <= idx < len(self.crossings):
                self.crossings.pop(idx)
                self.signs.pop(idx)

    def reidemeister_III(self, idx_a: int, idx_b: int, idx_c: int) -> None:
        """Perform a (simplified) Reidemeister III move by permuting crossings."""
        indices = [idx_a, idx_b, idx_c]
        valid = [i for i in indices if 0 <= i < len(self.crossings)]
        if len(valid) < 2:
            return
        # Swap two crossings as a proxy for the real move.
        i, j = valid[0], valid[1]
        self.crossings[i], self.crossings[j] = self.crossings[j], self.crossings[i]
        self.signs[i], self.signs[j] = self.signs[j], self.signs[i]


def generate_random_knot(crossings: int = 5) -> Knot:
    """Generate a simple random knot.

    Currently this generates a circular sequence of crossings with random signs.
    """
    if crossings < 0:
        crossings = 0

    # Each crossing is represented as (a, b, c, d) where indices are arbitrary.
    knot_crossings = [((i * 4) % (crossings * 4 + 1), (i * 4 + 1) % (crossings * 4 + 1), (i * 4 + 2) % (crossings * 4 + 1), (i * 4 + 3) % (crossings * 4 + 1)) for i in range(crossings)]
    signs = [random.choice([-1, 1]) for _ in range(crossings)]
    return Knot(crossings=knot_crossings, signs=signs)


def is_unknot(knot: Knot) -> bool:
    """Determine whether a knot is an unknot.

    This is a placeholder that treats a knot with zero crossings as the unknot.
    """
    return len(knot.crossings) == 0
