"""Utilities for representing and manipulating knots."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Knot:
    """A simple knot representation.

    Uses PD codes to represent knots
    Each crossing is represented by a 4-tuple of strand indices.
    Other possibilities:
    """

    crossings: List[Tuple[int, int, int, int]]
    signs: List[int]

    def to_graph(self) -> dict:
        """
        
        draws a picture of the knot given its PD code

        """

        raise NotImplementedError

        return {"adj": np.array(adj, dtype=np.float32), "node": np.array(node, dtype=np.float32)}

    def reidemeister_I(self, crossing_idx: int) -> None:
        """
        R1 Move
        """
        raise NotImplementedError

    def reidemeister_II(self, idx_a: int, idx_b: int) -> None:
        """
        R2 Move
        """
        raise NotImplementedError

    def reidemeister_III(self, idx_a: int, idx_b: int, idx_c: int) -> None:
        """R3 Move"""
        raise NotImplementedError

def generate_random_knot(crossings: int = 5) -> Knot:
    """Generate a simple random knot.

    Currently this generates a circular sequence of crossings with random signs.
    """
    raise NotImplementedError
