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

    def __init__(self, array):
        """

        Initialize a knot diagram as a planar diagram code

        Input: An array of 4-tuples [(),(),(),]
        (subject to conditions in Mastin - PD Codes...)
        """
        raise NotImplementedError

    def to_graph(self) -> dict:
        """
        
        draws a picture of the knot given its PD code

        """

        raise NotImplementedError

        return {"adj": np.array(adj, dtype=np.float32), "node": np.array(node, dtype=np.float32)}

    def reidemeister_I(self, crossing_idx: int) -> None:
        """
        R1 Move

        Input: self, arc, crossing sign
        Output: PD code representing a link after the specified R-move
        """
        raise NotImplementedError

    def reidemeister_II(self, idx_a: int, idx_b: int) -> None:
        """
        R2 Move

        Input: self, idx_a: over arc, idx_b: under arc
        Output: (if arcs are compatible, ie share a face) PD code representing a link after the specified R-move
        """
        raise NotImplementedError

    def reidemeister_III(self, idx_a: int, idx_b: int, idx_c: int) -> None:
        """
        R3 Move

        Input: self, idx_a: arc, idx_b: arc, idx_c: arc
        Output: (if arcs satisfy succession condition c.f. Mastin) PD code representing a link after the specified R-move
        """
        raise NotImplementedError

def generate_random_knot(crossings: int = 5) -> Knot:
    """Generate a simple random knot.

    Currently this generates a circular sequence of crossings with random signs.
    """
    raise NotImplementedError
