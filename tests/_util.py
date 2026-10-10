"""Shared fixtures-as-functions for the test suite."""

from __future__ import annotations

import random

from knotml import ALL_KINDS, LinkDiagram

# KnotInfo / LinkInfo PD codes with their writhe in that diagram and determinant.
KNOTS = {
    "3_1": ([[1, 5, 2, 4], [3, 1, 4, 6], [5, 3, 6, 2]], 3, 3),
    "4_1": ([[4, 2, 5, 1], [8, 6, 1, 5], [6, 3, 7, 4], [2, 7, 3, 8]], 0, 5),
    "5_1": ([[2, 8, 3, 7], [4, 10, 5, 9], [6, 2, 7, 1], [8, 4, 9, 3], [10, 6, 1, 5]], 5, 5),
    "7_4": (
        [[2, 10, 3, 9], [4, 12, 5, 11], [6, 14, 7, 13], [8, 4, 9, 3], [10, 2, 11, 1], [12, 8, 13, 7], [14, 6, 1, 5]],
        7,
        15,
    ),
}
LINKS = {  # PD code, linking number, determinant
    "L2a1{0}": ([[4, 1, 3, 2], [2, 3, 1, 4]], -1, 2),
    "L2a1{1}": ([[4, 2, 3, 1], [2, 4, 1, 3]], 1, 2),
    "L4a1{0}": ([[6, 1, 7, 2], [8, 3, 5, 4], [2, 5, 3, 6], [4, 7, 1, 8]], -2, 4),
}


def random_braid_word(rng: random.Random, max_gen: int = 3, max_len: int = 7) -> list[int]:
    return [rng.choice((1, -1)) * rng.randint(1, max_gen) for _ in range(rng.randint(1, max_len))]


def random_diagram(rng: random.Random, steps: int, kinds=ALL_KINDS) -> LinkDiagram:
    d = LinkDiagram.from_braid(random_braid_word(rng))
    for _ in range(steps):
        d.apply_inplace(d.random_move(rng, kinds))
    return d


def permuted(d: LinkDiagram, perm: list[int]) -> LinkDiagram:
    """The same diagram with crossing ``c`` renumbered ``perm[c]``."""
    nbr = [0] * len(d.nbr)
    sign = [0] * d.n_crossings
    for c, s in enumerate(d.signs):
        sign[perm[c]] = s
        for k in range(4):
            q = d.nbr[4 * c + k]
            nbr[4 * perm[c] + k] = 4 * perm[q >> 2] + (q & 3)
    return LinkDiagram(nbr, sign, d.n_unknots)
