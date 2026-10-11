"""Scramble-length curriculum: start from short scrambles, lengthen them as the agent succeeds."""

from __future__ import annotations

from collections import deque

from .config import CurriculumConfig

__all__ = ["Curriculum"]


class Curriculum:
    """Training starts use ``lo..hi`` construction moves; ``hi`` grows by ``step`` (up to
    ``max_scramble``) once ``window`` episodes started at the current level succeed at a rate of
    at least ``promote_at``.  ``lo`` stays, so short problems keep appearing."""

    def __init__(self, cfg: CurriculumConfig, scramble: tuple) -> None:
        self.cfg = cfg
        self.lo, self.hi = int(scramble[0]), int(scramble[1])
        self.version = 0
        self.results: deque = deque(maxlen=max(1, cfg.window))

    @property
    def range(self) -> tuple[int, int]:
        return self.lo, self.hi

    @property
    def success_rate(self) -> float:
        return sum(self.results) / len(self.results) if self.results else float("nan")

    def update(self, episodes: list[dict]) -> bool:
        """Record finished episodes; returns whether the level went up."""
        for e in episodes:
            if e["version"] == self.version:
                self.results.append(e["solved"])
        if not self.cfg.enabled or self.hi >= self.cfg.max_scramble:
            return False
        if len(self.results) == self.results.maxlen and self.success_rate >= self.cfg.promote_at:
            self.hi = min(self.hi + self.cfg.step, self.cfg.max_scramble)
            self.version += 1
            self.results.clear()
            return True
        return False

    def state_dict(self) -> dict:
        return {"lo": self.lo, "hi": self.hi, "version": self.version, "results": list(self.results)}

    def load_state_dict(self, state: dict) -> None:
        self.lo, self.hi, self.version = state["lo"], state["hi"], state["version"]
        self.results.clear()
        self.results.extend(state["results"])
