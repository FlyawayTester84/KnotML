"""A :class:`~knotml.envs.KirbyEnv` episode driven by hand: steps, undo, previews and the record."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from ..envs.kirby_env import Kirby, KirbyEnv, Transition
from ..envs.rewards import example_reward, make_reward
from ..handlebody import Handlebody
from ..kirby import AnyMove
from ..notation import move_spec, touched_crossings

__all__ = ["PlaySession", "Record", "components_of", "pd_summary", "example_reward", "load_reward"]


@dataclass
class Record:
    """One ``env.step`` taken in the session."""

    action: np.ndarray
    move: Optional[AnyMove]  # the decoded move (None if the action decodes to nothing)
    legal: bool
    reward: float
    terminated: bool
    truncated: bool
    marked: set = field(default_factory=set)  # crossings created or rearranged, circled on the drawing

    @property
    def label(self) -> str:
        text = move_spec(self.move) if self.move is not None else f"action {self.action.tolist()}"
        return text if self.legal else f"{text} (illegal)"


def components_of(obj: Kirby) -> dict:
    """``{label: framing}``, with ``None`` for a dotted circle."""
    return obj.components if isinstance(obj, Handlebody) else obj.framings


def pd_summary(obj: Kirby) -> dict:
    """The PD code of ``obj`` and what is needed to enter it again with ``--pd``.

    * ``pd``: :meth:`LinkDiagram.to_pd` (edges numbered 1, 2, ... along each component).
    * ``labels`` and ``framings``: the components in the order ``--pd`` reads them back,
      which is ``--framings`` order (``"dot"`` for a dotted circle).
    * ``circles``: ``{label: framing}`` of the crossingless components, which a PD
      code cannot hold.
    * ``args``: ``--pd ... --framings ...`` (plus ``--four``/``--closed``), or
      ``None`` when there are crossingless components or no crossings.
    """
    import shlex

    d, comps = obj.diagram, components_of(obj)
    labels = d.component_labels()
    framings = ["dot" if comps[lab] is None else str(comps[lab]) for lab in labels]
    circles = {lab: comps[lab] for lab in d.trivial_labels}
    args = None
    if labels and not circles:
        pd = str([list(q) for q in d.to_pd()]).replace(" ", "")
        args = f"--pd {shlex.quote(pd)} --framings {' '.join(framings)}"
        if isinstance(obj, Handlebody):
            args += " --closed" if obj.closed else (" --four" if "dot" not in framings else "")
    return {"pd": d.to_pd(), "labels": labels, "framings": framings, "circles": circles, "args": args}


class PlaySession:
    """An episode of ``env`` with undo.

    ``start`` fixes the start diagram (otherwise the env's ``start`` builds
    one); ``seed`` seeds the first reset and the session's own random moves.
    ``states[i]`` is the state after ``records[:i]``.
    """

    def __init__(self, env: KirbyEnv, start: Optional[Kirby] = None, seed: Optional[int] = None) -> None:
        self.env, self.start = env, start
        self.rng = random.Random(seed)
        self.reset(seed)

    # ------------------------------------------------------------------
    # Episode
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> None:
        """A new episode: the given start diagram, or a fresh one from the env's start distribution."""
        options = {"diagram": self.start} if self.start is not None else None
        self.obs, info = self.env.reset(seed=seed, options=options)
        self.seed = seed
        self.scramble: list = info.get("scramble", [])
        self.records: list[Record] = []
        self.states: list[Kirby] = [self.env.kirby.copy()]
        self.preview: Optional[tuple] = None  # (move, state after or None if illegal, marked crossings)
        self._moves: Optional[tuple] = None

    @property
    def state(self) -> Kirby:
        return self.env.kirby

    @property
    def done(self) -> bool:
        return bool(self.records) and (self.records[-1].terminated or self.records[-1].truncated)

    @property
    def total_return(self) -> float:
        return sum(r.reward for r in self.records)

    def moves(self) -> list:
        """The legal moves, in the order the page numbers them (cached per state)."""
        key = (self.state.diagram.stamp, id(self.state), len(self.records))
        if self._moves is None or self._moves[0] != key:
            self._moves = (key, self.env.legal_moves())
        return self._moves[1]

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def action_for(self, move: AnyMove) -> Optional[np.ndarray]:
        """The action vector of a (possibly illegal) move, or ``None`` if the env cannot express it."""
        try:
            vector = self.env.encode(move)
        except (KeyError, IndexError, ValueError):  # kind outside the calculus, or an anchor that does not exist
            return None
        return vector if self.env.action_space.contains(vector) else None

    def step(self, action: Sequence[int]) -> Record:
        """``env.step(action)``; illegal actions count as steps, as they do for an agent."""
        if self.done:
            raise RuntimeError("the episode is over: undo or reset")
        action = np.asarray(action, dtype=np.int64)
        before = self.state.diagram
        move = self.env.decode(action)
        legal = move is not None and self.env.is_legal(move)
        n_before = before.n_crossings
        touched = touched_crossings(before, move) if legal else set()
        self.obs, reward, terminated, truncated, _ = self.env.step(action)
        n_after = self.state.diagram.n_crossings
        marked = touched | set(range(n_before, n_after)) if legal else set()
        record = Record(action, move, legal, float(reward), bool(terminated), bool(truncated), marked)
        self.records.append(record)
        self.states.append(self.state.copy())
        self.preview = None
        return record

    def step_move(self, move: AnyMove) -> Record:
        vector = self.action_for(move)
        if vector is None:
            raise ValueError(f"{move_spec(move)} is not an action of this environment (see `moves`)")
        return self.step(vector)

    def step_index(self, i: int) -> Record:
        moves = self.moves()
        if not 0 <= i < len(moves):
            raise ValueError(f"no legal move {i} (there are {len(moves)})")
        return self.step(self.env.encode(moves[i]))

    def random_step(self, kinds=None) -> Optional[Record]:
        """A uniformly random legal move (of the given kinds); ``None`` if there is none."""
        moves = [m for m in self.moves() if kinds is None or m.kind in kinds]
        return self.step(self.env.encode(self.rng.choice(moves))) if moves else None

    def undo(self, n: int = 1) -> int:
        """Take back up to ``n`` steps; returns how many were taken back."""
        n = min(n, len(self.records))
        if n:
            del self.records[-n:]
            del self.states[-n:]
            self.obs = self.env.restore(self.states[-1], len(self.records))
            self.preview = None
        return n

    def preview_move(self, move: Optional[AnyMove]) -> None:
        """Show what ``move`` would do (``None`` clears the preview); the state is not changed."""
        if move is None:
            self.preview = None
            return
        if not self.env.is_legal(move):
            self.preview = (move, None, set())
            return
        d = self.state.diagram
        touched = touched_crossings(d, move)
        after = self.state.apply(move)
        self.preview = (move, after, touched | set(range(d.n_crossings, after.diagram.n_crossings)))

    def history(self) -> list:
        """The episode as ``(label, diagram, marked, components, status)`` frames for ``steps_page``."""
        frames = [("start", self.states[0].diagram, set(), components_of(self.states[0]), "")]
        for record, state in zip(self.records, self.states[1:], strict=True):
            status = f"reward {record.reward:+.3g}" + (", terminated" if record.terminated else "")
            frames.append((record.label, state.diagram, record.marked, components_of(state), status))
        return frames


def load_reward(name: str) -> Callable[[Transition], float]:
    """A reward by name: ``zero``, ``sparse``, ``potential``, ``example`` or ``module:function``
    (see :mod:`knotml.envs.rewards`)."""
    return make_reward(name)
