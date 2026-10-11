"""Gymnasium environment: reduce a Kirby diagram to the empty diagram with Kirby moves.

The state is a :class:`~knotml.kirby.FramedLink` (``calculus="3-manifold"``:
Reidemeister moves, blow-ups/downs, handle slides) or a
:class:`~knotml.handlebody.Handlebody` (``calculus="4-manifold"``: Reidemeister
moves, handle slides, 1/2-handle and, if ``closed``, 2/3-handle creation and
cancellation).  An episode terminates when no components are left: the empty
diagram, i.e. S³ (3-manifold) or B⁴ (4-manifold; S⁴ if closed).

Actions
-------
An action is ``(kind, a, b, param)`` (``MultiDiscrete``):

* ``kind`` indexes :attr:`KirbyEnv.kinds`;
* ``a``, ``b`` are *anchors*: ``0 <= i < 4 * max_crossings`` is a port (crossing
  ``i // 4``, slot ``i % 4``; it also names a side of an edge), and
  ``4 * max_crossings + k`` is the component in slot ``k`` (components are
  numbered in label order);
* ``param`` is ``0, 1, 2`` for ``-1, 0, +1``: the sign of ``R1_ADD`` and
  ``BLOW_UP``, the framing of ``CREATE_12``.

Fields a kind does not use are ignored.  An action that does not decode to a
legal move, or that would exceed ``max_crossings`` / ``max_components``,
leaves the state unchanged and is reported with ``info["legal"] = False``.
:meth:`KirbyEnv.legal_actions` lists exactly the legal actions (O(n + output));
:class:`CandidateActions` turns that list into a ``Discrete`` space with
``action_masks()`` for masked policy-gradient methods.

Reward
------
Left open: ``reward_fn(transition) -> float`` receives a :class:`Transition`
(state before, decoded move, state after, legality, termination).  The default
:func:`zero_reward` returns 0.

Start states
------------
``reset(options={"diagram": d})`` starts from a given framed link or handlebody.
Otherwise ``start(env, rng)`` builds one; the default :class:`ScrambleStart`
applies random construction moves to the empty diagram, so every start can be
reduced to the empty diagram.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Sequence, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core import Move, MoveKind
from ..handlebody import Handlebody
from ..kirby import AnyMove, FramedLink, KirbyKind, KirbyMove, linking_matrix

__all__ = ["KirbyEnv", "CandidateActions", "ScrambleStart", "Transition", "zero_reward"]

Kirby = Union[FramedLink, Handlebody]
_PARAM = (-1, 0, 1)
_INT_MIN, _INT_MAX = np.iinfo(np.int32).min, np.iinfo(np.int32).max
_REIDEMEISTER = (MoveKind.R1_REMOVE, MoveKind.R1_ADD, MoveKind.R2_REMOVE, MoveKind.R2_ADD, MoveKind.R3)
_SIMPLIFYING = frozenset(
    {MoveKind.R1_REMOVE, MoveKind.R2_REMOVE, MoveKind.R3, KirbyKind.BLOW_DOWN, KirbyKind.CANCEL_12, KirbyKind.CANCEL_23}
)


class Transition(NamedTuple):
    """What a reward function sees after one step."""

    before: Kirby  # a copy of the state before the step
    move: Optional[AnyMove]  # the decoded move, None if the action did not decode
    after: Kirby  # the live state after the step (copy it to keep it)
    legal: bool  # whether the move was applied
    terminated: bool  # whether ``after`` is the empty diagram


RewardFn = Callable[[Transition], float]


def zero_reward(transition: Transition) -> float:
    """Placeholder reward (the reward function is left open)."""
    return 0.0


class KirbyEnv(gym.Env):
    """Reduce a Kirby diagram to the empty diagram (see the module docstring)."""

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(
        self,
        calculus: str = "4-manifold",
        closed: bool = False,
        max_crossings: int = 64,
        max_components: int = 8,
        max_steps: int = 200,
        reward_fn: RewardFn = zero_reward,
        start: Optional[Callable[[KirbyEnv, np.random.Generator], Kirby]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        if calculus not in ("3-manifold", "4-manifold"):
            raise ValueError("calculus must be '3-manifold' or '4-manifold'")
        self.calculus, self.closed = calculus, closed
        self.max_crossings, self.max_components, self.max_steps = max_crossings, max_components, max_steps
        self.reward_fn = reward_fn
        self.start = start if start is not None else ScrambleStart()
        self.render_mode = render_mode
        if calculus == "3-manifold":
            kirby_kinds = (KirbyKind.HANDLE_SLIDE, KirbyKind.BLOW_UP, KirbyKind.BLOW_DOWN)
        else:
            kirby_kinds = (KirbyKind.HANDLE_SLIDE, KirbyKind.CANCEL_12, KirbyKind.CREATE_12)
            if closed:
                kirby_kinds += (KirbyKind.CANCEL_23, KirbyKind.CREATE_23)
        self.kinds: tuple = _REIDEMEISTER + kirby_kinds
        self._kind_index = {k: i for i, k in enumerate(self.kinds)}
        P, N, C = 4 * max_crossings, max_crossings, max_components
        self.n_anchors = P + C
        self.action_space = spaces.MultiDiscrete([len(self.kinds), self.n_anchors, self.n_anchors, 3])
        box = lambda low, high, shape: spaces.Box(low, high, shape, dtype=np.int64)  # noqa: E731
        self.observation_space = spaces.Dict(
            {
                "nbr": box(-1, P - 1, (P,)),  # port at the other end of each port's edge
                "sign": box(-1, 1, (N,)),  # crossing signs
                "strand_component": box(-1, C - 1, (2 * N,)),  # component slot of the under/over strand
                "side_face": box(-1, P - 1, (P,)),  # face on the side named by each port
                "side_piece": box(-1, N - 1, (P,)),  # connected piece of the projection
                "component_type": box(0, 2, (C,)),  # 0 absent, 1 framed (2-handle), 2 dotted (1-handle)
                "framing": box(_INT_MIN, _INT_MAX, (C,)),
                "crossingless": box(0, 1, (C,)),
                "linking": box(_INT_MIN, _INT_MAX, (C, C)),  # linking matrix, framings on the diagonal
                "n_crossings": box(0, N, (1,)),
                "n_components": box(0, C, (1,)),
            }
        )
        self._obj: Optional[Kirby] = None
        self._steps = 0
        self._cost_cache: tuple = (None, {})

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        info: dict[str, Any] = {}
        if options and "diagram" in options:
            obj = options["diagram"].copy()
            self._check_kind(obj)
        else:
            obj = self.start(self, self.np_random)
            info["scramble"] = list(getattr(self.start, "last_moves", []))
        self._check_capacity(obj)
        self._obj, self._steps = obj, 0
        return self._observation(), {**info, **self._info(None, True)}

    def restore(self, state: Kirby, steps: int = 0) -> dict[str, np.ndarray]:
        """Continue the episode from a copy of ``state`` with ``steps`` steps already taken.

        For undo and tree search: unlike :meth:`reset`, the random generator is
        left alone.  Returns the observation.
        """
        obj = state.copy()
        self._check_kind(obj)
        self._check_capacity(obj)
        self._obj, self._steps = obj, steps
        return self._observation()

    def step(self, action):
        if self._obj is None:
            raise RuntimeError("call reset() first")
        self._steps += 1
        move = self.decode(action)
        legal = move is not None and self.is_legal(move)
        before = self._obj.copy()
        if legal:
            self._obj.apply_inplace(move, check=False)
        terminated = self._obj.is_empty()
        truncated = not terminated and self._steps >= self.max_steps
        reward = float(self.reward_fn(Transition(before, move, self._obj, legal, terminated)))
        return self._observation(), reward, terminated, truncated, self._info(move, legal)

    def render(self):
        if self._obj is None:
            return None
        text = f"{self._obj!r}\nPD: {self._obj.diagram.to_pd()}"
        if self.render_mode == "human":
            print(text)
            return None
        return text

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    @property
    def kirby(self) -> Kirby:
        """The current framed link / handlebody (live object: copy it before changing it)."""
        return self._obj

    @property
    def steps(self) -> int:
        """Steps taken in this episode (truncation comes at ``max_steps``)."""
        return self._steps

    def decode(self, action: Sequence[int]) -> Optional[AnyMove]:
        """The move an action stands for, or ``None`` if it refers to nothing that exists."""
        k, a, b, p = (int(v) for v in action)
        if not (0 <= k < len(self.kinds) and 0 <= p < 3):
            return None
        kind, param = self.kinds[k], _PARAM[p]
        if kind in (MoveKind.R1_REMOVE, MoveKind.R2_REMOVE, MoveKind.R3):
            port = self._port(a)
            return None if port is None else Move(kind, port)
        if kind == MoveKind.R1_ADD:
            anchor = self._side_or_circle(a)
            return None if anchor is None else Move(kind, anchor, sign=param)
        if kind == MoveKind.R2_ADD:
            x, y = self._side_or_circle(a), self._side_or_circle(b)
            return None if x is None or y is None else Move(kind, x, y)
        if kind == KirbyKind.HANDLE_SLIDE:
            x, y = self._port(a), self._port(b)
            return None if x is None or y is None else KirbyMove(kind, x, y)
        if kind in (KirbyKind.BLOW_UP, KirbyKind.CREATE_12):
            return KirbyMove(kind, sign=param)
        if kind == KirbyKind.CREATE_23:
            return KirbyMove(kind)
        label = self._component(a)  # BLOW_DOWN, CANCEL_12, CANCEL_23
        return None if label is None else KirbyMove(kind, a=label)

    def encode(self, move: AnyMove) -> np.ndarray:
        """The canonical action vector of a move (unused fields are 0)."""
        kind = move.kind
        P = 4 * self.max_crossings
        slot = {lab: i for i, lab in enumerate(self._obj.labels())}
        trivial = self._obj.diagram.trivial_labels

        def anchor(x: int) -> int:  # a port, or a crossingless component's negative anchor
            return x if x >= 0 else P + slot[trivial[-1 - x]]

        a = b = p = 0
        if kind in (MoveKind.R1_REMOVE, MoveKind.R2_REMOVE, MoveKind.R3, KirbyKind.HANDLE_SLIDE, MoveKind.R2_ADD):
            a = anchor(move.a)
            if kind in (MoveKind.R2_ADD, KirbyKind.HANDLE_SLIDE):
                b = anchor(move.b)
        elif kind == MoveKind.R1_ADD:
            a, p = anchor(move.a), _PARAM.index(move.sign)
        elif kind in (KirbyKind.BLOW_UP, KirbyKind.CREATE_12):
            p = _PARAM.index(move.sign)
        elif kind in (KirbyKind.BLOW_DOWN, KirbyKind.CANCEL_12, KirbyKind.CANCEL_23):
            a = P + slot[move.a]
        return np.array([self._kind_index[kind], a, b, p], dtype=np.int64)

    def is_legal(self, move: AnyMove) -> bool:
        """Legal in the calculus and within the capacity: O(1) (after an O(n) pass per state)."""
        return move.kind in self._kind_index and self._obj.is_legal(move) and self._fits(move)

    def legal_moves(self) -> list:
        """Every legal move within the capacity, each once: O(n + output)."""
        return [m for m in self._obj.iter_legal_moves(self.kinds) if self._fits(m)]

    def legal_actions(self) -> np.ndarray:
        """Canonical action vectors of :meth:`legal_moves`, shape ``(L, 4)``."""
        moves = self.legal_moves()
        if not moves:
            return np.zeros((0, 4), dtype=np.int64)
        return np.stack([self.encode(m) for m in moves])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_kind(self, obj: Kirby) -> None:
        want = Handlebody if self.calculus == "4-manifold" else FramedLink
        if not isinstance(obj, want):
            raise TypeError(f"{self.calculus} calculus needs a {want.__name__}, got {type(obj).__name__}")
        if isinstance(obj, Handlebody) and obj.closed != self.closed:
            raise ValueError(f"handlebody has closed={obj.closed}, environment has closed={self.closed}")

    def _check_capacity(self, obj: Kirby) -> None:
        if obj.diagram.n_crossings > self.max_crossings or len(obj.labels()) > self.max_components:
            raise ValueError(f"start diagram exceeds the capacity ({self.max_crossings} crossings, "
                             f"{self.max_components} components)")

    def _port(self, i: int) -> Optional[int]:
        return i if 0 <= i < 4 * self._obj.diagram.n_crossings else None

    def _component(self, i: int) -> Optional[int]:
        labels = self._obj.labels()
        k = i - 4 * self.max_crossings
        return labels[k] if 0 <= k < len(labels) else None

    def _side_or_circle(self, i: int) -> Optional[int]:
        """A port, or the negative anchor of a crossingless component slot."""
        if i < 4 * self.max_crossings:
            return self._port(i)
        label = self._component(i)
        trivial = self._obj.diagram.trivial_labels
        return -1 - list(trivial).index(label) if label in trivial else None

    def _fits(self, move: AnyMove) -> bool:
        """Whether the result stays within ``max_crossings`` and ``max_components``."""
        kind = move.kind
        d = self._obj.diagram
        grow, new = 0, 0
        if kind == MoveKind.R1_ADD:
            grow = 1
        elif kind == MoveKind.R2_ADD:
            grow = 2
        elif kind == KirbyKind.HANDLE_SLIDE:
            grow = self._slide_cost(d.label_of_port(move.b))
        elif kind == KirbyKind.CREATE_12:
            grow, new = 2, 2
        elif kind in (KirbyKind.BLOW_UP, KirbyKind.CREATE_23):
            new = 1
        return d.n_crossings + grow <= self.max_crossings and len(self._obj.labels()) + new <= self.max_components

    def _slide_cost(self, label: int) -> int:
        """Crossings a handle slide over this component adds (exact; see ``kirby.band_slide``)."""
        d = self._obj.diagram
        stamp, costs = self._cost_cache
        if stamp != (d.stamp, id(self._obj)):
            writhe: dict[int, int] = {}
            costs = {}
            lab = d._label
            for c, s in enumerate(d.signs):
                u, o = lab[2 * c], lab[2 * c + 1]
                if u == o:
                    costs[u] = costs.get(u, 0) + 3
                    writhe[u] = writhe.get(u, 0) + s
                else:
                    costs[u] = costs.get(u, 0) + 1
                    costs[o] = costs.get(o, 0) + 1
            for lab_ in self._obj.labels():
                framing = 0 if isinstance(self._obj, Handlebody) and self._obj.is_dotted(lab_) else self._obj.framing(lab_)
                costs[lab_] = costs.get(lab_, 0) + 2 * abs(framing - writhe.get(lab_, 0))
            self._cost_cache = ((d.stamp, id(self._obj)), costs)
        return costs.get(label, 0)

    def _observation(self) -> dict[str, np.ndarray]:
        obj = self._obj
        d = obj.diagram
        n, N, C = d.n_crossings, self.max_crossings, self.max_components
        P = 4 * N
        labels = obj.labels()
        slot = {lab: i for i, lab in enumerate(labels)}
        full = lambda size, fill=-1: np.full(size, fill, dtype=np.int64)  # noqa: E731
        nbr, sign, strand, face, piece = full(P), full(N, 0), full(2 * N), full(P), full(P)
        nbr[: 4 * n] = d.nbr
        sign[:n] = d.signs
        strand[: 2 * n] = [slot[lab] for lab in d._label]
        face[: 4 * n] = d.face_of_port()
        dcomp = d.diagram_component_of_crossing()
        piece[: 4 * n] = np.repeat(np.asarray(dcomp, dtype=np.int64), 4) if n else []
        ctype, framing, crossingless = full(C, 0), full(C, 0), full(C, 0)
        trivial = set(d.trivial_labels)
        diagonal = {}
        for k, lab in enumerate(labels):
            dotted = isinstance(obj, Handlebody) and obj.is_dotted(lab)
            ctype[k] = 2 if dotted else 1
            framing[k] = 0 if dotted else obj.framing(lab)
            crossingless[k] = lab in trivial
            diagonal[lab] = framing[k]
        linking = np.zeros((C, C), dtype=np.int64)
        if labels:
            linking[: len(labels), : len(labels)] = linking_matrix(d, labels, diagonal)
        return {
            "nbr": nbr,
            "sign": sign,
            "strand_component": strand,
            "side_face": face,
            "side_piece": piece,
            "component_type": ctype,
            "framing": framing,
            "crossingless": crossingless,
            "linking": linking,
            "n_crossings": np.array([n], dtype=np.int64),
            "n_components": np.array([len(labels)], dtype=np.int64),
        }

    def _info(self, move: Optional[AnyMove], legal: bool) -> dict[str, Any]:
        return {
            "legal": legal,
            "move": move,
            "n_crossings": self._obj.diagram.n_crossings,
            "n_components": len(self._obj.labels()),
            "steps": self._steps,
        }


class ScrambleStart:
    """Start states built from the empty diagram by random construction moves.

    Each step picks a move kind uniformly among those with a legal move (so
    rare kinds are not drowned out by R2_ADD), then a move of that kind
    uniformly.  Every start can therefore be reduced back to the empty diagram.
    ``last_moves`` records the moves used (also in ``reset``'s ``info["scramble"]``).
    """

    def __init__(self, n_moves: tuple[int, int] = (2, 8), max_crossings: Optional[int] = None,
                 kinds: Optional[Sequence] = None) -> None:
        self.n_moves = n_moves
        self.max_crossings = max_crossings
        self.kinds = kinds
        self.last_moves: list = []

    def __call__(self, env: KirbyEnv, rng: np.random.Generator) -> Kirby:
        if env.calculus == "4-manifold":
            obj: Kirby = Handlebody.empty(closed=env.closed)
            default = (KirbyKind.CREATE_12, MoveKind.R1_ADD, MoveKind.R2_ADD, MoveKind.R3, KirbyKind.HANDLE_SLIDE)
            if env.closed:
                default += (KirbyKind.CREATE_23,)
        else:
            from ..core import LinkDiagram

            obj = FramedLink(LinkDiagram.unknot(0), {})
            default = (KirbyKind.BLOW_UP, MoveKind.R1_ADD, MoveKind.R2_ADD, MoveKind.R3, KirbyKind.HANDLE_SLIDE)
        kinds = tuple(self.kinds) if self.kinds is not None else default
        budget = min(self.max_crossings or env.max_crossings, env.max_crossings)
        self.last_moves = []
        saved = env._obj
        env._obj = obj  # reuse the environment's capacity check
        try:
            for _ in range(int(rng.integers(self.n_moves[0], self.n_moves[1] + 1))):
                by_kind: dict = {}
                for m in obj.iter_legal_moves(kinds):
                    if env._fits(m) and (m.kind in (KirbyKind.BLOW_UP, KirbyKind.CREATE_23) or
                                         obj.diagram.n_crossings + _growth(env, m) <= budget):
                        by_kind.setdefault(m.kind, []).append(m)
                if not by_kind:
                    break
                options = by_kind[sorted(by_kind)[int(rng.integers(len(by_kind)))]]
                move = options[int(rng.integers(len(options)))]
                obj.apply_inplace(move, check=False)
                self.last_moves.append(move)
        finally:
            env._obj = saved
        return obj


def _growth(env: KirbyEnv, move: AnyMove) -> int:
    kind = move.kind
    if kind == MoveKind.R1_ADD:
        return 1
    if kind in (MoveKind.R2_ADD, KirbyKind.CREATE_12):
        return 2
    if kind == KirbyKind.HANDLE_SLIDE:
        return env._slide_cost(env._obj.diagram.label_of_port(move.b))
    return 0


class CandidateActions(gym.Wrapper):
    """``Discrete(max_actions)`` over the current legal moves, with ``action_masks()``.

    Action ``i`` is the ``i``-th legal move (``candidates[i]`` in the observation,
    in :meth:`KirbyEnv.encode` form; -1 rows are padding).  When more than
    ``max_actions`` moves are legal, all simplifying moves (removals, R3,
    cancellations, blow-downs) are kept and the rest are sampled with the
    environment's random generator.  An index without a candidate is a no-op.
    """

    def __init__(self, env: gym.Env, max_actions: int = 1024) -> None:
        super().__init__(env)
        base: KirbyEnv = env.unwrapped  # type: ignore[assignment]
        self.max_actions = max_actions
        self.action_space = spaces.Discrete(max_actions)
        self.observation_space = spaces.Dict(
            {
                **base.observation_space.spaces,
                "candidates": spaces.Box(-1, base.n_anchors - 1, (max_actions, 4), dtype=np.int64),
                "action_mask": spaces.Box(0, 1, (max_actions,), dtype=np.int8),
            }
        )
        self._candidates = np.zeros((0, 4), dtype=np.int64)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs), info

    def step(self, action):
        i = int(action)
        if i < len(self._candidates):
            vector = self._candidates[i]
        else:  # no candidate: an action that decodes to nothing (an R1_REMOVE at a component slot)
            base: KirbyEnv = self.env.unwrapped  # type: ignore[assignment]
            vector = np.array([base.kinds.index(MoveKind.R1_REMOVE), base.n_anchors - 1, 0, 0], dtype=np.int64)
        obs, reward, terminated, truncated, info = self.env.step(vector)
        return self._augment(obs), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.max_actions, dtype=bool)
        mask[: len(self._candidates)] = True
        return mask

    def _augment(self, obs: dict) -> dict:
        base: KirbyEnv = self.env.unwrapped  # type: ignore[assignment]
        moves = base.legal_moves()
        if len(moves) > self.max_actions:
            keep = [m for m in moves if m.kind in _SIMPLIFYING][: self.max_actions]
            rest = [m for m in moves if m.kind not in _SIMPLIFYING]
            pick = base.np_random.choice(len(rest), size=self.max_actions - len(keep), replace=False)
            moves = keep + [rest[int(j)] for j in sorted(pick)]
        self._candidates = np.stack([base.encode(m) for m in moves]) if moves else np.zeros((0, 4), dtype=np.int64)
        padded = np.full((self.max_actions, 4), -1, dtype=np.int64)
        padded[: len(self._candidates)] = self._candidates
        mask = np.zeros(self.max_actions, dtype=np.int8)
        mask[: len(self._candidates)] = 1
        return {**obs, "candidates": padded, "action_mask": mask}
