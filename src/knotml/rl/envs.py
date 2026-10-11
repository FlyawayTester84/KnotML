"""Environments for training: :class:`KirbyEnv` + :class:`CandidateActions`, built from a config.

The agent sees ``Discrete(max_actions)``: action ``i`` is the ``i``-th legal
move listed in ``obs["candidates"]`` (padded with -1, ``obs["action_mask"]``
marks the real ones).  Vector environments autoreset in the same step
(``final_obs`` in the info), which keeps rollouts free of dummy reset steps.
"""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Optional

import gymnasium as gym
import numpy as np

from ..envs.kirby_env import CandidateActions, KirbyEnv, ScrambleStart
from ..envs.rewards import make_reward
from .config import Config

__all__ = ["ScrambleControl", "make_env", "make_vector_env", "EnvSpec", "env_spec", "OBS_KEYS"]

# Observation entries the policies read (``side_piece`` is implied by the faces of split diagrams).
OBS_KEYS = ("nbr", "sign", "strand_component", "side_face", "component_type", "framing", "crossingless", "linking",
            "n_crossings", "n_components", "candidates", "action_mask")


class ScrambleControl(gym.Wrapper):
    """Lets a vector environment change the scramble length of future starts:
    ``envs.call("set_scramble", lo, hi)``."""

    def set_scramble(self, lo: int, hi: int) -> None:
        start = self.env.unwrapped.start
        if isinstance(start, ScrambleStart):
            start.n_moves = (int(lo), int(hi))


def make_env(env: dict, reward: dict, gamma: float, scramble: Optional[tuple] = None) -> gym.Env:
    """One training environment from the ``env`` and ``reward`` config sections (as plain dicts, so it pickles)."""
    kwargs = dict(reward.get("kwargs") or {})
    if reward["name"] == "potential":
        kwargs.setdefault("gamma", gamma)
    base = KirbyEnv(
        calculus=env["calculus"],
        closed=env["closed"],
        max_crossings=env["max_crossings"],
        max_components=env["max_components"],
        max_steps=env["max_steps"],
        reward_fn=make_reward(reward["name"], **kwargs),
        start=ScrambleStart(n_moves=tuple(scramble or env["scramble"])),
    )
    return ScrambleControl(CandidateActions(base, max_actions=env["max_actions"]))


def _gamma(cfg: Config) -> float:
    return float(cfg.algorithm.params.get("gamma", 0.99))


def make_vector_env(cfg: Config, num_envs: Optional[int] = None, seed: Optional[int] = None,
                    vectorization: Optional[str] = None, scramble: Optional[tuple] = None) -> gym.vector.VectorEnv:
    """``num_envs`` copies (default ``cfg.env.num_envs``) in one process (``sync``) or one each (``async``)."""
    num_envs = num_envs or cfg.env.num_envs
    mode = vectorization or cfg.env.vectorization
    fn = partial(make_env, dataclasses.asdict(cfg.env), dataclasses.asdict(cfg.reward), _gamma(cfg), scramble)
    fns = [fn] * num_envs
    same_step = gym.vector.AutoresetMode.SAME_STEP
    if mode == "async" and num_envs > 1:
        envs = gym.vector.AsyncVectorEnv(fns, autoreset_mode=same_step)
    elif mode in ("sync", "async"):
        envs = gym.vector.SyncVectorEnv(fns, autoreset_mode=same_step)
    else:
        raise ValueError(f"env.vectorization must be sync or async, not {mode!r}")
    if seed is not None:
        envs.action_space.seed(seed)
    return envs


@dataclasses.dataclass(frozen=True)
class EnvSpec:
    """The sizes a policy is built for."""

    n_ports: int  # 4 * max_crossings
    n_components: int  # max_components
    n_actions: int  # max_actions
    kinds: tuple  # move kind names, in action order
    kind_fields: tuple  # per kind: (reads a, reads b, reads param)

    @property
    def n_kinds(self) -> int:
        return len(self.kinds)


def env_spec(cfg: Config) -> EnvSpec:
    env = make_env(dataclasses.asdict(cfg.env), dataclasses.asdict(cfg.reward), _gamma(cfg))
    base: KirbyEnv = env.unwrapped  # type: ignore[assignment]
    return EnvSpec(
        n_ports=4 * base.max_crossings,
        n_components=base.max_components,
        n_actions=cfg.env.max_actions,
        kinds=tuple(k.name for k in base.kinds),
        kind_fields=tuple(tuple(bool(x) for x in row) for row in base.action_fields()),
    )


def stack_obs(observations) -> dict:
    """Stack a list of single observations (dicts of arrays) into a batch."""
    return {k: np.stack([o[k] for o in observations]) for k in OBS_KEYS}
