"""Measuring a policy: success rate on fresh scrambles, and recorded episodes for HTML step pages."""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import numpy as np
import torch

from ..notation import move_spec, touched_crossings
from ..play.session import components_of
from .config import Config
from .envs import make_env, make_vector_env
from .rollout import obs_to_torch

__all__ = ["evaluate", "record_episodes"]


@torch.no_grad()
def evaluate(policy, cfg: Config, device: torch.device, episodes: int = 64, num_envs: int = 8,
             scramble: Optional[tuple] = None, deterministic: bool = True, seed: int = 10_000) -> dict:
    """Run ``episodes`` episodes (each env runs the same number, so short episodes are not
    over-counted) and report success rate, mean length of solved episodes and mean return."""
    num_envs = max(1, min(num_envs, episodes))
    quota = math.ceil(episodes / num_envs)
    envs = make_vector_env(cfg, num_envs=num_envs, seed=seed, vectorization="sync", scramble=scramble)
    obs, _ = envs.reset(seed=seed)
    counts = np.zeros(num_envs, dtype=np.int64)
    returns, lengths = np.zeros(num_envs), np.zeros(num_envs, dtype=np.int64)
    results = []
    was_training = policy.training
    policy.eval()
    try:
        while (counts < quota).any():
            action = policy.act(obs_to_torch(obs, device), deterministic=deterministic)[0]
            obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
            returns += reward
            lengths += 1
            for i in np.flatnonzero(terminated | truncated):
                if counts[i] < quota:
                    results.append((bool(terminated[i]), int(lengths[i]), float(returns[i])))
                    counts[i] += 1
                returns[i], lengths[i] = 0.0, 0
    finally:
        envs.close()
        policy.train(was_training)
    solved = [r for r in results if r[0]]
    return {
        "success": len(solved) / len(results),
        "length_solved": float(np.mean([r[1] for r in solved])) if solved else float("nan"),
        "return": float(np.mean([r[2] for r in results])),
        "episodes": len(results),
    }


@torch.no_grad()
def record_episodes(policy, cfg: Config, device: torch.device, n: int = 3, scramble: Optional[tuple] = None,
                    deterministic: bool = True, seed: int = 20_000) -> list[tuple[list, bool]]:
    """``n`` episodes as ``(frames, solved)``; ``frames`` is the history format of
    :func:`~knotml.visualization.steps_page`."""
    gamma = float(cfg.algorithm.params.get("gamma", 0.99))
    env = make_env(dataclasses.asdict(cfg.env), dataclasses.asdict(cfg.reward), gamma, scramble)
    base = env.unwrapped
    was_training = policy.training
    policy.eval()
    out = []
    try:
        for ep in range(n):
            obs, info = env.reset(seed=seed + ep)
            scrambled = ", ".join(move_spec(m) for m in info.get("scramble", []))
            frames = [("start", base.kirby.diagram.copy(), set(), dict(components_of(base.kirby)),
                       f"scramble: {scrambled}" if scrambled else "")]
            while True:
                batch = obs_to_torch({k: np.asarray(v)[None] for k, v in obs.items()}, device)
                index = int(policy.act(batch, deterministic=deterministic)[0][0])
                move = base.decode(obs["candidates"][index]) if obs["action_mask"][index] else None
                d = base.kirby.diagram
                n_before = d.n_crossings
                touched = touched_crossings(d, move) if move is not None else set()
                obs, reward, terminated, truncated, info = env.step(index)
                marked = touched | set(range(n_before, base.kirby.diagram.n_crossings)) if info["legal"] else set()
                label = move_spec(move) if move is not None else "no-op"
                frames.append((label, base.kirby.diagram.copy(), marked, dict(components_of(base.kirby)),
                               f"reward {reward:+.3g}"))
                if terminated or truncated:
                    out.append((frames, bool(terminated)))
                    break
    finally:
        policy.train(was_training)
    return out
