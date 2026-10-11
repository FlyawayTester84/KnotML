"""Collecting experience from a vector environment: rollout storage, GAE and episode statistics."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch

from .envs import OBS_KEYS, stack_obs

__all__ = ["obs_to_torch", "EpisodeTracker", "RolloutBuffer", "Collector"]

_I32 = np.iinfo(np.int32)


def obs_to_torch(obs: dict, device: torch.device) -> dict:
    """Observation arrays -> int32 tensors on ``device`` (policies cast indices to int64 themselves)."""
    out = {}
    for k in OBS_KEYS:
        a = np.asarray(obs[k])
        if a.dtype != np.int32:
            a = np.clip(a, _I32.min, _I32.max).astype(np.int32)
        out[k] = torch.from_numpy(a).to(device, non_blocking=True)
    return out


class EpisodeTracker:
    """Per-env return and length; finished episodes are kept until :meth:`pop`.

    ``version`` is the curriculum version an episode started under, so the
    curriculum can count only episodes from its current level.
    """

    def __init__(self, num_envs: int) -> None:
        self.returns = np.zeros(num_envs)
        self.lengths = np.zeros(num_envs, dtype=np.int64)
        self.version = np.zeros(num_envs, dtype=np.int64)
        self.finished: list[dict] = []

    def step(self, reward: np.ndarray, terminated: np.ndarray, truncated: np.ndarray, version: int) -> None:
        self.returns += reward
        self.lengths += 1
        for i in np.flatnonzero(terminated | truncated):
            self.finished.append({"return": float(self.returns[i]), "length": int(self.lengths[i]),
                                  "solved": bool(terminated[i]), "version": int(self.version[i])})
            self.returns[i], self.lengths[i], self.version[i] = 0.0, 0, version  # the env has already reset

    def pop(self) -> list[dict]:
        out, self.finished = self.finished, []
        return out


class RolloutBuffer:
    """``num_steps x num_envs`` transitions on ``device``, with GAE advantages and returns."""

    def __init__(self, num_steps: int, num_envs: int, obs_shapes: dict, device: torch.device) -> None:
        self.T, self.N, self.device = num_steps, num_envs, device
        self.obs = {k: torch.zeros((num_steps, num_envs, *shape), dtype=torch.int32, device=device)
                    for k, shape in obs_shapes.items()}
        z = lambda dtype=torch.float32: torch.zeros((num_steps, num_envs), dtype=dtype, device=device)  # noqa: E731
        self.actions, self.logprobs, self.values = z(torch.long), z(), z()
        self.rewards, self.ends = z(), z()
        self.advantages, self.returns = z(), z()

    def add(self, t: int, obs: dict, action, logprob, value, reward, ends) -> None:
        for k, v in obs.items():
            self.obs[k][t] = v
        self.actions[t], self.logprobs[t], self.values[t] = action, logprob, value
        self.rewards[t] = torch.as_tensor(reward, device=self.device)
        self.ends[t] = torch.as_tensor(ends, dtype=torch.float32, device=self.device)

    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float) -> None:
        """Generalised advantage estimation.  ``ends[t]`` marks that step ``t`` finished an episode;
        truncated episodes already carry ``gamma * V(final obs)`` in their reward."""
        gae = torch.zeros(self.N, device=self.device)
        for t in reversed(range(self.T)):
            next_value = last_value if t == self.T - 1 else self.values[t + 1]
            alive = 1.0 - self.ends[t]
            delta = self.rewards[t] + gamma * next_value * alive - self.values[t]
            gae = delta + gamma * lam * alive * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def minibatches(self, n_minibatches: int, generator: torch.Generator) -> Iterator[dict]:
        """Shuffled minibatches of flattened transitions."""
        total = self.T * self.N
        order = torch.randperm(total, generator=generator).to(self.device)
        size = max(1, total // n_minibatches)
        flat = lambda x: x.reshape(total, *x.shape[2:])  # noqa: E731
        data = {"obs": {k: flat(v) for k, v in self.obs.items()}, "actions": flat(self.actions),
                "logprobs": flat(self.logprobs), "values": flat(self.values),
                "advantages": flat(self.advantages), "returns": flat(self.returns)}
        for start in range(0, total - size + 1, size):
            idx = order[start:start + size]
            yield {"obs": {k: v[idx] for k, v in data["obs"].items()},
                   **{k: v[idx] for k, v in data.items() if k != "obs"}}


class Collector:
    """Steps a vector environment with a policy and keeps the current observation between rollouts."""

    def __init__(self, envs, device: torch.device, seed: int) -> None:
        self.envs, self.device = envs, device
        self.obs, _ = envs.reset(seed=seed)
        self.tracker = EpisodeTracker(envs.num_envs)
        self.version = 0  # bumped by the curriculum when the start distribution changes
        self.num_envs = envs.num_envs

    def obs_shapes(self) -> dict:
        return {k: tuple(np.asarray(self.obs[k]).shape[1:]) for k in OBS_KEYS}

    @torch.no_grad()
    def collect(self, policy, buffer: RolloutBuffer, gamma: float) -> dict:
        """Fill ``buffer`` with ``buffer.T`` steps from every env; returns rollout statistics."""
        n_legal = 0.0
        for t in range(buffer.T):
            obs = obs_to_torch(self.obs, self.device)
            action, logprob, _, value = policy.act(obs)
            next_obs, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
            reward = np.asarray(reward, dtype=np.float32).copy()
            self.tracker.step(reward, terminated, truncated, self.version)
            cut = np.flatnonzero(truncated & ~terminated)
            if len(cut):  # bootstrap a time-limit cut with the value of the state it stopped in
                final = stack_obs([info["final_obs"][i] for i in cut])
                reward[cut] += gamma * policy.value(obs_to_torch(final, self.device)).float().cpu().numpy()
            buffer.add(t, obs, action, logprob, value, reward, terminated | truncated)
            n_legal += float(obs["action_mask"].sum()) / self.num_envs
            self.obs = next_obs
        last_value = policy.value(obs_to_torch(self.obs, self.device))
        return {"last_value": last_value, "candidates": n_legal / buffer.T}
