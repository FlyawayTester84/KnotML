"""Proximal Policy Optimization (Schulman et al. 2017) with masked categorical actions.

The details follow the common "PPO implementation details" checklist:

* GAE advantages, normalised per minibatch;
* a clipped surrogate objective and a clipped value loss;
* an entropy bonus;
* global gradient-norm clipping;
* linear learning-rate annealing;
* early stopping of an update when the approximate KL exceeds ``target_kl``.

Time-limit truncations are bootstrapped (see :class:`~knotml.rl.rollout.Collector`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn

from ..rollout import RolloutBuffer
from .base import Algorithm, register_algorithm

__all__ = ["PPO"]


@register_algorithm("ppo")
class PPO(Algorithm):
    @dataclass
    class Hyperparameters:
        learning_rate: float = 3e-4
        anneal_lr: bool = True
        num_steps: int = 128  # rollout length per environment
        gamma: float = 0.99
        gae_lambda: float = 0.95
        num_minibatches: int = 8
        update_epochs: int = 4
        norm_adv: bool = True
        clip_coef: float = 0.2
        clip_vloss: bool = True
        ent_coef: float = 0.01
        vf_coef: float = 0.5
        max_grad_norm: float = 0.5
        target_kl: Optional[float] = 0.03  # None: no early stopping
        adam_eps: float = 1e-5
        weight_decay: float = 0.0

    def __init__(self, hp: Hyperparameters, policy, collector, device, total_timesteps: int, seed: int = 0) -> None:
        super().__init__(hp, policy, collector, device, total_timesteps, seed)
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=hp.learning_rate, eps=hp.adam_eps,
                                           weight_decay=hp.weight_decay)
        self.buffer = RolloutBuffer(hp.num_steps, collector.num_envs, collector.obs_shapes(), device)
        self.batch_size = hp.num_steps * collector.num_envs
        if self.batch_size < hp.num_minibatches:
            raise ValueError("num_steps * num_envs must be at least num_minibatches")

    def iteration(self, global_step: int) -> dict:
        hp = self.hp
        if hp.anneal_lr:
            frac = max(0.0, 1.0 - global_step / max(1, self.total_timesteps))
            for group in self.optimizer.param_groups:
                group["lr"] = frac * hp.learning_rate
        self.policy.eval()
        stats = self.collector.collect(self.policy, self.buffer, hp.gamma)
        self.buffer.compute_gae(stats["last_value"], hp.gamma, hp.gae_lambda)
        self.policy.train()
        return {"env_steps": self.batch_size, "candidates": stats["candidates"], **self._update()}

    def _update(self) -> dict:
        hp, buf = self.hp, self.buffer
        clipfracs, kls = [], []
        pg_loss = v_loss = entropy = torch.zeros((), device=self.device)
        stop = False
        for epoch in range(hp.update_epochs):
            for mb in buf.minibatches(hp.num_minibatches, self.generator):
                logprob, ent, value = self.policy.evaluate(mb["obs"], mb["actions"])
                log_ratio = logprob - mb["logprobs"]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()  # the low-variance k3 estimator
                    kls.append(float(approx_kl))
                    clipfracs.append(float(((ratio - 1.0).abs() > hp.clip_coef).float().mean()))
                adv = mb["advantages"]
                if hp.norm_adv and adv.numel() > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - hp.clip_coef, 1 + hp.clip_coef)).mean()
                if hp.clip_vloss:
                    clipped = mb["values"] + (value - mb["values"]).clamp(-hp.clip_coef, hp.clip_coef)
                    v_loss = 0.5 * torch.max((value - mb["returns"]) ** 2, (clipped - mb["returns"]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((value - mb["returns"]) ** 2).mean()
                entropy = ent.mean()
                loss = pg_loss - hp.ent_coef * entropy + hp.vf_coef * v_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), hp.max_grad_norm)
                self.optimizer.step()
                if hp.target_kl is not None and kls[-1] > hp.target_kl:
                    stop = True
                    break
            if stop:
                break
        values, returns = buf.values.flatten(), buf.returns.flatten()
        var = float(returns.var()) if returns.numel() > 1 else 0.0
        explained = float("nan") if var == 0 else 1.0 - float((returns - values).var()) / var
        return {
            "loss/policy": pg_loss.item(), "loss/value": v_loss.item(), "loss/entropy": entropy.item(),
            "approx_kl": float(np.mean(kls)) if kls else 0.0,
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "explained_variance": explained, "lr": self.optimizer.param_groups[0]["lr"],
            "update_epochs_run": epoch + 1,
        }

    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
