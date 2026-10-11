"""Gymnasium environments for knots and Kirby diagrams."""

import gymnasium as gym

from .kirby_env import CandidateActions, KirbyEnv, ScrambleStart, Transition, zero_reward
from .unknot_env import UnknotEnv

__all__ = ["KirbyEnv", "CandidateActions", "ScrambleStart", "Transition", "zero_reward", "UnknotEnv"]

if "knotml/KirbyReduce-v0" not in gym.registry:
    gym.register(id="knotml/KirbyReduce-v0", entry_point="knotml.envs.kirby_env:KirbyEnv")
