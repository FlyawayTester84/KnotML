"""Run a trained agent on a given Kirby diagram and record the moves it makes.

* :func:`list_agents` lists the trained agents saved under ``models/``.
* :class:`Agent` loads one by folder name or checkpoint path.
* :func:`solve` runs it from a start diagram for at most ``max_steps`` moves.
  The first attempt is greedy; further attempts sample from the policy.
  Every move is a legal Kirby move applied by :class:`~knotml.envs.KirbyEnv`.

The policy's weights do not depend on the environment's capacity. A diagram
larger than the agent saw in training can therefore still be run, with the
capacity raised to fit.  How well the agent then does is another matter: it
learnt on scrambles of the empty diagram.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch

from ..core import LinkDiagram
from ..handlebody import Handlebody
from ..kirby import AnyMove, FramedLink
from ..notation import move_spec, touched_crossings
from ..play.session import components_of
from .config import Config
from .envs import make_env
from .rollout import obs_to_torch
from .trainer import load_policy

__all__ = ["MODELS_DIR", "list_agents", "find_checkpoint", "Agent", "Step", "Attempt", "solve", "reachable"]

MODELS_DIR = Path("models")
Kirby = Union[FramedLink, Handlebody]


def find_checkpoint(agent: Union[str, Path], checkpoint: str = "best",
                    models_dir: Union[str, Path] = MODELS_DIR) -> Path:
    """The checkpoint file of ``agent``: a ``.pt`` path, a run folder, or a folder name under ``models_dir``."""
    path = Path(agent)
    if path.suffix == ".pt":
        if not path.exists():
            raise FileNotFoundError(f"no checkpoint {path}")
        return path
    folder = path if path.is_dir() else Path(models_dir) / str(agent)
    candidate = folder / "checkpoints" / f"{checkpoint}.pt"
    if not candidate.exists():
        names = ", ".join(a["name"] for a in list_agents(models_dir)) or "none"
        raise FileNotFoundError(f"no {checkpoint}.pt for agent {agent!r} (looked in {folder}/checkpoints); "
                                f"agents in {models_dir}/: {names}")
    return candidate


def list_agents(models_dir: Union[str, Path] = MODELS_DIR) -> list[dict]:
    """One summary per agent folder (``<name>/checkpoints/best.pt`` or ``latest.pt``) under ``models_dir``."""
    out = []
    root = Path(models_dir)
    if not root.is_dir():
        return out
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        path = next((folder / "checkpoints" / f for f in ("best.pt", "latest.pt")
                     if (folder / "checkpoints" / f).exists()), None)
        if path is None:
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)
        cfg = Config.from_dict(state["config"])
        curriculum = state.get("curriculum", {})
        out.append({
            "name": folder.name, "path": path, "calculus": cfg.env.calculus, "closed": cfg.env.closed,
            "steps": state.get("global_step", 0), "max_crossings": cfg.env.max_crossings,
            "max_components": cfg.env.max_components, "policy": cfg.policy.name, **cfg.policy.params,
            "scramble": (curriculum.get("lo"), curriculum.get("hi")),
        })
    return out


@dataclass
class Agent:
    """A trained policy with the configuration it was trained under."""

    name: str
    path: Path
    policy: torch.nn.Module
    config: Config
    trained: dict  # capacity and progress at training time

    @classmethod
    def load(cls, agent: Union[str, Path], checkpoint: str = "best", device: Optional[torch.device] = None,
             models_dir: Union[str, Path] = MODELS_DIR, max_crossings: Optional[int] = None,
             max_components: Optional[int] = None, max_actions: Optional[int] = None) -> Agent:
        path = find_checkpoint(agent, checkpoint, models_dir)
        state = torch.load(path, map_location="cpu", weights_only=False)
        saved = Config.from_dict(state["config"])
        policy, cfg = load_policy(str(path), device or torch.device("cpu"), max_crossings=max_crossings,
                                  max_components=max_components, max_actions=max_actions)
        policy.eval()
        trained = {"max_crossings": saved.env.max_crossings, "max_components": saved.env.max_components,
                   "max_actions": saved.env.max_actions, "steps": state.get("global_step", 0),
                   "scramble": tuple(state.get("curriculum", {}).get(k) for k in ("lo", "hi"))}
        name = path.parent.parent.name if path.parent.name == "checkpoints" else path.stem
        return cls(name, path, policy, cfg, trained)

    @property
    def calculus(self) -> str:
        return self.config.env.calculus

    @property
    def closed(self) -> bool:
        return self.config.env.closed

    def start_from(self, diagram: LinkDiagram, framings: Sequence[Optional[int]]) -> Kirby:
        """The Kirby diagram this agent works on: a Handlebody (4-manifold agent; ``None`` = dotted circle)
        or a FramedLink (3-manifold agent)."""
        if self.calculus == "4-manifold":
            return Handlebody(diagram, list(framings), closed=self.closed)
        if any(f is None for f in framings):
            raise ValueError(f"agent {self.name!r} plays the 3-manifold calculus: dotted circles are not allowed")
        return FramedLink(diagram, list(framings))

    def with_capacity_for(self, start: Kirby, crossings_room: int = 16, components_room: int = 2) -> Agent:
        """This agent, rebuilt with enough capacity for ``start`` (plus room to grow) if it needs more."""
        need_x = start.diagram.n_crossings + crossings_room
        need_c = len(start.labels()) + components_room
        env = self.config.env
        if need_x <= env.max_crossings and need_c <= env.max_components:
            return self
        return Agent.load(self.path, device=next(self.policy.parameters()).device,
                          max_crossings=max(env.max_crossings, need_x), max_components=max(env.max_components, need_c),
                          max_actions=env.max_actions)


@dataclass
class Step:
    move: AnyMove
    probability: float  # the policy's probability of this move
    value: float  # the value estimate of the state before the move
    n_crossings: int  # after the move
    n_components: int

    @property
    def spec(self) -> str:
        return move_spec(self.move)


@dataclass
class Attempt:
    sampled: bool  # False: greedy
    seed: int
    steps: list = field(default_factory=list)
    stop: str = "step limit"  # solved | step limit | cycle | no legal move
    frames: list = field(default_factory=list)  # for visualization.steps_page

    @property
    def solved(self) -> bool:
        return self.stop == "solved"


def reachable(start: Kirby) -> tuple[bool, str]:
    """Whether ``start`` passes the invariant checks the empty diagram passes (necessary, not sufficient)."""
    if isinstance(start, Handlebody):
        target = "S⁴" if start.closed else "B⁴"
        have, want = start.invariants(), Handlebody.empty(closed=start.closed).invariants()
        diff = {k: v for k, v in have.items() if want.get(k) != v}
        if diff:
            shown = ", ".join(f"{k} = {v} (empty: {want.get(k)})" for k, v in diff.items())
            return False, f"not {target}: {shown}"
        return True, (f"invariants agree with {target} ({', '.join(f'{k} = {v}' for k, v in have.items())}); "
                      "a necessary condition only (a contractible Mazur manifold passes too)")
    h1 = start.homology()
    if h1:
        return False, f"not S³: H₁ = {h1}"
    return True, "H₁ = 0, as for S³ (a necessary condition only)"


@torch.no_grad()
def solve(agent: Agent, start: Kirby, max_steps: int = 20, attempts: int = 1, seed: int = 0,
          stop_on_cycle: bool = True) -> list[Attempt]:
    """Run ``agent`` from ``start`` for at most ``max_steps`` moves, ``attempts`` times (the first
    greedy, the rest sampled).  A greedy attempt stops early when it revisits a diagram, since it would
    then repeat itself."""
    device = next(agent.policy.parameters()).device
    env_cfg = {**dataclasses.asdict(agent.config.env), "max_steps": max_steps}
    env = make_env(env_cfg, {"name": "zero", "kwargs": {}}, gamma=0.99)
    base = env.unwrapped
    out = []
    for k in range(attempts):
        attempt = Attempt(sampled=k > 0, seed=seed + k)
        generator = torch.Generator(device="cpu").manual_seed(seed + k)
        obs, _ = env.reset(seed=seed + k, options={"diagram": start})
        attempt.frames.append(("start", base.kirby.diagram.copy(), set(), dict(components_of(base.kirby)), ""))
        seen = {base.kirby.canonical_key()}
        for _ in range(max_steps):
            if not np.asarray(obs["action_mask"]).any():
                attempt.stop = "no legal move"
                break
            batch = obs_to_torch({key: np.asarray(v)[None] for key, v in obs.items()}, device)
            dist, value = agent.policy.distribution(batch)
            probs = dist.probs[0].float().cpu()
            index = int(torch.multinomial(probs, 1, generator=generator)) if attempt.sampled else int(probs.argmax())
            move = base.decode(obs["candidates"][index])
            d = base.kirby.diagram
            n_before, touched = d.n_crossings, touched_crossings(d, move)
            obs, _, terminated, _, _ = env.step(index)
            after = base.kirby
            marked = touched | set(range(n_before, after.diagram.n_crossings))
            step = Step(move, float(probs[index]), float(value[0]), after.diagram.n_crossings, len(after.labels()))
            attempt.steps.append(step)
            attempt.frames.append((step.spec, after.diagram.copy(), marked, dict(components_of(after)),
                                   f"agent: probability {step.probability:.2f}, value before the move "
                                   f"{step.value:+.2f}"))
            if terminated:
                attempt.stop = "solved"
                break
            key = after.canonical_key()
            if stop_on_cycle and not attempt.sampled and key in seen:
                attempt.stop = "cycle"
                break
            seen.add(key)
        out.append(attempt)
    return out
