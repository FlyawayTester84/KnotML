"""Training configuration: a YAML file (``configs/ppo.yaml``) read into dataclasses.

Sections ``run``, ``env``, ``reward``, ``curriculum`` and ``eval`` are fixed;
``policy`` and ``algorithm`` hold a ``name`` (looked up in the policy /
algorithm registry) plus that component's own hyperparameters, which the
component checks itself.  Unknown keys are errors, so typos do not pass
silently.  ``key.sub=value`` overrides (values parsed as YAML) are applied on
top of the file, e.g. ``algorithm.learning_rate=1e-4``.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

__all__ = ["Config", "load_config", "apply_overrides", "build_dataclass"]


@dataclass
class RunConfig:
    name: str = "kirby-ppo"
    seed: int = 0
    device: str = "auto"  # auto (cuda, else cpu), cuda, cuda:1, mps, cpu
    output_dir: str = "runs"
    total_timesteps: int = 1_000_000
    log_every: int = 10  # iterations between log lines on the terminal (metrics.csv records every iteration)
    progress: bool = True  # tqdm progress bar, shown when the output is a terminal
    checkpoint_every: int = 50  # iterations
    keep_checkpoints: int = 3  # step checkpoints kept besides latest.pt and best.pt
    tensorboard: bool = True  # also log to TensorBoard when it is installed
    torch_threads: int = 0  # 0: PyTorch's default
    tf32: bool = True  # TensorFloat-32 matmuls on Ampere+ GPUs
    compile: bool = False  # torch.compile the policy


@dataclass
class EnvConfig:
    calculus: str = "4-manifold"
    closed: bool = False
    max_crossings: int = 32
    max_components: int = 6
    max_steps: int = 40
    max_actions: int = 512  # candidate actions per state (CandidateActions)
    num_envs: int = 16
    vectorization: str = "async"  # async (a process per env) or sync
    scramble: tuple = (1, 3)  # construction moves per start (the curriculum raises the maximum)


@dataclass
class RewardConfig:
    name: str = "potential"
    kwargs: dict = field(default_factory=dict)


@dataclass
class CurriculumConfig:
    enabled: bool = True
    window: int = 200  # episodes, all started at the current level
    promote_at: float = 0.8  # success rate that raises the scramble maximum
    step: int = 1
    max_scramble: int = 12


@dataclass
class EvalConfig:
    every: int = 25  # iterations; 0 turns evaluation off
    episodes: int = 64
    num_envs: int = 8
    deterministic: bool = True  # greedy actions
    scramble: Optional[tuple] = None  # None: the current training range


@dataclass
class ComponentConfig:
    """A registry ``name`` plus that component's hyperparameters."""

    name: str
    params: dict = field(default_factory=dict)


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: ComponentConfig = field(default_factory=lambda: ComponentConfig("gnn"))
    algorithm: ComponentConfig = field(default_factory=lambda: ComponentConfig("ppo"))

    def to_dict(self) -> dict:
        """The plain-data form, as written to ``config.yaml`` (round-trips through :meth:`from_dict`)."""
        out = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, ComponentConfig):
                out[f.name] = {"name": value.name, **value.params}
            else:
                out[f.name] = {k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(value).items()}
        return out

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        data = dict(data or {})
        unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown config section(s): {', '.join(sorted(unknown))}")
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            section = data.get(f.name)
            if f.name in ("policy", "algorithm"):
                section = dict(section or {})
                default = getattr(cls(), f.name).name
                kwargs[f.name] = ComponentConfig(section.pop("name", default), section)
            else:
                kwargs[f.name] = build_dataclass(type(getattr(cls(), f.name)), section or {}, f.name)
        return cls(**kwargs)


def build_dataclass(cls, values: dict, where: str = ""):
    """``cls(**values)`` with a clear error for unknown keys; lists become tuples where the default is one."""
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(names)
    if unknown:
        raise ValueError(f"unknown key(s) in {where or cls.__name__}: {', '.join(sorted(unknown))} "
                         f"(allowed: {', '.join(names)})")
    defaults = cls()
    fixed = {}
    for k, v in values.items():
        tuple_field = isinstance(getattr(defaults, k), tuple) or names[k].type in ("tuple", "Optional[tuple]")
        if isinstance(v, list) and tuple_field:
            v = tuple(v)
        fixed[k] = v
    return cls(**fixed)


def apply_overrides(data: dict, overrides: Sequence[str]) -> dict:
    """Apply ``a.b=value`` strings to nested ``data`` (values are parsed as YAML: ``1e-4``, ``[1, 3]``, ``true``)."""
    import yaml

    data = {k: dict(v) if isinstance(v, dict) else v for k, v in (data or {}).items()}
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"override {item!r}: expected key.sub=value")
        path = key.strip().split(".")
        node = data
        for part in path[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"override {item!r}: {part} is not a section")
        value = yaml.safe_load(raw)
        if isinstance(value, str) and _looks_numeric(value):  # YAML 1.1 reads "1e-4" as a string
            value = float(value)
        node[path[-1]] = value
    return data


_FLOAT = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _looks_numeric(text: str) -> bool:
    return bool(_FLOAT.match(text.strip()))


def load_config(path: Union[str, Path, None] = None, overrides: Sequence[str] = ()) -> Config:
    """Read a YAML config (``None``: the built-in defaults) and apply ``overrides``."""
    import yaml

    data = {}
    if path is not None:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    data = _fix_numbers(apply_overrides(data, overrides))
    return Config.from_dict(data)


def _fix_numbers(node):
    """YAML 1.1 (PyYAML) reads ``3e-4`` as a string; turn such strings into floats."""
    if isinstance(node, dict):
        return {k: _fix_numbers(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_fix_numbers(v) for v in node]
    if isinstance(node, str) and _looks_numeric(node) and not node.strip().isdigit():
        return float(node)
    return node
