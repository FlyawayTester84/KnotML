"""The training loop: environments, policy, algorithm, curriculum, evaluation, logging and checkpoints.

A run lives in ``<output_dir>/<name>/``:

* ``config.yaml``: the full configuration used;
* ``metrics.csv``: one row per iteration;
* ``eval.csv``: one row per evaluation;
* ``checkpoints/``: ``latest.pt``, ``best.pt`` (best evaluation success) and the last
  ``keep_checkpoints`` periodic ``step_<n>.pt``;
* ``tb/``: TensorBoard events, when TensorBoard is installed.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import random
import signal
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .algorithms import make_algorithm
from .config import Config
from .curriculum import Curriculum
from .envs import EnvSpec, env_spec, make_vector_env
from .evaluate import evaluate
from .policies import make_policy
from .rollout import Collector

try:
    from tqdm.auto import tqdm
except ImportError:  # the bar is optional
    tqdm = None

__all__ = ["Trainer", "resolve_device", "load_policy"]

_TRAIN_KEYS = ("iteration", "step", "sps", "episodes", "return", "length", "success", "scramble_lo", "scramble_hi",
               "curriculum_success", "candidates")


def resolve_device(name: str) -> torch.device:
    """``auto``: CUDA when available, else CPU (MPS only on request: scatter-heavy small batches gain little)."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("run.device asks for CUDA, but torch.cuda.is_available() is False")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("run.device asks for MPS, which this PyTorch build or machine does not support")
    return device


def load_policy(path: str, device: Optional[torch.device] = None, **capacity: Optional[int]):
    """``(policy, config)`` from a checkpoint written by :class:`Trainer`.

    ``max_crossings``, ``max_components`` and ``max_actions`` rebuild the
    policy for another environment capacity (the returned config says which).
    The ``gnn`` weights do not depend on it; a policy whose weights do fails to
    load with a size mismatch.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(state["config"])
    spec = EnvSpec(**{k: tuple(map(tuple, v)) if k == "kind_fields" else (tuple(v) if isinstance(v, list) else v)
                      for k, v in state["spec"].items()})
    changes = {k: int(v) for k, v in capacity.items() if v is not None}
    if changes:
        unknown = set(changes) - {"max_crossings", "max_components", "max_actions"}
        if unknown:
            raise TypeError(f"unknown capacity argument(s): {', '.join(sorted(unknown))}")
        env = dataclasses.replace(cfg.env, **changes)
        cfg = dataclasses.replace(cfg, env=env)
        spec = dataclasses.replace(spec, n_ports=4 * env.max_crossings, n_components=env.max_components,
                                   n_actions=env.max_actions)
    policy = make_policy(cfg.policy.name, spec, cfg.policy.params)
    policy.load_state_dict(state["policy"])
    return policy.to(device or resolve_device(cfg.run.device)), cfg


class _Stop(BaseException):
    """Raised by the SIGTERM handler to leave the training loop (a BaseException, so nothing swallows it)."""


def _catch_sigterm():
    """Turn SIGTERM into :class:`_Stop` (main thread only); returns the previous handler, or ``None``."""
    if threading.current_thread() is not threading.main_thread():
        return None

    def handler(signum, frame):
        raise _Stop()

    return signal.signal(signal.SIGTERM, handler)


def _short(value: float, fmt: str = ".2f") -> str:
    return format(value, fmt) if isinstance(value, float) and math.isfinite(value) else "-"


class _Logger:
    def __init__(self, run_dir: Path, tensorboard: bool, resume: bool) -> None:
        self.run_dir = run_dir
        self.files: dict = {}
        self.resume = resume
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(str(run_dir / "tb"))
            except (ImportError, ModuleNotFoundError):
                print("TensorBoard is not installed; logging to CSV only (pip install tensorboard)")

    def write(self, name: str, row: dict, step: int) -> None:
        if name not in self.files:
            path = self.run_dir / f"{name}.csv"
            exists = path.exists() and self.resume
            f = open(path, "a" if exists else "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(f, fieldnames=list(row), extrasaction="ignore")
            if not exists:
                writer.writeheader()
            self.files[name] = (f, writer)
        f, writer = self.files[name]
        writer.writerow(row)
        f.flush()
        if self.tb is not None:
            for k, v in row.items():
                if isinstance(v, (int, float)) and k not in ("step", "iteration") and math.isfinite(v):
                    self.tb.add_scalar(f"{name}/{k}", v, step)

    def close(self) -> None:
        for f, _ in self.files.values():
            f.close()
        if self.tb is not None:
            self.tb.close()


class Trainer:
    """Build everything from a :class:`Config`; :meth:`train` runs until ``run.total_timesteps``."""

    def __init__(self, cfg: Config, resume: Optional[str] = None, quiet: bool = False) -> None:
        self.cfg, self.quiet = cfg, quiet
        self._bar = None
        self.stopped: Optional[str] = None  # why train() ended early (e.g. "SIGTERM"), None if it finished
        run = cfg.run
        random.seed(run.seed)
        np.random.seed(run.seed)
        torch.manual_seed(run.seed)
        self.device = resolve_device(run.device)
        if run.torch_threads:
            torch.set_num_threads(run.torch_threads)
        if run.tf32 and self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.spec = env_spec(cfg)
        self.policy = make_policy(cfg.policy.name, self.spec, cfg.policy.params).to(self.device)
        if run.compile:
            self.policy.compile()
        self.envs = make_vector_env(cfg, seed=run.seed)
        self.collector = Collector(self.envs, self.device, seed=run.seed)
        self.algorithm = make_algorithm(cfg.algorithm.name, cfg.algorithm.params, policy=self.policy,
                                        collector=self.collector, device=self.device,
                                        total_timesteps=run.total_timesteps, seed=run.seed)
        self.curriculum = Curriculum(cfg.curriculum, cfg.env.scramble)
        self.global_step = self.iteration = 0
        self.best = -math.inf
        if resume:
            self.run_dir = Path(resume).resolve().parent.parent
            self._load(resume)
        else:
            self.run_dir = self._new_run_dir(Path(run.output_dir) / run.name)
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self._write_config()
        self.logger = _Logger(self.run_dir, run.tensorboard, resume=bool(resume))

    # ------------------------------------------------------------------

    def train(self) -> dict:
        """Train until ``run.total_timesteps`` environment steps; returns the last metrics.

        SIGTERM (sent by Slurm before a time limit or on preemption, and by ``kill``)
        ends training after saving ``latest.pt``; :attr:`stopped` then says so, and
        ``--resume`` continues from there.
        """
        self._say(f"run {self.run_dir} | device {self.device} | {sum(p.numel() for p in self.policy.parameters()):,} "
                  f"parameters | {self.cfg.env.num_envs} envs ({self.cfg.env.vectorization})")
        self._bar = self._progress_bar()
        self._metrics: dict = {}
        previous = _catch_sigterm()
        try:
            self._loop()
        except _Stop:
            self.stopped = "SIGTERM"
        finally:
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)
            if self._bar is not None:
                self._bar.close()
                self._bar = None
            self.save()
            self.close()
        if self.stopped:
            self._say(f"stopped by {self.stopped} at step {self.global_step:,}; continue with "
                      f"--resume {self.run_dir / 'checkpoints' / 'latest.pt'}")
        return self._metrics

    def _loop(self) -> None:
        cfg, run = self.cfg, self.cfg.run
        recent: list[dict] = []
        while self.global_step < run.total_timesteps:
            start = time.perf_counter()
            metrics = self.algorithm.iteration(self.global_step)
            steps = int(metrics.pop("env_steps"))
            self.global_step += steps
            self.iteration += 1
            self._metrics = metrics
            episodes = self.collector.tracker.pop()
            recent = (recent + episodes)[-max(1, cfg.curriculum.window):]
            if self.curriculum.update(episodes):
                self.envs.call("set_scramble", *self.curriculum.range)
                self.collector.version = self.curriculum.version
            row = self._train_row(metrics, episodes, recent, steps / (time.perf_counter() - start))
            self.logger.write("metrics", row, self.global_step)
            if self._bar is not None:
                self._bar.update(steps)
                self._bar.set_postfix(scramble=f"{self.curriculum.lo}-{self.curriculum.hi}",
                                      success=_short(row["success"]), ret=_short(row["return"], "+.2f"),
                                      refresh=False)
            if self.iteration % max(1, run.log_every) == 0:
                self._say(self._format(row))
            if cfg.eval.every and self.iteration % cfg.eval.every == 0:
                self.run_evaluation()
            if run.checkpoint_every and self.iteration % run.checkpoint_every == 0:
                self.save(periodic=True)

    def _progress_bar(self):
        """A tqdm bar over the environment steps (``None`` when quiet, turned off or tqdm is missing).
        With ``disable=None`` tqdm hides it when the output is not a terminal (a log file, ``| tee``)."""
        run = self.cfg.run
        if self.quiet or not run.progress or tqdm is None:
            return None
        return tqdm(total=run.total_timesteps, initial=min(self.global_step, run.total_timesteps), unit="step",
                    unit_scale=True, dynamic_ncols=True, smoothing=0.1, desc=self.run_dir.name, disable=None)

    def _say(self, text: str) -> None:
        """Print a log line (above the progress bar, if there is one)."""
        if self.quiet:
            return
        if getattr(self, "_bar", None) is not None:
            tqdm.write(text)
        else:
            print(text, flush=True)

    def run_evaluation(self) -> dict:
        e = self.cfg.eval
        scramble = tuple(e.scramble) if e.scramble else self.curriculum.range
        result = evaluate(self.policy, self.cfg, self.device, episodes=e.episodes, num_envs=e.num_envs,
                          scramble=scramble, deterministic=e.deterministic, seed=10_000 + self.iteration)
        row = {"iteration": self.iteration, "step": self.global_step, "scramble_lo": scramble[0],
               "scramble_hi": scramble[1], **result}
        self.logger.write("eval", row, self.global_step)
        steps = result["length_solved"]
        solved = f"solved in {steps:.1f} steps" if math.isfinite(steps) else "none solved"
        self._say(f"  eval scramble {scramble[0]}-{scramble[1]}: success {result['success']:.2f}, {solved}, "
                  f"return {result['return']:+.3f}")
        # "best" compares like with like: success at the hardest level reached so far
        score = scramble[1] + result["success"]
        if score > self.best:
            self.best = score
            self.save(name="best.pt")
        return result

    # ------------------------------------------------------------------

    def save(self, periodic: bool = False, name: str = "latest.pt") -> Path:
        state = {
            "config": self.cfg.to_dict(),
            "spec": dataclasses.asdict(self.spec),
            "policy": self.policy.state_dict(),
            "algorithm": self.algorithm.state_dict(),
            "curriculum": self.curriculum.state_dict(),
            "global_step": self.global_step,
            "iteration": self.iteration,
            "best": self.best,
        }
        folder = self.run_dir / "checkpoints"
        path = folder / name
        torch.save(state, path)
        if periodic:
            torch.save(state, folder / f"step_{self.global_step}.pt")
            old = sorted(folder.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
            for p in old[: max(0, len(old) - self.cfg.run.keep_checkpoints)]:
                p.unlink()
        return path

    def _load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(state["policy"])
        self.algorithm.load_state_dict(state["algorithm"])
        self.curriculum.load_state_dict(state["curriculum"])
        self.global_step, self.iteration, self.best = state["global_step"], state["iteration"], state["best"]
        self.envs.call("set_scramble", *self.curriculum.range)
        self.collector.version = self.curriculum.version

    def close(self) -> None:
        try:
            self.envs.close()
        except Exception:  # env workers already gone (e.g. killed by the same SIGTERM)
            pass
        self.logger.close()

    # ------------------------------------------------------------------

    @staticmethod
    def _new_run_dir(base: Path) -> Path:
        path, k = base, 1
        while path.exists():
            k += 1
            path = base.with_name(f"{base.name}-{k}")
        return path

    def _write_config(self) -> None:
        import yaml

        with open(self.run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg.to_dict(), f, sort_keys=False)

    def _train_row(self, metrics: dict, episodes: list, recent: list, sps: float) -> dict:
        nan = float("nan")
        mean = lambda key, xs: float(np.mean([e[key] for e in xs])) if xs else nan  # noqa: E731
        row = {"iteration": self.iteration, "step": self.global_step, "sps": round(sps, 1),
               "episodes": len(episodes), "return": mean("return", recent), "length": mean("length", recent),
               "success": mean("solved", recent), "scramble_lo": self.curriculum.lo,
               "scramble_hi": self.curriculum.hi, "curriculum_success": self.curriculum.success_rate}
        row.update(metrics)
        return {k: row.get(k, nan) for k in (*_TRAIN_KEYS, *sorted(set(row) - set(_TRAIN_KEYS)))}

    @staticmethod
    def _format(row: dict) -> str:
        def num(v, fmt):
            return "-" if isinstance(v, float) and not math.isfinite(v) else format(v, fmt)

        return (f"it {row['iteration']:>5} | step {row['step']:>10,} | {num(row['sps'], ',.0f'):>6} sps | "
                f"scramble {row['scramble_lo']}-{row['scramble_hi']} | success {num(row['success'], '.2f')} | "
                f"len {num(row['length'], '.1f')} | ret {num(row['return'], '+.3f')} | "
                f"pg {num(row.get('loss/policy', math.nan), '+.3f')} v {num(row.get('loss/value', math.nan), '.3f')} "
                f"ent {num(row.get('loss/entropy', math.nan), '.2f')} kl {num(row.get('approx_kl', math.nan), '.4f')} "
                f"ev {num(row.get('explained_variance', math.nan), '.2f')}")
