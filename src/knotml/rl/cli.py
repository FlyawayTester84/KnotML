"""Command-line programs: ``scripts/train_agent.py`` and ``scripts/evaluate_agent.py``."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

__all__ = ["train_main", "evaluate_main"]


def train_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="train_agent.py",
        description="Train an agent to reduce Kirby diagrams to the empty diagram.",
        epilog="examples:\n"
        "  python scripts/train_agent.py --config configs/ppo.yaml\n"
        "  python scripts/train_agent.py --config configs/ppo.yaml --set run.device=cuda env.num_envs=32\n"
        "  python scripts/train_agent.py --resume runs/kirby-ppo/checkpoints/latest.pt "
        "--set run.total_timesteps=5000000",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/ppo.yaml", help="YAML hyperparameters (default configs/ppo.yaml)")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                        help="override config entries, e.g. algorithm.learning_rate=1e-4 env.scramble=[1,4]")
    parser.add_argument("--resume", help="continue a run from its checkpoint (its saved config, plus --set)")
    args = parser.parse_args(argv)

    from .config import Config, apply_overrides, load_config
    from .trainer import Trainer

    try:
        if args.resume:
            import torch

            saved = torch.load(args.resume, map_location="cpu", weights_only=False)["config"]
            cfg = Config.from_dict(apply_overrides(saved, args.set))
        else:
            cfg = load_config(args.config, args.set)
    except (ValueError, FileNotFoundError) as err:
        parser.error(str(err))
    trainer = Trainer(cfg, resume=args.resume)
    trainer.train()
    print(f"done: {trainer.global_step:,} steps; checkpoints in {trainer.run_dir / 'checkpoints'}")
    return 0


def evaluate_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evaluate_agent.py",
        description="Measure a trained agent and optionally record its episodes as HTML step pages.",
        epilog="examples:\n"
        "  python scripts/evaluate_agent.py runs/kirby-ppo/checkpoints/best.pt --scramble 1 6 --episodes 200\n"
        "  python scripts/evaluate_agent.py runs/kirby-ppo/checkpoints/best.pt --html outputs/agent.html --record 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--scramble", type=int, nargs=2, metavar=("MIN", "MAX"),
                        help="construction moves per start (default: the range training had reached)")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the most likely")
    parser.add_argument("--device", help="override the run's device (cpu, cuda, mps)")
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--html", help="also write recorded episodes as step-through pages (PATH, PATH-2, ...)")
    parser.add_argument("--record", type=int, default=3, help="episodes to record with --html (default 3)")
    args = parser.parse_args(argv)

    import torch

    from .evaluate import evaluate, record_episodes
    from .trainer import load_policy, resolve_device

    device = resolve_device(args.device) if args.device else None
    policy, cfg = load_policy(args.checkpoint, device)
    device = next(policy.parameters()).device
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    reached = state.get("curriculum", {})
    scramble = tuple(args.scramble) if args.scramble else (reached.get("lo", cfg.env.scramble[0]),
                                                           reached.get("hi", cfg.env.scramble[1]))
    result = evaluate(policy, cfg, device, episodes=args.episodes, num_envs=min(16, args.episodes),
                      scramble=scramble, deterministic=not args.stochastic, seed=args.seed)
    steps = result["length_solved"]
    solved = f"solved ones took {steps:.1f} steps on average" if steps == steps else "none solved"
    print(f"{args.checkpoint} (step {state.get('global_step', 0):,}), scramble {scramble[0]}-{scramble[1]}, "
          f"{'sampled' if args.stochastic else 'greedy'}: success {result['success']:.3f} over {result['episodes']} "
          f"episodes; {solved}; mean return {result['return']:+.3f}")
    if args.html:
        import os

        from ..visualization import write_steps_page

        calculus = "4" if cfg.env.calculus == "4-manifold" else "3"
        root, ext = os.path.splitext(args.html)
        for k, (frames, solved) in enumerate(record_episodes(policy, cfg, device, n=args.record, scramble=scramble,
                                                             deterministic=not args.stochastic, seed=args.seed)):
            path = args.html if k == 0 else f"{root}-{k + 1}{ext or '.html'}"
            write_steps_page(path, frames, calculus)
            print(f"wrote {path} ({'solved' if solved else 'not solved'} in {len(frames) - 1} steps)")
    return 0
