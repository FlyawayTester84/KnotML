"""Command-line programs: ``scripts/train_agent.py`` and ``scripts/evaluate_agent.py``."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

__all__ = ["train_main", "evaluate_main", "solve_main"]


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


def solve_main(argv: Optional[Sequence[str]] = None) -> int:
    """``scripts/solve_link.py``: let a trained agent reduce a given Kirby diagram, move by move."""
    from ..cli import add_diagram_arguments

    parser = argparse.ArgumentParser(
        prog="solve_link.py",
        description="Let a trained agent reduce a Kirby diagram to the empty diagram, at most --max-steps moves. "
        "Prints the moves (and a command replaying them) and writes them as a step-through HTML page. "
        "Exit status 0 when solved, 1 when not.",
        epilog="examples:\n"
        "  python scripts/solve_link.py --list-agents\n"
        "  python scripts/solve_link.py --knotinfo \"L2a1{1}\" --framings dot 3 --agent ppo-small-cpu\n"
        "  python scripts/solve_link.py --pd \"[[4,1,3,2],[2,3,1,4]]\" --framings dot 0 --max-steps 20\n"
        "  python scripts/solve_link.py --knotinfo \"L7a6{0}\" --framings dot -2 --attempts 10 --open",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_diagram_arguments(parser)
    parser.add_argument("--agent", help="a folder name under --models, or a checkpoint .pt "
                        "(default: the agent, if --models holds exactly one)")
    parser.add_argument("--checkpoint", default="best", choices=["best", "latest"], help="which checkpoint (best)")
    parser.add_argument("--models", default="models", help="folder of trained agents (default models)")
    parser.add_argument("--list-agents", action="store_true", help="list the agents in --models and exit")
    parser.add_argument("--max-steps", type=int, default=20, help="most moves the agent may try (default 20)")
    parser.add_argument("--attempts", type=int, default=1,
                        help="the first attempt is greedy, later ones sample from the policy; the shortest "
                        "solution is shown (default 1)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", help="cpu (default), cuda or mps")
    parser.add_argument("--max-crossings", type=int, help="environment capacity (default: the agent's, raised to fit)")
    parser.add_argument("--max-components", type=int, help="environment capacity (default: the agent's, raised to fit)")
    parser.add_argument("--max-actions", type=int, help="candidate moves shown to the agent (default: the agent's)")
    parser.add_argument("--html", default="outputs/solve.html",
                        help="the step-through page (default outputs/solve.html)")
    parser.add_argument("--no-html", action="store_true", help="do not write the page")
    parser.add_argument("--open", action="store_true", help="open the page in the browser")
    parser.add_argument("--trace", action="store_true", help="print the PD code after every move")
    args = parser.parse_args(argv)

    import shlex

    import torch

    from ..cli import diagram_from_args, parse_framings, source_args
    from ..notation import format_components
    from ..play.session import components_of
    from .solve import Agent, list_agents, reachable, solve

    agents = list_agents(args.models)
    if args.list_agents:
        if not agents:
            print(f"no agents in {args.models}/")
        for a in agents:
            closed = " (closed)" if a["closed"] else ""
            print(f"{a['name']:<20} {a['calculus']}{closed}, {a['steps']:,} training steps, "
                  f"scrambles up to {a['scramble'][1]}, capacity {a['max_crossings']} crossings / "
                  f"{a['max_components']} components, {a['policy']} hidden {a.get('hidden')} layers {a.get('layers')}"
                  f"  [{a['path']}]")
        return 0
    if not (args.braid or args.pd or args.knotinfo or args.snappy):
        parser.error("give a link: --knotinfo NAME, --pd CODE, --braid WORD or --snappy NAME")
    if not args.framings:
        parser.error("give --framings: one per component in label order, an integer or 'dot' (a 1-handle)")
    if args.agent is None:
        if len(agents) != 1:
            names = ", ".join(a["name"] for a in agents) or "none"
            parser.error(f"choose an agent with --agent (agents in {args.models}/: {names})")
        args.agent = agents[0]["name"]
    device = torch.device(args.device or "cpu")
    try:
        agent = Agent.load(args.agent, args.checkpoint, device, args.models, args.max_crossings,
                           args.max_components, args.max_actions)
        if args.closed and not agent.closed:
            parser.error(f"--closed, but agent {agent.name!r} was trained without 2/3-handle moves")
        start = agent.start_from(diagram_from_args(args, default=None), parse_framings(args.framings))
        if args.max_crossings is None and args.max_components is None:
            agent = agent.with_capacity_for(start)
    except (FileNotFoundError, ValueError) as err:
        parser.error(str(err))

    env, trained = agent.config.env, agent.trained
    print(f"agent {agent.name} ({agent.path}): {agent.calculus}{' (closed)' if agent.closed else ''}, "
          f"{trained['steps']:,} training steps on scrambles of up to {trained['scramble'][1]} moves")
    if (env.max_crossings, env.max_components) != (trained["max_crossings"], trained["max_components"]):
        print(f"  capacity raised to {env.max_crossings} crossings / {env.max_components} components "
              f"(trained with {trained['max_crossings']} / {trained['max_components']}): larger than it has seen")
    ok, note = reachable(start)
    print(f"start: {start.diagram.n_crossings} crossings, components {format_components(components_of(start))}")
    print(f"  {note}" + ("" if ok else " -> the empty diagram cannot be reached; running the agent anyway"))

    attempts = solve(agent, start, max_steps=args.max_steps, attempts=max(1, args.attempts), seed=args.seed)
    for k, a in enumerate(attempts, 1):
        how = "sampled" if a.sampled else "greedy"
        n = f"{len(a.steps)} move{'s' if len(a.steps) != 1 else ''}"
        verdict = {"solved": f"solved in {n}", "cycle": f"stopped after {n}: "
                   "back at an earlier diagram (a greedy agent would loop)", "no legal move": "no legal move left",
                   "step limit": f"not solved within {args.max_steps} moves"}[a.stop]
        print(f"attempt {k} ({how}): {verdict}")
    solved = [a for a in attempts if a.solved]
    best = min(solved, key=lambda a: len(a.steps)) if solved else attempts[0]
    plural = "s" if len(best.steps) != 1 else ""
    print(f"\n{'solution' if solved else 'greedy attempt'} ({len(best.steps)} move{plural}):")
    for i, (step, frame) in enumerate(zip(best.steps, best.frames[1:], strict=True), 1):
        print(f"  {i:>3}. {step.spec:<22} p {step.probability:.2f}  value {step.value:+.2f}  -> "
              f"{step.n_crossings} crossings, {step.n_components} components")
        if args.trace:
            print(f"       PD {frame[1].to_pd()}")
    if solved:
        flags = " --four" if agent.calculus == "4-manifold" and None not in parse_framings(args.framings) else ""
        flags += " --closed" if agent.closed else ""
        moves = " ".join(f"--move {shlex.quote(s.spec)}" for s in best.steps)
        print("\nreplay and check the invariants along the way:\n"
              f"  python scripts/inspect_diagram.py {source_args(args)} --framings {' '.join(args.framings)}{flags} "
              f"{moves} --verify")
    if not args.no_html:
        try:
            from ..visualization import write_steps_page

            write_steps_page(args.html, best.frames, "4" if agent.calculus == "4-manifold" else "3")
            print(f"\nwrote {args.html}")
            if args.open:
                import webbrowser
                from pathlib import Path

                webbrowser.open(Path(args.html).resolve().as_uri())
        except ImportError as err:  # drawing needs spherogram/plink (the oracles extra)
            print(f"\nno page: {err} (install the oracles extra to draw)")
    return 0 if solved else 1
