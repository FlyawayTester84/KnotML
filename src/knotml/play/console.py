"""Play a :class:`~knotml.envs.KirbyEnv` episode from the terminal, watching it on a static HTML page.

Run ``python scripts/play_env.py --help``.  Each command is one line; after
every command that changes the state, the page is rewritten and an open copy
reloads itself.
"""

from __future__ import annotations

import argparse
import os
import shlex
import webbrowser
from pathlib import Path
from typing import Optional

import numpy as np

from ..envs.kirby_env import KirbyEnv, ScrambleStart
from ..handlebody import Handlebody
from ..kirby import KirbyKind
from ..core import MoveKind
from ..notation import describe_move, move_spec, parse_move
from .page import COMMANDS, observation_text, write_page
from .session import PlaySession, Record, load_reward, pd_summary

__all__ = ["Console", "main"]

_KINDS = {**MoveKind.__members__, **KirbyKind.__members__}


class Console:
    """Turns command lines into session calls; :meth:`execute` returns the text to print."""

    def __init__(self, session: PlaySession, page: Optional[str] = None, max_rows: int = 200) -> None:
        self.session, self.page, self.max_rows = session, page, max_rows
        self.finished = False

    def render(self) -> None:
        if self.page:
            write_page(self.session, self.page, self.max_rows)

    def execute(self, line: str) -> tuple[str, bool]:
        """Run one command: ``(output, whether the page needs rewriting)``."""
        try:
            words = shlex.split(line)
        except ValueError as err:
            return f"error: {err}", False
        if not words:
            return "", False
        cmd, args = words[0].lower(), words[1:]
        try:
            if cmd.lstrip("-").isdigit() or cmd.upper() in _KINDS:  # bare "12" or "HANDLE_SLIDE 14 26" = step
                cmd, args = "step", words
            handler = getattr(self, f"cmd_{cmd}", None)
            if handler is None:
                return f"unknown command {cmd!r} (try help)", False
            return handler(args)
        except (ValueError, RuntimeError) as err:
            return f"error: {err}", False

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def cmd_help(self, args) -> tuple[str, bool]:
        width = max(len(c) for c, _ in COMMANDS)
        lines = [f"  {c.ljust(width)}  {t}" for c, t in COMMANDS]
        return "commands:\n" + "\n".join(lines), False

    cmd_h = cmd_help

    def cmd_step(self, args) -> tuple[str, bool]:
        if not args:
            raise ValueError("step N, or step SPEC (e.g. step HANDLE_SLIDE 14 26)")
        if len(args) == 1 and args[0].lstrip("-").isdigit():
            record = self.session.step_index(int(args[0]))
        else:
            record = self.session.step_move(parse_move(" ".join(args)))
        return self._report(record), True

    def cmd_action(self, args) -> tuple[str, bool]:
        space = self.session.env.action_space
        try:
            vector = np.array([int(v) for v in args], dtype=np.int64)
        except ValueError:
            raise ValueError("action K A B P takes four integers") from None
        if vector.shape != (4,) or not space.contains(vector):
            raise ValueError(f"an action is four integers below {space.nvec.tolist()}")
        return self._report(self.session.step(vector)), True

    def cmd_random(self, args) -> tuple[str, bool]:
        n, kinds = 1, None
        for a in args:
            if a.isdigit():
                n = int(a)
            elif a.upper() in _KINDS:
                kinds = {_KINDS[a.upper()]}
            else:
                raise ValueError("random [N] [KIND]")
        out = []
        for _ in range(n):
            if self.session.done:
                break
            record = self.session.random_step(kinds)
            if record is None:
                out.append("no legal move" + (f" of kind {next(iter(kinds)).name}" if kinds else ""))
                break
            out.append(self._report(record))
        return "\n".join(out), bool(out)

    def cmd_preview(self, args) -> tuple[str, bool]:
        if not args or args[0].lower() == "off":
            self.session.preview_move(None)
            return "preview cleared", True
        if len(args) == 1 and args[0].isdigit():
            moves = self.session.moves()
            i = int(args[0])
            if i >= len(moves):
                raise ValueError(f"no legal move {i} (there are {len(moves)})")
            move = moves[i]
        else:
            move = parse_move(" ".join(args))
        self.session.preview_move(move)
        _, after, _ = self.session.preview
        if after is None:
            return f"{move_spec(move)} is not legal here (preview shows nothing)", True
        n, k = after.diagram.n_crossings, len(after.labels())
        return f"preview {move_spec(move)}: {n} crossings, {k} components", True

    def cmd_moves(self, args) -> tuple[str, bool]:
        kinds = None
        if args:
            name = args[0].upper()
            if name not in _KINDS:
                raise ValueError(f"unknown kind {name}; choose from {', '.join(_KINDS)}")
            kinds = {_KINDS[name]}
        session = self.session
        moves = [(i, m) for i, m in enumerate(session.moves()) if kinds is None or m.kind in kinds]
        if kinds is None:  # a few of each kind; `moves KIND` lists one kind in full
            by_kind: dict = {}
            for i, m in moves:
                by_kind.setdefault(m.kind, []).append((i, m))
            lines = [f"{len(moves)} legal moves (moves KIND lists all of one kind)"]
            for kind, rows in by_kind.items():
                lines.append(f"{kind.name} ({len(rows)})")
                lines += [f"  {i:4d}  {move_spec(m):<22}  {describe_move(session.state, m)}" for i, m in rows[:5]]
                if len(rows) > 5:
                    lines.append(f"        ... {len(rows) - 5} more")
            return "\n".join(lines), False
        lines = [f"{len(moves)} legal moves"]
        lines += [f"  {i:4d}  {move_spec(m):<22}  {describe_move(session.state, m)}" for i, m in moves]
        return "\n".join(lines), False

    def cmd_undo(self, args) -> tuple[str, bool]:
        n = self.session.undo(int(args[0]) if args else 1)
        return (f"took back {n} step(s); {self._summary()}" if n else "nothing to undo"), bool(n)

    def cmd_reset(self, args) -> tuple[str, bool]:
        seed = int(args[0]) if args else None
        self.session.reset(seed)
        start = ", ".join(move_spec(m) for m in self.session.scramble)
        text = "new episode" + (f" (seed {seed})" if seed is not None else "")
        text += f", scramble: {start}" if start else ""
        return f"{text}; {self._summary()}", True

    def cmd_obs(self, args) -> tuple[str, bool]:
        return observation_text(self.session.obs), False

    def cmd_pd(self, args) -> tuple[str, bool]:
        info = pd_summary(self.session.state)
        if not info["labels"] and not info["circles"]:
            return "PD: [] (the empty diagram)", False
        lines = [f"PD: {info['pd']}"]
        if info["labels"]:
            names = " ".join(f"K{lab}" for lab in info["labels"])
            lines.append(f"framings, in --pd order ({names}): {' '.join(info['framings'])}")
        if info["circles"]:
            circles = ", ".join(f"K{lab} ({'dot' if f is None else f})" for lab, f in info["circles"].items())
            lines.append(f"crossingless, not in the PD code: {circles}")
        if info["args"]:
            lines.append(f"enter it again: python scripts/play_env.py {info['args']}")
        return "\n".join(lines), False

    def cmd_save(self, args) -> tuple[str, bool]:
        from ..visualization import write_steps_page

        root = os.path.splitext(self.page or "outputs/play.html")[0]
        path = args[0] if args else root + "_episode.html"
        calculus = "4" if isinstance(self.session.state, Handlebody) else "3"
        write_steps_page(path, self.session.history(), calculus)
        return f"wrote {path}", False

    def cmd_open(self, args) -> tuple[str, bool]:
        if not self.page:
            return "no page (started with --no-html)", False
        webbrowser.open(Path(self.page).resolve().as_uri())
        return f"opened {self.page}", False

    def cmd_quit(self, args) -> tuple[str, bool]:
        self.finished = True
        return "", False

    cmd_exit = cmd_q = cmd_quit

    # ------------------------------------------------------------------

    def _summary(self) -> str:
        s = self.session
        return (f"{s.state.diagram.n_crossings} crossings, {len(s.state.labels())} components, "
                f"{len(s.moves())} legal moves")

    def _report(self, record: Record) -> str:
        s = self.session
        text = (f"step {s.env.steps}: {record.label}  reward {record.reward:+.3g}  return {s.total_return:+.3g}  "
                f"-> {self._summary()}")
        if record.terminated:
            text += f"\nterminated: the empty diagram after {s.env.steps} steps (undo, or reset for a new episode)"
        elif record.truncated:
            text += f"\ntruncated: {s.env.max_steps} steps (undo, or reset for a new episode)"
        return text


def build(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Console:
    """The console the command-line options describe."""
    from ..cli import diagram_from_args, kirby_from_args

    start = None
    d = diagram_from_args(args, default=None)
    if d is not None:
        if not args.framings:
            parser.error("give --framings for the diagram (the environment plays Kirby diagrams)")
        try:
            start = kirby_from_args(args, d)
        except ValueError as err:
            parser.error(f"--framings: {err}")
        calculus = "4-manifold" if isinstance(start, Handlebody) else "3-manifold"
        closed = bool(getattr(start, "closed", False))
    else:
        calculus = "3-manifold" if args.calculus == "3" else "4-manifold"
        closed = args.closed
    try:
        reward = load_reward(args.reward)
    except (ValueError, ImportError, AttributeError) as err:
        parser.error(f"--reward: {err}")
    env = KirbyEnv(
        calculus=calculus,
        closed=closed,
        max_crossings=args.max_crossings,
        max_components=args.max_components,
        max_steps=args.max_steps,
        reward_fn=reward,
        start=ScrambleStart(n_moves=tuple(args.scramble)),
    )
    try:
        session = PlaySession(env, start=start, seed=args.seed)
    except ValueError as err:
        parser.error(str(err))
    return Console(session, None if args.no_html else args.html, args.max_rows)


def main(argv=None) -> int:
    from ..cli import add_diagram_arguments

    parser = argparse.ArgumentParser(
        prog="play_env.py",
        description="Play the Kirby environment from the terminal and watch it on an HTML page.",
        epilog="examples:\n"
        "  python scripts/play_env.py                                  # 4-manifold, scrambled start\n"
        "  python scripts/play_env.py --calculus 3 --seed 4 --reward example\n"
        "  python scripts/play_env.py --knotinfo \"L2a1{1}\" --framings dot 0   # a cancelling pair (B^4)\n"
        "  python scripts/play_env.py --knotinfo \"L7a6{0}\" --framings dot -2 --max-crossings 120",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_diagram_arguments(parser)
    parser.add_argument("--calculus", choices=["3", "4"], default="4",
                        help="without a diagram: 3-manifold (FramedLink) or 4-manifold (Handlebody) calculus")
    parser.add_argument("--seed", type=int, help="seed of the first episode and of `random`")
    parser.add_argument("--scramble", type=int, nargs=2, default=[2, 8], metavar=("MIN", "MAX"),
                        help="without a diagram: number of random construction moves for a start (default 2 8)")
    parser.add_argument("--max-crossings", type=int, default=64, help="environment capacity (default 64)")
    parser.add_argument("--max-components", type=int, default=8, help="environment capacity (default 8)")
    parser.add_argument("--max-steps", type=int, default=200, help="truncation (default 200)")
    parser.add_argument("--reward", default="zero",
                        help="zero (default), sparse, potential, my_reward (src/knotml/envs/my_rewards.py), "
                        "example, or module:function (knotml.envs.rewards)")
    parser.add_argument("--html", default="outputs/play.html", help="the page (default outputs/play.html)")
    parser.add_argument("--no-html", action="store_true", help="terminal only")
    parser.add_argument("--no-open", action="store_true", help="do not open the page in the browser")
    parser.add_argument("--max-rows", type=int, default=200, help="moves shown per kind on the page (default 200)")
    args = parser.parse_args(argv)
    console = build(args, parser)

    try:
        import readline  # noqa: F401  (line editing and history for input())
    except ImportError:
        pass
    session = console.session
    print(f"KirbyReduce, {session.env.calculus}{' (closed)' if session.env.closed else ''}: "
          f"reduce the diagram to the empty diagram. Type help for commands.")
    if session.scramble:
        print("scramble:", ", ".join(move_spec(m) for m in session.scramble))
    console.render()
    print(console._summary())
    if console.page:
        print(f"page: {console.page}")
        if not args.no_open:
            console.cmd_open([])
    while not console.finished:
        try:
            line = input("kirby> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        output, changed = console.execute(line)
        if output:
            print(output)
        if changed:
            console.render()
    return 0
