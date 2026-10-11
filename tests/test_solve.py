import re
from pathlib import Path

import pytest
from _util import LINKS

torch = pytest.importorskip("torch")
pytest.importorskip("yaml")

from knotml import Handlebody, KirbyKind, KirbyMove, LinkDiagram, Move, MoveKind  # noqa: E402
from knotml.notation import parse_move  # noqa: E402
from knotml.rl.cli import solve_main  # noqa: E402
from knotml.rl.solve import Agent, find_checkpoint, list_agents, reachable, solve  # noqa: E402

MODELS = Path(__file__).resolve().parent.parent / "models"
AGENT = "ppo-small-cpu"
L7A6 = [[8, 1, 9, 2], [10, 4, 11, 3], [14, 10, 7, 9], [12, 6, 13, 5], [2, 7, 3, 8], [4, 12, 5, 11], [6, 14, 1, 13]]

pytestmark = pytest.mark.skipif(not (MODELS / AGENT).exists(), reason="the saved agent is not in models/")


def pair(framing=0):
    return Handlebody.from_pd(LINKS["L2a1{1}"][0], [None, framing])  # a cancelling pair: B⁴


def test_agents_are_listed_and_found():
    names = {a["name"]: a for a in list_agents(MODELS)}
    assert names[AGENT]["calculus"] == "4-manifold" and names[AGENT]["steps"] > 0
    assert find_checkpoint(AGENT, "latest", MODELS).name == "latest.pt"
    assert find_checkpoint(MODELS / AGENT / "checkpoints" / "best.pt").name == "best.pt"
    with pytest.raises(FileNotFoundError, match=AGENT):  # the message lists what exists
        find_checkpoint("no-such-agent", models_dir=MODELS)


def test_solves_a_cancelling_pair_with_kinks_and_the_moves_replay():
    agent = Agent.load(AGENT, models_dir=MODELS)
    start = pair(3)
    start.apply_inplace(Move(MoveKind.R1_ADD, 0, sign=1))
    start.apply_inplace(Move(MoveKind.R1_ADD, 1, sign=-1))
    (attempt,) = solve(agent, start, max_steps=20)
    assert attempt.solved and not attempt.sampled
    assert len(attempt.frames) == len(attempt.steps) + 1 and attempt.frames[0][0] == "start"
    replay = start.copy()
    for step in attempt.steps:  # the printed specs are moves of the same diagram
        replay.apply_inplace(parse_move(step.spec))
    assert replay.is_empty()
    assert all(0.0 < s.probability <= 1.0 for s in attempt.steps)


def test_step_limit_and_cycles():
    agent = Agent.load(AGENT, models_dir=MODELS)
    mazur = Handlebody.from_pd(L7A6, [None, -2])  # contractible, but its boundary is not S³: never solvable
    ok, note = reachable(mazur)
    assert ok and "necessary" in note
    attempts = solve(agent, mazur, max_steps=5, attempts=3, seed=1, stop_on_cycle=False)
    assert [a.sampled for a in attempts] == [False, True, True]
    assert all(not a.solved and len(a.steps) <= 5 and a.stop == "step limit" for a in attempts)
    (greedy,) = solve(agent, mazur, max_steps=50)
    assert greedy.stop in ("cycle", "step limit") and len(greedy.steps) <= 50
    ok, note = reachable(Handlebody(LinkDiagram.unknot(1), [1]))  # CP² minus a ball: χ = 2
    assert not ok and "euler" in note


def test_capacity_is_raised_for_larger_diagrams():
    agent = Agent.load(AGENT, models_dir=MODELS)
    big = pair(0)
    for i in range(36):  # 38 crossings: more than the 32 the agent was trained with
        big.apply_inplace(Move(MoveKind.R1_ADD, 0, sign=1 if i % 2 else -1))
    roomy = agent.with_capacity_for(big)
    assert roomy.config.env.max_crossings >= 38 + 16 and roomy.trained["max_crossings"] == 32
    assert agent.with_capacity_for(pair()) is agent
    (attempt,) = solve(roomy, big, max_steps=3)
    assert len(attempt.steps) <= 3


def test_command_line(tmp_path, capsys):
    pd = str(LINKS["L2a1{1}"][0])
    common = ["--models", str(MODELS), "--agent", AGENT, "--no-html"]
    assert solve_main(["--pd", pd, "--framings", "dot", "5", *common]) == 0
    out = capsys.readouterr().out
    assert "solved in 1 move" in out and "--move 'CANCEL_12 0'" in out and "--verify" in out
    assert solve_main(["--pd", str(L7A6), "--framings", "dot", "-2", "--max-steps", "4", *common]) == 1
    assert "necessary condition only" in capsys.readouterr().out
    assert solve_main(["--list-agents", "--models", str(MODELS)]) == 0
    assert re.search(rf"^{AGENT}\s+4-manifold", capsys.readouterr().out, re.M)
    with pytest.raises(SystemExit):
        solve_main(["--pd", pd, *common])  # no --framings
    with pytest.raises(SystemExit):
        solve_main(["--pd", pd, "--framings", "dot", "0", *common, "--agent", "nope"])
    if len(list_agents(MODELS)) > 1:  # several agents: --agent is required, and the error names them
        with pytest.raises(SystemExit):
            solve_main(["--pd", pd, "--framings", "dot", "0", "--models", str(MODELS), "--no-html"])
        assert AGENT in capsys.readouterr().err
    assert KirbyMove(KirbyKind.CANCEL_12, a=0) == parse_move("CANCEL_12 0")
