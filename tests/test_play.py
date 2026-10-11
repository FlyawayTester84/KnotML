import re

import numpy as np
import pytest
from _util import LINKS

from knotml import ALL_KINDS, FramedLink, Handlebody, KirbyKind, KirbyMove, LinkDiagram, Move, MoveKind
from knotml.notation import describe_move, move_crossings, move_labels, move_spec, parse_move

gym = pytest.importorskip("gymnasium")
from knotml.envs import KirbyEnv, ScrambleStart  # noqa: E402
from knotml.play import Console, PlaySession, example_reward, load_reward  # noqa: E402

L7A6 = [[8, 1, 9, 2], [10, 4, 11, 3], [14, 10, 7, 9], [12, 6, 13, 5], [2, 7, 3, 8], [4, 12, 5, 11], [6, 14, 1, 13]]


def cancelling_pair():
    return Handlebody.from_pd(LINKS["L2a1{1}"][0], [None, 0])  # B⁴: one CANCEL_12 empties it


def test_specs_round_trip_and_describe():
    mazur = Handlebody.from_pd(L7A6, [None, -2]).apply(KirbyMove(KirbyKind.CREATE_12, sign=1))
    closed = Handlebody.from_pd(LINKS["L2a1{1}"][0], [None, 0], closed=True).apply(KirbyMove(KirbyKind.CREATE_23))
    framed = FramedLink(LinkDiagram.from_pd(LINKS["L2a1{1}"][0], n_unknots=1), [1, 2, -1])  # K2: a split -1 unknot
    knot = LinkDiagram.from_braid([1, 2, 1, -2, 1])
    knot.apply_inplace(Move(MoveKind.R1_ADD, 0, sign=1))
    seen = set()
    for obj, moves in [(mazur, mazur.legal_moves()), (closed, closed.legal_moves()), (framed, framed.legal_moves()),
                       (knot, knot.legal_moves(ALL_KINDS))]:
        for m in moves:
            assert parse_move(move_spec(m)) == m
            assert describe_move(obj, m)
            seen.add(m.kind)
    assert seen == set(MoveKind) | set(KirbyKind)  # every kind of move is covered
    assert parse_move("handle_slide 14, 26") == KirbyMove(KirbyKind.HANDLE_SLIDE, 14, 26)


@pytest.mark.parametrize("spec", ["BOGUS 1", "R3", "R1_ADD 3", "R2_ADD a b", "CREATE_23 1", ""])
def test_bad_specs_raise(spec):
    with pytest.raises(ValueError):
        parse_move(spec)


def test_what_a_move_points_at():
    H = Handlebody.from_pd(LINKS["L2a1{1}"][0], [None, 0], closed=True).apply(KirbyMove(KirbyKind.CREATE_23))
    (cancel,) = [m for m in H.legal_moves() if m.kind == KirbyKind.CANCEL_12]
    assert move_labels(H, cancel) == {0, 1} and move_crossings(H.diagram, cancel) == set()
    slide = next(m for m in H.legal_moves() if m.kind == KirbyKind.HANDLE_SLIDE)
    assert move_crossings(H.diagram, slide) == {p >> 2 for p in (slide.a, slide.b) if p >= 0}
    kink = Move(MoveKind.R1_ADD, -1, sign=1)  # on the crossingless 0-framed unknot, label 2
    assert move_labels(H, kink) == {2} and move_crossings(H.diagram, kink) == set()
    r2 = next(m for m in H.legal_moves() if m.kind == MoveKind.R2_ADD and m.a >= 0)
    assert len(move_crossings(H.diagram, r2)) in (1, 2) and move_labels(H, r2) <= {2}


def test_restore_continues_an_episode():
    env = KirbyEnv()
    env.reset(seed=3)
    saved, key = env.kirby.copy(), env.kirby.canonical_key()
    env.step(env.legal_actions()[0])
    obs = env.restore(saved, steps=7)
    assert env.kirby.canonical_key() == key and env.steps == 7 and env.observation_space.contains(obs)
    assert env.kirby is not saved
    with pytest.raises(TypeError):
        env.restore(FramedLink.unknot(1))


def test_session_steps_undo_and_preview():
    session = PlaySession(KirbyEnv(max_steps=10), start=cancelling_pair())
    key = session.state.canonical_key()
    (i,) = [k for k, m in enumerate(session.moves()) if m.kind == KirbyKind.CANCEL_12]
    session.preview_move(session.moves()[i])
    move, after, _ = session.preview
    assert after.is_empty() and session.state.canonical_key() == key  # previews change nothing
    session.preview_move(KirbyMove(KirbyKind.CANCEL_12, a=1))  # K1 is the 2-handle: illegal
    assert session.preview[1] is None

    bad = session.step_move(KirbyMove(KirbyKind.CANCEL_12, a=1))
    assert not bad.legal and session.env.steps == 1 and session.state.canonical_key() == key
    with pytest.raises(ValueError):
        session.step_move(KirbyMove(KirbyKind.BLOW_UP, sign=1))  # not in the 4-manifold calculus
    record = session.step_index(i)
    assert record.legal and record.terminated and session.done and session.state.is_empty()
    with pytest.raises(RuntimeError):
        session.step_index(0)
    assert [frame[0] for frame in session.history()] == ["start", "CANCEL_12 1 (illegal)", "CANCEL_12 0"]

    assert session.undo(5) == 2 and session.env.steps == 0 and session.state.canonical_key() == key
    assert not session.done and session.undo() == 0
    session.reset(seed=1)
    assert session.state.canonical_key() == key and not session.records


def test_console_commands():
    console = Console(PlaySession(KirbyEnv(calculus="3-manifold"), seed=2))
    run = lambda line: console.execute(line)  # noqa: E731
    assert "step N" in run("help")[0]
    out, changed = run("moves")
    assert "legal moves" in out and not changed
    n = len(console.session.moves())
    assert run("step 0")[1] and console.session.env.steps == 1
    assert run("0")[1] and console.session.env.steps == 2  # a bare index steps too
    assert run("undo 2")[0].startswith("took back 2") and console.session.env.steps == 0
    assert len(console.session.moves()) == n
    assert "reward" in run("random 3")[0] and console.session.env.steps == 3
    assert "illegal" in run("action 0 0 0 0")[0]
    assert "BLOW_UP" in run("BLOW_UP 1")[0]  # a bare spec steps too
    assert run("preview 0")[1] and console.session.preview is not None
    assert run("preview off")[1] and console.session.preview is None
    assert "n_crossings" in run("obs")[0]
    for line, error in [("step 99999", "no legal move"), ("action 1 2", "four integers"), ("frobnicate", "unknown"),
                        ("step R9 1", "unknown move kind"), ("moves NOPE", "unknown kind"), ("step 'x", "error")]:
        out, changed = run(line)
        assert error in out and not changed, line
    assert run("quit") == ("", False) and console.finished


def test_rewards():
    assert load_reward("zero") is not example_reward
    assert load_reward("example") is example_reward
    assert load_reward("knotml.play.session:example_reward") is example_reward
    with pytest.raises(ValueError):
        load_reward("nonsense")
    session = PlaySession(KirbyEnv(reward_fn=example_reward), start=cancelling_pair())
    assert session.step_move(KirbyMove(KirbyKind.CANCEL_12, a=1)).reward == -1.0
    assert session.step_move(KirbyMove(KirbyKind.CANCEL_12, a=0)).reward == 10.0


def test_page(tmp_path):
    pytest.importorskip("plink")
    pytest.importorskip("spherogram")
    page = tmp_path / "play.html"
    console = Console(PlaySession(KirbyEnv(), start=Handlebody.from_pd(L7A6, [None, -2])), str(page))
    console.render()
    text = page.read_text()
    version = re.search(r'"version": (\d+)', text).group(1)
    assert (tmp_path / "play.version.js").read_text().strip() == f"window.knotmlVersion = {version};"
    assert 'class="spot"' in text and 'data-cmd="step 0"' in text and "HANDLE_SLIDE (" in text
    console.execute("preview 0")
    console.render()
    text2 = page.read_text()
    assert "<b>Preview</b>" in text2 and re.search(r'"version": (\d+)', text2).group(1) != version
    out, _ = console.execute("save")
    assert out == f"wrote {tmp_path / 'play_episode.html'}" and (tmp_path / "play_episode.html").exists()
    console.session.step_index(int(np.argmax([m.kind == KirbyKind.HANDLE_SLIDE for m in console.session.moves()])))
    console.render()
    assert "last step: <code>HANDLE_SLIDE" in page.read_text()


def test_pd_command_reenters_the_same_diagram():
    import random

    from knotml.cli import parse_framings
    from knotml.play.session import pd_summary

    rng = random.Random(4)
    checked = 0
    for seed in range(6):
        session = PlaySession(KirbyEnv(), seed=seed)
        for _ in range(rng.randrange(4)):
            session.random_step()
        state, info = session.state, pd_summary(session.state)
        if info["args"] is None or state.diagram.is_pd_ambiguous():
            continue  # crossingless components, or an orientation a PD code cannot record
        again = Handlebody(LinkDiagram.from_pd(info["pd"]), parse_framings(info["framings"]))
        assert again.canonical_key() == state.canonical_key()
        assert "--pd" in info["args"] and "--framings" in info["args"]
        checked += 1
    assert checked >= 3
    console = Console(PlaySession(KirbyEnv(), start=cancelling_pair()))
    out, changed = console.execute("pd")
    assert out.startswith("PD: [") and "framings, in --pd order (K0 K1): dot 0" in out and not changed
    empty = Console(PlaySession(KirbyEnv(start=ScrambleStart(n_moves=(0, 0)))))
    assert empty.execute("pd")[0] == "PD: [] (the empty diagram)"
