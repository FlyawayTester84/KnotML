import numpy as np
import pytest
from _util import LINKS

gym = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env  # noqa: E402

from knotml import FramedLink, Handlebody, KirbyKind, LinkDiagram, MoveKind  # noqa: E402
from knotml.envs import CandidateActions, KirbyEnv, ScrambleStart, Transition  # noqa: E402

CONFIGS = [{}, {"calculus": "3-manifold"}, {"closed": True}]


@pytest.mark.parametrize("kwargs", CONFIGS)
def test_passes_the_gymnasium_checker(kwargs):
    check_env(KirbyEnv(**kwargs), skip_render_check=True)


def test_registered():
    import knotml.envs  # noqa: F401

    env = gym.make("knotml/KirbyReduce-v0", calculus="3-manifold")
    assert env.unwrapped.calculus == "3-manifold"


@pytest.mark.parametrize("kwargs", CONFIGS)
def test_reset_is_reproducible(kwargs):
    a, b = KirbyEnv(**kwargs), KirbyEnv(**kwargs)
    obs_a, info_a = a.reset(seed=7)
    obs_b, info_b = b.reset(seed=7)
    assert all(np.array_equal(obs_a[k], obs_b[k]) for k in obs_a)
    assert info_a["scramble"] == info_b["scramble"]


@pytest.mark.parametrize("kwargs", CONFIGS)
def test_legal_actions_are_legal_and_stay_in_bounds(kwargs):
    env = KirbyEnv(max_crossings=24, max_components=6, **kwargs)
    rng = np.random.default_rng(0)
    for seed in range(3):
        env.reset(seed=seed)
        for _ in range(15):
            actions = env.legal_actions()
            assert len(actions)
            for a in actions[:: max(1, len(actions) // 40)]:
                move = env.decode(a)
                assert move is not None and env.is_legal(move)
                assert np.array_equal(env.encode(move), a)
            obs, _, terminated, _, info = env.step(actions[rng.integers(len(actions))])
            assert info["legal"] and env.observation_space.contains(obs)
            assert env.kirby.diagram.n_crossings <= 24 and len(env.kirby.labels()) <= 6
            env.kirby.diagram.check()
            if terminated:
                break


def test_slide_size_is_predicted_exactly():
    env = KirbyEnv(max_crossings=400, max_components=8)
    env.reset(seed=3, options={"diagram": Handlebody.from_pd(LINKS["L2a1{1}"][0], [None, 2])})
    env.step(env.encode(next(m for m in env.legal_moves() if m.kind == KirbyKind.CREATE_12)))
    for move in [m for m in env.legal_moves() if m.kind == KirbyKind.HANDLE_SLIDE][:30]:
        before = env.kirby.diagram.n_crossings
        predicted = env._slide_cost(env.kirby.diagram.label_of_port(move.b))
        assert env.kirby.apply(move).diagram.n_crossings == before + predicted


def test_solving_terminates():
    env = KirbyEnv()
    env.reset(options={"diagram": Handlebody.from_pd(LINKS["L2a1{1}"][0], [None, 0])})  # cancelling pair = B⁴
    (cancel,) = [a for a in env.legal_actions() if env.kinds[a[0]] == KirbyKind.CANCEL_12]
    _, _, terminated, truncated, info = env.step(cancel)
    assert terminated and not truncated and info["legal"] and env.kirby.is_empty()
    env3 = KirbyEnv(calculus="3-manifold")
    env3.reset(options={"diagram": FramedLink.unknot(-1)})
    (down,) = [a for a in env3.legal_actions() if env3.kinds[a[0]] == KirbyKind.BLOW_DOWN]
    assert env3.step(down)[2]


def test_illegal_actions_change_nothing_and_reach_the_reward():
    seen = []
    env = KirbyEnv(reward_fn=lambda t: (seen.append(t), -1.0 if not t.legal else 0.5)[1])
    env.reset(seed=0)
    key = env.kirby.canonical_key()
    bad = np.array([env.kinds.index(MoveKind.R1_REMOVE), env.n_anchors - 1, 0, 0])  # a component slot: no port
    _, reward, terminated, _, info = env.step(bad)
    assert not info["legal"] and reward == -1.0 and not terminated
    assert env.kirby.canonical_key() == key
    good = env.legal_actions()[0]
    assert env.step(good)[1] == 0.5
    assert isinstance(seen[-1], Transition) and seen[-1].legal and seen[-1].move == env.decode(good) is not None


def test_truncates_after_max_steps():
    env = KirbyEnv(max_steps=3)
    env.reset(seed=0)
    bad = np.array([env.kinds.index(MoveKind.R1_REMOVE), env.n_anchors - 1, 0, 0])
    assert [env.step(bad)[3] for _ in range(3)] == [False, False, True]


@pytest.mark.parametrize("calculus", ["4-manifold", "3-manifold"])
def test_scrambled_starts_are_trivial(calculus):
    env = KirbyEnv(calculus=calculus, start=ScrambleStart(n_moves=(4, 10)))
    for seed in range(10):
        env.reset(seed=seed)
        k = env.kirby
        if calculus == "4-manifold":  # B⁴
            assert (k.euler_characteristic(), k.homology(), k.intersection_form()) == (1, [], [])
        assert k.boundary().homology() == [] if calculus == "4-manifold" else k.homology() == []


def test_start_diagram_is_validated():
    env = KirbyEnv()
    with pytest.raises(TypeError):
        env.reset(options={"diagram": FramedLink.unknot(1)})
    with pytest.raises(ValueError):
        KirbyEnv(max_components=1).reset(options={"diagram": Handlebody(LinkDiagram.unknot(2), [None, 0])})


def test_candidate_wrapper():
    env = CandidateActions(KirbyEnv(), max_actions=64)
    obs, _ = env.reset(seed=2)
    assert env.observation_space.contains(obs)
    mask = env.action_masks()
    n = int(mask.sum())
    assert n == int(obs["action_mask"].sum()) and np.all(obs["candidates"][n:] == -1)
    base = env.unwrapped
    assert n == min(64, len(base.legal_moves()))
    if len(base.legal_moves()) > 64:  # simplifying moves are always kept
        kept = {tuple(row) for row in obs["candidates"][:n]}
        assert all(tuple(base.encode(m)) in kept for m in base.legal_moves()
                   if m.kind in (MoveKind.R1_REMOVE, MoveKind.R2_REMOVE, KirbyKind.CANCEL_12))
    _, _, _, _, info = env.step(0)
    assert info["legal"]
    small = CandidateActions(KirbyEnv(), max_actions=4096)  # room to spare: some indices have no candidate
    small.reset(options={"diagram": Handlebody.from_pd(LINKS["L2a1{1}"][0], [None, 0])})
    empty_slot = int(np.flatnonzero(~small.action_masks())[0])
    key = small.unwrapped.kirby.canonical_key()
    obs, _, _, _, info = small.step(empty_slot)
    assert not info["legal"] and small.unwrapped.kirby.canonical_key() == key
    assert small.observation_space.contains(obs)


def test_my_reward_is_registered():
    import pickle

    from knotml.envs import make_reward
    from knotml.envs.my_rewards import my_reward

    assert make_reward("my_reward") is my_reward
    bound = make_reward("my_reward", bonus=2.0)  # config kwargs are passed on every call
    assert bound.func is my_reward and bound.keywords == {"bonus": 2.0}
    assert pickle.loads(pickle.dumps(bound)).keywords == {"bonus": 2.0}  # usable in subprocess envs
    env = KirbyEnv(reward_fn=bound)
    env.reset(seed=0)
    assert isinstance(env.step(env.legal_actions()[0])[1], float)
