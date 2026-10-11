import dataclasses
import math
import pickle

import numpy as np
import pytest
from _util import LINKS

torch = pytest.importorskip("torch")
pytest.importorskip("yaml")
pytest.importorskip("gymnasium")

from knotml import Handlebody, KirbyKind, KirbyMove  # noqa: E402
from knotml.envs import KirbyEnv, Potential, Sparse, make_reward  # noqa: E402
from knotml.envs.kirby_env import Transition  # noqa: E402
from knotml.rl import Trainer, evaluate, load_config, load_policy, make_policy, record_episodes  # noqa: E402
from knotml.rl.config import CurriculumConfig  # noqa: E402
from knotml.rl.curriculum import Curriculum  # noqa: E402
from knotml.rl.envs import env_spec, make_env, make_vector_env  # noqa: E402
from knotml.rl.rollout import RolloutBuffer, obs_to_torch  # noqa: E402

CPU = torch.device("cpu")


def tiny(tmp_path=None, *extra):
    """A config small enough for unit tests."""
    sets = ["run.device=cpu", "run.tensorboard=false", "env.num_envs=2", "env.vectorization=sync",
            "env.max_crossings=16", "env.max_actions=64", "env.max_steps=8", "algorithm.num_steps=8",
            "algorithm.num_minibatches=2", "algorithm.update_epochs=2", "policy.hidden=16", "policy.layers=1",
            "eval.episodes=4", "eval.num_envs=2", "eval.every=2", "run.checkpoint_every=2",
            "run.total_timesteps=64", "run.keep_checkpoints=1", *extra]
    if tmp_path is not None:
        sets += [f"run.output_dir={tmp_path}", "run.name=test"]
    return load_config("configs/ppo.yaml", sets)


def test_config_file_overrides_and_errors():
    overrides = ["algorithm.learning_rate=1e-4", "env.scramble=[2, 5]", "eval.scramble=[1,3]"]
    cfg = load_config("configs/ppo.yaml", overrides)
    assert cfg.algorithm.name == "ppo" and cfg.algorithm.params["learning_rate"] == 1e-4
    assert cfg.env.scramble == (2, 5) and cfg.eval.scramble == (1, 3) and cfg.policy.name == "gnn"
    assert load_config(None).to_dict()["env"]["scramble"] == [1, 3]
    assert type(cfg).from_dict(cfg.to_dict()) == cfg
    for bad in (["env.max_crosings=10"], ["nonsense.key=1"], ["env"]):
        with pytest.raises(ValueError):
            load_config("configs/ppo.yaml", bad)
    spec = env_spec(cfg)
    with pytest.raises(ValueError, match="hiden"):
        make_policy("gnn", spec, {"hiden": 3})
    with pytest.raises(ValueError, match="unknown policy"):
        make_policy("transformer", spec, {})


def test_rewards():
    H = Handlebody.from_pd(LINKS["L2a1{1}"][0], [None, 0])
    after = H.apply(KirbyMove(KirbyKind.CANCEL_12, a=0))
    solve = Transition(H, KirbyMove(KirbyKind.CANCEL_12, a=0), after, True, True)
    noop = Transition(H, None, H, False, False)
    assert Sparse()(solve) == 1.0 and Sparse()(noop) == -0.1
    pot = Potential(gamma=0.9)
    assert pot(solve) == pytest.approx(1.0 + 0.9 * 0 + (0.1 * 2 + 0.01 * 2))  # phi(empty) = 0
    assert pot(noop) == pytest.approx(-0.1 + (0.9 - 1.0) * pot.phi(H))
    assert isinstance(pickle.loads(pickle.dumps(pot)), Potential)  # works in subprocess vector envs
    assert make_reward("potential", gamma=0.5).gamma == 0.5
    with pytest.raises(ValueError):
        make_reward("nonsense")
    with pytest.raises(ValueError):
        make_reward("sparse", wrong=1)


def test_action_fields_match_the_encoding():
    env = KirbyEnv(closed=True)
    fields = env.action_fields()
    for seed in range(3):
        env.reset(seed=seed)
        for move in env.legal_moves():
            vector = env.encode(move)
            used = fields[env.kinds.index(move.kind)]
            assert all(vector[1 + j] == 0 for j in range(3) if not used[j]), move


def _batch(cfg, seeds=(0, 1, 2)):
    env = make_env(dataclasses.asdict(cfg.env), dataclasses.asdict(cfg.reward), 0.99, (3, 5))
    obs = [env.reset(seed=s)[0] for s in seeds]
    return {k: np.stack([o[k] for o in obs]) for k in obs[0]}


def test_policy_masks_and_ignores_crossing_numbering():
    cfg = tiny()
    spec = env_spec(cfg)
    torch.manual_seed(0)
    policy = make_policy("gnn", spec, {"hidden": 16, "layers": 2}).double().eval()
    batch = _batch(cfg)
    obs = obs_to_torch(batch, CPU)
    logits, mask, value = policy(obs)
    assert logits.shape == mask.shape == (3, spec.n_actions) and value.shape == (3,)
    dist, _ = policy.distribution(obs)
    assert torch.all(dist.probs[~mask] == 0)
    assert torch.allclose(dist.probs.sum(-1), torch.ones(3, dtype=dist.probs.dtype))
    action = policy.act(obs, deterministic=True)[0]
    assert torch.all(mask[torch.arange(3), action])

    # renumber the crossings of each diagram: same value, same logit for each candidate
    rng = np.random.default_rng(0)
    P = spec.n_ports
    perm = {k: v.copy() for k, v in batch.items()}
    for i in range(3):
        n = int(batch["n_crossings"][i, 0])
        pi = rng.permutation(n)
        port = lambda p: 4 * pi[p >> 2] + (p & 3) if 0 <= p < 4 * n else p  # noqa: E731
        for p in range(4 * n):
            q = port(p)
            perm["nbr"][i, q] = port(batch["nbr"][i, p])
            perm["side_face"][i, q] = batch["side_face"][i, p]
        for c in range(n):
            perm["sign"][i, pi[c]] = batch["sign"][i, c]
            perm["strand_component"][i, 2 * pi[c]: 2 * pi[c] + 2] = batch["strand_component"][i, 2 * c: 2 * c + 2]
        for row in perm["candidates"][i]:
            if row[0] >= 0:
                row[1], row[2] = port(row[1]) if row[1] < P else row[1], port(row[2]) if row[2] < P else row[2]
    logits2, _, value2 = policy(obs_to_torch(perm, CPU))
    assert torch.allclose(value, value2, atol=1e-9)
    assert torch.allclose(logits[mask], logits2[mask], atol=1e-9)


def test_gae_matches_a_direct_computation():
    buf = RolloutBuffer(3, 1, {}, CPU)
    buf.values[:, 0] = torch.tensor([0.5, 0.2, 0.1])
    buf.rewards[:, 0] = torch.tensor([1.0, 0.0, 2.0])
    buf.ends[:, 0] = torch.tensor([0.0, 1.0, 0.0])
    buf.compute_gae(torch.tensor([0.4]), gamma=0.9, lam=0.8)
    d2 = 2.0 + 0.9 * 0.4 - 0.1
    d1 = 0.0 - 0.2  # the episode ended at step 1: no bootstrap
    d0 = 1.0 + 0.9 * 0.2 - 0.5
    expected = [d0 + 0.9 * 0.8 * d1, d1, d2]
    assert torch.allclose(buf.advantages[:, 0], torch.tensor(expected))
    assert torch.allclose(buf.returns, buf.advantages + buf.values)


def test_curriculum_promotes_on_current_level_only():
    cur = Curriculum(CurriculumConfig(window=4, promote_at=0.75, max_scramble=4), (1, 2))
    assert not cur.update([{"solved": True, "version": 0}] * 3)
    assert cur.update([{"solved": False, "version": 0}]) and cur.range == (1, 3) and cur.version == 1
    assert not cur.update([{"solved": True, "version": 0}] * 4)  # episodes from the old level do not count
    assert cur.update([{"solved": True, "version": 1}] * 4) and cur.range == (1, 4)
    assert not cur.update([{"solved": True, "version": 2}] * 4) and cur.range == (1, 4)  # capped
    state = cur.state_dict()
    other = Curriculum(CurriculumConfig(), (1, 1))
    other.load_state_dict(state)
    assert other.range == (1, 4) and other.version == 2


def test_vector_env_autoresets_in_the_same_step():
    envs = make_vector_env(tiny(), num_envs=2, seed=0, vectorization="sync")
    obs, _ = envs.reset(seed=0)
    for _ in range(12):
        actions = np.zeros(2, dtype=np.int64)  # always the first candidate
        obs, _, term, trunc, info = envs.step(actions)
        if (term | trunc).any():
            assert "final_obs" in info
    envs.call("set_scramble", 2, 2)
    assert all(len(s) <= 2 for s in envs.reset(seed=1)[1]["scramble"])
    envs.close()


def test_training_checkpoints_resume_and_evaluation(tmp_path):
    cfg = tiny(tmp_path)
    trainer = Trainer(cfg, quiet=True)
    metrics = trainer.train()
    run = trainer.run_dir
    assert trainer.global_step == 64 and math.isfinite(metrics["loss/value"])
    assert (run / "config.yaml").exists() and (run / "metrics.csv").exists() and (run / "eval.csv").exists()
    names = sorted(p.name for p in (run / "checkpoints").iterdir())
    assert names == ["best.pt", "latest.pt", "step_64.pt"]  # older step checkpoints pruned
    assert len((run / "metrics.csv").read_text().strip().splitlines()) == 1 + 4

    cfg_more = dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, total_timesteps=96))
    resumed = Trainer(cfg_more, resume=str(run / "checkpoints" / "latest.pt"), quiet=True)
    assert resumed.global_step == 64 and resumed.run_dir == run
    resumed.train()
    assert resumed.global_step == 96
    assert len((run / "metrics.csv").read_text().strip().splitlines()) == 1 + 6  # appended, one header

    policy, loaded = load_policy(str(run / "checkpoints" / "latest.pt"), CPU)
    assert loaded == cfg_more
    result = evaluate(policy, loaded, CPU, episodes=4, num_envs=2, scramble=(1, 1))
    assert result["episodes"] == 4 and 0.0 <= result["success"] <= 1.0
    (frames, solved), = record_episodes(policy, loaded, CPU, n=1, scramble=(1, 1))
    assert frames[0][0] == "start" and len(frames) >= 2 and isinstance(solved, bool)


def test_evaluate_script(tmp_path, capsys):
    from knotml.rl.cli import evaluate_main, train_main

    assert train_main(["--set", "run.device=cpu", "run.tensorboard=false", "env.num_envs=2", "env.vectorization=sync",
                       "env.max_crossings=16", "env.max_actions=64", "env.max_steps=8", "algorithm.num_steps=8",
                       "algorithm.num_minibatches=2", "policy.hidden=16", "policy.layers=1", "eval.every=0",
                       "run.total_timesteps=16", f"run.output_dir={tmp_path}", "run.name=cli"]) == 0
    checkpoint = tmp_path / "cli" / "checkpoints" / "latest.pt"
    assert evaluate_main([str(checkpoint), "--episodes", "4", "--scramble", "1", "1", "--device", "cpu"]) == 0
    assert "success" in capsys.readouterr().out
