import pytest

from unknot_rl.envs.unknot_env import UnknotEnv


def test_env_reset_and_step():
    env = UnknotEnv(seed=0)
    obs = env.reset()
    assert isinstance(obs, dict)

    obs2, reward, done, info = env.step(0)
    assert isinstance(obs2, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
