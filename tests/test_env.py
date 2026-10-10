import pytest

pytest.skip(
    "UnknotEnv observation encoding and action decoding await the action-space design "
    "(planning/research-summary.md, sections 2-3)",
    allow_module_level=True,
)

from knotml.envs.unknot_env import UnknotEnv  # noqa: E402


def test_env_reset_and_step():
    env = UnknotEnv(seed=0)
    obs, info = env.reset()

    assert isinstance(obs, dict)
    assert "adj" in obs and "node" in obs
    assert obs["adj"].shape[0] == obs["adj"].shape[1]
    assert isinstance(info, dict)

    obs2, reward, terminated, truncated, info2 = env.step(0)
    assert isinstance(obs2, dict)
    assert "adj" in obs2 and "node" in obs2
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info2, dict)
