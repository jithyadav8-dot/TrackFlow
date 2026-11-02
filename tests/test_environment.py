import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.data.prepare_schedule import synthetic_schedule
from src.simulator.environment import TrainTrafficEnv
from tests.test_simulator import scenario


def test_gymnasium_checker():
    check_env(TrainTrafficEnv(schedule=synthetic_schedule()), skip_render_check=True)


def test_observations_masks_and_reward():
    env = TrainTrafficEnv(scenario())
    obs, _ = env.reset(seed=42)
    assert obs.shape == (14,) and obs.dtype == np.float32
    assert env.action_space.n == 3
    assert env.action_masks().tolist() == [True, True, True]
    _, reward, term, trunc, info = env.step(1)
    assert reward == pytest.approx(-0.1)  # one waiting priority-2 train for one minute
    assert info["reward_components"]["waiting"] == -0.1
    assert not term and not trunc
    while not env.section.done:
        masks = env.action_masks()
        obs, reward, _, _, _ = env.step(1 if masks[1] else 2 if masks[2] else 0)
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)


def test_termination_and_truncation():
    env = TrainTrafficEnv(scenario([], horizon=1, drain=0))
    env.reset()
    assert env.step(0)[2:4] == (True, False)
    env = TrainTrafficEnv(scenario(horizon=1, drain=0))
    env.reset()
    result = env.step(0)
    assert result[2:4] == (False, True)
    assert result[4]["reward_components"]["backlog"] == -4
    with pytest.raises(RuntimeError):
        env.step(0)


def test_reset_and_hidden_future_delays():
    import copy

    base = scenario([("a", 0, 10, 3, 1)])
    delayed = copy.deepcopy(base)
    delayed["trains"][0]["ready_time"] = 15
    a, b = TrainTrafficEnv(base), TrainTrafficEnv(delayed)
    np.testing.assert_array_equal(a.reset(seed=1)[0], b.reset(seed=1)[0])
    env = TrainTrafficEnv(schedule=synthetic_schedule())
    initial = env.reset(seed=123)[0]
    ids = [(t.ready_time, t.travel_duration) for t in env.section.trains]
    np.testing.assert_array_equal(initial, env.reset(seed=123)[0])
    assert ids == [(t.ready_time, t.travel_duration) for t in env.section.trains]


def test_rewards_do_not_count_cumulative_wait():
    env = TrainTrafficEnv(scenario())
    env.reset()
    first = env.step(0)[4]["reward_components"]["waiting"]
    second = env.step(0)[4]["reward_components"]["waiting"]
    assert first == second == pytest.approx(-0.15)
