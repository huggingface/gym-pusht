import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

import gym_pusht  # noqa: F401


@pytest.mark.parametrize(
    "env_task, obs_type",
    [
        ("PushT-v0", "state"),
        ("PushT-v0", "pixels"),
        ("PushT-v0", "pixels_agent_pos"),
        ("PushT-v0", "environment_state_agent_pos"),
    ],
)
def test_env(env_task, obs_type):
    env = gym.make(f"gym_pusht/{env_task}", obs_type=obs_type)
    check_env(env.unwrapped)
