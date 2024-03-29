import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import gym_pusht  # noqa


def test_env():
    env = gym.make("gym_pusht/PushT-v0")
    check_env(env.unwrapped, skip_render_check=True)
