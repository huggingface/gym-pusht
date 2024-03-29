import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PushTEnv(gym.Env):
    """PushT environment."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def step(self, action):
        observation = self.observation_space.sample()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.observation_space.seed(seed)
        observation = self.observation_space.sample()
        info = {}
        return observation, info

    def render(self):
        pass

    def close(self):
        pass


