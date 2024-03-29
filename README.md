# gym-pusht

A gymnasium environment PushT.

![render](https://github.com/huggingface/gym-pusht/assets/45557362/f5423c71-4777-4203-b3ed-81c50e07a0dc)

## Installation

```bash
pip install gym-pusht
```

## Usage

```python
import gymnasium as gym

env = gym.make("gym_pusht/PushT-v0", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```
