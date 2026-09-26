# gym-pusht

A gymnasium environment PushT.

<img src="http://remicadene.com/assets/gif/pusht_diffusion.gif" width="50%" alt="Diffusion policy on PushT env"/>


## Installation

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):
```bash
conda create -y -n pusht python=3.10 && conda activate pusht
```

Install gym-pusht:
```bash
pip install gym-pusht
```


## Quick start

```python
# example.py
import gymnasium as gym
import gym_pusht

env = gym.make("gym_pusht/PushT-v0", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

## Description

PushT environment.

The goal of the agent is to push the block to the goal zone. The agent is a circle and the block is a tee shape.

### Action Space

The action space is continuous and consists of two values: [x, y]. The values are in the range [0, 512] and
represent the target position of the agent.

### Observation Space

If `obs_type` is set to `state`, the observation space is a 5-dimensional vector representing the state of the
environment: [agent_x, agent_y, block_x, block_y, block_angle]. The values are in the range [0, 512] for the agent
and block positions and [0, 2*pi] for the block angle.

If `obs_type` is set to `environment_state_agent_pos` the observation space is a dictionary with:
    - `environment_state`: 16-dimensional vector representing the keypoint locations of the T (in [x0, y0, x1, y1, ...]
        format). The values are in the range [0, 512].
    - `agent_pos`: A 2-dimensional vector representing the position of the robot end-effector.

If `obs_type` is set to `pixels`, the observation space is a 96x96 RGB image of the environment.

### Rewards

The reward is the coverage of the block in the goal zone. The reward is 1.0 if the block is fully in the goal zone.

### Success Criteria

The environment is considered solved if the block is at least 95% in the goal zone.

### Starting State

The agent starts at a random position and the block starts at a random position and angle.

### Episode Termination

The episode terminates when the block is at least 95% in the goal zone.

### Arguments

```python
>>> import gymnasium as gym
>>> import gym_pusht
>>> env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
>>> env
<TimeLimit<OrderEnforcing<PassiveEnvChecker<PushTEnv<gym_pusht/PushT-v0>>>>>
```

* `obs_type`: (str) The observation type. Can be either `state`, `environment_state_agent_pos`, `pixels` or `pixels_agent_pos`. Default is `state`.

* `block_cog`: (tuple) The center of gravity of the block if different from the center of mass. Default is `None`.

* `damping`: (float) The damping factor of the environment if different from 0. Default is `None`.

* `render_mode`: (str) The rendering mode. Can be either `human` or `rgb_array`. Default is `rgb_array`.

* `observation_width`: (int) The width of the observed image. Default is `96`.

* `observation_height`: (int) The height of the observed image. Default is `96`.

* `visualization_width`: (int) The width of the visualized image. Default is `680`.

* `visualization_height`: (int) The height of the visualized image. Default is `680`.

### Reset Arguments

Passing the option `options["reset_to_state"]` will reset the environment to a specific state.

> [!WARNING]
> For legacy compatibility, the inner functioning has been preserved, and the state set is not the same as the
> the one passed in the argument.

```python
>>> import gymnasium as gym
>>> import gym_pusht
>>> env = gym.make("gym_pusht/PushT-v0")
>>> state, _ = env.reset(options={"reset_to_state": [0.0, 10.0, 20.0, 30.0, 1.0]})
>>> state
array([ 0.      , 10.      , 57.866196, 50.686398,  1.      ],
        dtype=float32)
```


## Version History

* v0: Original version
* v0.1.6: Added compatibility with pymunk 7.x

## Compatibility Notes

* Compatible with pymunk versions 6.6.0 and higher, including pymunk 7.x
* If you encounter any issues with pymunk 7.x, please report them on our GitHub issue tracker


## References

* TODO:


## Contribute

Instead of using `pip` directly, we use `poetry` for development purposes to easily track our dependencies.
If you don't have it already, follow the [instructions](https://python-poetry.org/docs/#installation) to install it.

Install the project with dev dependencies:
```bash
poetry install --all-extras
```

### Follow our style

```bash
# install pre-commit hooks
pre-commit install

# apply style and linter checks on staged files
pre-commit
```

## Acknowledgment

gym-pusht is adapted from [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
