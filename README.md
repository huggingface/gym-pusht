# gym-pusht

A gymnasium environment PushT.

![render](https://github.com/huggingface/gym-pusht/assets/45557362/f5423c71-4777-4203-b3ed-81c50e07a0dc)

## Installation

```bash
pip install gym-pusht
```

## Quick start

```python
import gymnasium as gym
import gym_pusht

env = gym.make("gym_pusht/PushT-v0", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

## Description

PushT environment.

The goal of the agent is to push the block to the goal zone. The agent is a circle and the block is a tee shape.

## Action Space

The action space is continuous and consists of two values: [x, y]. The values are in the range [0, 512] and
represent the target position of the agent.

## Observation Space

If `obs_type` is set to `state`, the observation space is a 5-dimensional vector representing the state of the
environment: [agent_x, agent_y, block_x, block_y, block_angle]. The values are in the range [0, 512] for the agent
and block positions and [0, 2*pi] for the block angle.

If `obs_type` is set to `pixels`, the observation space is a 96x96 RGB image of the environment.

## Rewards

The reward is the coverage of the block in the goal zone. The reward is 1.0 if the block is fully in the goal zone.

## Success Criteria

The environment is considered solved if the block is at least 95% in the goal zone.

## Starting State

The agent starts at a random position and the block starts at a random position and angle.

## Episode Termination

The episode terminates when the block is at least 95% in the goal zone.

## Arguments

```python
>>> import gymnasium as gym
>>> import gym_pusht
>>> env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
>>> env
<TimeLimit<OrderEnforcing<PassiveEnvChecker<PushTEnv<gym_pusht/PushT-v0>>>>>
```

* `obs_type`: (str) The observation type. Can be either `state` or `pixels`. Default is `state`.

* `block_cog`: (tuple) The center of gravity of the block if different from the center of mass. Default is `None`.

* `damping`: (float) The damping factor of the environment if different from 0. Default is `None`.

* `render_action`: (bool) Whether to render the action on the image. Default is `True`.

* `render_size`: (int) The size of the rendered image. Default is `96`.

* `render_mode`: (str) The rendering mode. Can be either `human` or `rgb_array`. Default is `None`.

## Reset Arguments

Passing the option `options["reset_to_state"]` will reset the environment to a specific state.

> [!WARNING]
> For legacy compatibility, the inner fonctionning has been preserved, and the state set is not the same as the
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

## References

* TODO:


## Contribute

Instead of using `pip` directly, we use `poetry` for development purposes to easily track our dependencies.
If you don't have it already, follow the [instructions](https://python-poetry.org/docs/#installation) to install it.

Install the project with dev dependencies:
```bash
poetry install --with dev
```

### Add dependencies

The equivalent of `pip install some-package` would just be:
```bash
poetry add some-package
```

### Follow our style

```bash
# install pre-commit hooks
pre-commit install

# apply style and linter checks on staged files
pre-commit
```