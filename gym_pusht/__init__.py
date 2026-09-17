from gymnasium.envs.registration import register

register(
    id="gym_pusht/PushT-v0",
    entry_point="gym_pusht.envs:PushTEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state", "randomize_goal": False},
)

# Register a version with randomized goal
register(
    id="gym_pusht/PushT-v1",
    entry_point="gym_pusht.envs:PushTEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state", "randomize_goal": True},
)
