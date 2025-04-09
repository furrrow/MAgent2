from __future__ import annotations
import random
import numpy as np
from pettingzoo.utils.env import AECEnv
# from magent2.environments import battle_v4, battlefield_v5
from magent2.environments.custom_map import battlefield

# env = battle_v4.env(render_mode='human')
env = battlefield.env(render_mode='human')

def random_demo(env: AECEnv, render: bool = True, episodes: int = 1) -> float:
    """Runs an env object with random actions."""
    total_reward = 0
    completed_episodes = 0

    while completed_episodes < episodes:
        env.reset()
        for agent in env.agent_iter():
            if render:
                env.render()

            obs, reward, termination, truncation, _ = env.last()
            total_reward += reward
            if termination or truncation:
                action = None
            elif isinstance(obs, dict) and "action_mask" in obs:
                action = random.choice(np.flatnonzero(obs["action_mask"]).tolist())
            else:
                action = env.action_space(agent).sample()
                action = 13
            env.step(action)

        completed_episodes += 1

    if render:
        env.close()

    print("Average total reward", total_reward / episodes)

    return total_reward
random_demo(env, render=True, episodes=1)