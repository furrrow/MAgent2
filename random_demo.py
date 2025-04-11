from __future__ import annotations
import random
import numpy as np
from pettingzoo.utils.env import AECEnv
from magent2.environments.custom_map import battlefield, naive

# env = battle_v4.env(render_mode='human')
# env = battlefield.env(render_mode='human')
env = naive.env(render_mode='human')
"""
actions for battlefield env:
0: formation up
1: up-left
2: up?
3: up-right
4: left
5: left
6: nothing
7: right
8: formation right?
9: down-left
10: down
11: down-right
12: formation-down? 
13: attack top left?
14: attack up
15: attack top right
16: attack left
17: attack right
18: attack bottom left
19: attack bottom
20: attack bottom right
"""

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
                if "blue" in agent:
                    action = 6
                else: # red
                    action = env.action_space(agent).sample()
                    action = 7
            env.step(action)

        completed_episodes += 1

    if render:
        env.close()

    print("Average total reward", total_reward / episodes)

    return total_reward
random_demo(env, render=True, episodes=1)