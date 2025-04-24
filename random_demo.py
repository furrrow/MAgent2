from __future__ import annotations
import random
import numpy as np
from pettingzoo.utils.env import AECEnv

from agents.dummy_agent import DummyAgent
from magent2.environments.custom_map import battlefield, naive, four_way
from agents import dummy_agent
import time

# env = battle_v4.env(render_mode='human')
# env = battlefield.env(render_mode='human')
# env = naive.env(render_mode='human')
env = four_way.env(render_mode='human')
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
obs_keys = {
    0: "obstacle_map",
    1: "team_0_presence",
    2: "team_0_hp",
    3: "team_1_presence",
    4: "team_1_hp",
}
def reward_modification(observation, reward):
    self_presence_map = observation[:, :, 1]
    opposite_presence_map = observation[:, :, 3]
    return reward

def random_demo(env: AECEnv, render: bool = True, episodes: int = 1) -> float:
    """Runs an env object with random actions."""
    total_reward = 0
    completed_episodes = 0
    do_nothing = 6
    env.reset()
    agents = {name:DummyAgent(name, do_nothing) for name in env.agents}
    while completed_episodes < episodes:
        env.reset()
        message_dict = {}
        for agent_name in env.agent_iter():
            if render:
                env.render()
                # time.sleep(2)

            obs, reward, termination, truncation, _ = env.last()
            custom_reward = reward_modification(obs, reward)
            total_reward += custom_reward
            d_agent = agents[agent_name]
            if termination or truncation:
                action = None
            elif isinstance(obs, dict) and "action_mask" in obs:
                action = random.choice(np.flatnonzero(obs["action_mask"]).tolist())
            else:
                action, message = d_agent.get_action(obs)
                message_dict[agent_name] = message
                print(f"{agent_name} says {message}")
            env.step(action)

        completed_episodes += 1

    if render:
        env.close()

    print("Average total reward", total_reward / episodes)

    return total_reward
random_demo(env, render=True, episodes=10)