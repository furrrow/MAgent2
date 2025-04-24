from __future__ import annotations
import random
import numpy as np
from pettingzoo.utils.env import AECEnv

from agents.ppo_agent import Agent
from magent2.environments.custom_map import battlefield, naive, four_way
from agents import dummy_agent
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

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
obs_keys = {
    0: "obstacle_map",
    1: "team_0_presence",
    2: "team_0_hp",
    3: "team_1_presence",
    4: "team_1_hp",
}
def reward_modification(observation, reward):
    # self_presence_map = observation[:, :, 1]
    # opposite_presence_map = observation[:, :, 3]
    return reward

def naive_walk_demo(env: AECEnv, render: bool = True, episodes: int = 1) -> float:
    """Runs an env object with random actions."""
    n_hidden = 32
    learning_rate = 1e-3
    num_steps = 128
    total_timesteps = 1_000_000
    device = torch.device("cpu")
    total_reward = 0
    completed_episodes = 0
    do_nothing = 6
    env.reset()
    agent_names = env.agents.copy()

    agents = {}
    optimizers = {}
    buffers = {}
    total_reward = {}
    next_done = {}
    for agent_name in agent_names:
        agents[agent_name] = Agent(env, agent_name, n_hidden, channel_last=True).to(device)
        optimizers[agent_name] = optim.Adam(agents[agent_name].parameters(), lr=learning_rate, eps=1e-5)
        obs_space_shape_swapped = env.observation_space(agent_name).shape
        obs_space_shape_swapped = list(obs_space_shape_swapped)[::-1]
        buffer = TensorDict({
            "obs": torch.zeros((num_steps,) + tuple(obs_space_shape_swapped)).to(device),
            "actions": torch.zeros((num_steps,) + env.action_space(agent_name).shape).to(device),
            "logprobs": torch.zeros(num_steps).to(device),
            "rewards": torch.zeros(num_steps).to(device),
            "dones": torch.zeros(num_steps).to(device),
            "values": torch.zeros(num_steps).to(device),
        })
        buffers[agent_name] = buffer
        total_reward[agent_name] = 0

    global_step = 0
    start_time = time.time()
    frame_list = []  # For creating a gif
    step = 0
    episode = 0
    while step < total_timesteps:
        env.reset()
        message_dict = {}
        for agent_name in env.agent_iter():
            next_done[agent_name] = 0
            total_reward[agent_name] = 0
            if render:
                env.render()
                # time.sleep(2)
            observation, reward, termination, truncation, info = env.last()
            obs_agent = torch.Tensor(env.observe(agent_name)).to(device)
            obs_agent = torch.swapaxes(obs_agent, 0, 2)
            buffers[agent_name]['obs'][step] = obs_agent
            buffers[agent_name]['dones'][step] = next_done[agent_name]
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agents[agent_name].get_action_and_value(obs_agent.unsqueeze(0))
                buffers[agent_name]["values"][step] = value.flatten()
                # TODO: action space is discrete...
                low_limit = torch.tensor(env.action_space(agent_name).start).to(device)
                high_limit = torch.tensor(env.action_space(agent_name).n).to(device)
                action[0] = action[0].clip(low_limit, high_limit)
            # buffers[agent_name]["actions"][step], actions[agent_name] = action, action[0].cpu().numpy()
            # buffers[agent_name]["logprobs"][step] = logprob

            # obs, reward, termination, truncation, _ = env.last()
            # message_dict[agent_name] = message
            # print(f"{agent_name} says {message}")
            env.step(action)

        completed_episodes += 1

    if render:
        env.close()

    print("Average total reward", total_reward / episodes)

    return total_reward
naive_walk_demo(env, render=True, episodes=10)