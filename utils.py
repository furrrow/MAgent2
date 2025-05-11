from __future__ import annotations

import numpy as np
import torch

action_dict = {
    0: "up_2",
    1: "up_left",
    2: "up",
    3: "upright",
    4: "left_2",
    5: "left",
    6: "nothing",
    7: "right",
    8: "right_2",
    9: "down_left",
    10: "down",
    11: "down_right",
    12: "down_2",
    13: "attack_topleft",
    14: "attack_up",
    15: "attack_top_right",
    16: "attack_left",
    17: "attack_right",
    18: "attack_bottom_left",
    19: "attack_bottom",
    20: "attack_bottom_right",
}
# only have actions with movements
custom_actions = {
    0: 1,           # up_left
    1: 2,           # up
    2: 3,           # up_right
    3: 5,           # left
    4: 6,           # nothing
    5: 7,           # right
    6: 9,           # down_left
    7: 10,          # down
    8: 11,          # down_right
}

obs_keys = {
    0: "obstacle_map",
    1: "team_0_presence",
    2: "team_0_hp",
    3: "team_1_presence",
    4: "team_1_hp",
}


def calculate_returns(args, step, terminal_or_truncated, buffers, agent_name, value, device):
    advantages = torch.zeros_like(buffers[agent_name]['rewards']).to(device)
    lastgaelam = 0
    for t in reversed(range(step)):
        if t == args.num_steps - 1:
            nextnonterminal = 1.0 - terminal_or_truncated
            nextvalues = value
        else:
            nextnonterminal = 1.0 - buffers[agent_name]['dones'][t + 1]
            nextvalues = buffers[agent_name]['values'][t + 1]
        delta = buffers[agent_name]['rewards'][t] + args.gamma * nextvalues * nextnonterminal - \
                buffers[agent_name]['values'][t]
        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + buffers[agent_name]['values']
    return advantages, returns


def square_distance_reward(env, threshold = 3):
    team_0_state = env.state()[:, :, 1]
    team_1_state = env.state()[:, :, 3]
    team0_y, team0_x = np.nonzero(team_0_state)
    team1_y, team1_x = np.nonzero(team_1_state)
    team_0_locs = np.array([team0_x, team0_y]).T
    team_1_locs = np.array([team1_x, team1_y]).T
    distance = team_0_locs - team_1_locs
    norm_distance = np.linalg.norm(distance, axis=1)
    custom_reward = - min(norm_distance)
    if min(norm_distance) < threshold:
        custom_reward = 1000
    return np.float64(custom_reward) / 10


def get_custom_obs(observation, r = 6, key_idx = 1):
    """
    script to artificially limit field of view.
    agent should be centered around idx (6, 6) so put a mask around it.
    """
    init_r = observation.shape[0]
    assert (r*2 + 1) <= init_r
    mask = np.zeros(observation.shape)
    low = 6 - r
    high = 6 + r + 1
    mask[low:high, low:high, :] = 1
    observation = observation * mask
    return observation


def get_all_custom_obs(env, agent_list, device, r):
    """
    get all observations for the centralized critic
    """
    all_observations = []
    for name in agent_list:
        raw_obs = env.observe(name)
        custom_obs = (get_custom_obs(raw_obs, r))
        custom_obs = torch.swapaxes(torch.Tensor(custom_obs).to(device), 0, 2).unsqueeze(0)
        all_observations.append(custom_obs)
    return all_observations
