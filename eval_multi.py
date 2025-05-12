from __future__ import annotations

import datetime
import random
import numpy as np
import yaml
import wandb
from agents.ppo_agent import CentralCritic, DecentActor
from magent2.environments.custom_map import naive_multi, four_way
import time
import tyro
from dataclasses import dataclass
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from utils import get_custom_obs, get_all_custom_obs, square_distance_reward, calculate_returns
from utils import action_dict, custom_actions, arrow_actions
import matplotlib.pyplot as plt


"""
load a saved weight and do plotting
"""


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 5
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    use_wandb: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "RL"
    """the wandb's project name"""
    wandb_entity: str = "jianyu34-university-of-maryland"
    """the entity (team) of wandb's project"""

    render: bool = False
    render_freq: int = 10
    """ how often to render training runs """
    eval_freq: int = 10
    n_eval: int = 1
    """ how many loop to run per eval"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    checkpoints_path: str = "./saves"  # Save path
    save_freq: int = 10
    checkpoint_path: str = "/home/jim/Documents/Projects/MAgent2/saves/naive_mappo__size40__3agents__2025-05-12_03-07-52/checkpoint_300.pt"  # Model load file name, "" doesn't load
    config_file: str = "/home/jim/Documents/Projects/MAgent2/saves/naive_mappo__size40__3agents__2025-05-12_03-07-52/config.yaml"  # Model load file name, "" doesn't load

    # Algorithm specific arguments
    env_id: str = "eval_multi"
    """the id of the environment"""
    map_size: int = 40
    """map_size"""
    n_agents: int = 2
    """the number of red agents in the map"""

def get_action_value_map(agent, observation):
    map_idx = []
    c, h, w = observation.shape
    map_elements = []
    x_list = []
    y_list = []
    r_h = h // 2
    r_w = w // 2
    idx = 0
    for i in range(h):
        if i < r_h:
            obs_row_0, obs_row_1 = 0, i + r_h + 1
            map_row_0, map_row_1 = r_h - i, h
        elif i > r_h:
            obs_row_0, obs_row_1 = i - r_h, h
            map_row_0, map_row_1 = 0, h - i + r_h
        else:
            obs_row_0, obs_row_1 = 0, h
            map_row_0, map_row_1 = 0, h
        for j in range(w):
            if j < r_w:
                obs_col_0, obs_col_1 = 0, j + r_w + 1
                map_col_0, map_col_1 = r_w - j, w
            elif j > r_w:
                obs_col_0, obs_col_1 = j - r_w, w
                map_col_0, map_col_1 = 0, w - j + r_w
            else:
                obs_col_0, obs_col_1 = 0, w
                map_col_0, map_col_1 = 0, w
            map_element = torch.zeros_like(observation)
            map_element[:, map_row_0: map_row_1, map_col_0: map_col_1] = observation[:, obs_row_0:obs_row_1,
                                                                         obs_col_0:obs_col_1]
            map_idx.append(idx)
            x_list.append(j)
            y_list.append(i)
            map_elements.append(map_element)
            idx += 1
    action_map = agent.get_greedy_action(torch.stack(map_elements))
    # value_map = agent.get_value(torch.stack(map_elements))
    sanity_check = torch.Tensor(np.array(map_idx)).reshape(h, w)
    arrows = [arrow_actions[a] for a in np.array(action_map)]

    return np.array(x_list), np.array(y_list), np.array(arrows), 0

if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.config_file is not None:
        print(f"yaml path: {args.config_file}")
        with open(args.config_file, "r") as f:
            config = yaml.safe_load(f)

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_id}__{args.exp_name}__{args.map_size}__{timestamp}"
    if args.checkpoints_path is not None:
        args.checkpoints_path = os.path.join(args.checkpoints_path, run_name)
        checkpoint = torch.load(args.checkpoint_path, weights_only=True)
    else:
        print("Checkpoint could not load, exiting")
        exit()
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    rgb_env = naive_multi.env(map_size=args.map_size, n_red_agents=args.n_agents, render_mode='rgb_array')
    render_env = naive_multi.env(map_size=args.map_size, n_red_agents=args.n_agents, render_mode='human')
    env = rgb_env
    env_name = "rgb_env"
    episodes = 0
    iteration = 0
    do_nothing = 6
    env.reset()
    agent_names = env.agents.copy()
    red_agents = [agent_name for agent_name in agent_names if 'red' in agent_name]
    actors = {}
    critics = {}
    actor_optimizers = {}
    critic_optimizers = {}
    buffers = {}
    total_reward = {}
    last_state = {}
    last_action = {}
    returns = {}
    """ setup agent buffer and initial observations """
    for agent_name in agent_names:
        action_size = len(custom_actions)
        actors[agent_name] = DecentActor(action_size, config['n_hidden']).to(device)
        critics[agent_name] = CentralCritic(config['n_hidden'], 5*len(red_agents)).to(device)
        actor_optimizers[agent_name] = optim.Adam(actors[agent_name].parameters(), lr=config['learning_rate'], eps=1e-5)
        critic_optimizers[agent_name] = optim.Adam(critics[agent_name].parameters(), lr=config['learning_rate'], eps=1e-5)
        obs_space_shape_swapped = env.observation_space(agent_name).shape
        obs_space_shape_swapped = list(obs_space_shape_swapped)[::-1]
        total_reward[agent_name] = 0

    for agent_name in red_agents:
        actors[agent_name].load_state_dict(checkpoint[agent_name])
        actors[agent_name].eval()

    global_step = 0
    step = 0
    start_time = time.time()
    frame_list = []  # For creating a gif
    arrow_frames = []
    n_eval = args.n_eval
    if args.capture_video:
        env = rgb_env
    if args.render:
        env = render_env
    for k in range(n_eval):
        for agent_name in agent_names:
            total_reward[agent_name] = 0
        env.reset()
        for agent_name in env.agent_iter():
            if args.render:
                env.render()
            # skip blue agents
            if "blue" in agent_name or "blue" in env.agent_selection:
                blue_observation, blue_reward, blue_termination, blue_truncation, info = env.last()
                # print(agent_name, step, blue_termination, blue_truncation, info)
                if blue_termination or blue_truncation:
                    env.step(None)
                    continue
                else:
                    env.step(do_nothing)
                    continue
            obs_all_tensor = get_all_custom_obs(env, red_agents, device=device, r=config["observation_radius"])
            observation, reward, termination, truncation, info = env.last()
            observation = get_custom_obs(observation, r=config["observation_radius"])
            try:
                state = env.state()
            except:
                print("error in retrieving state, resetting state...")
                env.reset()
            if reward > 2:
                termination = True
                for key in env.terminations:
                    env.terminations[key] = True
                for key in env.truncations:
                    env.truncations[key] = False
            obs_agent = torch.Tensor(observation).to(device)
            obs_agent = torch.swapaxes(obs_agent, 0, 2)
            reward = square_distance_reward(env, config['distance_threshold'])
            total_reward[agent_name] += reward
            if reward > 2:
                termination = True
                for key in env.terminations:
                    env.terminations[key] = True
                for key in env.truncations:
                    env.truncations[key] = False
            obs_agent = torch.Tensor(observation).to(device)
            obs_agent = torch.swapaxes(obs_agent, 0, 2)

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, entropy = actors[agent_name].get_action(obs_agent.unsqueeze(0))
                # value = critics[agent_name].get_value(torch.stack(obs_all_tensor))

            # new part, plot the policy map around the agent!
            xs, ys, arrows, _ = get_action_value_map(actors[agent_name], obs_agent)

            # now to plot this action map.....
            # Creating plot
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            arrow_plot = ax2.quiver(xs, ys, np.array(arrows)[:, 0], np.array(arrows)[:, 1])
            x_dots = np.linspace(0, 12, 13)
            x_dots = x_dots.reshape(1, 13).repeat(13, axis=0)
            y_dots = np.linspace(0, 12, 13)
            y_dots = y_dots.reshape(13, 1).repeat(13, axis=1)
            ax2.scatter(x_dots, y_dots, s=5)
            ax2.scatter(6, 6, s=7)
            ax2.xaxis.set_ticks([])
            ax2.yaxis.set_ticks([])
            ax2.grid()
            ax2.set_aspect('equal')
            # fig2.canvas.draw()
            plt.savefig(f'eval/eval{k}_{agent_name}_arrow{step}.png')

            # print(agent_name, step, termination, truncation, action, action_dict[int(action)])
            terminal_or_truncated = int(np.logical_or(termination, truncation))

            if termination or truncation:
                print(f"iteration {iteration} total_reward {total_reward[agent_name]:.4f}")
                env.step(None)
            else:
                env.step(custom_actions[int(action.cpu())])
            if agent_name == red_agents[-1]:
                step += 1
                global_step += 1
                if args.capture_video:

                        image = Image.fromarray(env.render())
                        frame_list.append(image)
                        image.save(f'eval/eval{k}_render{step}.png')
            env.close()
        if args.capture_video:
            if len(frame_list) > 0:
                frame_list[0].save(f'eval/{args.exp_name}_{env_name}_eval.gif', save_all=True,
                                   append_images=frame_list[1:],
                                   duration=3, loop=0)

        iteration += 1
        step = 0