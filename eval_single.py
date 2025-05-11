from __future__ import annotations

import datetime
import random
import numpy as np
import yaml
import wandb
from agents.ppo_agent import IppoAgent
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
"""
attacking without hitting the target does not appear to give rewards
"""
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
obs_keys = {
    0: "obstacle_map",
    1: "team_0_presence",
    2: "team_0_hp",
    3: "team_1_presence",
    4: "team_1_hp",
}

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
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    render: bool = False
    render_freq: int = 2
    checkpoints_path: str = "./saves"  # Save path
    save_freq: int = 10
    checkpoint_path: str = "/home/jim/Documents/Projects/MAgent2/saves/naive_single_size30__naive_single__5__2025-05-05_19-16-18/checkpoint_200.pt"  # Model load file name, "" doesn't load
    # checkpoint_path: str = "/home/jim/Documents/Projects/MAgent2/saves/naive_mappo_size40__naive_mappo__5__2025-05-05_22-33-51/checkpoint_1050.pt"  # Model load file name, "" doesn't load
    config_file: str = "/home/jim/Documents/Projects/MAgent2/saves/naive_single_size30__naive_single__5__2025-05-05_19-16-18/config.yaml"  # Model load file name, "" doesn't load
    # config_file: str = "/home/jim/Documents/Projects/MAgent2/saves/naive_mappo_size40__naive_mappo__5__2025-05-05_22-33-51/config.yaml"  # Model load file name, "" doesn't load

    # Algorithm specific arguments
    env_id: str = "naive_multi"
    """the id of the environment"""
    map_size: int = 25
    """map_size"""
    n_agents: int = 1
    """the number of red agents in the map"""
    distance_threshold: int = 3
    """how close does the red agent need to get to the blue agent to count as success"""
    observation_radius: int = 6 # default 6
    """observation radius for the agent"""
    total_timesteps: int = 5_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    n_hidden: int = 32
    """number of hidden layers in the network"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 4000
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4  # 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def calculate_returns(terminal_or_truncated, buffers, agent_name):
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

def get_custom_reward(env, threshold = 3):
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

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{timestamp}"
    if args.checkpoints_path is not None:
        args.checkpoints_path = os.path.join(args.checkpoints_path, run_name)

    if args.config_file is not None:
        print(f"yaml path: {args.config_file}")
        with open(args.config_file, "r") as f:
            config_dict = yaml.safe_load(f)
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.run.save()

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env = battle_v4.env(render_mode='human')
    # env = battlefield.env(render_mode='human')
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
    agents = {}
    optimizers = {}
    buffers = {}
    total_reward = {}
    last_state = {}
    last_action = {}
    returns = {}

    # Define the codec and create a VideoWriter object

    """ setup agent buffer and initial observations """
    for agent_name in agent_names:
        agents[agent_name] = IppoAgent(env, agent_name, args.n_hidden, channel_last=True).to(device)
        optimizers[agent_name] = optim.Adam(agents[agent_name].parameters(), lr=args.learning_rate, eps=1e-5)
        obs_space_shape_swapped = env.observation_space(agent_name).shape
        obs_space_shape_swapped = list(obs_space_shape_swapped)[::-1]
        buffer = {
            "obs": torch.zeros((args.num_steps,) + tuple(obs_space_shape_swapped)).to(device),
            "actions": torch.zeros((args.num_steps,) + env.action_space(agent_name).shape).to(device),
            "logprobs": torch.zeros(args.num_steps).to(device),
            "rewards": torch.zeros(args.num_steps).to(device),
            "dones": torch.zeros(args.num_steps).to(device),
            "values": torch.zeros(args.num_steps).to(device),
        }
        buffers[agent_name] = buffer
        total_reward[agent_name] = 0

    if args.checkpoint_path is not None:
        print(f"Checkpoints path: {args.checkpoint_path}")
        print(f"yaml path: {args.config_file}")
        with open(args.config_file, "r") as f:
            config_dict = yaml.safe_load(f)
        checkpoint = torch.load(args.checkpoint_path, weights_only=True)
        for agent_name in red_agents:
            agents[agent_name].load_state_dict(checkpoint[agent_name])
            agents[agent_name].eval()

    global_step = 0
    step = 0
    start_time = time.time()
    while iteration < 5:
        print(f"====== iteration {iteration} ======")
        env_name = "rgb_env"
        env = rgb_env
        for agent_name in agent_names:
            total_reward[agent_name] = 0
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizers[agent_name].param_groups[0]["lr"] = lrnow
        advantages = {}
        returns = {}
        blue_truncation, blue_termination = False, False
        frame_list = []  # For creating a gif
        while iteration < 5:
            total_reward[red_agents[0]] = 0
            env.reset()
            # print("env reset", env_name)
            for agent_name in env.agent_iter():
                image = Image.fromarray(env.render())
                frame_list.append(image)
                # if blue_truncation or blue_termination:
                    # print("blue agent", blue_truncation, blue_termination)
                if step == args.num_steps:
                    for key in env.truncations:
                        env.truncations[key] = True
                    break
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

                observation, reward, termination, truncation, info = env.last()
                # print(agent_name, step, termination, truncation, info)
                observation = get_custom_obs(observation, r=args.observation_radius)
                old_rwd = reward
                try:
                    state = env.state()
                except:
                    print("error in retrieving state, resetting state...")
                    env.reset()
                reward = get_custom_reward(env, args.distance_threshold)
                if reward > 2:
                    termination = True
                    for key in env.terminations:
                        env.terminations[key] = True
                    for key in env.truncations:
                        env.truncations[key] = False
                obs_agent = torch.Tensor(observation).to(device)
                obs_agent = torch.swapaxes(obs_agent, 0, 2)
                buffers[agent_name]['obs'][step] = obs_agent
                if step > 0:  # reward is the result of the last step
                    buffers[agent_name]['rewards'][step-1] = reward
                    total_reward[agent_name] += reward

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agents[agent_name].get_action_and_value(obs_agent.unsqueeze(0))
                buffers[agent_name]["values"][step] = value.flatten()
                buffers[agent_name]["actions"][step] = action
                buffers[agent_name]["logprobs"][step] = logprob
                print(agent_name, step, termination, truncation, action, action_dict[int(action)])
                terminal_or_truncated = int(np.logical_or(termination, truncation))
                buffers[agent_name]['dones'][step] = torch.Tensor([terminal_or_truncated]).to(device)

                if termination or truncation:
                    print(f"episode {episodes} total_reward {total_reward[agent_name]:.4f}")
                    writer.add_scalar(f"Charts/{agent_name}_total_rwd", total_reward[agent_name], global_step)
                    env.step(None)
                else:
                    env.step(int(action.cpu()))
                    # env.step(do_nothing)
                step += 1
                global_step += 1
            episodes += 1
            writer.add_scalar(f"Charts/episode", episodes, global_step)
            if args.render:
                env.close()
            env.reset()
            iteration += 1
        frame_list[0].save(f'{args.exp_name}_{env_name}_out.gif', save_all=True, append_images=frame_list[1:], duration=3,
                           loop=0)
        for agent_name in red_agents:
            if len(env.terminations) < 2:
                print("warning env.terminations dict less than 2", env.terminations)
                terminal_or_truncated = int(True)
            else:
                terminal, truncated = env.terminations[agent_name], env.truncations[agent_name]
                terminal_or_truncated = int(np.logical_or(terminal, truncated))
            advantages[agent_name], returns[agent_name] = calculate_returns(terminal_or_truncated, buffers, agent_name)

        iteration += 1
        step = 0