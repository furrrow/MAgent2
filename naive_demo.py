from __future__ import annotations

import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
import wandb
from agents.ppo_agent import Agent
from magent2.environments.custom_map import battlefield, naive, four_way
from agents import dummy_agent
import time
import tyro
from dataclasses import dataclass
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
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
    use_wandb: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "RL"
    """the wandb's project name"""
    wandb_entity: str = "jianyu34-university-of-maryland"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    render: bool = True
    render_freq: int = 5

    # Algorithm specific arguments
    env_id: str = "naive_custom_reward"
    """the id of the environment"""
    map_size: int = 25
    """map_size"""
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
    assert len(team0_x) == 1
    assert len(team1_x) == 1
    distance = np.array([team0_x, team0_y]) - np.array([team1_x, team1_y])
    norm_distance = np.linalg.norm(distance, 1)
    custom_reward = - norm_distance
    if norm_distance < threshold:
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
    scenario_name = "naive_scenario"

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
    rgb_env = naive.env(map_size=args.map_size, render_mode='rgb_array')
    render_env = naive.env(map_size=args.map_size, render_mode='human')
    env = rgb_env
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
    """ setup agent buffer and initial observations """
    for agent_name in agent_names:
        agents[agent_name] = Agent(env, agent_name, args.n_hidden, channel_last=True).to(device)
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

    global_step = 0
    step = 0
    start_time = time.time()
    frame_list = []  # For creating a gif
    while global_step < args.total_timesteps:
        print(f"====== iteration {iteration} ======")
        if args.render:
            if iteration % args.render_freq == 0:
                env = render_env
            else:
                env = rgb_env
        for agent_name in agent_names:
            total_reward[agent_name] = 0
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizers[agent_name].param_groups[0]["lr"] = lrnow
        advantages = None
        returns = None
        while step < args.num_steps:
            total_reward[red_agents[0]] = 0
            env.reset()
            for agent_name in env.agent_iter():
                if step == args.num_steps:
                    for key in env.truncations:
                        env.truncations[key] = True
                    break
                if args.render:
                    if episodes % args.render_freq == 0:
                        env.render()
                # skip blue agents
                if "blue" in agent_name or "blue" in env.agent_selection:
                    observation, reward, termination, truncation, info = env.last()
                    if termination or truncation:
                        # print(agent_name, step, termination, truncation, info)
                        env.step(None)
                        continue
                    else:
                        env.step(do_nothing)
                        continue
                observation, reward, termination, truncation, info = env.last()
                observation = get_custom_obs(observation, r=args.observation_radius)
                old_rwd = reward
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
                # print(agent_name, step, termination, truncation, action, action_dict[int(action)])
                terminal_or_truncated = int(np.logical_or(termination, truncation))
                buffers[agent_name]['dones'][step] = torch.Tensor([terminal_or_truncated]).to(device)

                if termination or truncation:
                    # print(f"episode {episodes} total_reward {total_reward[agent_name]:.4f}")
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

        agent_name = red_agents[0]
        if len(env.terminations) < 2:
            print("warning env.terminations dict less than 2", env.terminations)
            terminal_or_truncated = int(True)
            # terminal, truncated = env.terminations[agent_name], env.truncations[agent_name]
            # terminal_or_truncated = int(np.logical_or(terminal, truncated))
        advantages, returns = calculate_returns(terminal_or_truncated, buffers, agent_name)
        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agents[agent_name].get_action_and_value(
                    buffers[agent_name]["obs"][mb_inds], buffers[agent_name]["actions"][mb_inds])
                logratio = newlogprob - buffers[agent_name]["logprobs"][mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                    v_clipped = buffers[agent_name]["values"][mb_inds] + torch.clamp(
                        newvalue - buffers[agent_name]["values"][mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizers[agent_name].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agents[agent_name].parameters(), args.max_grad_norm)
                optimizers[agent_name].step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        for agent_name in red_agents:
            y_pred, y_true = buffers[agent_name]["values"].cpu().numpy(), returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar(f"Charts/{agent_name}_learning_rate", optimizers[agent_name].param_groups[0]["lr"],
                              global_step)
            writer.add_scalar(f"losses/{agent_name}_value_loss", v_loss.item(), global_step)
            print(f"losses/{agent_name}_value_loss, {v_loss.item():.3f}, global step {global_step}")
            writer.add_scalar(f"losses/{agent_name}_policy_loss", pg_loss.item(), global_step)
            print(f"losses/{agent_name}_policy_loss {pg_loss.item():.3f}, global step {global_step}")
            writer.add_scalar(f"losses/{agent_name}_entropy", entropy_loss.item(), global_step)
            writer.add_scalar(f"losses/{agent_name}_old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar(f"losses/{agent_name}_approx_kl", approx_kl.item(), global_step)
            writer.add_scalar(f"losses/{agent_name}_clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar(f"losses/{agent_name}_explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(f"Charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        iteration += 1
        step = 0