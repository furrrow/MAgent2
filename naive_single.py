from __future__ import annotations

import datetime
import random
import numpy as np
import yaml
import wandb
from agents.ppo_agent import IppoAgent
from magent2.environments.custom_map import naive_multi
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
from utils import action_dict, custom_actions

"""
attacking without hitting the target does not appear to give rewards
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

    render: bool = True
    render_freq: int = 10
    """ how often to render training runs """
    eval_freq: int = 10
    n_eval: int = 5
    """ how many loop to run per eval"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    checkpoints_path: str = "./saves"  # Save path
    save_freq: int = 50
    load_model: str = ""  # Model load file name, "" doesn't load

    # Algorithm specific arguments
    env_id: str = "naive_single"
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

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_id}__size{args.map_size}__{args.seed}__{timestamp}"
    if args.checkpoints_path is not None:
        args.checkpoints_path = os.path.join(args.checkpoints_path, run_name)

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
    """ setup agent buffer and initial observations """
    for agent_name in agent_names:
        # action_size = env.action_space(agent_name).n
        action_size = len(custom_actions)
        agents[agent_name] = IppoAgent(action_size, args.n_hidden, channel_last=True).to(device)
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

    if args.checkpoints_path is not None:
        print(f"Checkpoints path: {args.checkpoints_path}")
        os.makedirs(args.checkpoints_path, exist_ok=True)
        with open(os.path.join(args.checkpoints_path, "config.yaml"), "w") as f:
            yaml.dump(args, f)

    global_step = 0
    step = 0
    start_time = time.time()
    while global_step < args.total_timesteps:
        print(f"====== iteration {iteration} ======")
        for agent_name in agent_names:
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizers[agent_name].param_groups[0]["lr"] = lrnow
        advantages = {}
        returns = {}
        while step < args.num_steps:
            for agent_name in agent_names:
                total_reward[agent_name] = 0
            if args.render:
                if episodes % args.render_freq == 0:
                    env_name = "render_env"
                    env = render_env
                else:
                    env_name = "rgb_env"
                    env = rgb_env
            env.reset()
            for agent_name in env.agent_iter():
                if step == args.num_steps:
                    for key in env.truncations:
                        env.truncations[key] = True
                    break
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
                # print(agent_name, step, termination, truncation, info)
                observation = get_custom_obs(observation, r=args.observation_radius)
                old_rwd = reward
                try:
                    state = env.state()
                except:
                    print("error in retrieving state, resetting state...")
                    env.reset()
                reward = square_distance_reward(env, args.distance_threshold)
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
                # print(agent_name, step, termination, truncation, action, action_dict[custom_actions[int(action)]])
                terminal_or_truncated = int(np.logical_or(termination, truncation))
                buffers[agent_name]['dones'][step] = torch.Tensor([terminal_or_truncated]).to(device)

                if termination or truncation:
                    print(f"episode {episodes} total_reward {total_reward[agent_name]:.4f}")
                    writer.add_scalar(f"Charts/{agent_name}_total_rwd", total_reward[agent_name], global_step)
                    env.step(None)
                else:
                    env.step(custom_actions[int(action.cpu())])
                step += 1
                global_step += 1
            episodes += 1
            writer.add_scalar(f"Charts/episode", episodes, global_step)
            if args.render:
                env.close()

        for agent_name in red_agents:
            if len(env.terminations) < 2:
                print("warning env.terminations dict less than 2", env.terminations)
                terminal_or_truncated = int(True)
            else:
                terminal, truncated = env.terminations[agent_name], env.truncations[agent_name]
                terminal_or_truncated = int(np.logical_or(terminal, truncated))
            advantages[agent_name], returns[agent_name] = calculate_returns(
                args, step, terminal_or_truncated, buffers, agent_name, value, device)
        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for agent_name in red_agents:
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

                    mb_advantages = advantages[agent_name][mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - returns[agent_name][mb_inds]) ** 2
                        v_clipped = buffers[agent_name]["values"][mb_inds] + torch.clamp(
                            newvalue - buffers[agent_name]["values"][mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - returns[agent_name][mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - returns[agent_name][mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizers[agent_name].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agents[agent_name].parameters(), args.max_grad_norm)
                    optimizers[agent_name].step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
        for agent_name in red_agents:
            y_pred, y_true = buffers[agent_name]["values"].cpu().numpy(), returns[agent_name].cpu().numpy()
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


        # eval loop:
        if iteration % args.eval_freq == 0:
            env = rgb_env
            frame_list = []  # For creating a gif
            n_eval = args.n_eval
            for agent_name in agent_names:
                total_reward[agent_name] = 0
            for _ in range(n_eval):
                env.reset()
                for agent_name in env.agent_iter():
                    if args.capture_video:
                        image = Image.fromarray(env.render())
                        frame_list.append(image)
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
                    # print(agent_name, step, termination, truncation, info)
                    observation = get_custom_obs(observation, r=args.observation_radius)
                    old_rwd = reward
                    try:
                        state = env.state()
                    except:
                        print("error in retrieving state, resetting state...")
                        env.reset()
                    reward = square_distance_reward(env, args.distance_threshold)
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
                        action = agents[agent_name].get_greedy_action(obs_agent.unsqueeze(0))

                    if termination or truncation:
                        env.step(None)
                    else:
                        env.step(custom_actions[int(action.cpu())])
            for agent_name in red_agents:
                print(f"episode {episodes} eval average reward over {n_eval} episodes: {total_reward[agent_name]/n_eval:.4f}")
                writer.add_scalar(f"Charts/{agent_name}_eval_avg_rwd_{n_eval}", total_reward[agent_name]/n_eval, global_step)
            if args.render:
                if len(frame_list) > 0:
                    frame_list[0].save(f'{args.checkpoints_path}/iter{iteration}_out.gif', save_all=True,
                                       append_images=frame_list[1:], duration=5, loop=0)
            env.close()
            env.reset()

        iteration += 1
        step = 0

        if args.checkpoints_path:
            if iteration % args.save_freq == 0:
                save_dict = {}
                for agent_name in red_agents:
                    save_dict[f"{agent_name}"] = agents[agent_name].state_dict()
                torch.save(
                    save_dict,
                    os.path.join(args.checkpoints_path, f"checkpoint_{iteration}.pt"),
                )