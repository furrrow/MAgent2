import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env, agent_name, n_hidden=64, channel_last=False):
        super().__init__()
        self.n_hidden = n_hidden
        self.channel_last = channel_last
        self.obs_shape = env.observation_space(agent_name).shape
        self.action_size = env.action_space(agent_name).n
        self.conv_lin_size = (self.obs_shape[0] - 2) * (self.obs_shape[1] - 2) * self.n_hidden
        # self.conv_lin_size = 11
        self.dummy_msg = 0
        self.activation = nn.Tanh()
        self.convblock_actor = nn.Conv2d(self.obs_shape[-1], n_hidden, 3)
        self.convblock_critic = nn.Conv2d(self.obs_shape[-1], n_hidden, 3)
        self.critic_seq = nn.Sequential(
            layer_init(nn.Linear(self.conv_lin_size, n_hidden)),
            self.activation,
            layer_init(nn.Linear(n_hidden, n_hidden)),
            self.activation,
            layer_init(nn.Linear(n_hidden, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.conv_lin_size, n_hidden)),
            self.activation,
            layer_init(nn.Linear(n_hidden, n_hidden)),
            self.activation,
            layer_init(nn.Linear(n_hidden, np.prod(self.action_size)), std=0.01),
        )
        # self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(self.action_size)))

    def get_value(self, x):
        batch_size = x.shape[0]
        x = self.convblock_critic(x)
        x = F.tanh(x)
        return self.critic_seq(x.reshape(batch_size, -1))

    def get_action_and_value(self, x, action=None):
        x_actor = self.convblock_actor(x) # [1, 5, 13, 13]
        x_actor = F.tanh(x_actor) # [1, 32, 11, 11]
        batch_size = x_actor.shape[0]
        logits = self.actor(x_actor.reshape(batch_size, -1))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        x_critic = self.convblock_critic(x)
        x_critic = F.tanh(x_critic)
        return action, probs.log_prob(action), probs.entropy(), self.critic_seq(x_critic.reshape(batch_size, -1))