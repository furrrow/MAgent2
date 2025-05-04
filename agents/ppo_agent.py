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
        self.activation = nn.ReLU()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(5, 32, 3, stride=2)),
            self.activation,
            layer_init(nn.Conv2d(32, 64, 3, stride=1)),
            self.activation,
            nn.Flatten(),
            layer_init(nn.Linear(1024, 512)),
            self.activation,
        )
        self.actor = layer_init(nn.Linear(512, self.action_size), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        batch_size = x.shape[0]
        x = self.convblock_critic(x)
        x = F.tanh(x)
        return self.critic(self.network(x.reshape(batch_size, -1)))

    def get_action_and_value(self, x, action=None):
        batch_size = x.shape[0]
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden.reshape(batch_size, -1))