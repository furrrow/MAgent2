import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class GCN(torch.nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.conv1 = GCNConv(n_features, 16)
        self.conv2 = GCNConv(16, n_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class IppoAgent(nn.Module):
    def __init__(self, n_action, n_hidden=64, n_channel=5, channel_last=False):
        super().__init__()
        self.n_hidden = n_hidden
        self.channel_last = channel_last
        self.action_size = n_action
        self.dummy_msg = 0
        self.activation = nn.ReLU()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(n_channel, 32, 3, stride=2)),
            self.activation,
            layer_init(nn.Conv2d(32, 64, 3, stride=1)),
            self.activation,
            nn.Flatten(),
            layer_init(nn.Linear(1024, 512)),
            self.activation,
        )
        self.actor = layer_init(nn.Linear(512, n_action), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        batch_size = x.shape[0]
        hidden = self.network(x)
        return self.critic(hidden.reshape(batch_size, -1))

    def get_action_and_value(self, x, action=None):
        batch_size = x.shape[0]
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden.reshape(batch_size, -1))

    def get_greedy_action(self, x):
        hidden = self.network(x)
        logits = self.actor(hidden)
        action = torch.argmax(logits, dim=1)
        return action

class CentralCritic(nn.Module):
    def __init__(self, n_hidden=64, n_channel=5):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_agents = int(n_channel // 5)
        self.activation = nn.ReLU()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(n_channel, 32, 3, stride=2)),
            self.activation,
            layer_init(nn.Conv2d(32, 64, 3, stride=1)),
            self.activation,
            nn.Flatten(),
            layer_init(nn.Linear(1024, 512)),
            self.activation,
        )
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        batch, w, h = x.shape[1], x.shape[-2], x.shape[-1] # x: [n_agent, batch, 5, 13, 13]
        x_swap = torch.swapaxes(x, 0, 1)
        x_squash = x_swap.reshape(batch, -1, w, h)
        out = self.critic(self.network(x_squash))
        return out

class DecentActor(nn.Module):
    def __init__(self, action_space, n_hidden=64, n_channel=5):
        super().__init__()
        self.n_hidden = n_hidden
        self.action_size = action_space
        self.activation = nn.ReLU()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(n_channel, 32, 3, stride=2)),
            self.activation,
            layer_init(nn.Conv2d(32, 64, 3, stride=1)),
            self.activation,
            nn.Flatten(),
            layer_init(nn.Linear(1024, 512)),
            self.activation,
        )
        self.fc_mu = layer_init(nn.Linear(512, np.prod(action_space)), std=0.01)

    def get_action(self, x, action=None):
        hidden = self.network(x)
        logits = self.fc_mu(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_greedy_action(self, x):
        hidden = self.network(x)
        logits = self.fc_mu(hidden)
        action = torch.argmax(logits, dim=1)
        return action

class MessageActor(nn.Module):
    def __init__(self, action_space, n_agents, n_hidden=64, n_channel=5):
        super().__init__()
        self.n_hidden = n_hidden
        self.action_size = action_space
        self.n_agents = n_agents
        self.activation = nn.ReLU()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(n_channel, 32, 3, stride=2)),
            self.activation,
            layer_init(nn.Conv2d(32, 64, 3, stride=1)),
            self.activation,
            nn.Flatten(),
            layer_init(nn.Linear(1024, 512)),
            self.activation,
        )

        self.fc_mu = layer_init(nn.Linear(512 + n_agents, np.prod(action_space) + n_agents), std=0.01)
        self.msg_encoder = layer_init(nn.Linear(n_agents, n_agents), std=0.01)

    def get_action(self, x, x_msg, action=None):
        hidden = self.network(x)
        x_msg = self.msg_encoder(x_msg)
        logits = self.fc_mu(torch.cat([hidden, x_msg], axis=1))
        probs = Categorical(logits=logits[:, :-self.n_agents])
        msg_out = logits[:, -self.n_agents:]
        if action is None:
            action = probs.sample()
        return action, msg_out, probs.log_prob(action), probs.entropy()

    def get_greedy_action(self, x, x_msg):
        hidden = self.network(x)
        x_msg = self.msg_encoder(x_msg)
        logits = self.fc_mu(torch.cat([hidden, x_msg], axis=1))
        action = torch.argmax(logits[:, :-self.n_agents], dim=1)
        msg_out = logits[:, -self.n_agents:]
        return action, msg_out


class MessageCritic(nn.Module):
    def __init__(self, n_action, n_hidden=64, n_channel=5):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_agents = int(n_channel // 5)
        self.n_action = n_action
        self.activation = nn.ReLU()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(n_channel, 32, 3, stride=2)),
            self.activation,
            layer_init(nn.Conv2d(32, 64, 3, stride=1)),
            self.activation,
            nn.Flatten(),
            layer_init(nn.Linear(1024, 512)),
            self.activation,
        )
        self.msg_encoder = layer_init(nn.Linear(self.n_agents, self.n_agents), std=1)
        self.critic = layer_init(nn.Linear(512 + self.n_agents, 1), std=1)


    def get_value(self, x, x_msg):
        batch, w, h = x.shape[1], x.shape[-2], x.shape[-1] # x: [n_agent, batch, 5, 13, 13]
        x_swap = torch.swapaxes(x, 0, 1)
        x_squash = x_swap.reshape(batch, -1, w, h)
        x_obs = self.network(x_squash)
        x_msg = self.msg_encoder(x_msg)
        out = self.critic(torch.cat([x_obs, x_msg], axis=1))
        return out