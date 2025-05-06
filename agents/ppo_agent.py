import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

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
    def __init__(self, env, agent_name, n_hidden=64, n_channel=5, channel_last=False):
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
            layer_init(nn.Conv2d(n_channel, 32, 3, stride=2)),
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
        return self.critic(self.network(x.reshape(batch_size, -1)))

    def get_action_and_value(self, x, action=None):
        batch_size = x.shape[0]
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden.reshape(batch_size, -1))

class CentralCritic(nn.Module):
    def __init__(self, env, agent_list, n_hidden=64, n_channel=5):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_agents = int(n_channel // 5)
        self.observation_size_list = [env.observation_space(name).shape[0] for name in agent_list]
        self.total_act_size = sum([env.action_space(name).n for name in agent_list])
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
    def __init__(self, env, agent_name, n_hidden=64, n_channel=5):
        super().__init__()
        self.n_hidden = n_hidden
        self.obs_shape = env.observation_space(agent_name).shape
        self.action_size = env.action_space(agent_name).n
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
        self.fc_mu = layer_init(nn.Linear(512, np.prod(env.action_space(agent_name).n)), std=0.01)

    def get_action(self, x, action=None):
        hidden = self.network(x)
        logits = self.fc_mu(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

class MessageActor(nn.Module):
    def __init__(self, env, agent_name, n_agents, n_hidden=64, n_channel=5):
        super().__init__()
        self.n_hidden = n_hidden
        self.obs_shape = env.observation_space(agent_name).shape
        self.action_size = env.action_space(agent_name).n
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

        self.fc_mu = layer_init(nn.Linear(512, np.prod(env.action_space(agent_name).n)), std=0.01)
        self.msg_encoder = layer_init(nn.Linear(512, n_agents), std=0.01)
        self.gnn = GCN(2, n_agents)

    def get_action(self, x, x_msg, edge_index, action=None):
        hidden = self.network(x)
        logits = self.fc_mu(hidden)
        actor_encoding = self.msg_encoder(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        message_features = torch.stack([x_msg, actor_encoding[0]]).T  # [3, 2]
        msg_data = Data(x=message_features, edge_index=edge_index.t().contiguous())
        msg_out = self.gnn(msg_data).mean(axis=1)
        return action, msg_out, probs.log_prob(action), probs.entropy()

    def get_only_action(self, x, action=None):
        hidden = self.network(x)
        logits = self.fc_mu(hidden)
        actor_encoding = self.msg_encoder(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        message_features = torch.hstack([x_msg, actor_encoding]).T
        msg_data = Data(x=message_features, edge_index=edge_index.t().contiguous())
        msg_out = self.gnn(msg_data).mean(axis=1)
        return action, msg_out, probs.log_prob(action), probs.entropy()

class MessageCritic(nn.Module):
    def __init__(self, env, agent_list, n_hidden=64, n_channel=5):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_agents = int(n_channel // 5)
        self.agent_list = agent_list
        self.observation_size_list = [env.observation_space(name).shape[0] for name in agent_list]
        self.total_act_size = sum([env.action_space(name).n for name in agent_list])
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
        self.critic = layer_init(nn.Linear(512 + self.n_agents, 1), std=1)
        self.gnn = GCN(1, len(agent_list))

    def get_value(self, x, x_msg, edge_index):
        batch, w, h = x.shape[1], x.shape[-2], x.shape[-1] # x: [n_agent, batch, 5, 13, 13]
        x_swap = torch.swapaxes(x, 0, 1)
        x_squash = x_swap.reshape(batch, -1, w, h)
        x_vision = self.network(x_squash)
        msg_data = Data(x=x_msg.unsqueeze(1).float(), edge_index=edge_index.t().contiguous())
        msg_out = self.gnn(msg_data).mean(axis=1)
        out = self.critic(torch.cat((x_vision, msg_out.unsqueeze(0)), axis=1))
        return out