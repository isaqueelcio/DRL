# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Actor (Policy) Model com Layer Normalization."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256):
        """
        Initialize parameters and build model.
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.LayerNorm(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.LayerNorm(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """ 
        Reset model parameters.
        """
        self.fc1.weight.data.uniform_(*self.hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self.hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """ 
        Build an actor (policy) network that maps states -> actions.
        """
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        return torch.tanh(self.fc3(x))

    @staticmethod
    def hidden_init(layer):
        """ 
        Initialize hidden layers with a uniform distribution based on fan-in.
        """
        fan_in = layer.weight.data.size()[0]
        lim = 1. / (fan_in ** 0.5)
        return (-lim, lim)


class Critic(nn.Module):
    """Critic (Value) Model com 4 camadas e Layer Normalization."""

    def __init__(self, state_size, action_size, seed):
        """ 
        Initialize parameters and build model.
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, 256)
        self.bn1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256 + action_size, 256)
        self.bn2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.LayerNorm(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.LayerNorm(64)
        self.fc5 = nn.Linear(64, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset model parameters with a uniform distribution.
        """        
        self.fcs1.weight.data.uniform_(*self.hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*self.hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*self.hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*self.hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Build a critic (value) network that maps (state, action) pairs -> Q-values.
        """
        xs = F.relu(self.bn1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        return self.fc5(x)

    @staticmethod
    def hidden_init(layer):
        """ 
        Initialize hidden layers with a uniform distribution based on fan-in.
        """
        fan_in = layer.weight.data.size()[0]
        lim = 1. / (fan_in ** 0.5)
        return (-lim, lim)
