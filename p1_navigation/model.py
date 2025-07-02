import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Rede Neural simples para DQN."""

    def __init__(self, state_size, action_size, seed):
        """Inicializa os parâmetros.
        state_size: the size of the input (number of features in the state — here it's 37).
        action_size: the number of possible actions (here it's 4).
        seed: sets a seed for reproducibility.
        fc1, fc2, fc3: fully connected linear layers:
            fc1: from input state → 64 neurons
            fc2: 64 → 64 neurons
            fc3: 64 → number of actions (4 outputs)
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """Performs the forward pass.    
        self.fc1(state): Applies the first linear layer to the input state.
        F.relu(...): Applies the ReLU (Rectified Linear Unit) activation function to introduce non-linearity.
        self.fc2(...): Passes the result through the second linear layer, followed by ReLU.
        self.fc3(x): Outputs raw Q-values for each possible action (no activation function applied here,
                    as Q-values can take any real value).       
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
