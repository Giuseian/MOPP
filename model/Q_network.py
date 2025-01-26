import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super(QNetwork, self).__init__()
        input_dim = state_dim + action_dim  # Combine state and action dimensions
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))  # Single output for Q-value
        self.model = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)  # Concatenate state and action
        return self.model(x)