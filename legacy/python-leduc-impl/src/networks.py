"""Neural network definitions for NFSP and IQN agents."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple MLP for DQN best-response and average policy networks.

    Single hidden layer with ReLU activation, as per Heinrich & Silver (2016).
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class IQNNetwork(nn.Module):
    """Implicit Quantile Network.

    Uses cosine embedding for quantile input tau, combined with state features
    to produce quantile values for each action.
    """

    def __init__(self, input_size, hidden_size, output_size, embedding_dim=64):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim

        # State feature layer
        self.state_fc = nn.Linear(input_size, hidden_size)

        # Cosine embedding for tau
        # phi(tau) = ReLU(sum_i cos(i * pi * tau) * w_i + b) for i=0..n-1
        self.cos_embedding = nn.Linear(embedding_dim, hidden_size)

        # Output layer: from element-wise product to Q-values
        self.output_fc = nn.Linear(hidden_size, output_size)

    def forward(self, state, taus):
        """Forward pass.

        Args:
            state: (batch_size, input_size)
            taus: (batch_size, num_quantiles)

        Returns:
            quantile_values: (batch_size, num_quantiles, num_actions)
        """
        batch_size = state.shape[0]
        num_quantiles = taus.shape[1]

        # State features: (batch_size, hidden_size)
        state_features = F.relu(self.state_fc(state))

        # Cosine embedding: (batch_size, num_quantiles, embedding_dim)
        # i_pi = i * pi for i in [0, embedding_dim)
        i_pi = torch.arange(0, self.embedding_dim, device=taus.device).float() * torch.pi
        # taus: (batch_size, num_quantiles, 1) * i_pi: (embedding_dim,)
        # -> (batch_size, num_quantiles, embedding_dim)
        cos_input = taus.unsqueeze(-1) * i_pi.unsqueeze(0).unsqueeze(0)
        cos_features = torch.cos(cos_input)
        # (batch_size, num_quantiles, hidden_size)
        tau_features = F.relu(self.cos_embedding(cos_features))

        # Element-wise product: expand state_features to match
        # state_features: (batch_size, 1, hidden_size) * tau_features: (batch_size, num_quantiles, hidden_size)
        combined = state_features.unsqueeze(1) * tau_features

        # Output: (batch_size, num_quantiles, num_actions)
        quantile_values = self.output_fc(combined)

        return quantile_values
