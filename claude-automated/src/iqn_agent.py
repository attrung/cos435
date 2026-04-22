"""IQN-based best-response module for NFSP.

Implements Implicit Quantile Networks (Dabney et al., 2018) to replace DQN.
Target network updates are handled externally by the parent NFSP agent.
"""

import torch
import torch.nn.functional as F
import numpy as np

from src.networks import IQNNetwork
from src.replay_buffer import CircularReplayBuffer
from src.risk_distortion import get_distortion_fn


class IQNAgent:
    """IQN best-response agent that replaces DQN in NFSP.

    Learns quantile function Q_tau(s,a) instead of scalar Q(s,a).
    Target network updates are NOT handled internally — the parent
    NFSPIQNAgent calls soft_update_target() based on game step count.
    """

    def __init__(self, state_size, num_actions, config, device):
        self.state_size = state_size
        self.num_actions = num_actions
        self.device = device

        # IQN-specific params
        self.num_quantile_samples = config.get('iqn_num_quantiles', 8)
        self.num_quantile_eval = config.get('iqn_num_quantiles_eval', 32)
        self.embedding_dim = config.get('iqn_embedding_dim', 64)
        self.kappa = config.get('iqn_kappa', 1.0)

        # Standard params
        self.gamma = config.get('gamma', 1.0)
        self.lr = config.get('dqn_lr', 0.01)
        self.batch_size = config.get('batch_size', 128)
        self.min_buffer_size_to_learn = config.get('min_buffer_size_to_learn', 1000)
        hidden_size = config.get('hidden_size', 128)

        # Networks
        self.network = IQNNetwork(state_size, hidden_size, num_actions, self.embedding_dim).to(device)
        self.target_network = IQNNetwork(state_size, hidden_size, num_actions, self.embedding_dim).to(device)
        self.target_network.load_state_dict(self.network.state_dict())

        # Optimizer - Adam for IQN (quantile loss landscape is harder than MSE)
        optimizer_str = config.get('iqn_optimizer', 'adam')
        if optimizer_str == 'adam':
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        elif optimizer_str == 'sgd':
            self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr)
        else:
            raise ValueError(f'Unknown optimizer: {optimizer_str}')

        # Replay buffer
        buffer_size = config.get('dqn_buffer_size', 200000)
        self.buffer = CircularReplayBuffer(buffer_size, state_size, num_actions)

        # Risk distortion
        self.distortion_fn = get_distortion_fn(config)

        # Mean-variance utility: Q_adj = E[Q] - variance_penalty * Var[Q]
        self.variance_penalty = config.get('variance_penalty', 0.0)

    def select_action(self, state, legal_actions, epsilon):
        """Select action using quantile samples.

        Args:
            state: info state array
            legal_actions: list of legal action indices
            epsilon: current exploration rate

        Returns:
            action: selected action index
        """
        if np.random.random() < epsilon:
            return np.random.choice(legal_actions)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            taus = torch.rand(1, self.num_quantile_eval, device=self.device)
            taus = self.distortion_fn(taus)
            quantile_values = self.network(state_tensor, taus)

            if self.variance_penalty > 0:
                q_mean = quantile_values.mean(dim=1).squeeze(0)
                q_var = quantile_values.var(dim=1).squeeze(0)
                q_values = q_mean - self.variance_penalty * q_var
            else:
                q_values = quantile_values.mean(dim=1).squeeze(0)

            legal_mask = torch.full((self.num_actions,), float('-inf'), device=self.device)
            for a in legal_actions:
                legal_mask[a] = 0.0
            q_values = q_values + legal_mask
            return q_values.argmax().item()

    def select_action_with_probs(self, state, legal_actions, epsilon):
        """Select action and return probability vector for reservoir storage."""
        probs = np.zeros(self.num_actions, dtype=np.float32)

        if np.random.random() < epsilon:
            action = np.random.choice(legal_actions)
            for a in legal_actions:
                probs[a] = 1.0 / len(legal_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                taus = torch.rand(1, self.num_quantile_eval, device=self.device)
                taus = self.distortion_fn(taus)
                quantile_values = self.network(state_tensor, taus)

                if self.variance_penalty > 0:
                    q_mean = quantile_values.mean(dim=1).squeeze(0)
                    q_var = quantile_values.var(dim=1).squeeze(0)
                    q_values = q_mean - self.variance_penalty * q_var
                else:
                    q_values = quantile_values.mean(dim=1).squeeze(0)

                legal_mask = torch.full((self.num_actions,), float('-inf'), device=self.device)
                for a in legal_actions:
                    legal_mask[a] = 0.0
                q_values = q_values + legal_mask
                action = q_values.argmax().item()
            probs[action] = 1.0

        return action, probs

    def add_transition(self, state, action, reward, next_state, done, legal_actions_mask):
        self.buffer.add(state, action, reward, next_state, done, legal_actions_mask)

    def update(self):
        """IQN quantile Huber loss update. Returns loss value or None."""
        if len(self.buffer) < max(self.batch_size, self.min_buffer_size_to_learn):
            return None

        states, actions, rewards, next_states, dones, legal_masks = self.buffer.sample(self.batch_size)

        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards)
        next_states = torch.from_numpy(next_states)
        dones = torch.from_numpy(dones)
        legal_masks = torch.from_numpy(legal_masks)

        N = self.num_quantile_samples

        # Sample quantiles for current state
        taus = torch.rand(self.batch_size, N, device=self.device)

        # Current quantile values: (batch_size, N, num_actions)
        current_quantiles = self.network(states, taus)
        # Gather actions: (batch_size, N)
        current_quantiles = current_quantiles.gather(
            2, actions.unsqueeze(1).unsqueeze(2).expand(-1, N, -1)
        ).squeeze(2)

        # Target quantile values
        with torch.no_grad():
            taus_target = torch.rand(self.batch_size, N, device=self.device)
            next_quantiles = self.target_network(next_states, taus_target)

            # Best action from next state (average over quantiles)
            next_q_avg = next_quantiles.mean(dim=1)
            next_q_avg = next_q_avg + (legal_masks - 1.0) * 1e38
            best_actions = next_q_avg.argmax(dim=1)

            # Gather target quantiles for best action: (batch_size, N)
            next_quantiles = next_quantiles.gather(
                2, best_actions.unsqueeze(1).unsqueeze(2).expand(-1, N, -1)
            ).squeeze(2)

            # TD targets: (batch_size, N)
            targets = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * next_quantiles

        # Quantile Huber loss
        td_errors = targets.unsqueeze(1) - current_quantiles.unsqueeze(2)  # (batch_size, N, N)
        huber_loss = torch.where(
            td_errors.abs() <= self.kappa,
            0.5 * td_errors ** 2,
            self.kappa * (td_errors.abs() - 0.5 * self.kappa)
        )
        tau_weights = (taus.unsqueeze(2) - (td_errors.detach() < 0).float()).abs()
        loss = (tau_weights * huber_loss).mean(dim=2).mean(dim=1).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def soft_update_target(self, tau):
        """Soft (Polyak) update of target network."""
        for target_param, param in zip(self.target_network.parameters(),
                                       self.network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def get_state_dict(self):
        return {
            'iqn_network': self.network.state_dict(),
            'iqn_target_network': self.target_network.state_dict(),
            'iqn_optimizer': self.optimizer.state_dict(),
        }

    def load_state_dict(self, checkpoint):
        self.network.load_state_dict(checkpoint['iqn_network'])
        self.target_network.load_state_dict(checkpoint['iqn_target_network'])
        # Skip optimizer state — incompatible across PyTorch versions.
        # Network weights are sufficient; optimizer warms up quickly.
