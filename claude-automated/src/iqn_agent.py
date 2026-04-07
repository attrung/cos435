"""IQN-based best-response module for NFSP.

Implements Implicit Quantile Networks (Dabney et al., 2018) to replace DQN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.networks import IQNNetwork
from src.replay_buffer import CircularReplayBuffer
from src.risk_distortion import get_distortion_fn, identity


class IQNAgent:
    """IQN best-response agent that replaces DQN in NFSP.

    Learns quantile function Q_tau(s,a) instead of scalar Q(s,a).
    """

    def __init__(self, state_size, num_actions, config, device):
        self.state_size = state_size
        self.num_actions = num_actions
        self.device = device

        # IQN-specific params
        self.num_quantile_samples = config.get('iqn_num_quantiles', 8)  # N for training
        self.num_quantile_eval = config.get('iqn_num_quantiles_eval', 32)  # K for action selection
        self.embedding_dim = config.get('iqn_embedding_dim', 64)
        self.kappa = config.get('iqn_kappa', 1.0)

        # Standard params
        self.epsilon = config.get('dqn_epsilon', 0.06)
        self.gamma = config.get('gamma', 1.0)
        self.lr = config.get('dqn_lr', 0.1)
        self.batch_size = config.get('batch_size', 128)
        self.target_update_freq = config.get('target_update_freq', 1000)
        hidden_size = config.get('hidden_size', 128)

        # Networks
        self.network = IQNNetwork(state_size, hidden_size, num_actions, self.embedding_dim).to(device)
        self.target_network = IQNNetwork(state_size, hidden_size, num_actions, self.embedding_dim).to(device)
        self.target_network.load_state_dict(self.network.state_dict())

        # Optimizer - SGD to match baseline
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr)

        # Replay buffer (numpy-backed for fast sampling)
        buffer_size = config.get('dqn_buffer_size', 200000)
        self.buffer = CircularReplayBuffer(buffer_size, state_size, num_actions)

        # Risk distortion
        self.distortion_fn = get_distortion_fn(config)

        # Mean-variance utility: Q_adj = E[Q] - variance_penalty * Var[Q]
        self.variance_penalty = config.get('variance_penalty', 0.0)

        self.train_steps = 0

    def select_action(self, state, legal_actions):
        """Select action using quantile samples.

        Risk-neutral: argmax E[Q_tau(s,a)]
        CVaR/seeking: distort tau, then argmax E[Q_tau(s,a)]
        Mean-variance: argmax (E[Q] - A * Var[Q])
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(legal_actions)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # Sample K uniform quantiles, then distort
            taus = torch.rand(1, self.num_quantile_eval, device=self.device)
            taus = self.distortion_fn(taus)
            # (1, K, num_actions)
            quantile_values = self.network(state_tensor, taus)

            if self.variance_penalty > 0:
                # Mean-variance utility
                q_mean = quantile_values.mean(dim=1).squeeze(0)
                q_var = quantile_values.var(dim=1).squeeze(0)
                q_values = q_mean - self.variance_penalty * q_var
            else:
                # Standard: average over quantiles
                q_values = quantile_values.mean(dim=1).squeeze(0)

            # Mask illegal actions
            legal_mask = torch.full((self.num_actions,), float('-inf'), device=self.device)
            for a in legal_actions:
                legal_mask[a] = 0.0
            q_values = q_values + legal_mask
            return q_values.argmax().item()

    def add_transition(self, state, action, reward, next_state, done, legal_actions_mask):
        self.buffer.add(state, action, reward, next_state, done, legal_actions_mask)

    def update(self):
        """IQN quantile Huber loss update."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, legal_masks = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        legal_masks = torch.FloatTensor(legal_masks).to(self.device)

        batch_size = states.shape[0]
        N = self.num_quantile_samples

        # Sample quantiles for current state
        taus = torch.rand(batch_size, N, device=self.device)

        # Current quantile values: (batch_size, N, num_actions)
        current_quantiles = self.network(states, taus)
        # Gather actions: (batch_size, N)
        current_quantiles = current_quantiles.gather(
            2, actions.unsqueeze(1).unsqueeze(2).expand(-1, N, -1)
        ).squeeze(2)

        # Target quantile values
        with torch.no_grad():
            taus_target = torch.rand(batch_size, N, device=self.device)
            # (batch_size, N, num_actions)
            next_quantiles = self.target_network(next_states, taus_target)

            # Best action from next state (average over quantiles)
            next_q_avg = next_quantiles.mean(dim=1)  # (batch_size, num_actions)
            next_q_avg = next_q_avg + (legal_masks - 1.0) * 1e9
            best_actions = next_q_avg.argmax(dim=1)  # (batch_size,)

            # Gather target quantiles for best action: (batch_size, N)
            next_quantiles = next_quantiles.gather(
                2, best_actions.unsqueeze(1).unsqueeze(2).expand(-1, N, -1)
            ).squeeze(2)

            # TD targets: (batch_size, N)
            targets = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * next_quantiles

        # Quantile Huber loss
        # current: (batch_size, N, 1), targets: (batch_size, 1, N)
        td_errors = targets.unsqueeze(1) - current_quantiles.unsqueeze(2)  # (batch_size, N, N)
        huber_loss = torch.where(
            td_errors.abs() <= self.kappa,
            0.5 * td_errors ** 2,
            self.kappa * (td_errors.abs() - 0.5 * self.kappa)
        )
        # taus: (batch_size, N, 1)
        tau_weights = (taus.unsqueeze(2) - (td_errors.detach() < 0).float()).abs()
        loss = (tau_weights * huber_loss).sum(dim=2).mean(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        return loss.item()

    def get_state_dict(self):
        return {
            'iqn_network': self.network.state_dict(),
            'iqn_target_network': self.target_network.state_dict(),
            'iqn_optimizer': self.optimizer.state_dict(),
            'iqn_train_steps': self.train_steps,
        }

    def load_state_dict(self, checkpoint):
        self.network.load_state_dict(checkpoint['iqn_network'])
        self.target_network.load_state_dict(checkpoint['iqn_target_network'])
        self.optimizer.load_state_dict(checkpoint['iqn_optimizer'])
        self.train_steps = checkpoint['iqn_train_steps']
