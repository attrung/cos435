"""NFSP agent with IQN replacing DQN for best-response.

Everything else (average policy, reservoir buffer, eta) stays the same as baseline NFSP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.networks import MLP
from src.replay_buffer import ReservoirBuffer
from src.iqn_agent import IQNAgent


class NFSPIQNAgent:
    """NFSP Agent using IQN for best-response instead of DQN."""

    def __init__(self, player_id, state_size, num_actions, config, device):
        self.player_id = player_id
        self.state_size = state_size
        self.num_actions = num_actions
        self.device = device

        # Hyperparameters
        self.eta = config.get('eta', 0.1)
        self.avg_lr = config.get('avg_policy_lr', 0.01)
        self.batch_size = config.get('batch_size', 128)
        hidden_size = config.get('hidden_size', 128)

        # IQN best-response (replaces DQN)
        self.iqn = IQNAgent(state_size, num_actions, config, device)

        # Average policy network (same as baseline)
        self.avg_policy_network = MLP(state_size, hidden_size, num_actions).to(device)
        self.avg_optimizer = torch.optim.SGD(self.avg_policy_network.parameters(), lr=self.avg_lr)

        # Reservoir buffer for average policy (numpy-backed for fast sampling)
        reservoir_buffer_size = config.get('reservoir_buffer_size', 2000000)
        self.reservoir_buffer = ReservoirBuffer(reservoir_buffer_size, state_size)

        self._mode = None

    def step(self, state, legal_actions, is_evaluation=False):
        """Choose action using NFSP strategy."""
        if is_evaluation:
            return self._avg_policy_action(state, legal_actions), 'average_policy'

        if np.random.random() < self.eta:
            self._mode = 'best_response'
            return self.iqn.select_action(state, legal_actions), 'best_response'
        else:
            self._mode = 'average_policy'
            return self._avg_policy_action(state, legal_actions), 'average_policy'

    def _avg_policy_action(self, state, legal_actions):
        """Sample from average policy network."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.avg_policy_network(state_tensor).squeeze(0)
            legal_mask = torch.full((self.num_actions,), float('-inf'), device=self.device)
            for a in legal_actions:
                legal_mask[a] = 0.0
            logits = logits + legal_mask
            probs = F.softmax(logits, dim=0)
            return torch.multinomial(probs, 1).item()

    def add_transition(self, state, action, reward, next_state, done, legal_actions_mask):
        """Add to IQN replay buffer."""
        self.iqn.add_transition(state, action, reward, next_state, done, legal_actions_mask)

    def add_transition_batch(self, states, actions, rewards, next_states, dones, legal_masks):
        """Add batch of transitions to IQN replay buffer."""
        self.iqn.buffer.add_batch(states, actions, rewards, next_states, dones, legal_masks)

    def add_reservoir(self, state, action):
        """Add to reservoir buffer."""
        self.reservoir_buffer.add(state, action)

    def add_reservoir_batch(self, states, actions):
        """Add batch to reservoir buffer."""
        self.reservoir_buffer.add_batch(states, actions)

    def update(self):
        """Update IQN and average policy."""
        iqn_loss = self.iqn.update()
        avg_loss = self._update_avg_policy()
        return iqn_loss, avg_loss

    def _update_avg_policy(self):
        if len(self.reservoir_buffer) < self.batch_size:
            return None

        states, actions = self.reservoir_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)

        logits = self.avg_policy_network(states)
        loss = F.cross_entropy(logits, actions)

        self.avg_optimizer.zero_grad()
        loss.backward()
        self.avg_optimizer.step()

        return loss.item()

    def get_avg_policy_probs(self, state, legal_actions):
        """Get average policy probabilities for exploitability."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.avg_policy_network(state_tensor).squeeze(0)
            legal_mask = torch.full((self.num_actions,), float('-inf'), device=self.device)
            for a in legal_actions:
                legal_mask[a] = 0.0
            logits = logits + legal_mask
            probs = F.softmax(logits, dim=0)
            return probs.cpu().numpy()

    def get_state_dict(self):
        state = self.iqn.get_state_dict()
        state['avg_policy_network'] = self.avg_policy_network.state_dict()
        state['avg_optimizer'] = self.avg_optimizer.state_dict()
        return state

    def load_state_dict(self, checkpoint):
        self.iqn.load_state_dict(checkpoint)
        self.avg_policy_network.load_state_dict(checkpoint['avg_policy_network'])
        self.avg_optimizer.load_state_dict(checkpoint['avg_optimizer'])
