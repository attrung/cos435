"""NFSP agent with DQN best-response and average policy network.

Implements the Neural Fictitious Self-Play algorithm from Heinrich & Silver (2016).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from src.networks import MLP
from src.replay_buffer import CircularReplayBuffer, ReservoirBuffer


class NFSPAgent:
    """NFSP Agent for one player.

    Two components:
    1. Best-response (DQN): learns Q(s,a) via Q-learning
    2. Average policy: supervised learning on past behavior stored in reservoir buffer
    """

    def __init__(self, player_id, state_size, num_actions, config, device):
        self.player_id = player_id
        self.state_size = state_size
        self.num_actions = num_actions
        self.device = device

        # Hyperparameters
        self.eta = config.get('eta', 0.1)
        self.epsilon = config.get('dqn_epsilon', 0.06)
        self.gamma = config.get('gamma', 1.0)
        self.dqn_lr = config.get('dqn_lr', 0.1)
        self.avg_lr = config.get('avg_policy_lr', 0.01)
        self.batch_size = config.get('batch_size', 128)
        self.target_update_freq = config.get('target_update_freq', 1000)
        hidden_size = config.get('hidden_size', 128)

        # DQN networks
        self.q_network = MLP(state_size, hidden_size, num_actions).to(device)
        self.target_network = MLP(state_size, hidden_size, num_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Average policy network
        self.avg_policy_network = MLP(state_size, hidden_size, num_actions).to(device)

        # Optimizers - SGD as per original paper
        self.q_optimizer = torch.optim.SGD(self.q_network.parameters(), lr=self.dqn_lr)
        self.avg_optimizer = torch.optim.SGD(self.avg_policy_network.parameters(), lr=self.avg_lr)

        # Replay buffers (numpy-backed for fast sampling)
        dqn_buffer_size = config.get('dqn_buffer_size', 200000)
        reservoir_buffer_size = config.get('reservoir_buffer_size', 2000000)
        self.dqn_buffer = CircularReplayBuffer(dqn_buffer_size, state_size, num_actions)
        self.reservoir_buffer = ReservoirBuffer(reservoir_buffer_size, state_size)

        # Counters
        self.train_steps = 0
        self._mode = None  # 'best_response' or 'average_policy'

        # Pre-allocate reusable tensors for action selection (single sample)
        self._state_buf = torch.zeros(1, state_size, device=device)
        self._legal_mask_buf = torch.full((num_actions,), float('-inf'), device=device)

    def step(self, state, legal_actions, is_evaluation=False):
        """Choose action using NFSP strategy."""
        if is_evaluation:
            return self._avg_policy_action(state, legal_actions), 'average_policy'

        if np.random.random() < self.eta:
            self._mode = 'best_response'
            return self._dqn_action(state, legal_actions), 'best_response'
        else:
            self._mode = 'average_policy'
            return self._avg_policy_action(state, legal_actions), 'average_policy'

    def _dqn_action(self, state, legal_actions):
        """Epsilon-greedy action from DQN."""
        if np.random.random() < self.epsilon:
            return np.random.choice(legal_actions)

        with torch.no_grad():
            self._state_buf[0] = torch.as_tensor(state, dtype=torch.float32)
            q_values = self.q_network(self._state_buf).squeeze(0)
            self._legal_mask_buf.fill_(float('-inf'))
            for a in legal_actions:
                self._legal_mask_buf[a] = 0.0
            q_values = q_values + self._legal_mask_buf
            return q_values.argmax().item()

    def _avg_policy_action(self, state, legal_actions):
        """Sample action from average policy network."""
        with torch.no_grad():
            self._state_buf[0] = torch.as_tensor(state, dtype=torch.float32)
            logits = self.avg_policy_network(self._state_buf).squeeze(0)
            self._legal_mask_buf.fill_(float('-inf'))
            for a in legal_actions:
                self._legal_mask_buf[a] = 0.0
            logits = logits + self._legal_mask_buf
            probs = F.softmax(logits, dim=0)
            return torch.multinomial(probs, 1).item()

    def add_transition(self, state, action, reward, next_state, done, legal_actions_mask):
        """Add transition to DQN buffer."""
        self.dqn_buffer.add(state, action, reward, next_state, done, legal_actions_mask)

    def add_transition_batch(self, states, actions, rewards, next_states, dones, legal_masks):
        """Add batch of transitions to DQN buffer."""
        self.dqn_buffer.add_batch(states, actions, rewards, next_states, dones, legal_masks)

    def add_reservoir(self, state, action):
        """Add (state, action) to reservoir buffer for average policy training."""
        self.reservoir_buffer.add(state, action)

    def add_reservoir_batch(self, states, actions):
        """Add batch of (state, action) pairs to reservoir buffer."""
        self.reservoir_buffer.add_batch(states, actions)

    def update(self):
        """Update both DQN and average policy networks."""
        dqn_loss = self._update_dqn()
        avg_loss = self._update_avg_policy()
        self.train_steps += 1

        if self.train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return dqn_loss, avg_loss

    def _update_dqn(self):
        """Standard DQN update."""
        if len(self.dqn_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, legal_masks = self.dqn_buffer.sample(self.batch_size)

        states = torch.as_tensor(states, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        next_states = torch.as_tensor(next_states, device=self.device)
        dones = torch.as_tensor(dones, device=self.device)
        legal_masks = torch.as_tensor(legal_masks, device=self.device)

        # Current Q values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values + (legal_masks - 1.0) * 1e9
            next_q_max = next_q_values.max(dim=1)[0]
            targets = rewards + self.gamma * (1 - dones) * next_q_max

        loss = F.mse_loss(q_values, targets)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        return loss.item()

    def _update_avg_policy(self):
        """Supervised learning update for average policy."""
        if len(self.reservoir_buffer) < self.batch_size:
            return None

        states, actions = self.reservoir_buffer.sample(self.batch_size)

        states = torch.as_tensor(states, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)

        logits = self.avg_policy_network(states)
        loss = F.cross_entropy(logits, actions)

        self.avg_optimizer.zero_grad()
        loss.backward()
        self.avg_optimizer.step()

        return loss.item()

    def get_avg_policy_probs(self, state, legal_actions):
        """Get average policy action probabilities for exploitability computation."""
        with torch.no_grad():
            self._state_buf[0] = torch.as_tensor(state, dtype=torch.float32)
            logits = self.avg_policy_network(self._state_buf).squeeze(0)
            self._legal_mask_buf.fill_(float('-inf'))
            for a in legal_actions:
                self._legal_mask_buf[a] = 0.0
            logits = logits + self._legal_mask_buf
            probs = F.softmax(logits, dim=0)
            return probs.cpu().numpy()

    def get_state_dict(self):
        """Get state dict for checkpointing."""
        return {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'avg_policy_network': self.avg_policy_network.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'avg_optimizer': self.avg_optimizer.state_dict(),
            'train_steps': self.train_steps,
        }

    def load_state_dict(self, checkpoint):
        """Load state dict from checkpoint."""
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.avg_policy_network.load_state_dict(checkpoint['avg_policy_network'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.avg_optimizer.load_state_dict(checkpoint['avg_optimizer'])
        self.train_steps = checkpoint['train_steps']
