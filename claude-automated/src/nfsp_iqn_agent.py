"""NFSP agent with IQN replacing DQN for best-response.

Everything else (average policy, reservoir buffer, eta) stays the same as baseline NFSP.
"""

import torch
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

        # Reservoir buffer for average policy
        reservoir_buffer_size = config.get('reservoir_buffer_size', 2000000)
        self.reservoir_buffer = ReservoirBuffer(reservoir_buffer_size, state_size)

        # LR decay
        self.iqn_lr_init = self.iqn.lr
        self.avg_lr_init = self.avg_lr
        self.lr_decay_min = config.get('lr_decay_min', 1.0)

        # Pre-allocate reusable tensors for action selection
        self._state_buf = torch.zeros(1, state_size, device=device)
        self._legal_mask_buf = torch.full((num_actions,), float('-inf'), device=device)

    def step(self, state, legal_actions, is_evaluation=False):
        """Choose action using NFSP strategy."""
        if is_evaluation:
            return self._avg_policy_action(state, legal_actions), 'average_policy'

        if np.random.random() < self.eta:
            return self.iqn.select_action(state, legal_actions), 'best_response'
        else:
            return self._avg_policy_action(state, legal_actions), 'average_policy'

    def _avg_policy_action(self, state, legal_actions):
        """Sample from average policy network."""
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
        """Add to IQN replay buffer."""
        self.iqn.add_transition(state, action, reward, next_state, done, legal_actions_mask)

    def add_reservoir(self, state, action):
        """Add to reservoir buffer."""
        self.reservoir_buffer.add(state, action)

    def update(self):
        """Update IQN and average policy. Returns (iqn_loss, avg_loss)."""
        iqn_loss = self.iqn.update()
        avg_loss = self._update_avg_policy()
        return iqn_loss, avg_loss

    def _update_avg_policy(self):
        if len(self.reservoir_buffer) < self.batch_size:
            return None

        states, actions = self.reservoir_buffer.sample(self.batch_size)
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)

        logits = self.avg_policy_network(states)
        loss = F.cross_entropy(logits, actions)

        self.avg_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.avg_optimizer.step()

        return loss.item()

    def get_avg_policy_probs(self, state, legal_actions):
        """Get average policy probabilities for exploitability."""
        with torch.no_grad():
            self._state_buf[0] = torch.as_tensor(state, dtype=torch.float32)
            logits = self.avg_policy_network(self._state_buf).squeeze(0)
            self._legal_mask_buf.fill_(float('-inf'))
            for a in legal_actions:
                self._legal_mask_buf[a] = 0.0
            logits = logits + self._legal_mask_buf
            probs = F.softmax(logits, dim=0)
            return probs.cpu().numpy()

    def set_lr(self, progress):
        """Set learning rate based on training progress (0.0 to 1.0)."""
        factor = 1.0 - (1.0 - self.lr_decay_min) * progress
        for pg in self.iqn.optimizer.param_groups:
            pg['lr'] = self.iqn_lr_init * factor
        for pg in self.avg_optimizer.param_groups:
            pg['lr'] = self.avg_lr_init * factor

    def get_state_dict(self):
        state = self.iqn.get_state_dict()
        state['avg_policy_network'] = self.avg_policy_network.state_dict()
        state['avg_optimizer'] = self.avg_optimizer.state_dict()
        return state

    def load_state_dict(self, checkpoint):
        self.iqn.load_state_dict(checkpoint)
        self.avg_policy_network.load_state_dict(checkpoint['avg_policy_network'])
        self.avg_optimizer.load_state_dict(checkpoint['avg_optimizer'])
