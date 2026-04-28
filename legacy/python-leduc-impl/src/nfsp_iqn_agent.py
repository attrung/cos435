"""NFSP agent with IQN replacing DQN for best-response.

Everything else (average policy, reservoir buffer, eta) stays the same as baseline NFSP.
Matches OpenSpiel reference: per-episode mode, learn_every, soft targets, etc.
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
        self.dqn_lr_start = config.get('dqn_lr_start', config.get('dqn_lr', 0.01))
        self.dqn_lr_end = config.get('dqn_lr_end', self.dqn_lr_start)
        self.lr_decay_steps = config.get('lr_decay_episodes', 0) * 5  # approx game steps
        self.avg_lr = config.get('avg_policy_lr', 0.01)
        self.batch_size = config.get('batch_size', 128)
        hidden_size = config.get('hidden_size', 128)

        # Learning schedule (matching OpenSpiel reference)
        self.learn_every = config.get('learn_every', 64)
        self.min_buffer_size_to_learn = config.get('min_buffer_size_to_learn', 1000)
        self.target_update_freq = config.get('target_update_freq', 19200)
        self.target_update_tau = config.get('target_update_tau', 0.995)

        # Epsilon decay
        self.epsilon_start = config.get('epsilon_start', 0.06)
        self.epsilon_end = config.get('epsilon_end', 0.001)
        self.epsilon_decay_duration = config.get('epsilon_decay_duration', 20000000)

        # IQN best-response (replaces DQN)
        self.iqn = IQNAgent(state_size, num_actions, config, device)

        # Average policy network (same as baseline)
        self.avg_policy_network = MLP(state_size, hidden_size, num_actions).to(device)
        self.avg_optimizer = torch.optim.SGD(self.avg_policy_network.parameters(), lr=self.avg_lr)

        # Reservoir buffer for average policy (stores soft targets)
        reservoir_buffer_size = config.get('reservoir_buffer_size', 2000000)
        self.reservoir_buffer = ReservoirBuffer(reservoir_buffer_size, state_size, num_actions)

        # Per-episode mode
        self._mode = 'average_policy'
        self._sample_episode_mode()

        # Step counters (matching OpenSpiel's dual-counter system)
        self._game_steps = 0
        self._br_steps = torch.zeros(1, dtype=torch.long)

        # Loss tracking
        self._last_br_loss = None
        self._last_avg_loss = None

        # Pre-allocate reusable tensors for action selection
        self._state_buf = torch.zeros(1, state_size, device=device)
        self._legal_mask_buf = torch.full((num_actions,), float('-inf'), device=device)

    def _sample_episode_mode(self):
        """Sample mode for the next episode."""
        if np.random.random() < self.eta:
            self._mode = 'best_response'
        else:
            self._mode = 'average_policy'

    def sample_episode_mode(self):
        """Public interface to sample episode mode."""
        self._sample_episode_mode()

    def _get_epsilon(self):
        """Exponential epsilon decay based on BR-mode steps (matching OpenSpiel DQN counter)."""
        t = min(self._br_steps, self.epsilon_decay_duration)
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-t / self.epsilon_decay_duration)

    def _get_lr(self):
        """Linear LR decay from dqn_lr_start to dqn_lr_end over lr_decay_steps game steps."""
        if self.lr_decay_steps <= 0:
            return self.dqn_lr_start
        t = min(self._game_steps, self.lr_decay_steps)
        frac = t / self.lr_decay_steps
        return self.dqn_lr_start + (self.dqn_lr_end - self.dqn_lr_start) * frac

    def step(self, state, legal_actions, is_evaluation=False):
        """Choose action using NFSP strategy.

        Returns (action, mode, action_probs).
        """
        if is_evaluation:
            return self._avg_policy_action(state, legal_actions), 'average_policy', None

        if self._mode == 'best_response':
            epsilon = self._get_epsilon()
            action, probs = self.iqn.select_action_with_probs(state, legal_actions, epsilon)
            return action, 'best_response', probs
        else:
            return self._avg_policy_action(state, legal_actions), 'average_policy', None

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

    def add_reservoir(self, state, action_probs, legal_mask):
        """Add to reservoir buffer (soft targets)."""
        self.reservoir_buffer.add(state, action_probs, legal_mask)

    def increment_steps(self, n, is_br_mode=False):
        """Increment step counters and trigger learning/target updates when due.

        Uses dual counters matching OpenSpiel's architecture:
        - _game_steps (total, all modes): triggers IQN + avg policy learning
        - _br_steps (BR mode only): triggers epsilon decay + target network updates
        """
        old_steps = self._game_steps
        self._game_steps += n

        if is_br_mode:
            old_br = self._br_steps
            self._br_steps += n

        # Update learning rate
        lr = self._get_lr()
        for pg in self.iqn.optimizer.param_groups:
            pg['lr'] = lr

        # Learning triggers use total steps
        old_learn = old_steps // self.learn_every
        new_learn = self._game_steps // self.learn_every
        for _ in range(new_learn - old_learn):
            br_loss = self.iqn.update()
            avg_loss = self._update_avg_policy()
            if br_loss is not None:
                self._last_br_loss = br_loss
            if avg_loss is not None:
                self._last_avg_loss = avg_loss

        # Target update uses BR steps
        if is_br_mode:
            old_target = old_br // self.target_update_freq
            new_target = self._br_steps // self.target_update_freq
            if new_target > old_target:
                self.iqn.soft_update_target(self.target_update_tau)

    @property
    def losses(self):
        """Return (last_br_loss, last_avg_loss)."""
        return self._last_br_loss, self._last_avg_loss

    def _update_avg_policy(self):
        """Supervised learning update with soft targets and legal masking."""
        if len(self.reservoir_buffer) < max(self.batch_size, self.min_buffer_size_to_learn):
            return None

        states, action_probs, legal_masks = self.reservoir_buffer.sample(self.batch_size)
        states = torch.from_numpy(states)
        action_probs = torch.from_numpy(action_probs)
        legal_masks = torch.from_numpy(legal_masks)

        logits = self.avg_policy_network(states)
        # Mask illegal actions before computing loss
        logits = logits + (legal_masks - 1.0) * 1e38
        # Soft cross-entropy
        loss = F.cross_entropy(logits, action_probs)

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

    def get_weights(self):
        """Get network weights for syncing to workers."""
        return {
            'iqn_network': {k: v.clone() for k, v in self.iqn.network.state_dict().items()},
            'avg_policy_network': {k: v.clone() for k, v in self.avg_policy_network.state_dict().items()},
        }

    def load_weights(self, weights):
        """Load network weights from main process."""
        self.iqn.network.load_state_dict(weights['iqn_network'])
        self.avg_policy_network.load_state_dict(weights['avg_policy_network'])

    def get_state_dict(self):
        state = self.iqn.get_state_dict()
        state['avg_policy_network'] = self.avg_policy_network.state_dict()
        state['avg_optimizer'] = self.avg_optimizer.state_dict()
        state['game_steps'] = self._game_steps
        state['br_steps'] = self._br_steps
        state['dqn_buffer'] = self.iqn.buffer.state_dict()
        state['reservoir_buffer'] = self.reservoir_buffer.state_dict()
        return state

    def load_state_dict(self, checkpoint):
        self.iqn.load_state_dict(checkpoint)
        self.avg_policy_network.load_state_dict(checkpoint['avg_policy_network'])
        self.avg_optimizer.load_state_dict(checkpoint['avg_optimizer'])
        self._game_steps = checkpoint['game_steps']
        br = checkpoint.get('br_steps', 0)
        self._br_steps.fill_(int(br) if not isinstance(br, int) else br)
        if 'dqn_buffer' in checkpoint:
            self.iqn.buffer.load_state_dict(checkpoint['dqn_buffer'])
        if 'reservoir_buffer' in checkpoint:
            self.reservoir_buffer.load_state_dict(checkpoint['reservoir_buffer'])
