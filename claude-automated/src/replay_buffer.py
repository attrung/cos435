"""Replay buffers for NFSP.

Circular buffer for DQN, reservoir sampling for average policy.
All storage is contiguous numpy arrays.
"""

import numpy as np


class CircularReplayBuffer:
    """Circular (ring) replay buffer backed by pre-allocated numpy arrays."""

    def __init__(self, capacity, state_size, num_actions):
        self.capacity = capacity
        self.size = 0
        self.position = 0

        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.legal_masks = np.zeros((capacity, num_actions), dtype=np.float32)

    def add(self, state, action, reward, next_state, done, legal_actions_mask):
        i = self.position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done
        self.legal_masks[i] = legal_actions_mask
        self.position = (i + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size):
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.legal_masks[indices],
        )

    def __len__(self):
        return self.size

    def state_dict(self):
        return {'size': self.size, 'position': self.position,
                'states': self.states[:self.size].copy(),
                'actions': self.actions[:self.size].copy(),
                'rewards': self.rewards[:self.size].copy(),
                'next_states': self.next_states[:self.size].copy(),
                'dones': self.dones[:self.size].copy(),
                'legal_masks': self.legal_masks[:self.size].copy()}

    def load_state_dict(self, d):
        n = d['size']
        self.size = n
        self.position = d['position']
        self.states[:n] = d['states']
        self.actions[:n] = d['actions']
        self.rewards[:n] = d['rewards']
        self.next_states[:n] = d['next_states']
        self.dones[:n] = d['dones']
        self.legal_masks[:n] = d['legal_masks']


class ReservoirBuffer:
    """Reservoir sampling buffer (Algorithm R, Vitter 1985).

    Stores (state, action_probs, legal_mask) tuples for average policy
    training with soft cross-entropy targets.
    """

    def __init__(self, capacity, state_size, num_actions):
        self.capacity = capacity
        self.size = 0
        self.total_seen = 0

        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.action_probs = np.zeros((capacity, num_actions), dtype=np.float32)
        self.legal_masks = np.zeros((capacity, num_actions), dtype=np.float32)

    def add(self, state, action_probs, legal_mask):
        self.total_seen += 1
        if self.size < self.capacity:
            self.states[self.size] = state
            self.action_probs[self.size] = action_probs
            self.legal_masks[self.size] = legal_mask
            self.size += 1
        else:
            idx = np.random.randint(0, self.total_seen)
            if idx < self.capacity:
                self.states[idx] = state
                self.action_probs[idx] = action_probs
                self.legal_masks[idx] = legal_mask

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return self.states[indices], self.action_probs[indices], self.legal_masks[indices]

    def __len__(self):
        return self.size

    def state_dict(self):
        return {'size': self.size, 'total_seen': self.total_seen,
                'states': self.states[:self.size].copy(),
                'action_probs': self.action_probs[:self.size].copy(),
                'legal_masks': self.legal_masks[:self.size].copy()}

    def load_state_dict(self, d):
        n = d['size']
        self.size = n
        self.total_seen = d['total_seen']
        self.states[:n] = d['states']
        self.action_probs[:n] = d['action_probs']
        self.legal_masks[:n] = d['legal_masks']
