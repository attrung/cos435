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
        indices = np.random.randint(0, self.size, size=batch_size)
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


class ReservoirBuffer:
    """Reservoir sampling buffer (Algorithm R, Vitter 1985)."""

    def __init__(self, capacity, state_size):
        self.capacity = capacity
        self.size = 0
        self.total_seen = 0

        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)

    def add(self, state, action):
        self.total_seen += 1
        if self.size < self.capacity:
            self.states[self.size] = state
            self.actions[self.size] = action
            self.size += 1
        else:
            idx = np.random.randint(0, self.total_seen)
            if idx < self.capacity:
                self.states[idx] = state
                self.actions[idx] = action

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return self.states[indices], self.actions[indices]

    def __len__(self):
        return self.size
