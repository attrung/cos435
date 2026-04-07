"""Replay buffers for NFSP: numpy-backed for fast sampling.

Circular buffer for DQN, reservoir sampling for average policy.
All storage is contiguous numpy arrays — no Python lists of tuples.
Thread-safe: locks protect concurrent access from main + gradient threads.
"""

import numpy as np
import threading


class CircularReplayBuffer:
    """Circular (ring) replay buffer backed by pre-allocated numpy arrays.

    Sampling is pure numpy indexing — no Python loops, no zip, no conversion.
    Thread-safe via internal lock.
    """

    def __init__(self, capacity, state_size, num_actions):
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self._lock = threading.Lock()

        # Pre-allocate contiguous arrays
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.legal_masks = np.zeros((capacity, num_actions), dtype=np.float32)

    def add(self, state, action, reward, next_state, done, legal_actions_mask):
        with self._lock:
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

    def add_batch(self, states, actions, rewards, next_states, dones, legal_masks):
        """Add a batch using numpy slicing (handles wrap-around)."""
        n = len(states)
        if n == 0:
            return

        with self._lock:
            pos = self.position
            if pos + n <= self.capacity:
                # No wrap — single slice copy
                self.states[pos:pos + n] = states
                self.actions[pos:pos + n] = actions
                self.rewards[pos:pos + n] = rewards
                self.next_states[pos:pos + n] = next_states
                self.dones[pos:pos + n] = dones
                self.legal_masks[pos:pos + n] = legal_masks
            else:
                # Wrap around
                first = self.capacity - pos
                self.states[pos:] = states[:first]
                self.states[:n - first] = states[first:]
                self.actions[pos:] = actions[:first]
                self.actions[:n - first] = actions[first:]
                self.rewards[pos:] = rewards[:first]
                self.rewards[:n - first] = rewards[first:]
                self.next_states[pos:] = next_states[:first]
                self.next_states[:n - first] = next_states[first:]
                self.dones[pos:] = dones[:first]
                self.dones[:n - first] = dones[first:]
                self.legal_masks[pos:] = legal_masks[:first]
                self.legal_masks[:n - first] = legal_masks[first:]

            self.position = (pos + n) % self.capacity
            self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        """Sample batch — pure numpy indexing, no Python loops."""
        with self._lock:
            indices = np.random.randint(0, self.size, size=batch_size)
            return (
                self.states[indices].copy(),
                self.actions[indices].copy(),
                self.rewards[indices].copy(),
                self.next_states[indices].copy(),
                self.dones[indices].copy(),
                self.legal_masks[indices].copy(),
            )

    def __len__(self):
        return self.size


class ReservoirBuffer:
    """Reservoir sampling buffer backed by pre-allocated numpy arrays.

    Implements Algorithm R (Vitter, 1985) for uniform sampling from a stream.
    Thread-safe via internal lock.
    """

    def __init__(self, capacity, state_size):
        self.capacity = capacity
        self.size = 0
        self.total_seen = 0
        self._lock = threading.Lock()

        # Pre-allocate contiguous arrays
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)

    def add(self, state, action):
        with self._lock:
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

    def add_batch(self, states, actions):
        """Add batch of (state, action) pairs with reservoir sampling."""
        n = len(states)
        with self._lock:
            for i in range(n):
                self.total_seen += 1
                if self.size < self.capacity:
                    self.states[self.size] = states[i]
                    self.actions[self.size] = actions[i]
                    self.size += 1
                else:
                    idx = np.random.randint(0, self.total_seen)
                    if idx < self.capacity:
                        self.states[idx] = states[i]
                        self.actions[idx] = actions[i]

    def sample(self, batch_size):
        """Sample batch — pure numpy indexing."""
        with self._lock:
            indices = np.random.randint(0, self.size, size=batch_size)
            return self.states[indices].copy(), self.actions[indices].copy()

    def __len__(self):
        return self.size
