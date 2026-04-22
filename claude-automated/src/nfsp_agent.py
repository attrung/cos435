"""NFSP agent with DQN best-response and average policy network.

Implements the Neural Fictitious Self-Play algorithm from Heinrich & Silver (2016),
matching the OpenSpiel reference implementation's hyperparameters and structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.networks import MLP
from src.replay_buffer import CircularReplayBuffer, ReservoirBuffer


class NFSPAgent:
    """NFSP Agent for one player.

    Two components:
    1. Best-response (DQN): learns Q(s,a) via Q-learning
    2. Average policy: supervised learning on past behavior stored in reservoir buffer

    Key differences from naive implementation (matching OpenSpiel reference):
    - Per-episode mode sampling (not per-step)
    - learn_every=64 (not every step)
    - Soft target network update (tau=0.995, every 19200 steps)
    - Reservoir stores action probabilities (soft targets, not hard labels)
    - Illegal action masking in avg policy training
    - Epsilon decay (0.06 -> 0.001 over 20M steps)
    - min_buffer_size_to_learn=1000
    """

    def __init__(self, player_id, state_size, num_actions, config, device):
        self.player_id = player_id
        self.state_size = state_size
        self.num_actions = num_actions
        self.device = device

        # Hyperparameters
        self.eta = config.get('eta', 0.1)
        self.gamma = config.get('gamma', 1.0)
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

        # DQN networks 
        self.q_network = MLP(state_size, hidden_size, num_actions).to(device)
        self.target_network = MLP(state_size, hidden_size, num_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Average policy network
        self.avg_policy_network = MLP(state_size, hidden_size, num_actions).to(device)

        # Optimizers - SGD as per original paper
        self.q_optimizer = torch.optim.SGD(self.q_network.parameters(), lr=self.dqn_lr_start)
        self.avg_optimizer = torch.optim.SGD(self.avg_policy_network.parameters(), lr=self.avg_lr)

        # Replay buffers
        dqn_buffer_size = config.get('dqn_buffer_size', 200000)
        reservoir_buffer_size = config.get('reservoir_buffer_size', 2000000)
        self.dqn_buffer = CircularReplayBuffer(dqn_buffer_size, state_size, num_actions)
        self.reservoir_buffer = ReservoirBuffer(reservoir_buffer_size, state_size, num_actions)

        # Per-episode mode
        self._mode = 'average_policy'
        self._sample_episode_mode()

        # Step counters (matching OpenSpiel's dual-counter system)
        # _game_steps: total game steps (all modes) — for learning triggers
        # _br_steps: BR-mode game steps only — for epsilon decay & target updates
        #   Uses a tensor so it can be placed in shared memory for parallel workers
        #   (workers need current _br_steps to compute epsilon correctly)
        self._game_steps = 0
        self._br_steps = torch.zeros(1, dtype=torch.long)

        # Loss tracking
        self._last_br_loss = None
        self._last_avg_loss = None

        # Pre-allocate reusable tensors for action selection
        self._state_buf = torch.zeros(1, state_size, device=device)
        self._legal_mask_buf = torch.full((num_actions,), float('-inf'), device=device)

    def _sample_episode_mode(self):
        """Sample mode for the next episode (per-episode, not per-step)."""
        # at the start of each episode, the agent samples:
        # best_response with probability eta
        # average_policy otherwise
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

        Returns (action, mode, action_probs) where action_probs is the
        probability vector used for reservoir storage (only for best_response mode).
        """
        # If is_evaluation=True, the agent always uses the average policy.
        # when evaluating NFSP, we care about the learned long-run strategy, not the temporary best-response branch.
        if is_evaluation:
            return self._avg_policy_action(state, legal_actions), 'average_policy', None

        if self._mode == 'best_response':
            action, probs = self._dqn_action_with_probs(state, legal_actions)
            return action, 'best_response', probs
        else:
            return self._avg_policy_action(state, legal_actions), 'average_policy', None

    def _dqn_action_with_probs(self, state, legal_actions):
        """Epsilon-greedy action from DQN, returning action-probability vector of length num_action."""
        probs = np.zeros(self.num_actions, dtype=np.float32)
        epsilon = self._get_epsilon()

        # with probability epsilon 
        # pick a random action (exploration)
        # assign uniform probability across legal action in probs
        if np.random.random() < epsilon:
            action = np.random.choice(legal_actions)
            for a in legal_actions:
                probs[a] = 1.0 / len(legal_actions)
        
        # otherwise, greedy action
        # convert the state to a tensor
        # run q_network(state)
        # mask all illegal actions with -inf
        # choose argmax action and put probability 1.0 for that action
        else:
            with torch.no_grad():
                self._state_buf[0] = torch.as_tensor(state, dtype=torch.float32)
                q_values = self.q_network(self._state_buf).squeeze(0)
                self._legal_mask_buf.fill_(float('-inf'))
                for a in legal_actions:
                    self._legal_mask_buf[a] = 0.0
                q_values = q_values + self._legal_mask_buf
                action = q_values.argmax().item()
            probs[action] = 1.0

        return action, probs

    def _avg_policy_action(self, state, legal_actions):
        """Sample action from average policy network."""
        
        with torch.no_grad():
            self._state_buf[0] = torch.as_tensor(state, dtype=torch.float32)
            # compute logits (raw action scores for each action) from avg_policy_network
            logits = self.avg_policy_network(self._state_buf).squeeze(0)
            # mask illegal actions
            self._legal_mask_buf.fill_(float('-inf'))
            for a in legal_actions:
                self._legal_mask_buf[a] = 0.0
            logits = logits + self._legal_mask_buf
            # applies softmax to convert logits to probabilities
            # and sample an action stochastically from that distribution 
            probs = F.softmax(logits, dim=0)
            return torch.multinomial(probs, 1).item()

    def add_transition(self, state, action, reward, next_state, done, legal_actions_mask):
        """Add transition to DQN buffer."""
        self.dqn_buffer.add(state, action, reward, next_state, done, legal_actions_mask)

    def add_reservoir(self, state, action_probs, legal_mask):
        """Add (state, action_probs, legal_mask) to reservoir buffer."""
        self.reservoir_buffer.add(state, action_probs, legal_mask)

    def increment_steps(self, n, is_br_mode=False):
        """Increment step counters and trigger learning/target updates when due.

        Uses dual counters matching OpenSpiel's architecture:
        - _game_steps (total, all modes): triggers DQN + avg policy learning
        - _br_steps (BR mode only): triggers epsilon decay + target network updates
        """
        old_steps = self._game_steps
        self._game_steps += n

        if is_br_mode:
            old_br = self._br_steps
            self._br_steps += n

        # Update learning rate
        lr = self._get_lr()
        for pg in self.q_optimizer.param_groups:
            pg['lr'] = lr

        # Learning triggers use total steps (matches OpenSpiel's combined behavior)
        old_learn = old_steps // self.learn_every
        new_learn = self._game_steps // self.learn_every
        for _ in range(new_learn - old_learn):
            br_loss = self._update_dqn()
            avg_loss = self._update_avg_policy()
            if br_loss is not None:
                self._last_br_loss = br_loss
            if avg_loss is not None:
                self._last_avg_loss = avg_loss

        # Target update uses BR steps (matches OpenSpiel DQN counter)
        if is_br_mode:
            old_target = old_br // self.target_update_freq
            new_target = self._br_steps // self.target_update_freq
            if new_target > old_target:
                self._soft_update_target()

    def _soft_update_target(self):
        """Soft (Polyak) update of target network: target = tau*q + (1-tau)*target."""
        tau = self.target_update_tau
        for target_param, q_param in zip(self.target_network.parameters(),
                                         self.q_network.parameters()):
            target_param.data.copy_(tau * q_param.data + (1.0 - tau) * target_param.data)

    @property
    def losses(self):
        """Return (last_br_loss, last_avg_loss)."""
        return self._last_br_loss, self._last_avg_loss

    def _update_dqn(self):
        """Standard DQN update."""
        if len(self.dqn_buffer) < max(self.batch_size, self.min_buffer_size_to_learn):
            return None

        # sample a minibatch
        states, actions, rewards, next_states, dones, legal_masks = self.dqn_buffer.sample(self.batch_size)

        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards)
        next_states = torch.from_numpy(next_states)
        dones = torch.from_numpy(dones)
        legal_masks = torch.from_numpy(legal_masks)

        # Current Q values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        # r + gamma*(1−done)* max ​Q_target​(s′,a′)
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values + (legal_masks - 1.0) * 1e38
            next_q_max = next_q_values.max(dim=1)[0]
            targets = rewards + self.gamma * (1 - dones) * next_q_max

        # Loss and gradient step
        loss = F.mse_loss(q_values, targets) #mse loss between predicted q_value and target

        self.q_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.q_optimizer.step()

        return loss.item()

    def _update_avg_policy(self):
        """Supervised learning update for average policy with soft targets and legal masking.

            The average policy is trained like:
            At this state, reproduce the behavior distribution that the best-response branch used.
        
        """
        if len(self.reservoir_buffer) < max(self.batch_size, self.min_buffer_size_to_learn):
            return None
        
        # sample state, action-probs and legal action masks from the reservoir
        states, action_probs, legal_masks = self.reservoir_buffer.sample(self.batch_size)

        states = torch.from_numpy(states)
        action_probs = torch.from_numpy(action_probs)
        legal_masks = torch.from_numpy(legal_masks)

        # calculate logits from avg_policy_network(states)
        logits = self.avg_policy_network(states)

        # Mask illegal actions before computing loss
        logits = logits + (legal_masks - 1.0) * 1e38

        # Soft cross-entropy: -sum(target_probs * log_softmax(logits))
        loss = F.cross_entropy(logits, action_probs)

        self.avg_optimizer.zero_grad(set_to_none=True)
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

    def get_weights(self):
        """Get network weights for syncing to workers (lightweight, no optimizer state)."""
        return {
            'q_network': {k: v.clone() for k, v in self.q_network.state_dict().items()},
            'avg_policy_network': {k: v.clone() for k, v in self.avg_policy_network.state_dict().items()},
        }

    def load_weights(self, weights):
        """Load network weights from main process (workers only need networks, not optimizers)."""
        self.q_network.load_state_dict(weights['q_network'])
        self.avg_policy_network.load_state_dict(weights['avg_policy_network'])

    def get_state_dict(self):
        return {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'avg_policy_network': self.avg_policy_network.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'avg_optimizer': self.avg_optimizer.state_dict(),
            'game_steps': self._game_steps,
            'br_steps': self._br_steps,
            'dqn_buffer': self.dqn_buffer.state_dict(),
            'reservoir_buffer': self.reservoir_buffer.state_dict(),
        }

    def load_state_dict(self, checkpoint):
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.avg_policy_network.load_state_dict(checkpoint['avg_policy_network'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.avg_optimizer.load_state_dict(checkpoint['avg_optimizer'])
        self._game_steps = checkpoint['game_steps']
        br = checkpoint.get('br_steps', 0)
        self._br_steps.fill_(int(br) if not isinstance(br, int) else br)
        if 'dqn_buffer' in checkpoint:
            self.dqn_buffer.load_state_dict(checkpoint['dqn_buffer'])
        if 'reservoir_buffer' in checkpoint:
            self.reservoir_buffer.load_state_dict(checkpoint['reservoir_buffer'])
