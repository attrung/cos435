"""Exploitability computation and head-to-head evaluation."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pyspiel
import numpy as np
import torch
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import exploitability


class NFSPPolicy(policy_lib.Policy):
    """Wraps NFSP average policy network as an OpenSpiel Policy for exploitability computation."""

    def __init__(self, game, agents, device):
        """
        Args:
            game: OpenSpiel game
            agents: list of agents (one per player), each having get_avg_policy_probs()
            device: torch device
        """
        super().__init__(game, list(range(game.num_players())))
        self.game = game
        self.agents = agents
        self.device = device
        self.num_actions = game.num_distinct_actions()

    def action_probabilities(self, state, player_id=None):
        """Return dict of {action: probability} for the given state."""
        if player_id is None:
            player_id = state.current_player()

        legal_actions = state.legal_actions(player_id)
        info_state = state.information_state_tensor(player_id)

        probs = self.agents[player_id].get_avg_policy_probs(info_state, legal_actions)
        action_probs = {a: float(probs[a]) for a in legal_actions}

        # Normalize to handle numerical issues
        total = sum(action_probs.values())
        if total > 0:
            action_probs = {a: p / total for a, p in action_probs.items()}
        else:
            # Uniform fallback
            action_probs = {a: 1.0 / len(legal_actions) for a in legal_actions}

        return action_probs


def compute_exploitability(game, agents, device):
    """Compute NashConv exploitability of agents' average policies."""
    nfsp_policy = NFSPPolicy(game, agents, device)
    return exploitability.exploitability(game, nfsp_policy)


def head_to_head(game, agents_a, agents_b, device, num_games=1000):
    """Play two sets of agents against each other.

    agents_a plays as both player 0 and player 1 alternately.
    Returns average payoffs for each side.
    """
    total_returns_a = 0.0
    total_returns_b = 0.0

    for game_idx in range(num_games):
        state = game.new_initial_state()

        # Alternate who is player 0
        if game_idx % 2 == 0:
            # agents_a is player 0, agents_b is player 1
            current_agents = [agents_a[0], agents_b[1]]
            a_player = 0
        else:
            # agents_b is player 0, agents_a is player 1
            current_agents = [agents_b[0], agents_a[1]]
            a_player = 1

        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
            else:
                player = state.current_player()
                info_state = state.information_state_tensor(player)
                legal_actions = state.legal_actions()
                action, _, _ = current_agents[player].step(info_state, legal_actions, is_evaluation=True)
                state.apply_action(action)

        returns = state.returns()
        total_returns_a += returns[a_player]
        total_returns_b += returns[1 - a_player]

    avg_return_a = total_returns_a / num_games
    avg_return_b = total_returns_b / num_games

    return avg_return_a, avg_return_b
