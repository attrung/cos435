#!/usr/bin/env python3
"""Utility to load C++ NFSP trained models into Python for evaluation.

Usage:
    from eval.load_cpp_models import load_agents, compute_exploitability, head_to_head

    agents = load_agents("final_weights_baseline")
    expl = compute_exploitability(agents)
    payoff_a, payoff_b = head_to_head(agents_a, agents_b, num_games=10000)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import exploitability


class MLP(nn.Module):
    """Matches C++ and Python NFSP MLP architecture."""
    def __init__(self, input_size=30, hidden_size=128, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class NFSPPolicy(policy_lib.Policy):
    """Wraps avg policy networks as OpenSpiel Policy."""
    def __init__(self, game, models):
        super().__init__(game, list(range(game.num_players())))
        self.models = models
        self.num_actions = game.num_distinct_actions()

    def action_probabilities(self, state, player_id=None):
        if player_id is None:
            player_id = state.current_player()
        legal = state.legal_actions(player_id)
        info = state.information_state_tensor(player_id)
        with torch.no_grad():
            x = torch.FloatTensor(info).unsqueeze(0)
            logits = self.models[player_id](x).squeeze(0)
            mask = torch.full((self.num_actions,), float('-inf'))
            for a in legal:
                mask[a] = 0.0
            probs = F.softmax(logits + mask, dim=0).numpy()
        d = {a: float(probs[a]) for a in legal}
        t = sum(d.values())
        return {a: p / t for a, p in d.items()} if t > 0 else {a: 1.0 / len(legal) for a in legal}


def load_agents(weights_dir, hidden_size=128):
    """Load avg policy models from C++ training output.

    Args:
        weights_dir: directory containing p0_avg.pt, p1_avg.pt

    Returns:
        list of 2 MLP models (one per player)
    """
    game = pyspiel.load_game('leduc_poker')
    state_size = game.information_state_tensor_size()
    num_actions = game.num_distinct_actions()

    models = []
    for p in range(2):
        model = MLP(state_size, hidden_size, num_actions)
        path = os.path.join(weights_dir, f'p{p}_avg.pt')
        jit_model = torch.jit.load(path, map_location='cpu')
        model.load_state_dict(jit_model.state_dict())
        model.eval()
        models.append(model)
    return models


def compute_expl(models):
    """Compute exploitability of loaded models."""
    game = pyspiel.load_game('leduc_poker')
    policy = NFSPPolicy(game, models)
    return exploitability.exploitability(game, policy)


def head_to_head(models_a, models_b, num_games=10000):
    """Play two sets of agents against each other.

    Returns (avg_payoff_a, avg_payoff_b).
    """
    game = pyspiel.load_game('leduc_poker')
    num_actions = game.num_distinct_actions()
    total_a, total_b = 0.0, 0.0

    for g in range(num_games):
        state = game.new_initial_state()
        # Alternate who is player 0
        if g % 2 == 0:
            active = [models_a, models_b]
            a_player = 0
        else:
            active = [models_b, models_a]
            a_player = 1

        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                action = np.random.choice(actions, p=probs)
                state.apply_action(int(action))
            else:
                player = state.current_player()
                legal = state.legal_actions()
                info = state.information_state_tensor(player)
                with torch.no_grad():
                    x = torch.FloatTensor(info).unsqueeze(0)
                    logits = active[player][player](x).squeeze(0)
                    mask = torch.full((num_actions,), float('-inf'))
                    for a in legal:
                        mask[a] = 0.0
                    probs = F.softmax(logits + mask, dim=0).numpy()
                    probs = probs / probs.sum()
                action = np.random.choice(num_actions, p=probs)
                state.apply_action(action)

        returns = state.returns()
        total_a += returns[a_player]
        total_b += returns[1 - a_player]

    return total_a / num_games, total_b / num_games


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', required=True,
                        help='Weight directories to evaluate (e.g. final_weights_baseline final_weights_iqn_neutral)')
    parser.add_argument('--num-games', type=int, default=10000)
    args = parser.parse_args()

    print(f"{'Agent':<25} {'Exploitability':>15}")
    print("-" * 42)

    all_models = {}
    for w in args.weights:
        name = os.path.basename(w).replace('final_weights_', '')
        models = load_agents(w)
        expl = compute_expl(models)
        print(f"{name:<25} {expl:>15.6f}")
        all_models[name] = models

    if len(all_models) > 1:
        names = list(all_models.keys())
        print(f"\n{'Head-to-Head':<25} {'vs':<15} {'Payoff A':>10} {'Payoff B':>10}")
        print("-" * 62)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pa, pb = head_to_head(all_models[names[i]], all_models[names[j]],
                                      num_games=args.num_games)
                print(f"{names[i]:<25} {names[j]:<15} {pa:>10.4f} {pb:>10.4f}")
