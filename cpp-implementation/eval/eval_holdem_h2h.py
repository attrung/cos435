#!/usr/bin/env python3
"""Head-to-head evaluation for Hold'em NFSP models.

Loads LibTorch-saved avg policy weights, plays N games against a random opponent,
reports win rate. Called from C++ training loop.
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pyspiel


GAME_STRING = (
    "universal_poker(betting=limit,numPlayers=2,numRounds=4,"
    "blind=50 100,firstPlayer=2 1 1 1,numSuits=4,numRanks=13,"
    "numHoleCards=2,numBoardCards=0 3 1 1,"
    "raiseSize=100 100 200 200,maxRaises=3 3 3 3)"
)


class VarMLP(nn.Module):
    """Matches C++ VarMLP: arbitrary layer sizes (per-layer hidden dim)."""
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.layers = nn.ModuleList()
        in_sz = input_size
        for h in hidden_sizes:
            self.layers.append(nn.Linear(in_sz, h))
            in_sz = h
        self.output = nn.Linear(in_sz, output_size)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output(x)


def load_model(path, state_size, num_actions, **_):
    """Load a C++ torch::save'd MLP model, auto-detecting architecture from weights."""
    jit_model = torch.jit.load(path, map_location='cpu')
    sd = jit_model.state_dict()

    # C++ saves keys as fc0.weight, fc1.weight, ..., output.weight.
    # Discover hidden sizes by iterating fc0, fc1, ... in order.
    fc_indices = sorted(
        int(k.split('.')[0][2:]) for k in sd.keys()
        if k.startswith('fc') and k.endswith('.weight')
    )
    hidden_sizes = []
    for idx in fc_indices:
        w = sd[f'fc{idx}.weight']
        hidden_sizes.append(w.shape[0])  # output dim of this layer

    # Verify output layer
    if 'output.weight' not in sd:
        raise RuntimeError(f"Model {path} has no output layer")
    out_shape = sd['output.weight'].shape
    if out_shape[0] != num_actions:
        raise RuntimeError(f"Output dim mismatch: {out_shape[0]} != {num_actions}")

    model = VarMLP(state_size, hidden_sizes, num_actions)
    new_sd = {}
    for key, val in sd.items():
        if key.startswith('fc'):
            idx = int(key.split('.')[0][2:])
            suffix = key.split('.', 1)[1]  # weight or bias
            new_sd[f'layers.{idx}.{suffix}'] = val
        else:
            new_sd[key] = val
    model.load_state_dict(new_sd)
    model.eval()
    return model


def get_nfsp_action(model, state, player_id, num_actions):
    """Sample action from avg policy network."""
    legal_actions = state.legal_actions(player_id)
    info_state = state.information_state_tensor(player_id)

    with torch.no_grad():
        x = torch.FloatTensor(info_state).unsqueeze(0)
        logits = model(x).squeeze(0)
        mask = torch.full((num_actions,), float('-inf'))
        for a in legal_actions:
            mask[a] = 0.0
        logits = logits + mask
        probs = F.softmax(logits, dim=0).numpy()

    probs_legal = np.array([probs[a] for a in legal_actions])
    total = probs_legal.sum()
    if total > 0:
        probs_legal /= total
    else:
        probs_legal = np.ones(len(legal_actions)) / len(legal_actions)

    action = np.random.choice(legal_actions, p=probs_legal)
    return action


def get_random_action(state, player_id):
    """Uniform random legal action."""
    legal_actions = state.legal_actions(player_id)
    return np.random.choice(legal_actions)


def play_game(game, models, num_actions, nfsp_player=0):
    """Play one game: NFSP agent vs random."""
    state = game.new_initial_state()

    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = np.random.choice(actions, p=probs)
            state.apply_action(action)
        else:
            player = state.current_player()
            if player == nfsp_player:
                action = get_nfsp_action(models[player], state, player, num_actions)
            else:
                action = get_random_action(state, player)
            state.apply_action(action)

    return state.returns()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-dir', required=True)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--num-games', type=int, default=1000)
    parser.add_argument('--result-file', type=str, default=None)
    args = parser.parse_args()

    game = pyspiel.load_game(GAME_STRING)
    state_size = game.information_state_tensor_size()
    num_actions = game.num_distinct_actions()

    # Load both player models
    models = []
    for p in range(2):
        path = os.path.join(args.weights_dir, f'p{p}_avg.pt')
        model = load_model(path, state_size, num_actions)
        models.append(model)

    # Play games with NFSP as player 0 and player 1, average results
    total_reward = 0.0
    for i in range(args.num_games):
        nfsp_player = i % 2
        returns = play_game(game, models, num_actions, nfsp_player=nfsp_player)
        total_reward += returns[nfsp_player]

    avg_reward = total_reward / args.num_games

    if args.result_file:
        with open(args.result_file, 'w') as f:
            f.write(f'{avg_reward:.10f}\n')

    print(f"[H2H] ep={args.episode} | avg_reward={avg_reward:.4f} over {args.num_games} games")


if __name__ == '__main__':
    main()
