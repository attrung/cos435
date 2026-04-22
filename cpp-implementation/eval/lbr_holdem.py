#!/usr/bin/env python3
"""Local Best Response (LBR) approximation for Limit Hold'em.

Estimates exploitability by having a sampling-based best-response opponent
play against the evaluated agent's average policy.

At each decision point, the LBR opponent:
1. Tries each legal action
2. Runs M rollouts from the resulting state (both sides use eval policy)
3. Picks the action with highest average value

The LBR opponent's average payoff ≈ exploitability for that player.
Total exploitability = sum over both players of their LBR payoff.

Usage:
    python3 eval/lbr_holdem.py --weights eval_weights_holdem_baseline --rollouts 50 --games 500
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

GAME_STR = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=50 100,firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,raiseSize=100 100 200 200,maxRaises=3 3 3 3)"
NUM_ACTIONS = 3


sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model_utils import load_models as _load_models

def load_models(weights_dir, hidden=None):
    return _load_models(weights_dir)


def _load_models_compat(weights_dir, hidden=None):
    models = []
    for p in range(2):
        m = torch.jit.load(f'{weights_dir}/p{p}_avg.pt', map_location='cpu')
        m.eval()
        models.append(m)
    return models


def sample_action(model, state, player):
    """Sample action from avg policy network."""
    info = state.information_state_tensor(player)
    legal = state.legal_actions(player)
    with torch.no_grad():
        x = torch.FloatTensor(info).unsqueeze(0)
        logits = model(x).squeeze(0)
        mask = torch.full((NUM_ACTIONS,), float('-inf'))
        for a in legal:
            mask[a] = 0.0
        probs = F.softmax(logits + mask, dim=0).numpy()
        probs = probs / probs.sum()
    return int(np.random.choice(NUM_ACTIONS, p=probs))


def rollout_value(game, state, models, player):
    """Play out from state using both models, return value for player."""
    s = state.clone()
    while not s.is_terminal():
        if s.is_chance_node():
            outcomes = s.chance_outcomes()
            actions, probs = zip(*outcomes)
            s.apply_action(int(np.random.choice(actions, p=probs)))
        else:
            p = s.current_player()
            action = sample_action(models[p], s, p)
            s.apply_action(action)
    return s.returns()[player]


def lbr_game(game, models, lbr_player, num_rollouts):
    """Play one game where lbr_player uses LBR, opponent uses avg policy.
    Returns the LBR player's payoff."""
    state = game.new_initial_state()

    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            state.apply_action(int(np.random.choice(actions, p=probs)))
        else:
            player = state.current_player()
            if player == lbr_player:
                # LBR: try each action, estimate value via rollouts
                legal = state.legal_actions()
                best_action = legal[0]
                best_value = -1e9

                for action in legal:
                    child = state.clone()
                    child.apply_action(action)
                    # Estimate value via rollouts
                    total = 0.0
                    for _ in range(num_rollouts):
                        total += rollout_value(game, child, models, lbr_player)
                    avg_val = total / num_rollouts
                    if avg_val > best_value:
                        best_value = avg_val
                        best_action = action

                state.apply_action(best_action)
            else:
                # Opponent uses avg policy
                action = sample_action(models[player], state, player)
                state.apply_action(action)

    return state.returns()[lbr_player]


def compute_lbr(weights_dir, num_games=500, num_rollouts=50, hidden=256):
    """Compute LBR exploitability approximation."""
    game = pyspiel.load_game(GAME_STR)
    models = load_models(weights_dir, hidden)

    # LBR for player 0
    total_p0 = 0.0
    for g in range(num_games):
        total_p0 += lbr_game(game, models, 0, num_rollouts)
        if (g + 1) % 50 == 0:
            print(f'  P0 LBR: {g+1}/{num_games} games, avg={total_p0/(g+1):.2f} chips')
    lbr_p0 = total_p0 / num_games

    # LBR for player 1
    total_p1 = 0.0
    for g in range(num_games):
        total_p1 += lbr_game(game, models, 1, num_rollouts)
        if (g + 1) % 50 == 0:
            print(f'  P1 LBR: {g+1}/{num_games} games, avg={total_p1/(g+1):.2f} chips')
    lbr_p1 = total_p1 / num_games

    # Exploitability = average of both players' LBR values
    exploit = (lbr_p0 + lbr_p1) / 2.0
    exploit_mbb = exploit / 0.1  # convert to mbb/g (BB = 100 chips)

    return lbr_p0, lbr_p1, exploit, exploit_mbb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', required=True)
    parser.add_argument('--games', type=int, default=200,
                        help='Games per player for LBR (default 200)')
    parser.add_argument('--rollouts', type=int, default=30,
                        help='Rollouts per action per decision (default 30)')
    parser.add_argument('--hidden', type=int, default=256)
    args = parser.parse_args()

    print(f"LBR Exploitability (games={args.games}, rollouts={args.rollouts})")
    print("=" * 70)

    for w in args.weights:
        name = os.path.basename(w)
        print(f'\n--- {name} ---')
        lbr_p0, lbr_p1, exploit, exploit_mbb = compute_lbr(
            w, args.games, args.rollouts, args.hidden)
        print(f'  LBR P0: {lbr_p0:+.2f} chips/game')
        print(f'  LBR P1: {lbr_p1:+.2f} chips/game')
        print(f'  Exploitability: {exploit:.2f} chips/game = {exploit_mbb:.0f} mbb/g')

    print()


if __name__ == '__main__':
    main()
