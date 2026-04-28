#!/usr/bin/env python3
"""Head-to-head evaluation for Limit Hold'em NFSP agents.

Since exploitability is intractable for full Hold'em, we evaluate by:
1. Playing agents against each other (head-to-head payoffs)
2. Convergence proxy: br_loss and avg_policy_loss trends from training logs

Usage:
    # Evaluate latest checkpoints (all experiments)
    python3 scripts/eval_holdem.py

    # Evaluate with more games
    python3 scripts/eval_holdem.py --num-games 50000
"""

import argparse
import os
import sys
import glob
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import numpy as np
import pyspiel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.nfsp_agent import NFSPAgent
from src.nfsp_iqn_agent import NFSPIQNAgent
from src.utils import load_config, load_checkpoint

GAME_STR = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=50 100,firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,raiseSize=100 100 200 200,maxRaises=3 3 3 3)"


def load_agents_from_checkpoint(config_path):
    """Load agents from latest checkpoint."""
    config = load_config(config_path)
    game = pyspiel.load_game(GAME_STR)
    state_size = game.information_state_tensor_size()
    num_actions = game.num_distinct_actions()
    device = torch.device('cpu')

    agent_type = config.get('agent_type', 'nfsp')
    agents = []
    for pid in range(2):
        if agent_type == 'nfsp':
            agent = NFSPAgent(pid, state_size, num_actions, config, device)
        else:
            agent = NFSPIQNAgent(pid, state_size, num_actions, config, device)
        agents.append(agent)

    # Find latest checkpoint
    name = config['experiment_name']
    checkpoint_dir = config.get('checkpoint_dir', 'results/checkpoints')
    seed = 42
    latest_ep = 0
    for f in glob.glob(os.path.join(checkpoint_dir, f'{name}_seed{seed}_p0_ep*.pt')):
        try:
            ep = int(os.path.basename(f).split('_ep')[1].split('.pt')[0])
            latest_ep = max(latest_ep, ep)
        except (ValueError, IndexError):
            pass

    if latest_ep > 0:
        for p in range(2):
            path = os.path.join(checkpoint_dir, f'{name}_seed{seed}_p{p}_ep{latest_ep}.pt')
            if os.path.exists(path):
                load_checkpoint(path, agents[p])
        print(f"  Loaded {name} from ep {latest_ep:,}")
    else:
        print(f"  WARNING: no checkpoint found for {name}")

    return agents, latest_ep, name


def head_to_head(game, agents_a, agents_b, num_games=10000):
    """Play two agent sets against each other, alternating positions."""
    total_a, total_b = 0.0, 0.0
    num_actions = game.num_distinct_actions()

    for g in range(num_games):
        state = game.new_initial_state()
        if g % 2 == 0:
            current = [agents_a[0], agents_b[1]]
            a_player = 0
        else:
            current = [agents_b[0], agents_a[1]]
            a_player = 1

        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                state.apply_action(int(np.random.choice(actions, p=probs)))
            else:
                player = state.current_player()
                info = np.array(state.information_state_tensor(player), dtype=np.float32)
                legal = state.legal_actions()
                action, _, _ = current[player].step(info, legal, is_evaluation=True)
                state.apply_action(action)

        returns = state.returns()
        total_a += returns[a_player]
        total_b += returns[1 - a_player]

    return total_a / num_games, total_b / num_games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-games', type=int, default=10000)
    parser.add_argument('--configs', nargs='+', default=[
        'configs/holdem_baseline.yaml',
        'configs/holdem_iqn_neutral.yaml',
        'configs/holdem_iqn_mv01.yaml',
        'configs/holdem_iqn_mv05.yaml',
    ])
    args = parser.parse_args()

    game = pyspiel.load_game(GAME_STR)

    print("=" * 60)
    print("  Limit Hold'em — Head-to-Head Evaluation")
    print(f"  {args.num_games} games per matchup")
    print("=" * 60)

    # Load all agents
    all_agents = {}
    for cfg in args.configs:
        if not os.path.exists(cfg):
            print(f"  Skipping {cfg} (not found)")
            continue
        agents, ep, name = load_agents_from_checkpoint(cfg)
        all_agents[name] = (agents, ep)

    if len(all_agents) < 2:
        print("Need at least 2 trained agents for head-to-head.")
        return

    # Training progress summary
    print(f"\n{'Agent':<25} {'Episodes':>12}")
    print("-" * 40)
    for name, (agents, ep) in all_agents.items():
        print(f"{name:<25} {ep:>12,}")

    # Head-to-head
    names = list(all_agents.keys())
    print(f"\n{'Agent A':<25} {'Agent B':<20} {'Payoff A':>10} {'Payoff B':>10} {'mbb/g':>10}")
    print("-" * 78)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_agents = all_agents[names[i]][0]
            b_agents = all_agents[names[j]][0]
            pa, pb = head_to_head(game, a_agents, b_agents, args.num_games)
            # Convert to mbb/g: big blind = 100 chips in this game
            mbb = pa / 0.1  # 1 mbb = 0.001 BB = 0.1 chips
            print(f"{names[i]:<25} {names[j]:<20} {pa:>10.2f} {pb:>10.2f} {mbb:>+10.1f}")

    print()


if __name__ == '__main__':
    main()
