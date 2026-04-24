#!/usr/bin/env python3
"""Round-robin H2H tournament between all finished Hold'em variants.

Single-process, sequential. Loads each agent once. No multiprocessing → minimal RAM.
Each agent plays every other agent N games as P0 and N games as P1 (symmetric measure).

Output: symmetric NxN matrix of (agent_row's mbb/g winnings vs agent_col), averaged
over P0/P1 swap. Positive = row wins. Plus a ranking summary.
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn.functional as F
import pyspiel

GAME_STR = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=50 100,firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,raiseSize=100 100 200 200,maxRaises=3 3 3 3)"
NUM_ACTIONS = 3

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model_utils import load_models as _auto_load

# Agent name → weights dir on disk. Each dir must contain p0_avg.pt and p1_avg.pt.
AGENTS = {
    'baseline':    os.path.expanduser('~/cos435/cpp-implementation/final_weights_holdem_small_long'),
    'iqn_neutral': '/mnt/data/cos435/archive_20M/final_holdem_iqn_long_ep20M',
    'iqn_smaller': os.path.expanduser('~/cos435/cpp-implementation/eval_weights_holdem_iqn_smaller'),
    'iqn_averse':  os.path.expanduser('~/cos435/cpp-implementation/final_weights_holdem_iqn_averse'),
    'iqn_meanvar': os.path.expanduser('~/cos435/cpp-implementation/final_weights_holdem_iqn_meanvar'),
}


def sample_action(model, state, player):
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


def play_one_game(game, model_p0, model_p1):
    """Return the P0 return (positive = P0 won chips)."""
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            state.apply_action(int(np.random.choice(actions, p=probs)))
        else:
            p = state.current_player()
            model = model_p0 if p == 0 else model_p1
            state.apply_action(sample_action(model, state, p))
    return state.returns()[0]


def head_to_head(agent_a, agent_b, game, n_games=5000, seed=42):
    """
    agent_a, agent_b are dicts: {'p0': model, 'p1': model}.
    Returns (mean chip winnings of agent_a, SE) averaged over (A as P0) and (A as P1).
    """
    rng = np.random.RandomState(seed)
    # A as P0 vs B as P1
    np.random.seed(seed)
    torch.manual_seed(seed)
    vals_a_p0 = []
    for _ in range(n_games // 2):
        vals_a_p0.append(play_one_game(game, agent_a['p0'], agent_b['p1']))
    # A as P1 vs B as P0  — return here is P0's which is B's; negate for A
    np.random.seed(seed + 10000)
    torch.manual_seed(seed + 10000)
    vals_a_p1 = []
    for _ in range(n_games // 2):
        # P0 is B, P1 is A. P0's return is what B gets; A's = -return_p0
        vals_a_p1.append(-play_one_game(game, agent_b['p0'], agent_a['p1']))
    all_vals = np.array(vals_a_p0 + vals_a_p1)
    return all_vals.mean(), all_vals.std() / np.sqrt(len(all_vals))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=5000,
                        help='games per matchup (split half as P0 half as P1)')
    args = parser.parse_args()

    print(f"H2H tournament — {args.games} games per matchup (half as P0, half as P1)")
    print("=" * 80)

    torch.set_num_threads(1)  # be nice to training
    game = pyspiel.load_game(GAME_STR)

    # Load all agents
    t0 = time.time()
    loaded = {}
    for name, path in AGENTS.items():
        if not os.path.isdir(path):
            print(f'  SKIP {name}: no dir at {path}')
            continue
        try:
            models = _auto_load(path)
            loaded[name] = {'p0': models[0], 'p1': models[1]}
            print(f'  loaded {name} from {path}')
        except Exception as e:
            print(f'  FAIL {name}: {e}')
    print(f"  ({time.time()-t0:.1f}s to load all)")
    print()

    names = list(loaded.keys())
    n = len(names)
    matrix = np.zeros((n, n))
    se_matrix = np.zeros((n, n))

    total_pairs = n * (n - 1) // 2
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            t0 = time.time()
            mean, se = head_to_head(loaded[names[i]], loaded[names[j]], game, args.games)
            done += 1
            mbb = mean / 0.1  # chips → mbb/g (bb=100 chips, so 1 chip = 10 mbb)
            mbb_se = se / 0.1
            print(f"[{done}/{total_pairs}] {names[i]:>14} vs {names[j]:<14}  "
                  f"{names[i]}: {mbb:+7.1f} ± {mbb_se:.1f} mbb/g   ({time.time()-t0:.1f}s)")
            matrix[i, j] = mbb
            matrix[j, i] = -mbb
            se_matrix[i, j] = mbb_se
            se_matrix[j, i] = mbb_se

    print()
    print("=" * 80)
    print("Win matrix (row's mbb/g winnings vs column, averaged over P0/P1 swap):")
    print("=" * 80)
    header = " " * 15 + "  ".join(f"{n:>12}" for n in names)
    print(header)
    for i, row in enumerate(names):
        cells = "  ".join(f"{matrix[i,j]:+7.1f} ±{se_matrix[i,j]:4.1f}" if i != j else "      ---    " for j in range(n))
        print(f"{row:>14}  {cells}")

    print()
    print("Per-agent average (mean winnings vs all others):")
    for i, name in enumerate(names):
        others = [j for j in range(n) if j != i]
        avg = matrix[i, others].mean()
        print(f"  {name:>14}:  {avg:+7.1f} mbb/g avg")

    print()
    print("Ranking (best on top):")
    avgs = [(name, matrix[i, [j for j in range(n) if j != i]].mean()) for i, name in enumerate(names)]
    for rank, (name, v) in enumerate(sorted(avgs, key=lambda x: -x[1]), 1):
        print(f"  {rank}. {name:>14}  {v:+7.1f} mbb/g average vs others")

if __name__ == '__main__':
    main()
