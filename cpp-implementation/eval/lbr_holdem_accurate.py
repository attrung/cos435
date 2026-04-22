#!/usr/bin/env python3
"""Accurate LBR with parallel execution and confidence intervals."""

import argparse, os, sys, warnings, time
warnings.filterwarnings('ignore')
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyspiel

GAME_STR = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=50 100,firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,raiseSize=100 100 200 200,maxRaises=3 3 3 3)"
NUM_ACTIONS = 3

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model_utils import load_models as _auto_load

def load_models(weights_dir, hidden=None):
    return _auto_load(weights_dir)

def _unused(weights_dir, hidden=None):
    models = []
    for p in range(2):
        m = torch.jit.load(f'{weights_dir}/p{p}_avg.pt', map_location='cpu')
        m.eval()
        models.append(m)
    return models

def sample_action(model, state, player):
    info = state.information_state_tensor(player)
    legal = state.legal_actions(player)
    with torch.no_grad():
        x = torch.FloatTensor(info).unsqueeze(0)
        logits = model(x).squeeze(0)
        mask = torch.full((NUM_ACTIONS,), float('-inf'))
        for a in legal: mask[a] = 0.0
        probs = F.softmax(logits + mask, dim=0).numpy()
        probs = probs / probs.sum()
    return int(np.random.choice(NUM_ACTIONS, p=probs))

def rollout_value(game, state, models, player):
    s = state.clone()
    while not s.is_terminal():
        if s.is_chance_node():
            outcomes = s.chance_outcomes()
            actions, probs = zip(*outcomes)
            s.apply_action(int(np.random.choice(actions, p=probs)))
        else:
            s.apply_action(sample_action(models[s.current_player()], s, s.current_player()))
    return s.returns()[player]

def lbr_game(game, models, lbr_player, num_rollouts):
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            state.apply_action(int(np.random.choice(actions, p=probs)))
        else:
            player = state.current_player()
            if player == lbr_player:
                legal = state.legal_actions()
                best_action, best_value = legal[0], -1e9
                for action in legal:
                    child = state.clone()
                    child.apply_action(action)
                    val = sum(rollout_value(game, child, models, lbr_player)
                              for _ in range(num_rollouts)) / num_rollouts
                    if val > best_value:
                        best_value, best_action = val, action
                state.apply_action(best_action)
            else:
                state.apply_action(sample_action(models[player], state, player))
    return state.returns()[lbr_player]

def _worker(args):
    """Worker for parallel LBR computation."""
    weights_dir, lbr_player, num_games, num_rollouts, hidden, seed = args
    np.random.seed(seed)
    torch.set_num_threads(1)
    game = pyspiel.load_game(GAME_STR)
    models = load_models(weights_dir, hidden)
    results = []
    for _ in range(num_games):
        results.append(lbr_game(game, models, lbr_player, num_rollouts))
    return results

def compute_lbr_parallel(weights_dir, num_games=10000, num_rollouts=15,
                         hidden=256, num_workers=4):
    name = os.path.basename(weights_dir)
    results = {0: [], 1: []}

    for lbr_player in [0, 1]:
        games_per_worker = num_games // num_workers
        args = [(weights_dir, lbr_player, games_per_worker, num_rollouts,
                 hidden, 42 + i + lbr_player * 1000)
                for i in range(num_workers)]

        t0 = time.time()
        with mp.Pool(num_workers) as pool:
            worker_results = pool.map(_worker, args)
        elapsed = time.time() - t0

        for wr in worker_results:
            results[lbr_player].extend(wr)

        vals = np.array(results[lbr_player])
        print(f'  {name} P{lbr_player} LBR: mean={vals.mean():.1f} '
              f'std={vals.std():.1f} SE={vals.std()/np.sqrt(len(vals)):.1f} '
              f'({elapsed:.0f}s, {len(vals)} games)')

    p0 = np.array(results[0])
    p1 = np.array(results[1])
    exploit = (p0.mean() + p1.mean()) / 2.0
    # SE of exploit = sqrt(SE_p0² + SE_p1²) / 2
    se = np.sqrt((p0.std()**2/len(p0) + p1.std()**2/len(p1))) / 2
    mbb = exploit / 0.1
    mbb_se = se / 0.1

    return exploit, se, mbb, mbb_se, p0, p1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', required=True)
    parser.add_argument('--games', type=int, default=10000)
    parser.add_argument('--rollouts', type=int, default=15)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--hidden', type=int, default=256)
    args = parser.parse_args()

    print(f"LBR Exploitability (games={args.games}, rollouts={args.rollouts}, workers={args.workers})")
    print("=" * 75)

    results = []
    for w in args.weights:
        name = os.path.basename(w)
        print(f'\n--- {name} ---')
        exploit, se, mbb, mbb_se, p0, p1 = compute_lbr_parallel(
            w, args.games, args.rollouts, args.hidden, args.workers)
        results.append((name, exploit, se, mbb, mbb_se))

    print(f'\n{"="*75}')
    print(f'{"Agent":<30} {"Exploit (chips)":>16} {"mbb/g":>12} {"± SE":>10}')
    print(f'{"-"*75}')
    for name, exploit, se, mbb, mbb_se in results:
        print(f'{name:<30} {exploit:>12.1f} ± {se:>4.1f} {mbb:>10.0f} ± {mbb_se:>5.0f}')

if __name__ == '__main__':
    main()
