#!/usr/bin/env python3
"""LBR sanity check: exploit a uniform-random policy.

If LBR is working and units are right, exploitability of uniform-random in
limit Hold'em should be several thousand mbb/g (well above trained-agent
numbers like ~1900). If this returns a small number, something is wrong in
the LBR pipeline itself (rollout logic, unit conversion, etc.).
"""
import argparse, os, sys, time, warnings
warnings.filterwarnings('ignore')
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyspiel

GAME_STR = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=50 100,firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,raiseSize=100 100 200 200,maxRaises=3 3 3 3)"
NUM_ACTIONS = 3


class UniformModel(nn.Module):
    """Returns zero logits so softmax-over-legal gives uniform."""
    def forward(self, x):
        return torch.zeros(x.shape[0], NUM_ACTIONS)


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
    lbr_player, num_games, num_rollouts, seed = args
    np.random.seed(seed)
    torch.set_num_threads(1)
    game = pyspiel.load_game(GAME_STR)
    models = [UniformModel(), UniformModel()]
    return [lbr_game(game, models, lbr_player, num_rollouts) for _ in range(num_games)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=2000)
    parser.add_argument('--rollouts', type=int, default=15)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    print(f"LBR vs UNIFORM-RANDOM (games={args.games}, rollouts={args.rollouts}, workers={args.workers})")
    print("=" * 75)
    print("Expected: several thousand mbb/g (random is ~maximally exploitable).")
    print("If we see ~1000-2000 range, LBR/units are broken.")
    print("=" * 75)

    results = {0: [], 1: []}
    for lbr_player in [0, 1]:
        games_per_worker = args.games // args.workers
        worker_args = [(lbr_player, games_per_worker, args.rollouts, 42 + i + lbr_player * 1000)
                       for i in range(args.workers)]
        t0 = time.time()
        with mp.Pool(args.workers) as pool:
            out = pool.map(_worker, worker_args)
        elapsed = time.time() - t0
        for wr in out:
            results[lbr_player].extend(wr)
        vals = np.array(results[lbr_player])
        print(f'  uniform-random P{lbr_player} LBR: mean={vals.mean():.1f} '
              f'std={vals.std():.1f} SE={vals.std()/np.sqrt(len(vals)):.1f} '
              f'({elapsed:.0f}s, {len(vals)} games)')

    p0 = np.array(results[0]); p1 = np.array(results[1])
    exploit = (p0.mean() + p1.mean()) / 2.0
    se = np.sqrt((p0.std()**2/len(p0) + p1.std()**2/len(p1))) / 2
    mbb = exploit / 0.1
    mbb_se = se / 0.1
    print('=' * 75)
    print(f'uniform-random: exploit = {exploit:.1f} ± {se:.1f} chips | {mbb:.0f} ± {mbb_se:.0f} mbb/g')


if __name__ == '__main__':
    main()
