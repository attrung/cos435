#!/usr/bin/env python3
"""H2H: holdem_64x64_baseline vs holdem_64x64_meanvar at final (50M) weights.

Plays N/2 games with baseline as P0 vs meanvar as P1, and N/2 games swapped
(meanvar as P0 vs baseline as P1). Reports baseline's average chip/mbb winnings.
"""
import argparse, os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn.functional as F
import pyspiel

GAME_STR = ("universal_poker(betting=limit,numPlayers=2,numRounds=4,"
            "blind=50 100,firstPlayer=2 1 1 1,numSuits=4,numRanks=13,"
            "numHoleCards=2,numBoardCards=0 3 1 1,"
            "raiseSize=100 100 200 200,maxRaises=3 3 3 3)")
NUM_ACTIONS = 3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'eval'))
from model_utils import load_models as _auto_load


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


def play_one(game, m_p0, m_p1):
    s = game.new_initial_state()
    while not s.is_terminal():
        if s.is_chance_node():
            outcomes = s.chance_outcomes()
            acts, probs = zip(*outcomes)
            s.apply_action(int(np.random.choice(acts, p=probs)))
        else:
            p = s.current_player()
            m = m_p0 if p == 0 else m_p1
            s.apply_action(sample_action(m, s, p))
    return s.returns()[0]


def h2h(game, a_name, a_models, b_name, b_models, n_games, seed):
    # A as P0 vs B as P1
    np.random.seed(seed)
    torch.manual_seed(seed)
    v1 = [play_one(game, a_models[0], b_models[1]) for _ in range(n_games // 2)]
    # A as P1 vs B as P0. returns[0] is P0's (=B's); negate for A.
    np.random.seed(seed + 10000)
    torch.manual_seed(seed + 10000)
    v2 = [-play_one(game, b_models[0], a_models[1]) for _ in range(n_games // 2)]
    vals = np.array(v1 + v2)
    return vals.mean(), vals.std() / np.sqrt(len(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a-dir', default='final_weights_holdem_64x64_baseline')
    parser.add_argument('--a-name', default='baseline')
    parser.add_argument('--b-dir', default='final_weights_holdem_64x64_meanvar')
    parser.add_argument('--b-name', default='meanvar')
    parser.add_argument('--games', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.set_num_threads(1)
    game = pyspiel.load_game(GAME_STR)

    a = _auto_load(args.a_dir)
    b = _auto_load(args.b_dir)

    print(f"H2H: {args.a_name} ({args.a_dir}) vs {args.b_name} ({args.b_dir})")
    print(f"games={args.games} (half each as P0), seed={args.seed}")
    print("=" * 75)

    t0 = time.time()
    mean_a, se_a = h2h(game, args.a_name, a, args.b_name, b, args.games, args.seed)
    dt = time.time() - t0

    mbb = mean_a / 0.1   # 1 chip = 10 mbb (BB = 100 chips)
    mbb_se = se_a / 0.1
    sign = '+' if mean_a >= 0 else ''
    print(f"\n{args.a_name} avg chips vs {args.b_name}: {sign}{mean_a:.2f} ± {se_a:.2f}"
          f"   ({sign}{mbb:.1f} ± {mbb_se:.1f} mbb/g)   [{dt:.0f}s]")
    if mean_a > 0:
        print(f"  -> {args.a_name} wins")
    elif mean_a < 0:
        print(f"  -> {args.b_name} wins")
    else:
        print("  -> tie")


if __name__ == '__main__':
    main()
