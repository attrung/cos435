#!/usr/bin/env python3
"""Round-robin H2H tournament between Hold'em variants — BR HEADS (not avg policies).

Each agent plays using its best-response head:
  - NFSP baseline: argmax over its scalar Q-net.
  - IQN variants:  argmax over mean quantile value from its IQN net (K=32 samples).
                   For risk-distorted variants, the tau distortion applied during
                   training (CVaR α scales taus into the lower α-quantile) is
                   applied here too, matching the C++ compute_mean_q_snapshot.

Pure argmax (no ε-greedy). Output: N×N matrix of mbb/g + ranking.

Uses the HIGHEST-ITERATION model available for each variant:
  baseline   40M  (p*_q.pt       in final_weights_holdem_small_long)
  iqn_long   35M  (p*_iqn_net.pt in /mnt/data/cos435/weights/final_holdem_iqn_long;
                   note: the iqn_net file was staged from ep30M, avg from 35.14M)
  iqn_smaller 20M (iqn_net.pt    in /mnt/data/cos435/checkpoints/..._ep20000000_p*)
  iqn_averse  20M (iqn_net.pt    in /mnt/data/cos435/checkpoints/..._ep20000000_p*)
  iqn_meanvar 20M (iqn_net.pt    in /mnt/data/cos435/checkpoints/..._ep20000000_p*)
"""
import os, sys, time, warnings, math
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import pyspiel

GAME_STR = ("universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=50 100,"
            "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,"
            "numBoardCards=0 3 1 1,raiseSize=100 100 200 200,maxRaises=3 3 3 3)")
NUM_ACTIONS = 3
INFO_SIZE = 208

K_EVAL = 32          # number of tau samples per inference call
IQN_EMB_DIM = 64


# ── Network definitions matching C++ ─────────────────────────────────────

class VarMLP(nn.Module):
    """Scalar Q / avg net: per-layer hidden sizes, ReLU, linear output."""
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.layers = nn.ModuleList()
        in_sz = input_size
        for h in hidden_sizes:
            self.layers.append(nn.Linear(in_sz, h))
            in_sz = h
        self.output = nn.Linear(in_sz, output_size)

    def forward(self, x):
        for l in self.layers:
            x = torch.relu(l(x))
        return self.output(x)


class IQNNetwork(nn.Module):
    """Mirrors C++ IQNNetworkImpl."""
    def __init__(self, input_size, hidden_size, output_size, embedding_dim=IQN_EMB_DIM):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.state_fc = nn.Linear(input_size, hidden_size)
        self.cos_embedding = nn.Linear(embedding_dim, hidden_size)
        self.output_fc = nn.Linear(hidden_size, output_size)

    def forward(self, state, taus):
        # state: (B, IN), taus: (B, K) → (B, K, OUT)
        sf = torch.relu(self.state_fc(state))                        # (B, H)
        i_pi = torch.arange(self.embedding_dim, dtype=torch.float32) * math.pi
        cos_input = taus.unsqueeze(-1) * i_pi.unsqueeze(0).unsqueeze(0)  # (B,K,E)
        tf = torch.relu(self.cos_embedding(torch.cos(cos_input)))    # (B,K,H)
        combined = sf.unsqueeze(1) * tf                              # (B,K,H)
        return self.output_fc(combined)                              # (B,K,OUT)


# ── Loaders ──────────────────────────────────────────────────────────────

def _load_jit_sd(path):
    jm = torch.jit.load(path, map_location='cpu')
    return jm.state_dict()


def load_nfsp_q(path):
    """Load a scalar Q-net saved via C++ torch::save(q_net_, ...). Returns (module, arch)."""
    sd = _load_jit_sd(path)
    fc_idx = sorted(int(k.split('.')[0][2:]) for k in sd if k.startswith('fc') and k.endswith('.weight'))
    arch = [sd[f'fc{i}.weight'].shape[0] for i in fc_idx]
    m = VarMLP(INFO_SIZE, arch, NUM_ACTIONS)
    new_sd = {}
    for k, v in sd.items():
        if k.startswith('fc'):
            idx = k.split('.')[0][2:]
            rest = k.split('.', 1)[1]
            new_sd[f'layers.{idx}.{rest}'] = v
        else:
            new_sd[k] = v
    m.load_state_dict(new_sd)
    m.eval()
    return m, arch


def load_iqn_net(path):
    """Load an IQN net saved via C++ torch::save(iqn_net_, ...). Returns (module, hidden)."""
    sd = _load_jit_sd(path)
    # state_fc.weight shape = (hidden, INFO_SIZE)
    hidden = sd['state_fc.weight'].shape[0]
    m = IQNNetwork(INFO_SIZE, hidden, NUM_ACTIONS)
    m.load_state_dict(sd)
    m.eval()
    return m, hidden


# ── Action selection ─────────────────────────────────────────────────────

class NFSPPlayer:
    """Plays argmax over Q."""
    def __init__(self, q_net):
        self.q = q_net

    @torch.no_grad()
    def action(self, info_state, legal):
        x = torch.FloatTensor(info_state).unsqueeze(0)
        q = self.q(x).squeeze(0).numpy()
        best, best_q = legal[0], -1e18
        for a in legal:
            if q[a] > best_q:
                best_q = q[a]; best = a
        return best


class IQNPlayer:
    """Plays argmax over mean-Q across K tau samples, with optional tau distortion."""
    def __init__(self, iqn_net, distortion='none', p1=1.0, p2=1.0):
        self.net = iqn_net
        self.dist = distortion   # 'none' | 'cvar' | 'seeking'
        self.p1 = p1
        self.p2 = p2

    def _taus(self):
        taus = torch.rand(1, K_EVAL)
        if self.dist == 'cvar':
            taus = taus * self.p1
        elif self.dist == 'seeking':
            taus = self.p1 + taus * (self.p2 - self.p1)
        return taus

    @torch.no_grad()
    def action(self, info_state, legal):
        x = torch.FloatTensor(info_state).unsqueeze(0)
        taus = self._taus()
        q = self.net(x, taus).mean(dim=1).squeeze(0).numpy()   # mean over K
        best, best_q = legal[0], -1e18
        for a in legal:
            if q[a] > best_q:
                best_q = q[a]; best = a
        return best


# ── Game loop ────────────────────────────────────────────────────────────

def play_one(game, player_p0, player_p1):
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            state.apply_action(int(np.random.choice(actions, p=probs)))
        else:
            p = state.current_player()
            info = state.information_state_tensor(p)
            legal = state.legal_actions(p)
            act = (player_p0 if p == 0 else player_p1).action(info, legal)
            state.apply_action(act)
    return state.returns()[0]


def head_to_head(a, b, game, n_games, seed):
    # A as P0
    np.random.seed(seed)
    torch.manual_seed(seed)
    half = n_games // 2
    v1 = [play_one(game, a, b) for _ in range(half)]
    # A as P1
    np.random.seed(seed + 10000)
    torch.manual_seed(seed + 10000)
    v2 = [-play_one(game, b, a) for _ in range(half)]
    vals = np.array(v1 + v2)
    return vals.mean(), vals.std() / np.sqrt(len(vals))


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=2000, help='games per matchup (half P0, half P1)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--only', type=str, default='',
                        help='comma-sep subset of agent names to keep (e.g. '
                             '"baseline_40M,iqn_neutral_35M,iqn_averse_20M,iqn_meanvar_20M")')
    args = parser.parse_args()

    torch.set_num_threads(1)

    # (name, player_builder) — each builder returns (player_p0, player_p1)
    ROOT = os.path.expanduser('~/cos435/cpp-implementation')
    SHARED = '/mnt/data/cos435'

    def build_nfsp(dir_, tag):
        q0, arch = load_nfsp_q(f'{dir_}/p0_q.pt')
        q1, _ = load_nfsp_q(f'{dir_}/p1_q.pt')
        print(f'  loaded {tag} (NFSP Q, arch={arch})')
        return NFSPPlayer(q0), NFSPPlayer(q1)

    def build_iqn(path_p0, path_p1, tag, distortion='none', p1=1.0, p2=1.0):
        n0, h = load_iqn_net(path_p0)
        n1, _ = load_iqn_net(path_p1)
        print(f'  loaded {tag} (IQN, hidden={h}, dist={distortion})')
        return IQNPlayer(n0, distortion, p1, p2), IQNPlayer(n1, distortion, p1, p2)

    AGENTS = []   # list of (name, (player_p0, player_p1))

    # 1. baseline 40M (NFSP)
    AGENTS.append(('baseline_40M',
                   build_nfsp(f'{ROOT}/final_weights_holdem_small_long', 'baseline_40M')))

    # 2. iqn_long 35M (IQN neutral, BR head staged from ep30M)
    d = f'{SHARED}/weights/final_holdem_iqn_long'
    AGENTS.append(('iqn_neutral_35M',
                   build_iqn(f'{d}/p0_iqn_net.pt', f'{d}/p1_iqn_net.pt',
                             'iqn_neutral_35M', 'none')))

    # 3. iqn_smaller 20M (IQN neutral, small arch)
    AGENTS.append(('iqn_smaller_20M',
                   build_iqn(f'{SHARED}/checkpoints/holdem_iqn_smaller_ep20000000_p0/iqn_net.pt',
                             f'{SHARED}/checkpoints/holdem_iqn_smaller_ep20000000_p1/iqn_net.pt',
                             'iqn_smaller_20M', 'none')))

    # 4. iqn_averse 20M (CVaR α=0.25 → taus * 0.25)
    AGENTS.append(('iqn_averse_20M',
                   build_iqn(f'{SHARED}/checkpoints/holdem_iqn_averse_ep20000000_p0/iqn_net.pt',
                             f'{SHARED}/checkpoints/holdem_iqn_averse_ep20000000_p1/iqn_net.pt',
                             'iqn_averse_20M', 'cvar', 0.25)))

    # 5. iqn_meanvar 20M (mean-variance: no tau distortion at inference)
    AGENTS.append(('iqn_meanvar_20M',
                   build_iqn(f'{SHARED}/checkpoints/holdem_iqn_meanvar_ep20000000_p0/iqn_net.pt',
                             f'{SHARED}/checkpoints/holdem_iqn_meanvar_ep20000000_p1/iqn_net.pt',
                             'iqn_meanvar_20M', 'none')))

    if args.only:
        keep = set(s.strip() for s in args.only.split(','))
        AGENTS = [a for a in AGENTS if a[0] in keep]
        print(f'\nFiltered to: {[n for n,_ in AGENTS]}')

    print()
    game = pyspiel.load_game(GAME_STR)

    names = [n for n, _ in AGENTS]
    n = len(names)
    mat = np.zeros((n, n))
    se = np.zeros((n, n))

    total = n * (n - 1) // 2
    done = 0
    t_all = time.time()
    print(f"BR-head H2H tournament — {args.games} games per matchup (split P0/P1)")
    print("=" * 80)
    for i in range(n):
        for j in range(i + 1, n):
            t0 = time.time()
            (a0, a1) = AGENTS[i][1]
            (b0, b1) = AGENTS[j][1]
            # i as P0 (a0) vs j as P1 (b1), then swap → head_to_head's internal logic
            # We'll treat "a" = AGENTS[i] so a.p0=a0, a.p1=a1; play as defined.
            class Wrap:
                def __init__(self, p0, p1): self.p0=p0; self.p1=p1
            # head_to_head takes "player" objects; but our play_one takes per-seat.
            # So adapt:
            ai_p0, ai_p1 = a0, a1
            aj_p0, aj_p1 = b0, b1
            # i as P0
            np.random.seed(args.seed); torch.manual_seed(args.seed)
            v = [play_one(game, ai_p0, aj_p1) for _ in range(args.games // 2)]
            np.random.seed(args.seed + 10000); torch.manual_seed(args.seed + 10000)
            v += [-play_one(game, aj_p0, ai_p1) for _ in range(args.games // 2)]
            vals = np.array(v)
            mean = vals.mean(); ss = vals.std() / np.sqrt(len(vals))
            mbb = mean / 0.1   # chips → mbb/g (1 chip = 10 mbb; BB=100 chips)
            mbb_se = ss / 0.1
            mat[i, j] = mbb; mat[j, i] = -mbb
            se[i, j] = mbb_se; se[j, i] = mbb_se
            done += 1
            print(f"[{done}/{total}] {names[i]:>17} vs {names[j]:<17}  "
                  f"{names[i]}: {mbb:+7.1f} ± {mbb_se:4.1f} mbb/g   ({time.time()-t0:.1f}s)")

    print()
    print("=" * 80)
    print("BR-head win matrix (row's mbb/g winnings vs column):")
    print("=" * 80)
    header = " " * 18 + "  ".join(f"{x:>17}" for x in names)
    print(header)
    for i, row in enumerate(names):
        cells = []
        for j in range(n):
            if i == j: cells.append(f"{'---':>12}      ")
            else:      cells.append(f"{mat[i,j]:+8.1f} ±{se[i,j]:4.1f}")
        print(f"{row:>17}  {'  '.join(cells)}")

    print()
    print("Per-agent average (mean winnings vs all others):")
    avgs = []
    for i, name in enumerate(names):
        others = [j for j in range(n) if j != i]
        a = mat[i, others].mean()
        avgs.append((name, a))
        print(f"  {name:>17}:  {a:+8.1f} mbb/g avg")

    print()
    print("Ranking (best on top):")
    for rank, (name, v) in enumerate(sorted(avgs, key=lambda x: -x[1]), 1):
        print(f"  {rank}. {name:>17}  {v:+8.1f} mbb/g")

    print(f"\n(Total wall time: {time.time()-t_all:.1f}s, games={args.games}, seed={args.seed})")


if __name__ == '__main__':
    main()
