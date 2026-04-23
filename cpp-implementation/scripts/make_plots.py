#!/usr/bin/env python3
"""
Generate plots for both Leduc and Hold'em NFSP/IQN experiments.

Data sources:
  - logs/*.log — parse [Ep N/M] lines for episode/br_loss/avg_loss/h2h trajectories
  - results/logs/*_seed42_exploitability.csv — Leduc exact exploitability over episodes
  - results/logs/*_seed42_h2h.csv — Hold'em H2H-vs-random milestones (may be partial after resume)

Outputs:
  figures/leduc/*.png
  figures/holdem/*.png
"""
import os, re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
LOGS = ROOT / 'logs'
CSVS = ROOT / 'results' / 'logs'
FIGS = ROOT.parent / 'figures'
(FIGS / 'leduc').mkdir(parents=True, exist_ok=True)
(FIGS / 'holdem').mkdir(parents=True, exist_ok=True)

EP_LINE = re.compile(r"\[Ep\s+(\d+)/\s*\d+\]\s+avg_r=\s*([-\d.]+)\s+\|\s+(?:h2h|exploit)=\s*([-\d.]+)\s+\|\s+br=\s*([-\d.]+)\s+\|\s+avg=\s*([-\d.]+)")
H2H_LINE = re.compile(r">>> H2H WINRATE @ ep\s+(\d+):\s+([-\d.]+)")

def parse_log(path):
    """Return dict with np arrays: eps, avg_r, inline (h2h or exploit), br_loss, avg_loss, h2h_ep, h2h_val."""
    eps, ar, inline, br, av = [], [], [], [], []
    h2h_ep, h2h_val = [], []
    with open(path) as f:
        for line in f:
            m = EP_LINE.search(line)
            if m:
                eps.append(int(m.group(1)))
                ar.append(float(m.group(2)))
                inline.append(float(m.group(3)))
                br.append(float(m.group(4)))
                av.append(float(m.group(5)))
                continue
            m = H2H_LINE.search(line)
            if m:
                h2h_ep.append(int(m.group(1)))
                h2h_val.append(float(m.group(2)))
    return {
        'eps': np.array(eps),
        'avg_r': np.array(ar),
        'inline': np.array(inline),   # for Leduc this is exploit; for Hold'em it's h2h
        'br_loss': np.array(br),
        'avg_loss': np.array(av),
        'h2h_ep': np.array(h2h_ep),
        'h2h': np.array(h2h_val),
    }

def parse_expl_csv(path):
    """Leduc exact exploitability trajectory."""
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    return data[:, 0].astype(int), data[:, 1]


def smooth(y, window=50):
    if len(y) < window: return y
    return np.convolve(y, np.ones(window)/window, mode='valid')

def smoothed_x(x, window=50):
    if len(x) < window: return x
    return x[window-1:]

# ─── Leduc ─────────────────────────────────────────────────────
LEDUC_RUNS = [
    ('baseline',    'NFSP baseline', 'C0', '-'),
    ('iqn_neutral', 'IQN neutral',   'C1', '-'),
    ('iqn_mv01',    'IQN MV β=0.1',  'C2', '--'),
    ('iqn_mv05',    'IQN MV β=0.5',  'C2', '-'),
    ('iqn_averse',  'IQN CVaR α=0.25 (averse)', 'C3', '-'),
    ('iqn_seeking', 'IQN CVaR seeking', 'C4', '-'),
]

def leduc_full_trajectory(name):
    """Combine 0-40M data from the log (inline `exploit=`) with 40M-50M data from CSV.

    The logs have inline exploitability for IQN variants from ~ep 250K onwards. The CSV is
    only the resumed-training segment (ep 40M onwards). Baseline log was truncated on
    resume so we only have CSV data for it.

    Returns np arrays (episodes, exploitability_mbb) or (None, None) if no data.
    """
    eps_log, val_log = np.array([]), np.array([])
    log = LOGS / f'{name}.log'
    if log.exists():
        d = parse_log(log)
        if len(d['eps']):
            # Drop sentinel 999 (eval not yet computed)
            mask = d['inline'] < 100
            eps_log = d['eps'][mask]
            val_log = d['inline'][mask]
    eps_csv, val_csv = np.array([]), np.array([])
    csv = CSVS / f'{name}_seed42_exploitability.csv'
    if csv.exists():
        eps_csv, val_csv = parse_expl_csv(csv)
    # Concatenate. If both present, prefer the later CSV values when eps overlap.
    if len(eps_log) == 0 and len(eps_csv) == 0:
        return None, None
    if len(eps_csv) == 0:
        return eps_log, val_log
    if len(eps_log) == 0:
        return eps_csv, val_csv
    # Keep log data < first CSV episode, then append CSV
    cut = eps_csv.min()
    mask = eps_log < cut
    return np.concatenate([eps_log[mask], eps_csv]), np.concatenate([val_log[mask], val_csv])


def plot_leduc():
    # Exploitability (full 0 → 50M trajectory)
    plt.figure(figsize=(9, 5))
    for name, label, color, ls in LEDUC_RUNS:
        ep, val = leduc_full_trajectory(name)
        if ep is None: print(f'no data for {name}'); continue
        # Light downsample to avoid overplotting
        if len(ep) > 1000:
            step = max(1, len(ep) // 1000)
            ep, val = ep[::step], val[::step]
        plt.semilogy(ep/1e6, val, color=color, ls=ls, label=label, lw=1.2, alpha=0.85)
    plt.xlabel('Episodes (millions)')
    plt.ylabel('Exploitability (mbb/g, log scale)')
    plt.title("Leduc Hold'em — Exact Exploitability (full trajectory)")
    plt.legend(fontsize=8, loc='best')
    plt.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(FIGS / 'leduc' / 'exploitability.png', dpi=140)
    plt.close()
    print(f'wrote {FIGS}/leduc/exploitability.png')

    # Linear-scale (converged-zoom)
    plt.figure(figsize=(9, 5))
    for name, label, color, ls in LEDUC_RUNS:
        ep, val = leduc_full_trajectory(name)
        if ep is None: continue
        if len(ep) > 1000:
            step = max(1, len(ep) // 1000)
            ep, val = ep[::step], val[::step]
        plt.plot(ep/1e6, val, color=color, ls=ls, label=label, lw=1.2, alpha=0.85)
    plt.xlabel('Episodes (millions)')
    plt.ylabel('Exploitability (mbb/g)')
    plt.title("Leduc Hold'em — Exact Exploitability (linear, full trajectory)")
    plt.legend(fontsize=8, loc='best')
    plt.grid(alpha=0.3)
    plt.ylim(0, 2.5)
    plt.tight_layout()
    plt.savefig(FIGS / 'leduc' / 'exploitability_linear.png', dpi=140)
    plt.close()

    # Loss curves (br_loss) — from training logs
    plt.figure(figsize=(8, 5))
    for name, label, color, ls in LEDUC_RUNS:
        log = LOGS / f'{name}.log'
        if not log.exists():
            print(f'missing {log}'); continue
        d = parse_log(log)
        if len(d['eps']) == 0: continue
        w = min(100, max(1, len(d['br_loss']) // 2))
        if len(d['br_loss']) >= w and w > 1:
            y = smooth(d['br_loss'], w); x = d['eps'][w-1:]
        else:
            y = d['br_loss']; x = d['eps']
        if len(x) != len(y): continue
        plt.plot(x/1e6, y, color=color, ls=ls, label=label, lw=1.2, alpha=0.8)
    plt.xlabel('Episodes (millions)')
    plt.ylabel('BR loss (Q-learning / quantile Huber)')
    plt.title("Leduc Hold'em — Best-Response Loss Trajectory")
    plt.legend(fontsize=8, loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / 'leduc' / 'br_loss.png', dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, label, color, ls in LEDUC_RUNS:
        log = LOGS / f'{name}.log'
        if not log.exists(): continue
        d = parse_log(log)
        if len(d['eps']) == 0: continue
        w = min(100, max(1, len(d['avg_loss']) // 2))
        if len(d['avg_loss']) >= w and w > 1:
            y = smooth(d['avg_loss'], w); x = d['eps'][w-1:]
        else:
            y = d['avg_loss']; x = d['eps']
        if len(x) != len(y): continue
        plt.plot(x/1e6, y, color=color, ls=ls, label=label, lw=1.2, alpha=0.8)
    plt.xlabel('Episodes (millions)')
    plt.ylabel('Avg-policy loss (cross-entropy)')
    plt.title("Leduc Hold'em — Average-Policy Loss Trajectory")
    plt.legend(fontsize=8, loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / 'leduc' / 'avg_loss.png', dpi=140)
    plt.close()

    # Final-value bar chart
    names, finals = [], []
    for name, label, color, ls in LEDUC_RUNS:
        csv = CSVS / f'{name}_seed42_exploitability.csv'
        if not csv.exists(): continue
        _, val = parse_expl_csv(csv)
        names.append(label); finals.append(val[-1])
    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(len(names)), finals, color=[c for _,_,c,_ in LEDUC_RUNS[:len(names)]])
    plt.xticks(range(len(names)), names, rotation=20, ha='right', fontsize=8)
    plt.ylabel('Final exploitability (mbb/g, lower is better)')
    plt.title("Leduc Hold'em — Final Exploitability by Variant")
    for b, v in zip(bars, finals):
        plt.text(b.get_x()+b.get_width()/2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FIGS / 'leduc' / 'final_exploitability_bar.png', dpi=140)
    plt.close()

# ─── Hold'em ────────────────────────────────────────────────────
HOLDEM_RUNS = [
    ('holdem_small_long',   'NFSP baseline',     'C0', '-'),
    ('holdem_iqn_long',     'IQN neutral',       'C1', '-'),
    ('holdem_iqn_smaller',  'IQN neutral (small net)', 'C1', ':'),
    ('holdem_iqn_averse',   'IQN CVaR α=0.25 (averse)', 'C3', '-'),
    ('holdem_iqn_meanvar',  'IQN MV β=0.5',      'C2', '-'),
    ('holdem_iqn_seeking',  'IQN CVaR seeking',  'C4', '-'),
]

# Final LBR numbers at rollouts=100 (re-run with higher accuracy).
# Prior rollouts=15 numbers are preserved in REPORT.md for reference.
HOLDEM_LBR = {
    'holdem_small_long':  (1819, 109),  # 40M, r=100
    'holdem_iqn_long':    (2091, 123),  # 20M (archive), r=100
    # 'holdem_iqn_smaller': (pending r=100),
    'holdem_iqn_averse':  (605,  44),   # r=100; CAVEAT: P1 negative (still), see report
    'holdem_iqn_meanvar': (2258, 121),  # r=100
    # holdem_iqn_seeking: training in progress
}

def plot_holdem():
    # H2H vs random over training
    plt.figure(figsize=(9, 5))
    for name, label, color, ls in HOLDEM_RUNS:
        log = LOGS / f'{name}.log'
        if not log.exists(): continue
        d = parse_log(log)
        if len(d['h2h_ep']) == 0: continue
        plt.plot(d['h2h_ep']/1e6, d['h2h'], color=color, ls=ls, label=label, lw=1.3, alpha=0.85)
    plt.xlabel('Episodes (millions)')
    plt.ylabel('H2H win rate vs uniform random (mbb/g)')
    plt.title("Hold'em — Head-to-Head vs Random Policy")
    plt.legend(fontsize=8, loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / 'holdem' / 'h2h.png', dpi=140)
    plt.close()
    print(f'wrote {FIGS}/holdem/h2h.png')

    # BR loss
    plt.figure(figsize=(9, 5))
    for name, label, color, ls in HOLDEM_RUNS:
        log = LOGS / f'{name}.log'
        if not log.exists(): continue
        d = parse_log(log)
        if len(d['eps']) == 0: continue
        w = min(50, max(1, len(d['br_loss']) // 2))
        if len(d['br_loss']) >= w and w > 1:
            y = smooth(d['br_loss'], w); x = d['eps'][w-1:]
        else:
            y = d['br_loss']; x = d['eps']
        if len(x) != len(y): continue
        plt.plot(x/1e6, y, color=color, ls=ls, label=label, lw=1.2, alpha=0.85)
    plt.xlabel('Episodes (millions)')
    plt.ylabel('BR loss (smoothed)')
    plt.title("Hold'em — Best-Response Loss Trajectory")
    plt.legend(fontsize=8, loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / 'holdem' / 'br_loss.png', dpi=140)
    plt.close()

    # Avg policy loss
    plt.figure(figsize=(9, 5))
    for name, label, color, ls in HOLDEM_RUNS:
        log = LOGS / f'{name}.log'
        if not log.exists(): continue
        d = parse_log(log)
        if len(d['eps']) == 0: continue
        w = min(50, max(1, len(d['avg_loss']) // 2))
        if len(d['avg_loss']) >= w and w > 1:
            y = smooth(d['avg_loss'], w); x = d['eps'][w-1:]
        else:
            y = d['avg_loss']; x = d['eps']
        if len(x) != len(y): continue
        plt.plot(x/1e6, y, color=color, ls=ls, label=label, lw=1.2, alpha=0.85)
    plt.xlabel('Episodes (millions)')
    plt.ylabel('Avg-policy cross-entropy loss (smoothed)')
    plt.title("Hold'em — Average-Policy Loss Trajectory")
    plt.legend(fontsize=8, loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / 'holdem' / 'avg_loss.png', dpi=140)
    plt.close()

    # Final LBR bar chart
    names, means, ses = [], [], []
    colors = []
    for name, label, color, ls in HOLDEM_RUNS:
        if name not in HOLDEM_LBR: continue
        m, se = HOLDEM_LBR[name]
        names.append(label); means.append(m); ses.append(se)
        colors.append(color)
    plt.figure(figsize=(9, 5.5))
    bars = plt.bar(range(len(names)), means, yerr=ses, capsize=4, color=colors, alpha=0.85)
    plt.xticks(range(len(names)), names, rotation=20, ha='right', fontsize=8)
    plt.ylabel('LBR exploitability (mbb/g, lower is better)')
    plt.title("Hold'em — LBR Exploitability by Variant (rollouts=100, ± SE)")
    # Random reference line
    plt.axhline(2800, color='gray', ls='--', lw=1)
    plt.text(len(names)-0.5, 2810, 'uniform random ≈ 2800', ha='right', va='bottom', fontsize=8, color='gray')
    for b, v, s in zip(bars, means, ses):
        plt.text(b.get_x()+b.get_width()/2, v+s+30, f'{v:.0f}', ha='center', va='bottom', fontsize=9)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FIGS / 'holdem' / 'final_lbr_bar.png', dpi=140)
    plt.close()
    print(f'wrote {FIGS}/holdem/final_lbr_bar.png')

if __name__ == '__main__':
    plot_leduc()
    plot_holdem()
    print('All plots written.')
