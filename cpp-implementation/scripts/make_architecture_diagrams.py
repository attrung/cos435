#!/usr/bin/env python3
"""Generate architecture diagrams for the NFSP+IQN project.

Outputs:
  figures/architecture/nfsp_iqn_overall.png  — how IQN slots into NFSP
  figures/architecture/iqn_network.png       — internals of the IQN quantile network

Plain matplotlib, no graphviz dependency.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTDIR = os.path.expanduser('~/cos435/figures/architecture')
os.makedirs(OUTDIR, exist_ok=True)


def box(ax, xy, wh, text, fc='#e8f0ff', ec='#234', rounding=0.02, fontsize=10, weight='normal'):
    x, y = xy; w, h = wh
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f'round,pad=0.0,rounding_size={rounding}',
                       linewidth=1.2, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, weight=weight, wrap=True)


def arrow(ax, src, dst, color='#234', text=None, style='-|>',
          rad=0.0, lw=1.3, text_offset=(0, 0.0), fontsize=9, text_bg=True):
    a = FancyArrowPatch(src, dst,
                        arrowstyle=style, mutation_scale=14,
                        color=color, linewidth=lw,
                        connectionstyle=f'arc3,rad={rad}')
    ax.add_patch(a)
    if text:
        mx, my = (src[0] + dst[0]) / 2, (src[1] + dst[1]) / 2
        kwargs = dict(fontsize=fontsize, ha='center', va='center', color=color)
        if text_bg:
            kwargs['bbox'] = dict(boxstyle='round,pad=0.2', fc='white', ec='none')
        ax.text(mx + text_offset[0], my + text_offset[1], text, **kwargs)


# ── Diagram 1: NFSP+IQN overall ───────────────────────────────────────────

def make_overall():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10)
    ax.axis('off')

    IQN = '#ffd7b5'
    AVG = '#d4edd4'
    ENV = '#eaeaea'
    BUF = '#fff2b3'
    LEARN = '#d9e8f5'

    ax.text(7, 9.55,
            'NFSP + IQN architecture',
            ha='center', fontsize=16, weight='bold')
    ax.text(7, 9.15,
            'Swap: scalar Q-net (DQN)  →  distributional IQN  as the best-response head',
            ha='center', fontsize=11, style='italic', color='#555')

    # Row 1 — state + η-coin
    box(ax, (0.5, 7.3), (2.4, 0.9), 'Info-state  s\n(208-dim)',
        fc='#f5f5f5', fontsize=11, weight='bold')
    box(ax, (3.7, 7.3), (2.4, 0.9),
        'η-coin\n(anticipatory)', fc='#fff6e1', fontsize=10)

    # Two heads
    box(ax, (7.0, 7.3), (3.0, 0.9),
        'IQN    (BR head)\ndistributional Q(s, τ, a)',
        fc=IQN, fontsize=11, weight='bold')
    box(ax, (10.6, 7.3), (3.0, 0.9),
        'Avg-policy net\nsupervised  π̄(s)',
        fc=AVG, fontsize=11, weight='bold')

    # Decision
    box(ax, (7.1, 5.7), (2.8, 0.7),
        'a = argmax_a   mean_τ  Q(s, τ, a)', fc=IQN, fontsize=10)
    box(ax, (10.6, 5.7), (3.0, 0.7),
        'a ~ softmax  π̄(s)', fc=AVG, fontsize=10)

    # Action → env
    box(ax, (8.3, 4.3), (2.0, 0.8),
        'Action  a', fc='#f5f5f5', fontsize=12, weight='bold')
    box(ax, (8.3, 3.1), (2.0, 0.8),
        'Environment\n(Hold\'em)', fc=ENV, fontsize=10)

    # Buffers
    box(ax, (0.5, 3.1), (3.2, 0.9),
        'DQN replay buffer\n(s, a, r, sʹ, mask)', fc=BUF, fontsize=10)
    box(ax, (4.2, 3.1), (3.2, 0.9),
        'Reservoir buffer\n(s, π̄(s))', fc=BUF, fontsize=10)

    # Learners
    box(ax, (0.5, 1.4), (3.2, 1.1),
        'IQN training\nquantile Huber τ-loss\n(Dabney 2018)',
        fc=LEARN, fontsize=10, weight='bold')
    box(ax, (4.2, 1.4), (3.2, 1.1),
        'Avg-policy training\ncross-entropy on\nmask-filtered logits',
        fc=LEARN, fontsize=10)

    # Training updates
    box(ax, (0.5, 0.1), (3.2, 0.7),
        'updates IQN weights', fc=IQN, fontsize=9)
    box(ax, (4.2, 0.1), (3.2, 0.7),
        'updates Avg-policy weights', fc=AVG, fontsize=9)

    # ── arrows ──
    arrow(ax, (2.9, 7.75), (3.7, 7.75))
    arrow(ax, (6.1, 7.95), (7.0, 7.95), text='w.p. η', text_offset=(0, 0.15))
    arrow(ax, (6.1, 7.55), (10.6, 7.55), text='w.p. 1−η', text_offset=(0, -0.15))
    arrow(ax, (8.5, 7.3), (8.5, 6.4))
    arrow(ax, (12.1, 7.3), (12.1, 6.4))
    arrow(ax, (8.5, 5.7), (9.0, 5.1))
    arrow(ax, (12.1, 5.7), (9.6, 5.1))
    arrow(ax, (9.3, 4.3), (9.3, 3.9))
    # env → buffers (curving left)
    arrow(ax, (8.3, 3.45), (7.4, 3.55), rad=0.1)
    arrow(ax, (8.3, 3.25), (3.7, 3.35), rad=0.15)
    # buffers → learners
    arrow(ax, (2.1, 3.1), (2.1, 2.5))
    arrow(ax, (5.8, 3.1), (5.8, 2.5))
    # learners → updates
    arrow(ax, (2.1, 1.4), (2.1, 0.8))
    arrow(ax, (5.8, 1.4), (5.8, 0.8))
    # feedback loops to heads (bold red/green curves)
    arrow(ax, (2.1, 0.4), (7.1, 7.4), rad=-0.35, color='#b03a2e', lw=1.5)
    arrow(ax, (5.8, 0.4), (10.9, 7.4), rad=-0.3, color='#1e7e34', lw=1.5)

    # Legend
    ax.text(0.5, 4.7,
            'Legend:',
            fontsize=10, weight='bold')
    box(ax, (0.5, 4.3), (0.6, 0.3), '', fc=IQN, fontsize=8)
    ax.text(1.2, 4.45, 'IQN (BR head) — replaces DQN', fontsize=9, va='center')
    box(ax, (0.5, 3.9), (0.6, 0.3), '', fc=AVG, fontsize=8)
    ax.text(1.2, 4.05, 'Avg-policy head', fontsize=9, va='center')

    plt.savefig(os.path.join(OUTDIR, 'nfsp_iqn_overall.png'),
                dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('wrote', os.path.join(OUTDIR, 'nfsp_iqn_overall.png'))


# ── Diagram 2: IQN network internals ─────────────────────────────────────

def make_iqn_network():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14); ax.set_ylim(0, 11)
    ax.axis('off')

    STATE = '#e1ecff'
    TAU = '#ffd7b5'
    HID = '#ebe3ff'
    OUT = '#fff2b3'
    RISK = '#ffd0d0'
    FINAL = '#cfe6cf'

    ax.text(7, 10.4,
            'IQN network — BR head internals',
            ha='center', fontsize=16, weight='bold')
    ax.text(7, 10.0,
            'Combines  state features  with  τ-embedding  to produce quantile values Q(s, τ, a)',
            ha='center', fontsize=11, style='italic', color='#555')

    # LEFT stream: state
    box(ax, (0.6, 8.5), (3.0, 0.9),
        'state  s  (208-dim)', fc=STATE, fontsize=12, weight='bold')
    box(ax, (0.6, 7.2), (3.0, 0.9),
        'Linear  208 → hidden', fc=STATE, fontsize=11)
    box(ax, (0.6, 6.2), (3.0, 0.7), 'ReLU', fc=STATE, fontsize=11)
    box(ax, (0.6, 4.9), (3.0, 0.9),
        's_feat   (hidden,)', fc=STATE, fontsize=11, weight='bold')

    # RIGHT stream: tau
    box(ax, (10.4, 8.5), (3.0, 0.9),
        'τ ~ U(0,1)^K\n(K=32 at inference)', fc=TAU, fontsize=11, weight='bold')
    box(ax, (9.7, 7.0), (3.8, 1.1),
        'Optional τ distortion\nCVaR α:   τ ← α·τ\nCVaR seeking:   τ ← (1−α)+α·τ',
        fc=RISK, fontsize=10)
    box(ax, (10.4, 5.8), (3.0, 0.8),
        'cos(i·π·τ)    i = 0..63', fc=TAU, fontsize=11)
    box(ax, (10.4, 4.8), (3.0, 0.7),
        'Linear  64 → hidden', fc=TAU, fontsize=11)
    box(ax, (10.4, 3.9), (3.0, 0.6), 'ReLU', fc=TAU, fontsize=11)
    box(ax, (10.4, 2.8), (3.0, 0.9),
        'τ_feat   (K, hidden)', fc=TAU, fontsize=11, weight='bold')

    # MERGE
    box(ax, (5.0, 4.9), (4.0, 0.9),
        's_feat   ⊙   τ_feat\n(element-wise, broadcast over K)',
        fc=HID, fontsize=11, weight='bold')
    box(ax, (5.0, 3.6), (4.0, 0.8),
        'combined   (K, hidden)', fc=HID, fontsize=11)
    box(ax, (5.0, 2.3), (4.0, 0.9),
        'Linear   hidden → 3\n(num_actions)', fc=OUT, fontsize=11)
    box(ax, (5.0, 1.0), (4.0, 0.9),
        'Q(s, τ, a)   quantile values   (K × 3)',
        fc=OUT, fontsize=11, weight='bold')

    # Final: decision
    box(ax, (4.5, -0.4), (5.0, 0.9),
        'BR action  =  argmax_a   (1/K) · Σ_τ  Q(s, τ, a)',
        fc=FINAL, fontsize=12, weight='bold')

    # ── arrows ──
    # State stream
    arrow(ax, (2.1, 8.5), (2.1, 8.1))
    arrow(ax, (2.1, 7.2), (2.1, 6.9))
    arrow(ax, (2.1, 6.2), (2.1, 5.8))
    arrow(ax, (2.1, 4.9), (2.1, 4.6))
    # Tau stream
    arrow(ax, (11.9, 8.5), (11.9, 8.1))
    arrow(ax, (11.9, 7.0), (11.9, 6.6))
    arrow(ax, (11.9, 5.8), (11.9, 5.5))
    arrow(ax, (11.9, 4.8), (11.9, 4.5))
    arrow(ax, (11.9, 3.9), (11.9, 3.7))
    arrow(ax, (11.9, 2.8), (11.9, 2.5))

    # Streams → merge
    arrow(ax, (3.6, 5.35), (5.0, 5.35))
    arrow(ax, (10.4, 3.25), (9.0, 4.9))

    # merge → Q
    arrow(ax, (7.0, 4.9), (7.0, 4.4))
    arrow(ax, (7.0, 3.6), (7.0, 3.2))
    arrow(ax, (7.0, 2.3), (7.0, 1.9))

    # Q → decision
    arrow(ax, (7.0, 1.0), (7.0, 0.5))

    # annotation for batch
    ax.text(2.1, 5.4,
            '(broadcast over\nK quantiles)',
            ha='center', fontsize=8, color='#555', style='italic')

    plt.savefig(os.path.join(OUTDIR, 'iqn_network.png'),
                dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('wrote', os.path.join(OUTDIR, 'iqn_network.png'))


if __name__ == '__main__':
    make_overall()
    make_iqn_network()
