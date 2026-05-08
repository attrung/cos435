"""Generate the three figures driven by our own results.

  1. holdem_ranking_trend.png   -- avg mbb/g vs the 6 others, per agent, x five checkpoints.
  2. holdem_h2h_heatmap_50M.png -- 7x7 anti-symmetric matrix at the 50M checkpoint.
  3. frac_action_changed.png    -- fraction of training-batch rows where the std penalty
                                   flipped the next-action argmax, smoothed, vs episodes.
"""
import json
import os
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(ROOT, "..", "results", "h2h", "h2h_7way.csv")
JSONL_DIR = os.path.join(ROOT, "..", "results", "training_logs")
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})

AGENTS = ["baseline", "neutral",
          "mv_t_std_05", "mv_t_std_10", "mv_t_std_20", "mv_t_std_full"]
PRETTY = {
    "baseline":      "baseline",
    "neutral":       "iqn-neutral",
    "mv_t_std_05":   "iqn-std-0.5",
    "mv_t_std_10":   "iqn-std-1.0",
    "mv_t_std_20":   "iqn-std-2.0",
    "mv_t_std_full": "iqn-full-1.0",
}
COLORS = {
    "baseline":      "#1f77b4",
    "neutral":       "#2ca02c",
    "mv_t_std_05":   "#ff7f0e",
    "mv_t_std_10":   "#d62728",
    "mv_t_std_20":   "#8c564b",
    "mv_t_std_full": "#7f7f7f",
}


# =========================================================================
# Figure 1 / 2: head-to-head trend + heatmap (driven by h2h_7way.csv)
# =========================================================================
df = pd.read_csv(CSV)
# Drop the legacy no-op control: it is not part of the paper's narrative.
df = df[(df.a_name != "mv05_legacy") & (df.b_name != "mv05_legacy")].copy()
EPISODES = sorted(df.episode.unique())


def avg_mbb(ep):
    sub = df[df.episode == ep]
    out = {a: [] for a in AGENTS}
    for _, row in sub.iterrows():
        out[row.a_name].append(row.a_mbb)
        out[row.b_name].append(-row.a_mbb)
    return {a: float(np.mean(v)) for a, v in out.items()}


fig, ax = plt.subplots(figsize=(6.6, 3.8))
trend = {a: [avg_mbb(ep)[a] for ep in EPISODES] for a in AGENTS}
order = sorted(AGENTS, key=lambda a: -trend[a][-1])
x = [e / 1e6 for e in EPISODES]
for a in order:
    ax.plot(x, trend[a], marker="o", color=COLORS[a],
            label=f"{PRETTY[a]} ({trend[a][-1]:+.0f})", linewidth=2, markersize=5)
ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
ax.set_xlabel("Training episodes (millions)")
ax.set_ylabel("Avg winnings vs 5 others (mbb/g)")
ax.set_title("Head-to-head ranking across the 50M sweep ([64,64] MLP)")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
          fontsize=8, frameon=False)
ax.grid(alpha=0.3)
ax.set_xticks(x)
plt.tight_layout()
out1 = os.path.join(OUT, "holdem_ranking_trend.png")
plt.savefig(out1, dpi=180, bbox_inches="tight")
plt.close(fig)
print("wrote", out1)


ep = 50_000_000
sub = df[df.episode == ep]
M = np.zeros((len(AGENTS), len(AGENTS)))
for _, row in sub.iterrows():
    i = AGENTS.index(row.a_name)
    j = AGENTS.index(row.b_name)
    M[i, j] = row.a_mbb
    M[j, i] = -row.a_mbb
fig, ax = plt.subplots(figsize=(6.0, 4.6))
vmax = float(np.abs(M).max())
im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
short = ["baseline", "iqn-neutral",
         "iqn-std-0.5", "iqn-std-1.0",
         "iqn-std-2.0", "iqn-full-1.0"]
ax.set_xticks(range(len(AGENTS)))
ax.set_yticks(range(len(AGENTS)))
ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(short, fontsize=8)
ax.set_xlabel("Opponent (column)")
ax.set_ylabel("Row agent's winnings, mbb/g")
ax.set_title("H2H matrix at episode 50M (5000 hands/cell)")
for i in range(len(AGENTS)):
    for j in range(len(AGENTS)):
        if i == j:
            ax.text(j, i, "—", ha="center", va="center", fontsize=7)
        else:
            ax.text(j, i, f"{M[i,j]:+.0f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(M[i, j]) > 0.55 * vmax else "black")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mbb/g")
plt.tight_layout()
out2 = os.path.join(OUT, "holdem_h2h_heatmap_50M.png")
plt.savefig(out2, dpi=180, bbox_inches="tight")
plt.close(fig)
print("wrote", out2)


# =========================================================================
# Figure 3: frac_action_changed mechanism plot
# =========================================================================
def load_jsonl(path):
    rows = []
    for line in open(path):
        d = json.loads(line)
        rows.append(d)
    return pd.DataFrame(rows)


frac_files = {
    "baseline":      "holdem_64x64_baseline_seed42.jsonl",
    "neutral":       "holdem_64x64_iqn_neutral_seed42.jsonl",
    "mv_t_std_05":   "holdem_64x64_mv_train_std_05_seed42.jsonl",
    "mv_t_std_10":   "holdem_64x64_mv_train_std_10_seed42.jsonl",
    "mv_t_std_20":   "holdem_64x64_mv_train_std_20_seed42.jsonl",
    "mv_t_std_full": "holdem_64x64_mv_train_std_full_seed42.jsonl",
}

fig, ax = plt.subplots(figsize=(6.6, 3.4))
order = ["mv_t_std_full", "mv_t_std_20", "mv_t_std_10", "mv_t_std_05",
         "neutral", "baseline"]
for a in order:
    p = os.path.join(JSONL_DIR, frac_files[a])
    if not os.path.exists(p):
        continue
    d = load_jsonl(p)
    if "frac_action_changed" not in d.columns:
        continue
    d = d[d["br_loss"] > 0]  # post-warmup
    eps = d["episode"].values / 1e6
    f = d["frac_action_changed"].values
    if len(f) > 50:
        # rolling mean for legibility
        win = 25
        f = pd.Series(f).rolling(win, min_periods=1).mean().values
    final = float(f[-50:].mean()) if len(f) > 50 else float(np.mean(f))
    ax.plot(eps, f, color=COLORS[a],
            label=f"{PRETTY[a]} (final {final:.2f})",
            linewidth=1.8 if a == "mv_t_std_full" else 1.4,
            alpha=0.95)
ax.set_xlabel("Training episodes (millions)")
ax.set_ylabel("frac. argmax flipped by penalty")
ax.set_title(r"Mechanism: fraction of train-batch rows where $\widehat\sigma$"
             r" penalty changes the next-action argmax")
ax.set_ylim(-0.03, 0.85)
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
          fontsize=8, frameon=False)
ax.grid(alpha=0.3)
plt.tight_layout()
out3 = os.path.join(OUT, "frac_action_changed.png")
plt.savefig(out3, dpi=180, bbox_inches="tight")
plt.close(fig)
print("wrote", out3)
