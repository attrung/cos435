# Results — for paper figures and analysis

All artifacts here are committed to git so figures and tables can be regenerated without re-training.

## Files

### `h2h/` — head-to-head tournament data

| File | What |
|---|---|
| `h2h_7way.csv` | The headline result. 21 pairs × 5 checkpoints = 105 rows. Columns: `episode, a_name, b_name, a_avg_chips, a_se_chips, a_mbb, a_mbb_se, games, seconds`. Matrix is anti-symmetric (`a_mbb[i,j] = −a_mbb[j,i]`); only the upper triangle is stored. |
| `h2h_3way.csv` | Earlier 3-way tournament (baseline / iqn_neutral / mv05_legacy). Used in REPORT.md trend plots. |
| `holdem_64x64_<agent>_seed42_h2h.csv` | In-training H2H vs random opponent (one row per `eval_freq=10M` mark). Useful for sanity but not the headline number. |
| `watcher_logs/h2h_3way.log`, `watcher_logs/h2h_7way.log` | Full stdout of the H2H watcher scripts; contains every `play_one_game` summary line and per-pair timing. |

### `training_logs/` — per-run metrics

| File | What |
|---|---|
| `<agent>_seed42.jsonl` | One JSON object per `log_freq=50K` episodes. Keys: `episode, avg_reward, h2h_winrate, br_loss, avg_pol_loss, eps_per_sec, elapsed_min, frac_action_changed, timestamp, experiment`. The `frac_action_changed` field (added for the mean-variance runs) is the fraction of training-batch rows where the std penalty changed the next-action argmax — 0 for non-IQN agents. |
| `raw_stdout/<agent>.log` | Full console output, including startup banner with hyperparameters. Useful for traceability. |

### `legacy_experiments/`

Exploitability CSVs from earlier (pre-64x64) Hold'em runs that are referenced in `cpp-implementation/NOTES_FOR_PAPER.md`.

## Regenerating figures

The training JSONLs are line-delimited JSON; each is small (~250 KB). Quick example to plot `frac_action_changed` over time for the 4 mean-variance runs:

```python
import json, glob
import matplotlib.pyplot as plt

for path in sorted(glob.glob("results/training_logs/holdem_64x64_mv_train_std_*_seed42.jsonl")):
    eps, fc = [], []
    for line in open(path):
        d = json.loads(line)
        if "frac_action_changed" in d and d.get("br_loss", 0) > 0:
            eps.append(d["episode"]); fc.append(d["frac_action_changed"])
    plt.plot(eps, fc, label=path.split("/")[-1].split("_seed")[0].replace("holdem_64x64_", ""))

plt.xlabel("Training episodes"); plt.ylabel("frac argmax flipped by penalty")
plt.legend(); plt.savefig("figures/frac_changed.png")
```

The H2H matrix CSV plots cleanly as a heatmap or as per-pair trend lines (as in REPORT.md):

```python
import pandas as pd
df = pd.read_csv("results/h2h/h2h_7way.csv")
final = df[df.episode == 50_000_000]
# pivot into a 7×7 matrix; flip lower triangle by sign
```

## Provenance — which script produced what

| Result file | Producer |
|---|---|
| `h2h_7way.csv` | `cpp-implementation/scripts/eval/h2h_7way_watcher.sh` |
| `h2h_3way.csv` | `cpp-implementation/scripts/eval/h2h_3way_watcher.sh` |
| `<agent>_seed42.jsonl`, `raw_stdout/<agent>.log` | `cpp-implementation/scripts/train/run_holdem_64x64_<agent>.sh` |
| `legacy_experiments/baseline_*_exploitability.csv` | `cpp-implementation/scripts/legacy/run_overnight_baseline.sh` (and resume variant) |
