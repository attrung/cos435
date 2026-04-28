# Distributional NFSP for Limit Hold'em

COS 435 final project. We extend Heinrich & Silver's NFSP (Neural Fictitious Self-Play) with a distributional best-response head (IQN) and study whether risk-sensitive (mean-variance) action selection in self-play improves head-to-head play against a vanilla DQN-based NFSP baseline.

**TL;DR — negative result.** Across five training checkpoints (10M, 20M, 30M, 40M, 50M episodes) and four mean-variance variants (β = 0.5, 1.0, 2.0 in the train target, plus a "full" Markowitz variant that shifts every quantile by −β·std), the DQN baseline beats every IQN variant by 600–1100 mbb/g in head-to-head, and the mean-variance penalty is monotonically harmful: stronger β → worse against neutral. See [REPORT.md](REPORT.md) and [results/h2h/h2h_7way.csv](results/h2h/h2h_7way.csv).

## Repository layout

```
.
├── README.md                  # this file
├── REPORT.md                  # full writeup
├── NOTES_FOR_PAPER.md         # implementation notes — see cpp-implementation/
├── cpp-implementation/        # the research code (C++ training binary + Python eval)
│   ├── src/                   # C++ headers + train_holdem.cpp
│   ├── eval/                  # Python H2H / LBR / model-loading utilities
│   ├── scripts/
│   │   ├── train/             # 7 paper-run launch scripts (the 64x64 sweep)
│   │   ├── eval/              # H2H watcher, exploitability sweep
│   │   └── legacy/            # earlier experiments referenced in NOTES_FOR_PAPER.md
│   ├── CMakeLists.txt
│   ├── build_holdem.sh        # one-shot g++ build wrapper
│   └── NOTES_FOR_PAPER.md     # implementation notes (Leduc + Hold'em)
├── results/
│   ├── h2h/
│   │   ├── h2h_7way.csv       # 21-pair × 5-checkpoint matrix (the headline result)
│   │   ├── h2h_3way.csv       # earlier 3-way (baseline vs neutral vs mv05_legacy)
│   │   ├── holdem_64x64_*_seed42_h2h.csv   # in-training H2H vs random
│   │   └── watcher_logs/      # raw watcher stdout for traceability
│   ├── training_logs/
│   │   ├── *_seed42.jsonl     # per-50K-episode metrics: br_loss, avg_pol_loss, frac_action_changed, ep/s
│   │   └── raw_stdout/        # full training console output (one .log per run)
│   └── legacy_experiments/    # exploitability CSVs from earlier runs
├── weights/                   # paper-relevant trained weights (avg policies)
│   ├── README.md              # format + how to load
│   └── holdem_64x64_<agent>/ep<10..50>M/p{0,1}_avg.pt
├── figures/                   # architecture diagrams, plots
└── legacy/
    └── python-leduc-impl/     # initial Python NFSP impl (Leduc only) — superseded
```

## Quick reproduce

```bash
# 1. Clone
git clone git@github.com:attrung/cos435.git && cd cos435

# 2. Build OpenSpiel into cpp-implementation/third_party/  (see cpp-implementation/README.md)
# 3. Build the training binary
cd cpp-implementation && bash build_holdem.sh

# 4. Re-run any of the 7 agents (50M episodes, [64,64] arch, seed=42)
bash scripts/train/run_holdem_64x64_baseline.sh
bash scripts/train/run_holdem_64x64_iqn_neutral.sh
bash scripts/train/run_holdem_64x64_meanvar.sh
bash scripts/train/run_holdem_64x64_mv_train_std_05.sh
bash scripts/train/run_holdem_64x64_mv_train_std_10.sh
bash scripts/train/run_holdem_64x64_mv_train_std_20.sh
bash scripts/train/run_holdem_64x64_mv_train_std_full.sh

# 5. After all 7 finish (or use the committed weights/), run the H2H tournament
bash scripts/eval/h2h_7way_watcher.sh
```

Each training run takes ~5–7 hours on an 8-vCPU CPU box (4-way parallel). The committed `weights/` directory lets you skip step 4 and go straight to H2H.

## Agents

| Tag | Algorithm | Hyperparam |
|---|---|---|
| `baseline` | NFSP with vanilla DQN best-response | — |
| `iqn_neutral` | NFSP with risk-neutral IQN best-response | N=8 quantiles, K=32 eval |
| `meanvar` (mv05_legacy) | NFSP-IQN with `--var-penalty 0.5` | known-no-op (see REPORT.md §Bug analysis) |
| `mv_train_std_05` | NFSP-IQN, train-target Markowitz argmax | β=0.5, std-form |
| `mv_train_std_10` | NFSP-IQN, train-target Markowitz argmax | β=1.0, std-form |
| `mv_train_std_20` | NFSP-IQN, train-target Markowitz argmax | β=2.0, std-form |
| `mv_train_std_full` | NFSP-IQN, full Markowitz on every per-quantile target | β=1.0 (Tamar et al. style) |

All trained at architecture `[64, 64]` (2-layer MLP, 64 units each), 50M episodes, seed 42, on `universal_poker(betting=limit, numPlayers=2, numRounds=4, blind=50/100, ...)` via OpenSpiel.

## Final ranking

Average winnings vs all six other agents at the 50M checkpoint (mbb/g; positive = won chips):

| Rank | Agent | Avg mbb/g |
|---|---|---|
| 1 | baseline | **+844** |
| 2 | mv05_legacy | +287 |
| 2 | iqn_neutral | +287 |
| 4 | mv_train_std_05 | +222 |
| 5 | mv_train_std_10 | −139 |
| 6 | mv_train_std_20 | −611 |
| 7 | mv_train_std_full | −889 |

See `results/h2h/h2h_7way.csv` for the full 21-pair × 5-checkpoint matrix.

## Citation / contact

Anthony Trung (`ngtrung2901@gmail.com`). COS 435, Spring 2026.
