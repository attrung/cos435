# Trained weights — `holdem_64x64_*` agents

Seven NFSP agents at five checkpoints each (10M, 20M, 30M, 40M, 50M episodes), seed 42, architecture `[64, 64]`. Total: 70 files, ~5.4 MB.

## Layout

```
weights/
└── holdem_64x64_<agent>/
    └── ep<10|20|30|40|50>000000/
        ├── p0_avg.pt    # player 0 average-policy network (TorchScript)
        └── p1_avg.pt    # player 1 average-policy network (TorchScript)
```

The seven agent directories:
- `holdem_64x64_baseline` — NFSP with vanilla DQN best-response
- `holdem_64x64_iqn_neutral` — NFSP with risk-neutral IQN
- `holdem_64x64_meanvar` — NFSP-IQN with `--var-penalty 0.5` (this was a no-op due to the bug fixed in our `mv_train_std_*` runs; kept as redundant control)
- `holdem_64x64_mv_train_std_05` — train-target Markowitz, β=0.5 (std)
- `holdem_64x64_mv_train_std_10` — train-target Markowitz, β=1.0 (std)
- `holdem_64x64_mv_train_std_20` — train-target Markowitz, β=2.0 (std)
- `holdem_64x64_mv_train_std_full` — full Markowitz on every per-quantile target, β=1.0

## What's in each `.pt` file

A TorchScript module representing the average-policy MLP. Architecture is auto-detected at load time from the saved weights (`fc0.weight.shape[1]` → input size, `fc<i>.weight.shape[0]` → hidden sizes, `output.weight.shape[0]` → action count).

For these runs:
- input size = 208 (Limit Hold'em info-state dimension via OpenSpiel `universal_poker`)
- hidden sizes = `[64, 64]`
- output size = 3 (fold / call / raise)

## Loading from Python

```python
import sys
sys.path.insert(0, "cpp-implementation/eval")
from model_utils import load_models

p0_model, p1_model = load_models("weights/holdem_64x64_baseline/ep50000000")
# p0_model and p1_model are nn.Module subclasses — call as model(info_state_tensor)
```

## Why only the average policy?

In NFSP, the average-policy network is the policy that converges to a Nash equilibrium under self-play; the best-response head (DQN or IQN) is a transient learning signal. All H2H games and exploitability-vs-LBR evaluations in this paper use the average policy. We did not commit the BR-head weights (`q_net.pt` / `iqn_net.pt`) because they are not used downstream and would only inflate the repo.

## Reproducing the H2H tournament from these weights

```bash
cd cpp-implementation
# Symlink expected paths to weights/ (the tournament watcher reads from snapshots/)
mkdir -p snapshots
for d in ../weights/holdem_64x64_*; do
    ln -sf "$(realpath ${d})" snapshots/$(basename ${d})
done
bash scripts/eval/h2h_7way_watcher.sh
# Output: results/logs/h2h_7way.csv (or copy back to top-level results/h2h/)
```
