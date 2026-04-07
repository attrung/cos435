#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=""
cd "$(dirname "$0")/.."

mkdir -p results/logs results/checkpoints results/figures

echo "============================================================"
echo "  NFSP Leduc Poker — Full Experiment Suite (all parallel)"
echo "  Started: $(date)"
echo "============================================================"
echo ""

# --- Helper: check if experiment truly completed (checkpoint has enough episodes) ---
is_truly_complete() {
    local ckpt="$1"
    local required_ep="$2"
    [ -f "$ckpt" ] && python3 -c "
import torch, sys
ckpt = torch.load('$ckpt', map_location='cpu', weights_only=False)
sys.exit(0 if ckpt.get('episode', 0) >= $required_ep else 1)
" 2>/dev/null
}

# --- Helper: clean up all artifacts for an experiment ---
clean_experiment() {
    local name="$1"
    echo "  Cleaning stale results for ${name}..."
    rm -f "results/logs/${name}_seed42.jsonl"
    rm -f "results/logs/${name}_seed42_exploitability.npy"
    rm -f "results/logs/${name}.log"
    rm -f "results/checkpoints/${name}_latest_weights.pt"
    rm -f results/checkpoints/${name}_seed42_p*_final.pt
    # Keep periodic checkpoints as they may be useful for analysis,
    # but remove them too if we want a truly clean slate
    rm -f results/checkpoints/${name}_seed42_p*_ep*.pt
}

# --- Helper: run experiment, cleaning first if incomplete ---
run_if_needed() {
    local config="$1"
    local name="$2"
    local label="$3"
    local required_episodes="$4"
    local final_ckpt="results/checkpoints/${name}_seed42_p0_final.pt"

    if is_truly_complete "$final_ckpt" "$required_episodes"; then
        echo "=== $label — COMPLETE (verified), skipping ==="
        # Regenerate npy from jsonl if npy is missing/empty
        python3 -c "
import numpy as np, json, os
npy_path = 'results/logs/${name}_seed42_exploitability.npy'
jsonl_path = 'results/logs/${name}_seed42.jsonl'
try:
    data = np.load(npy_path)
    if data.size > 0:
        print(f'  npy OK: {len(data)} points')
        exit()
except: pass
# Regenerate from jsonl
if os.path.exists(jsonl_path):
    points = []
    with open(jsonl_path) as f:
        for line in f:
            e = json.loads(line.strip())
            if 'exploitability_eval' in e:
                points.append([e['episode'], e['exploitability_eval']])
    if points:
        np.save(npy_path, np.array(points))
        print(f'  Regenerated npy from jsonl: {len(points)} points')
    else:
        print('  WARNING: no exploitability data in jsonl')
else:
    print('  WARNING: no jsonl file found')
"
        echo ""
    else
        echo "=== $label ==="
        clean_experiment "$name"
        python3 src/train.py --config "$config" --parallel 2>&1 | tee "results/logs/${name}.log"
        echo ""
    fi
}

run_if_needed configs/nfsp_baseline.yaml       nfsp_baseline      "[1/5] BASELINE NFSP-DQN (10M episodes, parallel)"            10000000
run_if_needed configs/nfsp_iqn_neutral.yaml    nfsp_iqn_neutral   "[2/5] NFSP-IQN RISK-NEUTRAL (10M episodes, parallel)"        10000000
run_if_needed configs/nfsp_iqn_mean_var.yaml   nfsp_iqn_mean_var  "[3/5] NFSP-IQN MEAN-VARIANCE A=0.5 (10M episodes, parallel)" 10000000
run_if_needed configs/nfsp_iqn_averse.yaml     nfsp_iqn_averse    "[4/5] NFSP-IQN RISK-AVERSE CVaR alpha=0.25 (10M episodes, parallel)" 10000000
run_if_needed configs/nfsp_iqn_seeking.yaml    nfsp_iqn_seeking   "[5/5] NFSP-IQN RISK-SEEKING (10M episodes, parallel)"        10000000

echo "=== GENERATING RESULTS ==="
python3 scripts/generate_results.py
echo ""

echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Finished: $(date)"
echo "  Logs:    results/logs/"
echo "  Figures: results/figures/"
echo "  Summary: results/full_comparison.md"
echo "============================================================"
