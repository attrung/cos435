#!/bin/bash
# H2H wrapper — baseline vs meanvar final (50M) checkpoints.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs
GAMES=${GAMES:-10000}

python3 h2h_64x64_baseline_vs_meanvar.py \
    --a-dir final_weights_holdem_64x64_baseline --a-name baseline \
    --b-dir final_weights_holdem_64x64_meanvar  --b-name meanvar \
    --games "${GAMES}" 2>&1 | tee logs/h2h_64x64.log
