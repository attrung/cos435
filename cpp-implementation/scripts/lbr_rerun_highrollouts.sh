#!/bin/bash
# Re-run LBR on all finished Hold'em models with more rollouts (rollouts=100 vs prior 15)
# for a tighter lower bound on exploitability.
# Low-CPU settings (workers=2, games=1500) to coexist with ongoing training.
cd "$(dirname "$0")/.."

ROLLOUTS=100
GAMES=1500
WORKERS=2
OUTDIR=logs/lbr_rerun_r100
mkdir -p "$OUTDIR"

RUNS=(
    final_weights_holdem_small_long
    final_weights_holdem_iqn_long
    final_weights_holdem_iqn_smaller
    final_weights_holdem_iqn_averse
    final_weights_holdem_iqn_meanvar
)

echo "=== LBR rerun: rollouts=$ROLLOUTS games=$GAMES workers=$WORKERS ==="
echo "started: $(date)"

for w in "${RUNS[@]}"; do
    name=$(basename "$w")
    echo ""
    echo "--- $name ---"
    echo "start: $(date)"
    python3 eval/lbr_holdem_accurate.py --weights "$w" --games $GAMES --rollouts $ROLLOUTS --workers $WORKERS \
        > "$OUTDIR/${name}.log" 2>&1
    echo "end: $(date)"
    # Concise one-line summary
    grep -A1 "Agent " "$OUTDIR/${name}.log" | tail -1 || echo "    [no result line found]"
done

echo ""
echo "=== ALL DONE: $(date) ==="
