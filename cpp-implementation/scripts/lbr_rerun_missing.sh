#!/bin/bash
# Re-run LBR on iqn_long (20M archive) and iqn_smaller (eval weights) at rollouts=100.
# Waits for prior rerun orchestrator to finish to avoid CPU thrashing.
cd "$(dirname "$0")/.."

ROLLOUTS=100
GAMES=1500
WORKERS=2
OUTDIR=logs/lbr_rerun_r100
mkdir -p "$OUTDIR"

# Wait for prior rerun to exit (parent bash script, if still running)
while pgrep -f "lbr_rerun_highrollouts.sh" > /dev/null; do
    sleep 30
done

echo "=== LBR rerun (missing runs): rollouts=$ROLLOUTS games=$GAMES workers=$WORKERS ==="
echo "started: $(date)"

# iqn_long @ 20M — use the archived weights
echo "--- iqn_long (20M archive) ---"
python3 eval/lbr_holdem_accurate.py --weights /mnt/data/cos435/archive_20M/final_holdem_iqn_long_ep20M \
    --games $GAMES --rollouts $ROLLOUTS --workers $WORKERS \
    > "$OUTDIR/final_holdem_iqn_long_ep20M.log" 2>&1
echo "  $(grep -A1 'Agent ' "$OUTDIR/final_holdem_iqn_long_ep20M.log" | tail -1)"

# iqn_smaller — use eval_weights (final_weights was never written due to shutdown hang)
echo "--- iqn_smaller (eval_weights) ---"
python3 eval/lbr_holdem_accurate.py --weights eval_weights_holdem_iqn_smaller \
    --games $GAMES --rollouts $ROLLOUTS --workers $WORKERS \
    > "$OUTDIR/eval_weights_holdem_iqn_smaller.log" 2>&1
echo "  $(grep -A1 'Agent ' "$OUTDIR/eval_weights_holdem_iqn_smaller.log" | tail -1)"

echo "=== ALL DONE: $(date) ==="
