#!/bin/bash
# Poll training progress and run LBR exploitability at every 10M episode milestone.
# Usage: bash scripts/poll_lbr.sh
cd "$(dirname "$0")/.."

GAMES=5000
ROLLOUTS=15
WEIGHTS_DIR="eval_weights_holdem_baseline"
LOG="results/logs/holdem_baseline_seed42.jsonl"
LBR_LOG="results/logs/holdem_baseline_lbr.txt"

# Track which milestones we've already evaluated
LAST_MILESTONE=0

echo "=== LBR Polling Started: $(date) ==="
echo "  Checking every 60s for 10M episode milestones"
echo "  LBR: $GAMES games, $ROLLOUTS rollouts"
echo "  Results logged to: $LBR_LOG"
echo ""

while true; do
    # Get latest episode from the JSONL log
    if [ ! -f "$LOG" ]; then
        sleep 60
        continue
    fi

    # Find the highest episode number in the log
    LATEST_EP=$(grep -o '"episode": [0-9]*' "$LOG" 2>/dev/null | grep -o '[0-9]*' | sort -n | tail -1)

    if [ -z "$LATEST_EP" ]; then
        sleep 60
        continue
    fi

    # Calculate current milestone (floor to nearest 10M)
    MILESTONE=$(( (LATEST_EP / 10000000) * 10000000 ))

    if [ "$MILESTONE" -gt "$LAST_MILESTONE" ] && [ "$MILESTONE" -gt 0 ]; then
        echo "[$(date +%H:%M:%S)] Episode $LATEST_EP — running LBR at ${MILESTONE}M milestone..."

        # Run LBR
        RESULT=$(python3 eval/lbr_holdem_accurate.py \
            --weights "$WEIGHTS_DIR" \
            --games "$GAMES" \
            --rollouts "$ROLLOUTS" 2>&1)

        echo "$RESULT"
        echo "[milestone=${MILESTONE}] $(date) | $RESULT" >> "$LBR_LOG"
        echo ""

        LAST_MILESTONE=$MILESTONE
    fi

    # Check if training is done (no train_holdem process running)
    if ! pgrep -f "train_holdem" > /dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] Training process not found. Running final LBR on final weights..."
        RESULT=$(python3 eval/lbr_holdem_accurate.py \
            --weights "final_weights_holdem_baseline" \
            --games "$GAMES" \
            --rollouts "$ROLLOUTS" 2>&1)
        echo "$RESULT"
        echo "[final] $(date) | $RESULT" >> "$LBR_LOG"
        echo ""
        echo "=== LBR Polling Done: $(date) ==="
        break
    fi

    sleep 60
done
