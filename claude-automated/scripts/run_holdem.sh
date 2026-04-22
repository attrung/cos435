#!/bin/bash
# Run 4 Limit Hold'em NFSP experiments in parallel
# Each uses 2 workers (8 cores / 4 experiments = 2 each)
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=""
mkdir -p results/logs results/checkpoints

echo "============================================================"
echo "  Limit Hold'em NFSP — 4 experiments (2 workers each)"
echo "  Started: $(date)"
echo "============================================================"
echo ""

PIDS=()
NAMES=()
LOGS=()

launch() {
    local config="$1"; local name="$2"; local label="$3"
    local logfile="results/logs/${name}.log"
    echo "  $label — launching (log: $logfile)"
    python3 -u src/train.py --config "$config" --num-workers 2 \
        > "$logfile" 2>&1 &
    PIDS+=($!)
    NAMES+=("$name")
    LOGS+=("$logfile")
}

launch configs/holdem_baseline.yaml      holdem_baseline      "[1/4] BASELINE"
launch configs/holdem_iqn_neutral.yaml   holdem_iqn_neutral   "[2/4] IQN-NEUTRAL"
launch configs/holdem_iqn_mv01.yaml      holdem_iqn_mv01      "[3/4] IQN-MV-0.1"
launch configs/holdem_iqn_mv05.yaml      holdem_iqn_mv05      "[4/4] IQN-MV-0.5"

echo ""
echo "  4 experiments running. PIDs: ${PIDS[*]}"
echo "  Press Ctrl+C to stop all."
echo ""

cleanup() {
    echo "  Stopping all..."
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null; done
    sleep 3
    for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null; done
    wait 2>/dev/null
    echo "  Done."
}
trap cleanup EXIT INT TERM

while true; do
    ALL_DONE=true
    for i in "${!PIDS[@]}"; do
        kill -0 "${PIDS[$i]}" 2>/dev/null && ALL_DONE=false
    done

    echo "--- $(date +%H:%M:%S) ---"
    for i in "${!NAMES[@]}"; do
        last=$(grep 'EXPLOITABILITY\|eps/s' "${LOGS[$i]}" 2>/dev/null | tail -1)
        [ -n "$last" ] && echo "  ${NAMES[$i]}: $last"
    done

    $ALL_DONE && break
    sleep 60
done

trap - EXIT

FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "  FAILED: ${NAMES[$i]} (exit $rc)"
        FAILED=$((FAILED+1))
    else
        echo "  DONE: ${NAMES[$i]}"
    fi
done

echo ""
echo "============================================================"
echo "  ALL DONE: $(date)"
echo "============================================================"
