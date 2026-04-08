#!/bin/bash
export CUDA_VISIBLE_DEVICES=""
cd "$(dirname "$0")/.."

mkdir -p results/logs results/checkpoints results/figures

cleanup() {
    echo ""
    echo "  Cleaning up..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null
    done
    wait 2>/dev/null
    echo "  All processes stopped."
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo "  NFSP Leduc Poker — 4 experiments (single-process each)"
echo "  Started: $(date)"
echo "============================================================"
echo ""

PIDS=()
NAMES=()

launch() {
    local config="$1"
    local name="$2"
    local label="$3"

    rm -f "results/logs/${name}_seed42.jsonl" 2>/dev/null
    rm -f "results/logs/${name}_seed42_exploitability.npy" 2>/dev/null
    rm -f "results/logs/${name}.log" 2>/dev/null

    echo "  $label — launching (log: results/logs/${name}.log)"
    python3 src/train.py --config "$config" \
        > "results/logs/${name}.log" 2>&1 &
    PIDS+=($!)
    NAMES+=("$name")
}

launch configs/nfsp_baseline.yaml     nfsp_baseline     "[1/4] BASELINE"
launch configs/nfsp_iqn_neutral.yaml  nfsp_iqn_neutral  "[2/4] IQN-NEUTRAL"
launch configs/nfsp_iqn_mean_var.yaml nfsp_iqn_mean_var "[3/4] IQN-MEAN-VAR"
launch configs/nfsp_iqn_averse.yaml   nfsp_iqn_averse   "[4/4] IQN-AVERSE"

echo ""
echo "  ${#PIDS[@]} experiments running."
echo "  Press Ctrl+C to stop all."
echo ""

FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${NAMES[$i]}"
    if wait "$pid"; then
        echo "  DONE: $name ($(date))"
    else
        echo "  FAILED: $name (exit code $?) — check results/logs/${name}.log"
        FAILED=$((FAILED + 1))
    fi
done

trap - EXIT

echo ""
if [ $FAILED -eq 0 ]; then
    echo "=== GENERATING RESULTS ==="
    python3 scripts/generate_results.py
    echo ""
    echo "============================================================"
    echo "  ALL DONE: $(date)"
    echo "  Logs:    results/logs/"
    echo "  Figures: results/figures/"
    echo "============================================================"
else
    echo "  $FAILED experiment(s) failed. Check logs."
    exit 1
fi
