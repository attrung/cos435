#!/bin/bash
# Run all 6 NFSP experiments in parallel with C++ implementation
# ~10K ep/s each, ~1-2 cores per experiment, fits on 8 cores
cd "$(dirname "$0")"

# Default: 5 workers per experiment with batched episodes (fast, slight staleness)
# Pass --sequential to use 1 worker + batch=1 (matches OpenSpiel exactly, slower)
WORKERS=5
WORKER_BATCH=16
EPISODES=40000000

if [ "$1" = "--sequential" ]; then
    WORKERS=1
    WORKER_BATCH=1
    EPISODES=45000000
    echo "  Mode: SEQUENTIAL (1 worker, batch=1) — matches OpenSpiel"
fi

EVAL_FREQ=200000
CHECKPOINT_FREQ=1000000

mkdir -p logs eval_weights_baseline eval_weights_iqn_neutral \
  eval_weights_iqn_mv01 eval_weights_iqn_mv05 \
  eval_weights_iqn_averse eval_weights_iqn_seeking \
  final_weights_baseline final_weights_iqn_neutral \
  final_weights_iqn_mv01 final_weights_iqn_mv05 \
  final_weights_iqn_averse final_weights_iqn_seeking

echo "============================================================"
echo "  NFSP C++ — 6 experiments ($WORKERS workers each)"
echo "  Started: $(date)"
echo "============================================================"
echo ""

PIDS=()
NAMES=()

launch() {
    local name="$1"; local label="$2"; shift 2
    echo "  $label — launching"
    ./build/train --name "$name" --episodes $EPISODES --workers $WORKERS \
        --worker-batch $WORKER_BATCH --eval-freq $EVAL_FREQ \
        --checkpoint-freq $CHECKPOINT_FREQ "$@" > "logs/${name}.log" 2>&1 &
    PIDS+=($!)
    NAMES+=("$name")
}

launch baseline      "[1/6] BASELINE"     --agent nfsp     --dqn-lr 0.01
launch iqn_neutral    "[2/6] IQN-NEUTRAL"  --agent nfsp_iqn --dqn-lr 0.0005 --risk none
launch iqn_mv01       "[3/6] IQN-MV-0.1"   --agent nfsp_iqn --dqn-lr 0.0005 --risk none --var-penalty 0.1
launch iqn_mv05       "[4/6] IQN-MV-0.5"   --agent nfsp_iqn --dqn-lr 0.0005 --risk none --var-penalty 0.5
launch iqn_averse     "[5/6] IQN-AVERSE"   --agent nfsp_iqn --dqn-lr 0.0005 --risk cvar --risk-p1 0.25
launch iqn_seeking    "[6/6] IQN-SEEKING"  --agent nfsp_iqn --dqn-lr 0.0005 --risk seeking --risk-p1 0.75 --risk-p2 1.0

echo ""
echo "  6 experiments running. PIDs: ${PIDS[*]}"
echo ""

cleanup() {
    echo "  Stopping..."
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null; done
    sleep 2
    for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null; done
    wait 2>/dev/null
    echo "  Done."
}
trap cleanup EXIT INT TERM

# Monitor every 30s
while true; do
    DONE=true
    for i in "${!PIDS[@]}"; do
        kill -0 "${PIDS[$i]}" 2>/dev/null && DONE=false
    done

    echo "--- $(date +%H:%M:%S) ---"
    for i in "${!NAMES[@]}"; do
        last=$(grep 'ep/s' "logs/${NAMES[$i]}.log" 2>/dev/null | tail -1)
        [ -n "$last" ] && echo "  ${NAMES[$i]}: $last"
    done

    $DONE && break
    sleep 30
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
echo "  Logs: logs/"
echo "============================================================"
