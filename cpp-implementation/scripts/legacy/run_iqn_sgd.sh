#!/bin/bash
# Resume IQN experiments with SGD optimizer (was Adam)
# Reuses existing Adam-trained network weights but resets optimizer momentum.
# SGD-after-Adam often finds flatter minima → better generalization.
cd "$(dirname "$0")"

# Resume from current 40M checkpoints, train another 10M with SGD
EPISODES=50000000
WORKERS=5
WORKER_BATCH=16
EVAL_FREQ=200000
CHECKPOINT_FREQ=1000000
IQN_N=8
SGD_LR=0.001  # Conservative — SGD lr=0.01 would diverge

mkdir -p logs

echo "============================================================"
echo "  NFSP C++ — IQN SGD fine-tune (lr=$SGD_LR, momentum=0.9)"
echo "  Resuming from existing 40M checkpoints"
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
        --checkpoint-freq $CHECKPOINT_FREQ --iqn-n $IQN_N "$@" \
        > "logs/${name}_sgd.log" 2>&1 &
    PIDS+=($!)
    NAMES+=("$name")
}

launch iqn_neutral    "[1/5] IQN-NEUTRAL"  --agent nfsp_iqn --dqn-lr $SGD_LR --risk none
launch iqn_mv01       "[2/5] IQN-MV-0.1"   --agent nfsp_iqn --dqn-lr $SGD_LR --risk none --var-penalty 0.1
launch iqn_mv05       "[3/5] IQN-MV-0.5"   --agent nfsp_iqn --dqn-lr $SGD_LR --risk none --var-penalty 0.5
launch iqn_averse     "[4/5] IQN-AVERSE"   --agent nfsp_iqn --dqn-lr $SGD_LR --risk cvar --risk-p1 0.25
launch iqn_seeking    "[5/5] IQN-SEEKING"  --agent nfsp_iqn --dqn-lr $SGD_LR --risk seeking --risk-p1 0.75 --risk-p2 1.0

echo ""
echo "  5 experiments running. PIDs: ${PIDS[*]}"
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

while true; do
    DONE=true
    for i in "${!PIDS[@]}"; do
        kill -0 "${PIDS[$i]}" 2>/dev/null && DONE=false
    done
    echo "--- $(date +%H:%M:%S) ---"
    for i in "${!NAMES[@]}"; do
        last=$(grep 'ep/s' "logs/${NAMES[$i]}_sgd.log" 2>/dev/null | tail -1)
        [ -n "$last" ] && echo "  ${NAMES[$i]}: $last"
    done
    $DONE && break
    sleep 30
done

trap - EXIT
echo ""
echo "============================================================"
echo "  ALL DONE: $(date)"
echo "============================================================"
