#!/bin/bash
# NFSP baseline — 64x64 architecture, 50M episodes, snapshot every 10M.
#
# Checkpoints eval-policy weights (p{0,1}_avg.pt) to
#   snapshots/holdem_64x64_baseline/ep{10,20,30,40,50}M
# for later exploitability evaluation. Training's built-in checkpoint dir is
# rotated by the binary itself and is not used by the eval pipeline.
set -euo pipefail
cd "$(dirname "$0")"

NAME=holdem_64x64_baseline
HIDDEN="64,64"
WORKERS=3
WORKER_BATCH=4
EPISODES=50000000

DQN_LR=0.002
AVG_LR=0.0002

EVAL_FREQ=10000000      # save p*_avg.pt + run in-training H2H at each 10M
CHECKPOINT_FREQ=10000000
LOG_FREQ=50000

ARTIFACT_ROOT=/mnt/data/cos435/weights
SNAPSHOT_ROOT="$(pwd)/snapshots/${NAME}"

mkdir -p logs results/logs "${ARTIFACT_ROOT}/eval_${NAME}" "${ARTIFACT_ROOT}/final_${NAME}" "${SNAPSHOT_ROOT}"
rm -rf eval_weights_${NAME} final_weights_${NAME}
ln -sfn "${ARTIFACT_ROOT}/eval_${NAME}"  "eval_weights_${NAME}"
ln -sfn "${ARTIFACT_ROOT}/final_${NAME}" "final_weights_${NAME}"

# Fresh run — user said no need to preserve state for rerun.
rm -f logs/${NAME}.log results/logs/${NAME}_seed42.jsonl
rm -rf checkpoints/${NAME}_*
rm -f "${ARTIFACT_ROOT}/eval_${NAME}"/* "${ARTIFACT_ROOT}/final_${NAME}"/*
rm -rf "${SNAPSHOT_ROOT}"/*

# ── Snapshot watcher: copies eval_weights_{name}/p{0,1}_avg.pt into a
#    permanent per-episode dir each time the binary prints a CHECKPOINT line.
(
    seen=""
    while true; do
        if [ -f "logs/${NAME}.log" ]; then
            for ep in $(grep -oE 'CHECKPOINT\] saved at ep [0-9]+' "logs/${NAME}.log" \
                          | awk '{print $NF}' | sort -u); do
                dest="${SNAPSHOT_ROOT}/ep${ep}"
                if [ ! -f "${dest}/p1_avg.pt" ]; then
                    mkdir -p "${dest}"
                    cp -f "eval_weights_${NAME}/p0_avg.pt" "${dest}/p0_avg.pt" 2>/dev/null || true
                    cp -f "eval_weights_${NAME}/p1_avg.pt" "${dest}/p1_avg.pt" 2>/dev/null || true
                    if [ -f "${dest}/p1_avg.pt" ]; then
                        echo "[snapshot] ep=${ep} -> ${dest}" | tee -a "logs/${NAME}.log"
                    fi
                fi
            done
        fi
        sleep 30
    done
) &
WATCHER_PID=$!
trap "kill ${WATCHER_PID} 2>/dev/null || true" EXIT

echo "============================================================"
echo "  NFSP C++ Hold'em — BASELINE 64x64 (50M)"
echo "  Arch: [${HIDDEN}], Adam lr=${DQN_LR}/${AVG_LR}"
echo "  ${WORKERS} workers, ${EPISODES} episodes"
echo "  Snapshot dir: ${SNAPSHOT_ROOT}/ep{10,20,30,40,50}M"
echo "  Started: $(date)"
echo "============================================================"

./build/train_holdem --name ${NAME} --agent nfsp \
    --episodes ${EPISODES} --workers ${WORKERS} \
    --worker-batch ${WORKER_BATCH} --eval-freq ${EVAL_FREQ} \
    --checkpoint-freq ${CHECKPOINT_FREQ} --log-freq ${LOG_FREQ} \
    --dqn-lr ${DQN_LR} --avg-lr ${AVG_LR} \
    --hidden "${HIDDEN}" \
    --res-buf 15000000 \
    2>&1 | tee -a logs/${NAME}.log

# One last snapshot pass to catch the 50M checkpoint after training ended.
sleep 5
for ep in $(grep -oE 'CHECKPOINT\] saved at ep [0-9]+' "logs/${NAME}.log" \
              | awk '{print $NF}' | sort -u); do
    dest="${SNAPSHOT_ROOT}/ep${ep}"
    if [ ! -f "${dest}/p1_avg.pt" ]; then
        mkdir -p "${dest}"
        cp -f "eval_weights_${NAME}/p0_avg.pt" "${dest}/p0_avg.pt" 2>/dev/null || true
        cp -f "eval_weights_${NAME}/p1_avg.pt" "${dest}/p1_avg.pt" 2>/dev/null || true
    fi
done
# Also stash the final_weights/ copy as the authoritative 50M.
cp -f "final_weights_${NAME}/p0_avg.pt" "${SNAPSHOT_ROOT}/ep${EPISODES}/p0_avg.pt" 2>/dev/null || true
cp -f "final_weights_${NAME}/p1_avg.pt" "${SNAPSHOT_ROOT}/ep${EPISODES}/p1_avg.pt" 2>/dev/null || true

echo "============================================================"
echo "  DONE ${NAME}: $(date)"
echo "  Log:       logs/${NAME}.log"
echo "  Snapshots: ${SNAPSHOT_ROOT}/ep*"
echo "============================================================"
