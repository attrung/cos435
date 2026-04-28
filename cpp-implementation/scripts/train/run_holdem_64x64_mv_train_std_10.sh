#!/bin/bash
# NFSP-IQN mean-variance, train-target, std-form β=0.5. 64x64 arch, 50M episodes.
set -euo pipefail
cd "$(dirname "$0")"

NAME=holdem_64x64_mv_train_std_10
HIDDEN="64,64"
WORKERS=3
WORKER_BATCH=4
EPISODES=50000000

DQN_LR=0.002
AVG_LR=0.0002
IQN_N=8
VAR_MODE=train
BETA=1.0

EVAL_FREQ=10000000
CHECKPOINT_FREQ=10000000
LOG_FREQ=50000

ARTIFACT_ROOT=/mnt/data/cos435/weights
SNAPSHOT_ROOT="$(pwd)/snapshots/${NAME}"

mkdir -p logs results/logs "${ARTIFACT_ROOT}/eval_${NAME}" "${ARTIFACT_ROOT}/final_${NAME}" "${SNAPSHOT_ROOT}"
rm -rf eval_weights_${NAME} final_weights_${NAME}
ln -sfn "${ARTIFACT_ROOT}/eval_${NAME}"  "eval_weights_${NAME}"
ln -sfn "${ARTIFACT_ROOT}/final_${NAME}" "final_weights_${NAME}"

rm -f logs/${NAME}.log results/logs/${NAME}_seed42.jsonl
rm -rf checkpoints/${NAME}_*
rm -f "${ARTIFACT_ROOT}/eval_${NAME}"/* "${ARTIFACT_ROOT}/final_${NAME}"/*
rm -rf "${SNAPSHOT_ROOT}"/*

(
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
echo "  NFSP-IQN Hold'em — MV TRAIN STD β=${BETA} 64x64 (50M)"
echo "  Arch: [${HIDDEN}], Adam lr=${DQN_LR}/${AVG_LR}, IQN N=${IQN_N}"
echo "  --var-mode ${VAR_MODE} --beta ${BETA}"
echo "  ${WORKERS} workers, ${EPISODES} episodes"
echo "  Started: $(date)"
echo "============================================================"

./build/train_holdem --name ${NAME} --agent nfsp_iqn \
    --episodes ${EPISODES} --workers ${WORKERS} \
    --worker-batch ${WORKER_BATCH} --eval-freq ${EVAL_FREQ} \
    --checkpoint-freq ${CHECKPOINT_FREQ} --log-freq ${LOG_FREQ} \
    --iqn-n ${IQN_N} --risk none --var-mode ${VAR_MODE} --beta ${BETA} \
    --dqn-lr ${DQN_LR} --avg-lr ${AVG_LR} \
    --hidden "${HIDDEN}" \
    --res-buf 15000000 \
    2>&1 | tee -a logs/${NAME}.log

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
cp -f "final_weights_${NAME}/p0_avg.pt" "${SNAPSHOT_ROOT}/ep${EPISODES}/p0_avg.pt" 2>/dev/null || true
cp -f "final_weights_${NAME}/p1_avg.pt" "${SNAPSHOT_ROOT}/ep${EPISODES}/p1_avg.pt" 2>/dev/null || true

echo "============================================================"
echo "  DONE ${NAME}: $(date)"
echo "============================================================"
