#!/bin/bash
# Overnight NFSP-IQN (risk-neutral) — SMALLER net
# Arch: [128,64,128,64] (overridden via --hidden), Adam lr=0.002/0.0002
# IQN: N=8 quantiles, risk=none
# Purpose: see if IQN retains edge at even smaller capacity.
# FRESH run. Name: holdem_iqn_smaller.
cd "$(dirname "$0")"

NAME=holdem_iqn_smaller
WORKERS=3
WORKER_BATCH=4
EPISODES=20000000

DQN_LR=0.002
AVG_LR=0.0002
HIDDEN="128,64,128,64"

EVAL_FREQ=200000
CHECKPOINT_FREQ=5000000
LOG_FREQ=20000
IQN_N=8

# Weights live on /mnt/data (big disk); symlink in from repo for scripts that expect cwd-relative paths
ARTIFACT_ROOT=/mnt/data/cos435/weights
mkdir -p logs results/logs "${ARTIFACT_ROOT}/eval_${NAME}" "${ARTIFACT_ROOT}/final_${NAME}"
rm -rf eval_weights_${NAME} final_weights_${NAME}
ln -sfn "${ARTIFACT_ROOT}/eval_${NAME}"  "eval_weights_${NAME}"
ln -sfn "${ARTIFACT_ROOT}/final_${NAME}" "final_weights_${NAME}"

# FRESH: clear any stale state for this name
rm -f logs/${NAME}.log results/logs/${NAME}_seed42.jsonl
rm -rf checkpoints/${NAME}_*
rm -f "${ARTIFACT_ROOT}/eval_${NAME}"/* "${ARTIFACT_ROOT}/final_${NAME}"/*

echo "============================================================"
echo "  NFSP-IQN C++ Hold'em — IQN SMALLER NET"
echo "  Arch: [${HIDDEN}], Adam lr=${DQN_LR}/${AVG_LR}, IQN N=$IQN_N"
echo "  $WORKERS workers, $EPISODES episodes"
echo "  Started: $(date)"
echo "============================================================"
echo ""

./build/train_holdem --name ${NAME} --agent nfsp_iqn \
    --episodes $EPISODES --workers $WORKERS \
    --worker-batch $WORKER_BATCH --eval-freq $EVAL_FREQ \
    --checkpoint-freq $CHECKPOINT_FREQ --log-freq $LOG_FREQ \
    --iqn-n $IQN_N --risk none \
    --dqn-lr $DQN_LR --avg-lr $AVG_LR \
    --hidden "${HIDDEN}" \
    2>&1 | tee -a logs/${NAME}.log

echo ""
echo "============================================================"
echo "  DONE: $(date)"
echo "  Log: logs/${NAME}.log"
echo "============================================================"
