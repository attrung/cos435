#!/bin/bash
# Resume IQN-neutral (holdem_iqn_long) from ep 20M → 40M.
# DOES NOT wipe checkpoints or weights — relies on auto-resume in binary.
cd "$(dirname "$0")"

NAME=holdem_iqn_long
WORKERS=3
WORKER_BATCH=4
EPISODES=40000000

DQN_LR=0.002
AVG_LR=0.0002

EVAL_FREQ=200000
CHECKPOINT_FREQ=5000000
LOG_FREQ=20000
IQN_N=8

ARTIFACT_ROOT=/mnt/data/cos435/weights
mkdir -p logs "${ARTIFACT_ROOT}/eval_${NAME}" "${ARTIFACT_ROOT}/final_${NAME}"
[ -L "eval_weights_${NAME}" ]  || ln -sfn "${ARTIFACT_ROOT}/eval_${NAME}"  "eval_weights_${NAME}"
[ -L "final_weights_${NAME}" ] || ln -sfn "${ARTIFACT_ROOT}/final_${NAME}" "final_weights_${NAME}"

echo "============================================================"
echo "  RESUME: $NAME (20M → $EPISODES)"
echo "  lr=${DQN_LR}/${AVG_LR}, IQN N=$IQN_N, risk=none"
echo "  Expected: binary auto-resumes from checkpoints/${NAME}_ep20000000_*"
echo "  Started: $(date)"
echo "============================================================"

./build/train_holdem --name ${NAME} --agent nfsp_iqn \
    --episodes $EPISODES --workers $WORKERS \
    --worker-batch $WORKER_BATCH --eval-freq $EVAL_FREQ \
    --checkpoint-freq $CHECKPOINT_FREQ --log-freq $LOG_FREQ \
    --iqn-n $IQN_N --risk none \
    --dqn-lr $DQN_LR --avg-lr $AVG_LR \
    2>&1 | tee -a logs/${NAME}.log

echo "============================================================"
echo "  RESUME DONE: $(date)"
echo "============================================================"
