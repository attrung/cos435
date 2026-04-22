#!/bin/bash
# Resume NFSP baseline (holdem_small_long) from ep 20M → 40M.
# DOES NOT wipe checkpoints or weights — relies on auto-resume in binary.
cd "$(dirname "$0")"

NAME=holdem_small_long
WORKERS=3
WORKER_BATCH=4
EPISODES=40000000

DQN_LR=0.002
AVG_LR=0.0002

EVAL_FREQ=200000
CHECKPOINT_FREQ=5000000
LOG_FREQ=20000

# Symlinks should already exist from prior run, but re-link defensively
ARTIFACT_ROOT=/mnt/data/cos435/weights
mkdir -p logs "${ARTIFACT_ROOT}/eval_${NAME}" "${ARTIFACT_ROOT}/final_${NAME}"
[ -L "eval_weights_${NAME}" ]  || ln -sfn "${ARTIFACT_ROOT}/eval_${NAME}"  "eval_weights_${NAME}"
[ -L "final_weights_${NAME}" ] || ln -sfn "${ARTIFACT_ROOT}/final_${NAME}" "final_weights_${NAME}"

echo "============================================================"
echo "  RESUME: $NAME (20M → $EPISODES)"
echo "  Arch (auto-detected from ckpt): [256,128,256,128]"
echo "  lr=${DQN_LR}/${AVG_LR} (unchanged)"
echo "  Expected: binary auto-resumes from checkpoints/${NAME}_ep20000000_*"
echo "  Started: $(date)"
echo "============================================================"

./build/train_holdem --name ${NAME} --agent nfsp \
    --episodes $EPISODES --workers $WORKERS \
    --worker-batch $WORKER_BATCH --eval-freq $EVAL_FREQ \
    --checkpoint-freq $CHECKPOINT_FREQ --log-freq $LOG_FREQ \
    --dqn-lr $DQN_LR --avg-lr $AVG_LR \
    2>&1 | tee -a logs/${NAME}.log

echo "============================================================"
echo "  RESUME DONE: $(date)"
echo "============================================================"
