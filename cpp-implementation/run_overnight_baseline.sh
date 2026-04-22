#!/bin/bash
# Overnight NFSP baseline — small net + longer training
# Arch: [256,128,256,128] (built into binary), Adam lr=0.002/0.0002
# lr rationale: 0.001 plateau'd br_loss at 17.5 (medium net, slow).
#               0.003 plateau'd br_loss at 18.5 (small net, too coarse late).
#               0.002 = middle, biased toward progress for 12M-ep run.
# FRESH run. Name: holdem_small_long. Preserves holdem_small (5M) artifacts.
cd "$(dirname "$0")"

NAME=holdem_small_long
WORKERS=3
WORKER_BATCH=4
EPISODES=20000000

DQN_LR=0.002
AVG_LR=0.0002

EVAL_FREQ=200000
CHECKPOINT_FREQ=5000000
LOG_FREQ=20000

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
echo "  NFSP C++ Hold'em — BASELINE OVERNIGHT"
echo "  Arch: [256,128,256,128], Adam lr=${DQN_LR}/${AVG_LR}"
echo "  $WORKERS workers, $EPISODES episodes"
echo "  Started: $(date)"
echo "============================================================"
echo ""

./build/train_holdem --name ${NAME} --agent nfsp \
    --episodes $EPISODES --workers $WORKERS \
    --worker-batch $WORKER_BATCH --eval-freq $EVAL_FREQ \
    --checkpoint-freq $CHECKPOINT_FREQ --log-freq $LOG_FREQ \
    --dqn-lr $DQN_LR --avg-lr $AVG_LR \
    2>&1 | tee -a logs/${NAME}.log

echo ""
echo "============================================================"
echo "  DONE: $(date)"
echo "  Log: logs/${NAME}.log"
echo "============================================================"
