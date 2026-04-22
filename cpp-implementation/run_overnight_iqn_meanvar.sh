#!/bin/bash
# NFSP-IQN Mean-Variance — Hold'em
# Arch: [256,128,256,128], lr=0.002/0.0002 (matched to baseline + iqn_neutral)
# Risk: mean-variance with var_penalty=0.5 (moderate variance penalty on value)
cd "$(dirname "$0")"

NAME=holdem_iqn_meanvar
WORKERS=3
WORKER_BATCH=4
EPISODES=20000000

DQN_LR=0.002
AVG_LR=0.0002

EVAL_FREQ=200000
CHECKPOINT_FREQ=5000000
LOG_FREQ=20000
IQN_N=8
VAR_PENALTY=0.5

ARTIFACT_ROOT=/mnt/data/cos435/weights
mkdir -p logs results/logs "${ARTIFACT_ROOT}/eval_${NAME}" "${ARTIFACT_ROOT}/final_${NAME}"
rm -rf eval_weights_${NAME} final_weights_${NAME}
ln -sfn "${ARTIFACT_ROOT}/eval_${NAME}"  "eval_weights_${NAME}"
ln -sfn "${ARTIFACT_ROOT}/final_${NAME}" "final_weights_${NAME}"

rm -f logs/${NAME}.log results/logs/${NAME}_seed42.jsonl
rm -rf checkpoints/${NAME}_*
rm -f "${ARTIFACT_ROOT}/eval_${NAME}"/* "${ARTIFACT_ROOT}/final_${NAME}"/*

echo "============================================================"
echo "  NFSP-IQN Hold'em — MEAN-VARIANCE (var_penalty=$VAR_PENALTY)"
echo "  Arch: [256,128,256,128], Adam lr=${DQN_LR}/${AVG_LR}, IQN N=$IQN_N"
echo "  $WORKERS workers, $EPISODES episodes"
echo "  Started: $(date)"
echo "============================================================"
echo ""

./build/train_holdem --name ${NAME} --agent nfsp_iqn \
    --episodes $EPISODES --workers $WORKERS \
    --worker-batch $WORKER_BATCH --eval-freq $EVAL_FREQ \
    --checkpoint-freq $CHECKPOINT_FREQ --log-freq $LOG_FREQ \
    --iqn-n $IQN_N --risk none --var-penalty $VAR_PENALTY \
    --dqn-lr $DQN_LR --avg-lr $AVG_LR \
    2>&1 | tee -a logs/${NAME}.log

echo ""
echo "============================================================"
echo "  DONE: $(date)"
echo "============================================================"
