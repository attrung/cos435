#!/bin/bash
# NFSP-IQN Risk-Seeking — Hold'em
# Arch: [256,128,256,128], lr=0.002/0.0002 (matched to baseline + variants)
# Risk: seeking with p1=0.75, p2=1.0 (upper-quantile-weighted — opposite of CVaR-averse)
cd "$(dirname "$0")"

NAME=holdem_iqn_seeking
WORKERS=3
WORKER_BATCH=4
EPISODES=20000000

DQN_LR=0.002
AVG_LR=0.0002

EVAL_FREQ=200000
CHECKPOINT_FREQ=10000000
LOG_FREQ=20000
IQN_N=8
RISK_P1=0.75
RISK_P2=1.0

ARTIFACT_ROOT=/mnt/data/cos435/weights
mkdir -p logs results/logs "${ARTIFACT_ROOT}/eval_${NAME}" "${ARTIFACT_ROOT}/final_${NAME}"
rm -rf eval_weights_${NAME} final_weights_${NAME}
ln -sfn "${ARTIFACT_ROOT}/eval_${NAME}"  "eval_weights_${NAME}"
ln -sfn "${ARTIFACT_ROOT}/final_${NAME}" "final_weights_${NAME}"

rm -f logs/${NAME}.log results/logs/${NAME}_seed42.jsonl
rm -rf checkpoints/${NAME}_*
rm -f "${ARTIFACT_ROOT}/eval_${NAME}"/* "${ARTIFACT_ROOT}/final_${NAME}"/*

echo "============================================================"
echo "  NFSP-IQN Hold'em — RISK-SEEKING (p1=$RISK_P1 p2=$RISK_P2)"
echo "  Arch: [256,128,256,128], Adam lr=${DQN_LR}/${AVG_LR}, IQN N=$IQN_N"
echo "  $WORKERS workers, $EPISODES episodes"
echo "  Started: $(date)"
echo "============================================================"
echo ""

./build/train_holdem --name ${NAME} --agent nfsp_iqn \
    --episodes $EPISODES --workers $WORKERS \
    --worker-batch $WORKER_BATCH --eval-freq $EVAL_FREQ \
    --checkpoint-freq $CHECKPOINT_FREQ --log-freq $LOG_FREQ \
    --iqn-n $IQN_N --risk seeking --risk-p1 $RISK_P1 --risk-p2 $RISK_P2 \
    --dqn-lr $DQN_LR --avg-lr $AVG_LR \
    2>&1 | tee -a logs/${NAME}.log

echo ""
echo "============================================================"
echo "  DONE: $(date)"
echo "============================================================"
