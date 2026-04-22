#!/bin/bash
# Run Hold'em NFSP small — smaller net + higher lr experiment
# Arch: [256,128,256,128], Adam lr=0.003/0.0003, reward /100
#
# Auto-resume: if checkpoints exist, training resumes from latest.
# To start fresh, pass --fresh as the first argument.
cd "$(dirname "$0")"

NAME=holdem_small
WORKERS=7
WORKER_BATCH=4
EPISODES=5000000

EVAL_FREQ=100000
CHECKPOINT_FREQ=500000
LOG_FREQ=10000

mkdir -p logs results/logs results/figures \
  eval_weights_${NAME} final_weights_${NAME} \
  checkpoints

# Check for existing checkpoints to decide resume vs fresh
LATEST_CKPT=$(ls -d checkpoints/${NAME}_ep* 2>/dev/null | sed 's/.*_ep//' | sed 's/_p.*//' | sort -n | tail -1)

FORCE_FRESH=0
if [ "$1" = "--fresh" ]; then
    FORCE_FRESH=1
fi

if [ $FORCE_FRESH -eq 1 ]; then
    echo "  --fresh flag set: clearing all state"
    rm -f logs/${NAME}.log results/logs/${NAME}_seed42.jsonl
    rm -rf checkpoints/${NAME}_*
    rm -rf eval_weights_${NAME}/* final_weights_${NAME}/*
    LATEST_CKPT=""
elif [ -n "$LATEST_CKPT" ]; then
    echo "  Resuming from checkpoint at episode $LATEST_CKPT"
    # Append to existing log instead of overwriting
else
    echo "  No checkpoint found — starting fresh"
    rm -f logs/${NAME}.log results/logs/${NAME}_seed42.jsonl
fi

echo "============================================================"
echo "  NFSP C++ Hold'em — SMALL NET (smaller + higher lr)"
echo "  Arch: [256,128,256,128], Adam lr=0.003/0.0003"
echo "  Reward norm /100, DQN buf=600K, Reservoir=15M"
echo "  learn_every=256, target=128K (paper-matched)"
echo "  Epsilon 0.08 → 0.001 over 20M steps"
echo "  $WORKERS workers, $EPISODES episodes"
if [ -n "$LATEST_CKPT" ] && [ $FORCE_FRESH -eq 0 ]; then
    echo "  RESUMING from ep $LATEST_CKPT"
fi
echo "  Started: $(date)"
echo "============================================================"
echo ""

./build/train_holdem --name ${NAME} --agent nfsp \
    --episodes $EPISODES --workers $WORKERS \
    --worker-batch $WORKER_BATCH --eval-freq $EVAL_FREQ \
    --checkpoint-freq $CHECKPOINT_FREQ --log-freq $LOG_FREQ \
    2>&1 | tee -a logs/${NAME}.log

echo ""
echo "============================================================"
echo "  DONE: $(date)"
echo "  Log: logs/${NAME}.log"
echo "============================================================"
