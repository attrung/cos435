#!/bin/bash
# Overnight launcher: baseline + IQN in parallel now, then IQN-smaller after baseline finishes.
# All detached with nohup so this survives SSH disconnect.
cd "$(dirname "$0")"

mkdir -p logs

echo "============================================================"
echo "  OVERNIGHT LAUNCH: $(date)"
echo "============================================================"

# 1. Baseline NFSP — starts now
nohup bash run_overnight_baseline.sh > logs/_overnight_baseline.out 2>&1 &
BASELINE_PID=$!
echo "  BASELINE launched: PID $BASELINE_PID"
echo "    tail -f logs/holdem_small_long.log"

sleep 2

# 2. IQN neutral — starts now, parallel with baseline
nohup bash run_overnight_iqn.sh > logs/_overnight_iqn.out 2>&1 &
IQN_PID=$!
echo "  IQN-NEUTRAL launched: PID $IQN_PID"
echo "    tail -f logs/holdem_iqn_long.log"

# 3. IQN smaller — waits for baseline to exit, then launches.
#    Detached via nohup so ssh disconnect doesn't kill the waiter.
nohup bash -c "
  while kill -0 $BASELINE_PID 2>/dev/null; do sleep 60; done
  echo '[$(date)] baseline exited, launching iqn_smaller' >> logs/_overnight_smaller.out
  bash run_overnight_iqn_smaller.sh >> logs/_overnight_smaller.out 2>&1
" > /dev/null 2>&1 &
WAITER_PID=$!
echo "  IQN-SMALLER waiter: PID $WAITER_PID (will auto-start when baseline PID $BASELINE_PID exits)"
echo "    tail -f logs/holdem_iqn_smaller.log  (once it starts)"

echo ""
echo "  All detached. Safe to close terminal."
echo "  PIDs: baseline=$BASELINE_PID  iqn-neutral=$IQN_PID  waiter=$WAITER_PID"
echo "============================================================"
