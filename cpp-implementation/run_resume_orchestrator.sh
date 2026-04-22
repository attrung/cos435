#!/bin/bash
# Orchestrator that:
#   1. Waits for current IQN variant runs (PIDs $PID_AVERSE, $PID_MEANVAR) to finish
#   2. LBRs those finished runs
#   3. Archives 20M final_weights for baseline + iqn_neutral (to preserve for 20M-vs-40M comparison)
#   4. Launches resume of baseline + iqn_neutral to 40M in parallel
#   5. When resumes finish, LBRs them
cd "$(dirname "$0")"

PID_AVERSE=${1:?need averse PID}
PID_MEANVAR=${2:?need meanvar PID}

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/_resume_orch.log; }

log "orchestrator started — waiting for averse=$PID_AVERSE meanvar=$PID_MEANVAR"
while kill -0 $PID_AVERSE 2>/dev/null || kill -0 $PID_MEANVAR 2>/dev/null; do
    sleep 60
done
log "current IQN variants finished; kicking off LBR on averse + meanvar"

# LBR averse + meanvar in parallel (uses 4+4=8 cores for ~10 min each)
python3 eval/lbr_holdem_accurate.py --weights final_weights_holdem_iqn_averse  --games 5000 --rollouts 15 --workers 4 > logs/lbr_holdem_iqn_averse_final.log  2>&1 &
LBR_A=$!
python3 eval/lbr_holdem_accurate.py --weights final_weights_holdem_iqn_meanvar --games 5000 --rollouts 15 --workers 4 > logs/lbr_holdem_iqn_meanvar_final.log 2>&1 &
LBR_M=$!
log "LBR PIDs: averse=$LBR_A meanvar=$LBR_M — waiting"
wait $LBR_A $LBR_M
log "LBRs done. Archiving 20M weights before resume overwrites them."

# Archive 20M final_weights so we still have them after resume
ARCH=/mnt/data/cos435/archive_20M
mkdir -p $ARCH
cp -r /mnt/data/cos435/weights/final_holdem_small_long $ARCH/final_holdem_small_long_ep20M 2>/dev/null || log "WARN copy baseline weights failed"
cp -r /mnt/data/cos435/weights/final_holdem_iqn_long  $ARCH/final_holdem_iqn_long_ep20M  2>/dev/null || log "WARN copy iqn weights failed"
# Also save the 20M LBR numbers (already produced; just reference them)
echo "20M LBR: baseline=1440 mbb/g, iqn_neutral=1993 mbb/g" > $ARCH/20M_LBR_numbers.txt

log "archived to $ARCH; launching resume of baseline + iqn_neutral to 40M"

# Launch resumes in parallel
nohup bash run_resume_baseline.sh > logs/_resume_baseline.out 2>&1 &
RESUME_BASELINE=$!
sleep 2
nohup bash run_resume_iqn.sh > logs/_resume_iqn.out 2>&1 &
RESUME_IQN=$!
log "resume PIDs: baseline=$RESUME_BASELINE iqn=$RESUME_IQN"

# Wait for resumes to finish
while kill -0 $RESUME_BASELINE 2>/dev/null || kill -0 $RESUME_IQN 2>/dev/null; do
    sleep 60
done
log "resumes finished; LBRing 40M weights"

# Final LBR on the 40M weights
python3 eval/lbr_holdem_accurate.py --weights final_weights_holdem_small_long --games 5000 --rollouts 15 --workers 4 > logs/lbr_holdem_small_long_40M.log 2>&1 &
LBR_B40=$!
python3 eval/lbr_holdem_accurate.py --weights final_weights_holdem_iqn_long   --games 5000 --rollouts 15 --workers 4 > logs/lbr_holdem_iqn_long_40M.log   2>&1 &
LBR_I40=$!
wait $LBR_B40 $LBR_I40
log "ALL DONE. 20M and 40M LBR numbers saved. Check logs/lbr_*_40M.log and $ARCH/20M_LBR_numbers.txt"
