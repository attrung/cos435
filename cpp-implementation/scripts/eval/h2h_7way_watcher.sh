#!/bin/bash
# At each 10M-episode mark, when all 4 new mv_train_std_* variants have a snapshot,
# run a 7-way pairwise H2H against {baseline, neutral, mv05_legacy} (which already
# have all 5 snapshots from the prior run). 21 pairs × 5 marks = 105 H2H games.
#
# Each pair: 5000 games, ~15s. Total per 10M batch: ~5 min.
set -u
cd "$(dirname "$0")"

mkdir -p logs results/logs
OUT_CSV="results/logs/h2h_7way.csv"
LOG="logs/h2h_7way.log"
if [ ! -f "${OUT_CSV}" ]; then
    echo "episode,a_name,b_name,a_avg_chips,a_se_chips,a_mbb,a_mbb_se,games,seconds" > "${OUT_CSV}"
fi

# 7 agents, ordered. SHORT[i] is used as label and column key.
AGENTS=(
    "baseline:holdem_64x64_baseline"
    "neutral:holdem_64x64_iqn_neutral"
    "mv05_legacy:holdem_64x64_meanvar"
    "mv_t_std_05:holdem_64x64_mv_train_std_05"
    "mv_t_std_10:holdem_64x64_mv_train_std_10"
    "mv_t_std_20:holdem_64x64_mv_train_std_20"
    "mv_t_std_full:holdem_64x64_mv_train_std_full"
)

GAMES=${H2H_GAMES:-5000}

run_pair() {
    local ep=$1 ai=$2 bi=$3
    IFS=':' read -r a_short a_dir <<< "${AGENTS[$ai]}"
    IFS=':' read -r b_short b_dir <<< "${AGENTS[$bi]}"
    local a_path=snapshots/${a_dir}/ep${ep}
    local b_path=snapshots/${b_dir}/ep${ep}
    if [ ! -f "${a_path}/p1_avg.pt" ] || [ ! -f "${b_path}/p1_avg.pt" ]; then
        echo "SKIP ep=${ep} ${a_short} vs ${b_short} (missing)" | tee -a "${LOG}"
        return
    fi
    local t0=$SECONDS
    local out
    out=$(python3 h2h_64x64_baseline_vs_meanvar.py \
            --a-dir "${a_path}" --a-name "${a_short}_${ep}" \
            --b-dir "${b_path}" --b-name "${b_short}_${ep}" \
            --games "${GAMES}" --seed 42 2>&1)
    local dt=$((SECONDS - t0))
    echo "[ep=${ep}] ${a_short} vs ${b_short}: $(echo "${out}" | tail -2 | head -1)" | tee -a "${LOG}"
    local summary
    summary=$(echo "${out}" | grep -E "avg chips vs ${b_short}_${ep}:" | head -1)
    local chips chips_se mbb mbb_se
    chips=$(echo "${summary}"    | sed -nE 's/.*: *([+-]?[0-9.]+) *±.*/\1/p')
    chips_se=$(echo "${summary}" | sed -nE 's/.* ± *([0-9.]+) *\(.*/\1/p')
    mbb=$(echo "${summary}"      | sed -nE 's/.*\(([+-]?[0-9.]+) *±.*mbb.*/\1/p')
    mbb_se=$(echo "${summary}"   | sed -nE 's/.*\([+-]?[0-9.]+ *± *([0-9.]+) *mbb.*/\1/p')
    echo "${ep},${a_short},${b_short},${chips},${chips_se},${mbb},${mbb_se},${GAMES},${dt}" >> "${OUT_CSV}"
}

echo "=== 7-way H2H watcher started $(date) ===" | tee -a "${LOG}"

# Names of the 4 new variants we're waiting on.
NEW_VARIANTS=(
    holdem_64x64_mv_train_std_05
    holdem_64x64_mv_train_std_10
    holdem_64x64_mv_train_std_20
    holdem_64x64_mv_train_std_full
)

for ep in 10000000 20000000 30000000 40000000 50000000; do
    echo "waiting for all 4 new-variant snapshots @ ep=${ep}..." | tee -a "${LOG}"
    while true; do
        all=1
        for n in "${NEW_VARIANTS[@]}"; do
            if [ ! -f "snapshots/${n}/ep${ep}/p1_avg.pt" ]; then
                all=0
                break
            fi
        done
        [ "${all}" = "1" ] && break
        sleep 30
    done
    echo "all 4 new-variant snapshots present at ep=${ep} ($(date))" | tee -a "${LOG}"

    # All 21 pairs (i<j over all 7 agents).
    n_agents=${#AGENTS[@]}
    for ((i=0; i<n_agents; i++)); do
        for ((j=i+1; j<n_agents; j++)); do
            run_pair "${ep}" "${i}" "${j}"
        done
    done
done

echo "=== 7-way H2H watcher finished $(date) ===" | tee -a "${LOG}"
