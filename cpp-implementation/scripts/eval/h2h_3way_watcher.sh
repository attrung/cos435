#!/bin/bash
# Watches snapshots/ for every-10M marks; when all three runs have a snapshot
# for a given episode, runs 3-way H2H (baseline vs meanvar, baseline vs neutral,
# meanvar vs neutral) at 5000 games each and appends to results/logs/h2h_3way.csv.
set -u
cd "$(dirname "$0")"

mkdir -p logs results/logs
OUT_CSV="results/logs/h2h_3way.csv"
LOG="logs/h2h_3way.log"
if [ ! -f "${OUT_CSV}" ]; then
    echo "episode,a_name,b_name,a_avg_chips,a_se_chips,a_mbb,a_mbb_se,games,seconds" > "${OUT_CSV}"
fi

NAMES=(holdem_64x64_baseline holdem_64x64_meanvar holdem_64x64_iqn_neutral)
SHORT=(baseline meanvar neutral)
GAMES=${H2H_GAMES:-5000}
MAX_EP=${MAX_EP:-50000000}
STEP=10000000

run_h2h () {
    local ep=$1 ai=$2 bi=$3
    local a_name=${SHORT[$ai]} b_name=${SHORT[$bi]}
    local a_dir=snapshots/${NAMES[$ai]}/ep${ep}
    local b_dir=snapshots/${NAMES[$bi]}/ep${ep}
    local t0=$SECONDS
    local out
    out=$(python3 h2h_64x64_baseline_vs_meanvar.py \
            --a-dir "${a_dir}" --a-name "${a_name}_${ep}" \
            --b-dir "${b_dir}" --b-name "${b_name}_${ep}" \
            --games "${GAMES}" --seed 42 2>&1)
    local dt=$((SECONDS - t0))
    echo "[ep=${ep}] ${a_name} vs ${b_name}" | tee -a "${LOG}"
    echo "${out}" | tee -a "${LOG}"
    # parse summary: e.g. "baseline_10000000 avg chips vs meanvar_10000000: +57.06 ± 7.12   (+570.6 ± 71.2 mbb/g)   [55s]"
    local parsed
    parsed=$(echo "${out}" | awk -v a="${a_name}_${ep}" -v b="${b_name}_${ep}" '
        $0 ~ a " avg chips vs " b ":" {
            # collapse whitespace and pull numeric fields
            gsub(/[():\+\[\]s]/,"",$0);
            # expect: "...: X ± Y   (Z ± W mbb/g)"
            n = split($0, f, / +/);
            # parse from the right: last 4 non-symbol numbers before "mbbg" token
            # simpler: regex-less — use sub/match
        }
    ')
    # Simpler parse via sed
    local summary
    summary=$(echo "${out}" | grep -E "avg chips vs ${b_name}_${ep}:" | head -1)
    local chips chips_se mbb mbb_se
    chips=$(echo "${summary}" | sed -nE 's/.*: *([+-]?[0-9.]+) *±.*/\1/p')
    chips_se=$(echo "${summary}" | sed -nE 's/.* ± *([0-9.]+) *\(.*/\1/p')
    mbb=$(echo "${summary}" | sed -nE 's/.*\(([+-]?[0-9.]+) *±.*mbb.*/\1/p')
    mbb_se=$(echo "${summary}" | sed -nE 's/.*\([+-]?[0-9.]+ *± *([0-9.]+) *mbb.*/\1/p')
    echo "${ep},${a_name},${b_name},${chips},${chips_se},${mbb},${mbb_se},${GAMES},${dt}" >> "${OUT_CSV}"
}

echo "=== H2H 3-way watcher started $(date) ===" | tee -a "${LOG}"

for ep in 10000000 20000000 30000000 40000000 50000000; do
    [ "${ep}" -gt "${MAX_EP}" ] && break
    # Wait until all three snapshots for this ep exist.
    echo "waiting for all 3 snapshots @ ep=${ep}..." | tee -a "${LOG}"
    while true; do
        all_present=1
        for n in "${NAMES[@]}"; do
            if [ ! -f "snapshots/${n}/ep${ep}/p1_avg.pt" ]; then
                all_present=0
                break
            fi
        done
        if [ "${all_present}" = "1" ]; then break; fi
        sleep 30
    done
    echo "all 3 snapshots present for ep=${ep} at $(date)" | tee -a "${LOG}"

    # 3 pairs
    run_h2h "${ep}" 0 1   # baseline vs meanvar
    run_h2h "${ep}" 0 2   # baseline vs neutral
    run_h2h "${ep}" 1 2   # meanvar vs neutral
done

echo "=== H2H 3-way watcher finished $(date) ===" | tee -a "${LOG}"
