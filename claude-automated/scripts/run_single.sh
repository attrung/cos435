#!/bin/bash
# Usage: ./scripts/run_single.sh <config_file> [--parallel]
# Example: ./scripts/run_single.sh configs/nfsp_baseline.yaml --parallel
export CUDA_VISIBLE_DEVICES=""
cd "$(dirname "$0")/.."
mkdir -p results/logs results/checkpoints
python3 src/train.py --config "$1" ${2:+--parallel} 2>&1 | tee "results/logs/$(basename $1 .yaml).log"
