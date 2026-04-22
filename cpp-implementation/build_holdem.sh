#!/bin/bash
# Build the Hold'em training binary
# Uses: g++ with LibTorch + OpenSpiel shared library
set -e

cd "$(dirname "$0")"

TORCH_DIR=/home/anthonyn/miniconda3/envs/cos435/lib/python3.11/site-packages/torch
OSSPIEL=third_party/open_spiel

if [ ! -f "${OSSPIEL}/lib/libopen_spiel.so" ]; then
    echo "ERROR: OpenSpiel library not found at ${OSSPIEL}/lib/libopen_spiel.so"
    echo "Run the OpenSpiel build steps first (see task description)."
    exit 1
fi

mkdir -p build

echo "Building train_holdem..."
g++ -std=c++17 -O3 -march=native -pthread \
  -Isrc \
  -I${OSSPIEL}/include \
  -I${OSSPIEL}/include/open_spiel/abseil-cpp \
  -I${OSSPIEL}/include/open_spiel/json/include \
  -I${TORCH_DIR}/include -I${TORCH_DIR}/include/torch/csrc/api/include \
  -L${TORCH_DIR}/lib -L${OSSPIEL}/lib \
  -o build/train_holdem src/train_holdem.cpp \
  -lopen_spiel -ltorch -ltorch_cpu -lc10 \
  -Wl,-rpath,${TORCH_DIR}/lib \
  -Wl,-rpath,${OSSPIEL}/lib \
  -D_GLIBCXX_USE_CXX11_ABI=1

echo "Done: build/train_holdem"
echo ""
echo "Usage:"
echo "  ./build/train_holdem --name holdem_baseline --episodes 40000000 --workers 3"
echo "  ./build/train_holdem --name holdem_iqn --agent nfsp_iqn --dqn-lr 0.0005 --risk cvar --risk-p1 0.25"
