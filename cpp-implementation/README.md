# cpp-implementation/ — research code

C++ NFSP training binary for 2-player poker (Leduc + Limit Hold'em via OpenSpiel `universal_poker`), with a Python evaluation harness.

## Building

### 1. Install OpenSpiel (one-time)

The training binary links against `libopen_spiel.so` and OpenSpiel's headers must be on the include path. We expect them under `cpp-implementation/third_party/open_spiel`:

```
cpp-implementation/third_party/open_spiel/
├── lib/libopen_spiel.so
└── include/
    ├── open_spiel/        # game headers
    └── open_spiel/abseil-cpp/
```

To build OpenSpiel from source (pinned to whatever commit you ran the experiments on — see `git log` of the upstream repo):

```bash
cd cpp-implementation
git clone https://github.com/google-deepmind/open_spiel.git third_party/open_spiel
cd third_party/open_spiel
./install.sh
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ../open_spiel
make -j$(nproc) open_spiel
# Then collect the .so + headers into the layout above (the install.sh helper handles this on Linux).
```

### 2. Build the training binary

LibTorch is the other dependency. The build script reads it from a conda env path; adjust `TORCH_DIR` in `build_holdem.sh` to your environment.

```bash
cd cpp-implementation
bash build_holdem.sh
# Output: build/train_holdem
```

If you want CMake instead of the one-shot g++ wrapper, see `CMakeLists.txt` (a thin alternative).

## Running a 50M-episode training (one variant)

Each run script is fully self-contained — sets up artifact dirs, launches a snapshot watcher, then invokes `build/train_holdem`. Example:

```bash
cd cpp-implementation
bash scripts/train/run_holdem_64x64_mv_train_std_10.sh
```

Wall clock: ~6 hrs for one run, ~6 hrs for 4 runs in parallel (4-way × 5 threads on 8 vCPUs). RAM: ~26 GB per process with `--res-buf 15000000` (default for these scripts).

## Running the H2H tournament

After all 7 trainings have populated `cpp-implementation/snapshots/holdem_64x64_*/ep<N>M/`:

```bash
bash scripts/eval/h2h_7way_watcher.sh
# Output: results/logs/h2h_7way.csv
```

Or to use the committed paper weights instead of re-training:

```bash
mkdir -p snapshots
for d in ../weights/holdem_64x64_*; do
    ln -sf "$(realpath ${d})" snapshots/$(basename ${d})
done
bash scripts/eval/h2h_7way_watcher.sh
```

## Layout

```
cpp-implementation/
├── README.md              # this file
├── NOTES_FOR_PAPER.md     # implementation notes (Leduc + Hold'em)
├── CMakeLists.txt
├── build_holdem.sh        # one-shot g++ build wrapper
├── src/
│   ├── train_holdem.cpp   # main entry point — parses CLI, orchestrates workers + learners
│   ├── train.cpp          # Leduc training (older pathway)
│   ├── holdem.h           # OpenSpiel `universal_poker` wrapper, info-state encoding
│   ├── leduc_poker.h      # custom Leduc implementation
│   ├── nfsp_agent.h, nfsp_agent_holdem.h          # NFSP with DQN best-response
│   ├── nfsp_iqn_agent.h, nfsp_iqn_agent_holdem.h  # NFSP with IQN best-response
│   ├── networks.h         # IQNNetwork, VarMLP (LibTorch modules)
│   ├── fast_mlp.h         # raw-array MLP for fast worker-thread inference
│   └── replay_buffer.h    # circular + reservoir
├── eval/
│   ├── lbr_holdem_accurate.py    # parallel local best-response exploitability
│   ├── h2h_tournament.py         # round-robin H2H
│   ├── eval_holdem_h2h.py        # in-training H2H-vs-random helper
│   └── model_utils.py            # auto-detect-arch loader for saved C++ weights
└── scripts/
    ├── train/             # 7 paper-run launch scripts
    ├── eval/              # H2H watchers + exploitability sweep wrappers
    └── legacy/            # earlier run scripts referenced in NOTES_FOR_PAPER.md
```

## CLI surface (`build/train_holdem`)

Key flags used in the paper runs:

| Flag | Default | Purpose |
|---|---|---|
| `--agent {nfsp\|nfsp_iqn}` | `nfsp` | Best-response head |
| `--episodes <N>` | 5M | Total training episodes |
| `--name <str>` | `holdem_baseline` | Run name (used for artifact paths) |
| `--hidden "<csv>"` | `256,128,256,128` | MLP hidden sizes (paper runs use `64,64`) |
| `--workers <N>` | 1 | Episode-generator threads |
| `--res-buf <N>` | 30M | Reservoir buffer size (use 15M for 4-way parallel to fit in 125 GB RAM) |
| `--var-mode {none\|train\|train_full}` | `none` | Mean-variance application site (paper) |
| `--beta <float>` | 0 | Aversion coefficient β |
| `--var-penalty <float>` | 0 | (legacy alias, was a no-op on Hold'em workers — see REPORT.md) |
| `--eval-freq <N>` | 100K | Episodes between in-training H2H-vs-random + weight save |
| `--checkpoint-freq <N>` | 500K | Episodes between full state checkpoints (rotated) |

See `src/train_holdem.cpp` (`int main`, ~line 360 onwards) for the complete list.
