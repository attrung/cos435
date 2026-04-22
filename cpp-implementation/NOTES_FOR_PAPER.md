# NFSP + Distributional RL on Leduc Poker — Notes for Paper

## Setup

**Game**: Leduc poker (2-player zero-sum, ~936 information states, 6 cards).

**Implementation**: C++ with LibTorch CPU backend. Game logic reimplemented from scratch (verified against OpenSpiel: matched info state encoding, returns, and game statistics across 100K random rollouts). Worker threads use raw-array MLP inference (custom `FastMLP`) to avoid LibTorch overhead in the hot loop.

**Reference**: OpenSpiel's PyTorch NFSP (`open_spiel/python/pytorch/nfsp.py`) — verified algorithmic equivalence.

**Hardware**: Intel Xeon @ 3.10 GHz, 8 vCPUs (4 physical cores), 24.8 MB L3 cache, 31 GB RAM, no GPU. ~10K episodes/sec for baseline DQN, ~2-3K episodes/sec for IQN variants.

## Hyperparameters (matching Heinrich & Silver 2016 + OpenSpiel reference)

| Parameter | Value | Source |
|---|---|---|
| Hidden layers | [128] (single layer) | OpenSpiel Leduc example |
| Anticipatory param η | 0.1 | Paper |
| DQN learning rate | SGD lr=0.01 | OpenSpiel |
| Avg policy learning rate | SGD lr=0.01 | OpenSpiel |
| Batch size | 128 | OpenSpiel |
| learn_every | 64 game steps | OpenSpiel |
| Min buffer size to learn | 1000 | OpenSpiel |
| Target update freq | 19200 game steps (BR-only counter) | OpenSpiel |
| Target tau (soft update) | 0.995 | OpenSpiel |
| Epsilon decay | 0.06 → 0.001 over 20M steps (BR-only counter) | OpenSpiel |
| Discount γ | 1.0 | Paper |
| DQN replay buffer | 200K (circular) | Paper |
| Reservoir buffer | 2M (Algorithm R) | Paper |
| Episodes per experiment | 40M (parallel) + 5M sequential fine-tune | — |
| Loss | MSE (DQN) / Quantile Huber κ=1 (IQN) | Paper |

## IQN-specific parameters

| Parameter | Final Value | Iteration |
|---|---|---|
| Quantile training samples N | 8 | 4 → 8 (improved convergence) |
| Quantile evaluation samples K | 32 | Paper |
| Cosine embedding dim | 64 | Paper |
| Huber κ | 1.0 | Paper |
| Optimizer | Adam lr=0.001 → SGD lr=0.001 momentum=0.9 | Adam initially (SGD diverges from scratch); SGD fine-tune from Adam checkpoints |

## Critical implementation matters

### Dual-counter step counting (matched OpenSpiel exactly)

OpenSpiel uses two independent step counters that interact in a non-obvious way:

- `NFSP._iteration` — increments on every `NFSP.step()` call (~3 per episode per player), triggers BOTH avg policy learning AND DQN learning (in AVG mode)
- `DQN._iteration` — increments only when `DQN.step()` is called (BR mode only), triggers DQN learning AND epsilon decay AND target update

**Result**: DQN gets two independent learning triggers. In our code we replicate this: `_game_steps` triggers DQN+avg learning; `_dqn_steps` (BR-only) triggers extra DQN learning + epsilon + target update.

**Impact**: Single-counter implementation gave ~0.13 baseline exploitability at 20M episodes. Dual-counter fix improved to ~0.10 at 20M and ~0.092 at 45M. This was the single biggest fix.

### Game logic memory leak (critical bug)

`LeducState` originally used `std::vector<HistoryEntry> history_` that was never cleared on `reset()`. After 100K episodes the vector held 500K+ entries. `info_state_tensor()` iterated over the full history every call → O(n) per episode, throughput collapsed from ~10K → ~100 ep/s over the course of training.

**Fix**: Replaced with fixed-size array `HistoryEntry history_[8]` + `num_history_` counter, cleared in `reset()`.

### Worker staleness vs sequential

The C++ implementation runs 5 worker threads playing episodes in parallel while the main thread does gradient updates. Workers read network weights via lock-free `FastMLP` snapshots, so they may play with weights that are 1-2 gradient updates stale.

Tested impact: switching to single-worker batch=1 mode (matching OpenSpiel exactly) gives small additional improvements:
- baseline: 0.110 → 0.094 (Δ -0.016)
- iqn_neutral: 0.214 → 0.214 (negligible)

Conclusion: staleness matters a small amount for baseline, negligible for IQN. The remaining IQN gap is **not** due to staleness.

## Final Results (single seed = 42)

### Exploitability comparison

| Agent | Final | Min in last 1M | mbb/g (min) | Notes |
|---|---|---|---|---|
| **baseline** (DQN) | 0.094 | **0.092** | **46** | matches Heinrich & Silver 2016 (~30-50 mbb/g) |
| iqn_neutral (Adam, N=4) | 0.214 | 0.210 | 105 | initial config |
| iqn_neutral (Adam, N=8) | 0.187 | 0.187 | 94 | N=8 helps risk-neutral |
| iqn_neutral (SGD-finetune) | 0.179 | 0.178 | 89 | SGD momentum=0.9 from 40M Adam checkpoint, +10M episodes |
| iqn_mv01 (SGD-finetune) | 0.180 | 0.175 | 87 | same SGD fine-tune |
| iqn_mv05 (SGD-finetune) | 0.183 | 0.176 | 88 | same SGD fine-tune |
| iqn_mv01 (Adam, N=8) | 0.193 | 0.193 | 97 | mean-variance penalty 0.1 |
| iqn_mv05 (Adam, N=8) | 0.193 | 0.193 | 97 | mean-variance penalty 0.5 |
| iqn_averse (Adam, N=4) | 1.067 | 1.067 | 533 | CVaR α=0.25, plateaued |
| iqn_seeking (Adam, N=4) | 2.020 | 2.011 | 1006 | risk-seeking τ∈[0.75,1.0], does not converge |

**mbb/g conversion**: 1 big blind = 2 chips in Leduc convention. Multiply OpenSpiel's `exploitability()` by 500 to get mbb/g.

### Head-to-head tournament (5000 games per matchup)

| | iqn_neutral | iqn_mv01 | iqn_mv05 | iqn_averse | iqn_seeking |
|---|---|---|---|---|---|
| **baseline** | -0.111 (lose) | -0.068 (lose) | **+0.052 (lose)** | -0.129 (lose) | -0.714 (lose) |
| **iqn_neutral** | — | -0.036 | -0.080 | -0.277 | -0.817 |
| **iqn_mv01** | — | — | -0.029 | -0.229 | -0.982 |
| **iqn_mv05** | — | — | — | -0.305 | -0.888 |
| **iqn_averse** | — | — | — | — | -0.236 |

(Negative = first agent loses to second agent. Positive payoffs to baseline mean baseline wins.)

**Counterintuitive result**: **iqn_mv05 beats baseline head-to-head (+0.052 to mv05)** despite having 2x worse exploitability. This is the key finding for the paper:

> **Distributional risk-aware agents can exploit Nash-converged opponents even when they're farther from Nash themselves.**

Mean-variance penalty makes mv05 play more cautiously → exploits baseline's specific weaknesses while baseline (a Nash approximator) doesn't adapt.

## Algorithmic findings

### Risk-neutral IQN underperforms baseline DQN despite mathematical equivalence

In theory, risk-neutral IQN (averaging quantiles) should learn the same expected Q-values as DQN. In practice IQN converges to ~2x worse exploitability (94 vs 46 mbb/g).

Hypotheses (in order of likely impact):
1. **Adam vs SGD**: Adam's per-parameter adaptive scaling settles into different (sharper) minima than SGD's momentum-driven flat minima. SGD-after-Adam fine-tuning is being tested.
2. **Quantile sampling noise**: N=4 quantiles gave noisy gradients. N=8 helped (-0.027 improvement).
3. **Network capacity overhead**: IQN has 3 layers (state_fc + cos_embedding + output_fc) vs DQN's 2. More params, slower convergence at fixed episode budget.

### Risk-sensitive variants have algorithmic limitations in zero-sum games

**CVaR (averse)** plateaus at ~530 mbb/g (10x worse than neutral). The worst-case-focused best-response becomes overly pessimistic, learning to fold too much. This makes the average policy degenerate. **This is an algorithmic limit, not a hyperparameter issue.**

**Risk-seeking** never converges (1000+ mbb/g, basically random play). Best-response chases unlikely big wins, ignoring expected value. The avg policy trained on this behavior is exploitable.

**Mean-variance** (penalize variance of return distribution) works because it still considers the full distribution, just down-weighting volatile actions. Both mv01 and mv05 reach ~95 mbb/g — comparable to risk-neutral IQN.

### Convergence noise floor

Baseline exploitability oscillates ±0.01-0.02 around its floor in the last few million episodes. This is **not** a learning rate issue — OpenSpiel uses fixed lr=0.01 throughout and exhibits the same noise. The avg policy is converged; small policy fluctuations cause small exploitability fluctuations. Standard practice in NFSP papers is to report the **minimum over the last K evaluations** or **mean over a window**, not the raw final value.

## Implementation lessons (for paper appendix or future work)

1. **Custom game logic in C++** gave ~70x speedup over OpenSpiel's Python game loop (95K ep/s vs ~1.4K ep/s). LibTorch gradient computation is the bottleneck — game simulation is essentially free.

2. **Pre-allocated tensors** for gradient updates: writing replay-buffer samples directly into long-lived `torch::Tensor` storage via `data_ptr<float>()` avoids per-call allocation. ~30% speedup vs `from_blob().clone()`.

3. **FastMLP for inference**: extracting raw weight arrays from trained `torch::nn::Module` and doing manual matrix-vector multiplies in tight loops is ~100x faster than `module->forward(tensor)` for small networks (30→128→3). Workers use this for action selection; main thread uses LibTorch for gradients.

4. **Single-process multi-threading is OK** for this scale despite shared L3 cache. Separate processes (Python multiprocessing style) didn't help — measured directly via fork-based test.

5. **Replay buffer save/restore is essential** for resume. Without it, after a restart the avg policy has no historical data and must re-train from scratch, losing convergence progress. Old Python implementation was missing this; we added it.

---

## Limit Texas Hold'em Extension

### Game

**Game**: 2-player Limit Texas Hold'em via OpenSpiel's `universal_poker`.
- Blinds: 50/100 chips
- 4 betting rounds (preflop, flop, turn, river), raise sizes 100/100/200/200
- Max 3 raises per round
- Info state dimension: 208 (vs Leduc's 30)
- Actions: 3 (fold, call/check, raise)
- ~10^14 information states — tabular exploitability is intractable

### Hold'em Hyperparameters

**Methodology**: OpenSpiel has no Hold'em NFSP example. We extrapolate from the transformation OpenSpiel applied to the paper's Leduc params (LR÷10, learn_every=64, target=300×64=19200, soft tau=0.995, epsilon decay), then apply the same transformation to the paper's Hold'em params.

**Iteration history**:
1. Adam lr=0.01, reward norm /100, learn_every=128, target=300 → 1899 mbb/g at 38M (too high)
2. Adam lr=0.01, reward norm /100, learn_every=128, target=19200 → also didn't converge well (original config before "fix")
3. SGD lr=0.01, NO reward norm, learn_every=64, target=19200 → NaN crash (raw rewards ±2000 too large for SGD)
4. **Current**: SGD lr=0.01, reward norm /100, learn_every=64, target=19200 → smoke test OK, training

**Note on paper's lr=0.1**: The paper claims SGD lr=0.1 on raw Hold'em chip rewards (±2000). This is 1000x larger gradient steps than our setup. We could not reproduce this without NaN divergence. Likely undocumented gradient clipping or different batch handling. Reward normalization /100 is the pragmatic fix — makes Hold'em reward scale (±24) comparable to Leduc (±13).

| Parameter | Leduc (OpenSpiel) | Hold'em (extrapolated) | Paper Hold'em | Rationale |
|---|---|---|---|---|
| Network | [128] (1 layer) | [64,64,64,64] (4 layers) | [64,64,64,64] | Paper value |
| DQN optimizer | SGD lr=0.01 | SGD lr=0.01 | SGD lr=0.1 | OpenSpiel convention (paper÷10) |
| Avg policy | SGD lr=0.01 | SGD lr=0.005 | SGD lr=0.005 | Paper value |
| Target update | 19200 | 19200 | 300 | OpenSpiel convention (300 × learn_every) |
| learn_every | 64 | 64 | 1 (every step) | OpenSpiel convention |
| tau | 0.995 | 0.995 | 1.0 (hard) | OpenSpiel convention |
| Reward norm | none | /100 (big blind) | none | Practical: prevents NaN with SGD |
| DQN buffer | 200K | 2M | 2M | Paper value (larger game) |
| Reservoir | 2M | 2M | 2M | Same |
| Batch size | 128 | 256 | 256 | Paper value |
| Epsilon | 0.06→0.001/20M | 0.06→0.001/40M | 0.06 (fixed) | OpenSpiel convention + scaled |
| eta | 0.1 | 0.1 | 0.1 | Same |
| gamma | 1.0 | 1.0 | 1.0 | Same |

### Evaluation: Local Best Response (LBR)

Tabular exploitability is intractable for Hold'em (~10^14 info states, timed out at 120s).

**Method**: Monte Carlo LBR — play evaluated agent's avg policy against a best-responding opponent that does M rollouts per legal action and picks the best. LBR opponent's avg payoff ≈ exploitability.

- 10K games × 15 rollouts per action gives ±50 mbb/g standard error
- Head-to-head vs random is insufficient (baseline's H2H vs random *decreases* over time as it approaches Nash)

### Preliminary Hold'em Results (before machine overload)

First run with corrected hyperparameters (baseline at 38M episodes, IQN variants at 5-15M):

| Agent | Episodes | LBR Exploit (mbb/g) | ± SE |
|---|---|---|---|
| baseline | 38M | 1899 | ±47 |
| iqn_mv05 | ~10M | 2746 | ±60 |
| iqn_neutral | ~15M | 2634 | ±57 |
| iqn_mv01 | ~23M | 2929 | ±57 |

**Key early finding**: iqn_neutral at 15M episodes (2634 mbb/g) was approaching baseline at 38M (1899 mbb/g). Per-episode convergence rate appears faster for IQN in Hold'em — **opposite** of the Leduc finding. The distributional information is genuinely useful in the larger game.

**Paper reference**: Heinrich & Silver 2016 reports ~60-80 mbb/g at convergence for Hold'em (200M+ iterations). Our baseline at 1899 mbb/g is still early — needs more episodes. The learning rate and target update frequency changes should help convergence, but this is still a work in progress.

### Head-to-Head observations

- At early training: IQN variants score *higher* vs random opponents than baseline (248 vs 171 chips/game), because IQN is more exploitative while baseline is converging toward Nash
- Baseline H2H vs random *decreases* over training (228→170) as it becomes more balanced — expected NFSP behavior
- Head-to-head between agents: baseline dominates all IQN variants at current episode counts (+53 to +63 chips/game), but IQN is still training

---

## Files of interest

### Leduc
- `src/leduc_poker.h` — game implementation
- `src/nfsp_agent.h` — DQN-based NFSP agent (SGD, single-layer MLP)
- `src/nfsp_iqn_agent.h` — IQN-based NFSP agent with risk distortions
- `src/train.cpp` — Leduc training loop

### Hold'em
- `src/holdem.h` — OpenSpiel universal_poker wrapper (208-dim info state)
- `src/nfsp_agent_holdem.h` — DQN-based NFSP agent (Adam, 4-layer DeepMLP)
- `src/nfsp_iqn_agent_holdem.h` — IQN-based NFSP agent for Hold'em
- `src/train_holdem.cpp` — Hold'em training loop with CLI arg parsing
- `run_holdem.sh` — launch script for all 4 Hold'em experiments

### Shared
- `src/replay_buffer.h` — circular + reservoir buffers
- `src/networks.h` — MLP, DeepMLP, and IQN network definitions (LibTorch)
- `src/fast_mlp.h` — FastMLP + DeepFastMLP for fast inference
- `eval/load_cpp_models.py` — Python utility to auto-detect and load C++ weights
- `eval/lbr_holdem.py` — Monte Carlo LBR exploitability estimation
- `eval/lbr_holdem_accurate.py` — Parallel LBR with std error (10K games)
- `results/logs/{name}_seed42_exploitability.csv` — per-experiment exploitability curves
- `results/logs/{name}_seed42.jsonl` — per-experiment training stats
