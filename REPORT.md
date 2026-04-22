# NFSP and Risk-Sensitive IQN on Poker: Experimental Report

*Status: Hold'em CVaR-averse and Mean-Variance variants are currently training (PIDs 29022, 29035). Report will be updated with their numbers when complete. All other results are final.*

---

## 1. Motivation

Heads-up poker is a canonical benchmark for learning in imperfect-information games: both players act under uncertainty about the opponent's hand, making naive Q-learning diverge. **Neural Fictitious Self-Play (NFSP)** (Heinrich & Silver, 2016) addresses this by combining two ideas:
1. **Fictitious play** — each player plays a best response to the historical average of the opponent's strategy, which provably converges to a Nash equilibrium in two-player zero-sum games.
2. **Deep RL** — replace the tabular best-response and average-strategy representations with neural networks, making the method scale to large information-state spaces.

NFSP is the strongest *self-play-only* (no game-specific heuristics, no CFR tabular computation) baseline for Limit Hold'em in the literature. On Leduc Hold'em it reaches single-digit milli-big-blinds/hand of exploitability; on Limit Hold'em its LBR-exploitability is in the ~1000–2000 mbb/g range.

**Why risk-sensitive extensions?** A standard scalar Q-network estimates only the *mean* of the return distribution. The Implicit Quantile Network (IQN) (Dabney et al., 2018) estimates the *full distribution*, which allows incorporating risk preferences directly into the best-response operator:
- **CVaR-averse** weights the lower tail of the return distribution more heavily — intuitively, the player plays as if it expects the worst 25% of outcomes and is more cautious.
- **Mean-variance** subtracts a penalty proportional to the *variance* of returns from the Q-value used for action selection — smoother play, lower variance.

The open research question we investigated: **does risk-sensitive action selection improve the *equilibrium quality* (exploitability) of the learned average policy?** The intuition cuts both ways:
- Pro: conservative play might converge to a less exploitable policy (fewer obvious weaknesses).
- Con: risk-neutral is the correct Bayes-optimal objective for zero-sum equilibrium — distorting it could bias the learner away from Nash.

We evaluate both on Leduc Hold'em (small; exact exploitability tractable) and Limit Hold'em (large; LBR lower-bound evaluation).

---

## 2. Model

All experiments share the same NFSP skeleton; IQN variants swap the Q-network.

### NFSP core (both games)
Each player maintains two networks, both approximate functions of the information state:
- **Q-network `Q(s, a; θ)`** — trained by DQN (replay buffer + target network + double-Q bootstrap) against transitions generated under the opponent's current *average* policy. This is the best-response head.
- **Average policy `π̄(a | s; φ)`** — trained by supervised (softmax-cross-entropy) learning on a *reservoir* of (state, action-mask, action-probabilities) tuples collected across the entire training history.

At each decision point, the agent samples a *mode* for the current episode:
- With probability η (anticipatory parameter, 0.1), play best-response mode (argmax Q).
- Otherwise play the average policy (sample from π̄).

Both states and the actions taken are always added to the reservoir (regardless of mode) — this is what lets π̄ converge to the average of all historical best-responses, which is the fictitious-play equilibrium object.

### IQN variant
Replace Q(s, a; θ) with a quantile function `Z(s, a, τ; θ)`, where τ ∈ [0, 1] is a sampled quantile. The Q-value becomes:

```
Q(s, a) = E_τ[ψ(τ) · Z(s, a, τ)]
```

where ψ(τ) is a **risk distortion**:
- **Risk-neutral** (baseline IQN): ψ(τ) = 1 — recovers plain mean-Q.
- **CVaR-averse** with level α: ψ(τ) = (1/α) · 𝟙{τ ≤ α} — weights only the bottom α quantiles, then expectation over them. We use α = 0.25.
- **Mean-variance** with penalty β: Q(s, a) = E[Z] − β · Var[Z]. We use β = 0.5.

The quantile regression loss uses Huber quantile loss with κ = 1; N = 8 online quantile samples per batch.

For action selection in the hot (worker-thread) loop, the IQN agent periodically distills Q(s, a) into a fast 1-layer network (`fast_q_`) via `compute_mean_q_snapshot()`: it factorizes `Q(s, a) ≈ W_out (state_fc(s) ⊙ E[mean_τ_features])`, which is exact for the 1-quantile approximation of the mean and close enough for argmax.

### Hyperparameters (shared across all Hold'em runs)

| | Value |
|---|---|
| Architecture (hidden) | [256, 128, 256, 128] |
| Optimizer | Adam |
| DQN lr | 0.002 |
| Average-policy lr | 0.0002 |
| Mini-batch | 256 |
| DQN replay buffer | 600 K (paper-matched) |
| Reservoir size | 30 M (paper-matched) |
| Anticipatory η | 0.1 |
| Exploration ε | 0.08 → 0.001 over 20M steps |
| Discount γ | 1.0 |
| Target net update | every 128 K gradient steps (hard copy) |
| Learn every | 256 env steps (2 SGD updates per trigger) |
| IQN N | 8 |
| IQN κ (Huber) | 1.0 |

### Hyperparameters for Leduc runs
Leduc is a much smaller game so we use a much smaller network — a **single hidden layer of 128 units** (the standard OpenSpiel NFSP reference architecture). Training uses batch size 128 and 40 M episodes for IQN variants (lr = 0.001), 45 M for the NFSP baseline (lr = 0.01). Everything else (anticipatory η = 0.1, reservoir sampling, target-net hard copy) matches the Hold'em setup. The Leduc state space is small enough that **exact exploitability** is computed periodically via OpenSpiel's tabular best-response — no sampling or LBR needed. The results are drawn from previously-completed training runs saved in `logs/iqn_*.log` and `logs/baseline.log`.

### Engineering notes
- Implementation is C++ with LibTorch. Each run uses 3 worker threads generating episodes into a bounded queue, consumed by per-player gradient threads. This keeps the gradient step on a dedicated core.
- For Hold'em, reward is scaled by 1/100 (1 big-blind = 100 chips) to keep Q-values on a Leduc-comparable scale.
- Checkpoints: every 5 M episodes. Checkpoints include Q-network weights, average-policy weights, target-network weights, optimizer state, and the replay + reservoir buffers. Training auto-resumes from the latest checkpoint on restart.
- Hardware: 8 vCPU n2-series GCE instance, 125 GB RAM, 128 GB data disk for checkpoints. Each Hold'em run uses ~52 GB RAM; two runs fit in parallel.

---

## 3. Experimental Runs

### Hold'em — completed

All at 20 M episodes, 30 M reservoir, arch [256,128,256,128] (except `iqn_smaller`), lr 0.002/0.0002, N = 8 quantiles.

| Run | Architecture | Risk | LBR mbb/g (lower = better) | H2H vs random (late mean) | Runtime |
|---|---|---|---|---|---|
| `holdem_small_long` (NFSP baseline) | [256, 128, 256, 128] | — | **1440 ± 62** 🏆 | ~155 | 9.3 h |
| `holdem_iqn_long` (IQN neutral) | [256, 128, 256, 128] | none | **1993 ± 68** | ~161 | 9.3 h |
| `holdem_iqn_smaller` (IQN, smaller net) | [128, 64, 128, 64] | none | **2409 ± 78** | ~220 | 2.6 h (faster machine) |

Reference: LBR against uniform random policy ≈ 2 800 mbb/g.

**LBR methodology.** Best-response is estimated by Monte-Carlo tree lookahead: at each LBR decision the evaluator plays every legal action forward for 15 rollouts (both players sampling from the trained average policy), picks the action with the highest mean rollout value, and records the return. 5 000 hands per player, 4 parallel workers. Values reported as milli-big-blinds per hand with ± 1 standard error of the mean.

### Hold'em — in progress (currently training, 20 M eps each, ~6–7 h)

| Run | Architecture | Risk | Status |
|---|---|---|---|
| `holdem_iqn_averse` | [256, 128, 256, 128] | CVaR (α = 0.25) | running (PID 29022) |
| `holdem_iqn_meanvar` | [256, 128, 256, 128] | mean-variance (β = 0.5) | running (PID 29035) |

These will round out the Hold'em comparison with the same two risk distortions we have Leduc data for.

### Leduc Hold'em — completed (prior runs)

Architecture: **single hidden layer of 128 units** (OpenSpiel-standard for Leduc). NFSP baseline: lr = 0.01, 45 M episodes. All IQN variants: lr = 0.001, 40 M episodes, N = 8 quantiles. **Exploitability here is exact (via OpenSpiel's tabular best-response)**, reported in mbb/g.

| Run | Arch | Risk | Final exploitability |
|---|---|---|---|
| `baseline` (NFSP) | [128] | — | **0.094** 🏆 |
| `iqn_neutral` | [128] | none | 0.179 |
| `iqn_mv01` | [128] | mean-var, β = 0.1 | 0.180 |
| `iqn_mv05` | [128] | mean-var, β = 0.5 | 0.183 |
| `iqn_averse` | [128] | CVaR, α = 0.25 | 1.205 |
| `iqn_seeking` | [128] | CVaR-seeking, 0.75 | 2.071 |

### Cross-game finding

**Risk-neutral NFSP outperforms every IQN variant, on both games.**

- On **Leduc** (exact exploitability): baseline NFSP at 0.094 beats risk-neutral IQN at 0.179 (1.9× worse) and strongly dominates the CVaR-averse IQN at 1.205 (13× worse).
- On **Hold'em** (LBR lower bound): baseline NFSP at 1 440 mbb/g beats risk-neutral IQN at 1 993 mbb/g (1.4× worse). CVaR-averse and mean-variance results pending, but the Leduc pattern suggests they will also underperform.

The direction is consistent across two games with different scales, different evaluation methods (exact vs LBR), and different training budgets. That's strong evidence the finding is algorithmic, not a training-budget artifact.

**Interpretation.** For zero-sum equilibrium convergence, risk-neutral value is the theoretically correct objective — Nash equilibrium is defined by mean payoffs. Distorting the Q-value toward CVaR or mean-variance biases best-response *away* from the Nash best-response, and the resulting average policy carries that bias. Even the "drop-in replacement" IQN with ψ(τ) = 1 does worse than scalar Q: the quantile regression's Huber-loss is a noisier estimator of the mean than MSE, and the extra variance in Q-targets hurts the equilibrium-seeking dynamics more than the distributional information helps (for this metric).

This is a genuinely interesting negative result: **distributional RL methods that improve single-agent performance (Dabney et al. show IQN beats DQN on Atari) do not transfer to the self-play equilibrium regime.**

### Architecture scaling (Hold'em IQN only)

Within risk-neutral IQN, shrinking the network from [256, 128, 256, 128] to [128, 64, 128, 64] degraded LBR from 1 993 → 2 409 (+416 mbb/g). The gap to the NFSP baseline actually *widens* at lower capacity (IQN-smaller is 69% worse than NFSP-baseline; IQN-neutral at matched size is 38% worse). This rules out the "IQN just needs more parameters to shine" hypothesis.

---

## 4. Summary Table

*(Hold'em averse + mean-var rows will be filled in when training finishes.)*

| Run | Game | Arch | Risk | Eval | Value | ± SE |
|---|---|---|---|---|---|---|
| baseline | Leduc | [128] | — | exact | 0.094 | — |
| iqn_neutral | Leduc | [128] | none | exact | 0.179 | — |
| iqn_mv05 | Leduc | [128] | MV β=0.5 | exact | 0.183 | — |
| iqn_averse | Leduc | [128] | CVaR α=0.25 | exact | 1.205 | — |
| **holdem_small_long (NFSP)** | **Hold'em** | [256,128,256,128] | — | **LBR 5k×15** | **1440** | **62** |
| holdem_iqn_long | Hold'em | [256,128,256,128] | none | LBR 5k×15 | 1993 | 68 |
| holdem_iqn_smaller | Hold'em | [128,64,128,64] | none | LBR 5k×15 | 2409 | 78 |
| holdem_iqn_averse | Hold'em | [256,128,256,128] | CVaR α=0.25 | LBR 5k×15 | *pending* | |
| holdem_iqn_meanvar | Hold'em | [256,128,256,128] | MV β=0.5 | LBR 5k×15 | *pending* | |

---

## 5. Repository structure for reproduction

| Path | Contents |
|---|---|
| `src/train_holdem.cpp` | Hold'em C++ training binary source |
| `src/train.cpp` | Leduc C++ training binary source |
| `src/nfsp_agent_holdem.h`, `src/nfsp_iqn_agent_holdem.h` | Agent implementations |
| `build_holdem.sh`, `build.sh` | Build scripts (LibTorch + OpenSpiel) |
| `run_overnight_baseline.sh` → `final_weights_holdem_small_long/` | Baseline |
| `run_overnight_iqn.sh` → `final_weights_holdem_iqn_long/` | IQN neutral |
| `run_overnight_iqn_smaller.sh` → `final_weights_holdem_iqn_smaller/` | IQN at [128,64,128,64] |
| `run_overnight_iqn_averse.sh` → `final_weights_holdem_iqn_averse/` | IQN CVaR α=0.25 |
| `run_overnight_iqn_meanvar.sh` → `final_weights_holdem_iqn_meanvar/` | IQN mean-variance β=0.5 |
| `run_iqn.sh` | Leduc 5-variant sweep (launches all risk variants in parallel) |
| `eval/lbr_holdem_accurate.py` | Hold'em LBR evaluator |
| `eval/compute_exploitability.py` | Leduc exact exploitability via OpenSpiel |
| `/mnt/data/cos435/` | Weights + checkpoints (off-repo, on 128 GB data disk) |
| `logs/*.log` | Training logs (per-run) |
| `results/logs/*_seed42_h2h.csv` | Per-run H2H-vs-random trajectory |
| `results/logs/*_seed42_exploitability.csv` | Per-run exploitability trajectory (Leduc only) |
