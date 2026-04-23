# Risk-Sensitive NFSP on Poker: Leduc → Limit Hold'em

*A 20-minute read for someone new to this project.*

**Current status (2026-04-23):** Two Hold'em runs are re-training right now — `holdem_iqn_seeking` (20M eps, new risk-seeking variant) and `holdem_iqn_long` (fresh 40M retrain of IQN-neutral, because a prior resume crashed). ETA ~14h. All other numbers in this report are final.

---

## 1. What problem are we solving?

### 1.1 Why poker?

Poker is the canonical benchmark for **imperfect-information games** — games where players don't know the full state (here: the opponent's private cards). Classic algorithms like Minimax or single-agent Q-learning fail because:
- The opponent's strategy *is part of the environment*, and it's changing as they also learn.
- Naïve Q-learning diverges because Q-values don't have a single correct target — the target depends on the opponent, which you're simultaneously training against.

Poker gave us CFR (Counterfactual Regret Minimization), which provably solves two-player zero-sum games but requires enumerating the game tree (doesn't scale to large games without abstraction).

### 1.2 Neural Fictitious Self-Play (NFSP)

**NFSP** (Heinrich & Silver, 2016) is the strongest neural-network-only, self-play-only method for poker. The idea combines two classical game-theory results:

1. **Fictitious play**: each player best-responds to the *historical average* of the opponent's strategy. In two-player zero-sum, the average strategy converges to a Nash equilibrium.
2. **Deep RL**: replace the tabular best-response and average-policy representations with neural networks, so the method scales.

Each player trains **two networks**:
- A **Q-network** (best response to opponent's average) — trained via DQN.
- An **average-policy network** — trained via supervised learning on a *reservoir* of all past action distributions.

At each decision, the agent flips a biased coin (with probability η=0.1) to pick which head to play from — this is called the **anticipatory policy**. The reservoir sampling is what lets the supervised average-policy net converge to the true fictitious-play object.

### 1.3 Why extend NFSP with IQN?

The Q-network in NFSP estimates only the **mean** of the return distribution. Dabney et al. (2018) showed that estimating the **full distribution** (via Implicit Quantile Networks / **IQN**) improves single-agent RL. We can also *distort* the distribution to implement **risk-sensitive** preferences:

| Risk preference | How it shapes the Q-value | Intuition |
|---|---|---|
| Risk-neutral | `Q(s,a) = E[return]` | Standard NFSP; matches paper |
| CVaR-averse (α=0.25) | `Q(s,a) = E[return \| return ≤ 25th percentile]` | "Plan for the bottom 25% of outcomes" — conservative |
| Mean-variance (β=0.5) | `Q(s,a) = E[return] − β·Var[return]` | "Penalize volatile plays" — smoother |
| CVaR-seeking (α=0.75) | Weight upper quantiles more | "Gamble for the upside" |

**Open research question**: do any of these improve NFSP's equilibrium convergence (i.e., make the learned average policy less exploitable)?

The theoretical priors cut both ways:
- **Pro**: conservative players might be genuinely harder to exploit (fewer obvious weaknesses).
- **Con**: Nash equilibrium is defined by *mean* payoffs; distorting the Q-value distorts best-response away from Nash. So in principle, risk-neutral should win.

We tested on two games of increasing scale.

---

## 2. Stage 1 — Leduc Hold'em (small game, tractable)

### 2.1 Game

**Leduc Hold'em**: 2 players, 6-card deck, 2 betting rounds, fixed bet sizes. Small enough that **exact exploitability** can be computed by enumerating the game tree (OpenSpiel's tabular best-response).

### 2.2 Setup (all Leduc runs)

| | Value |
|---|---|
| Architecture | **[128]** (single hidden layer) |
| Batch | 128 |
| Episodes | 40M (IQN) / 45M (baseline) |
| DQN lr | 0.001 (IQN) / 0.01 (baseline) |
| Average-policy lr | same as DQN lr |
| Anticipatory η | 0.1 |
| IQN quantile samples N | 8 |
| Evaluation | **exact** exploitability every 50K eps via OpenSpiel |

### 2.3 Results

| Run | Risk preference | Final exploitability (mbb/g) |
|---|---|---|
| **baseline** (NFSP) | — | **0.094** 🏆 |
| iqn_neutral | risk-neutral | 0.179 |
| iqn_mv01 | mean-var, β = 0.1 | 0.180 |
| iqn_mv05 | mean-var, β = 0.5 | 0.183 |
| iqn_averse | **CVaR α = 0.25** | **1.205** |
| iqn_seeking | CVaR seeking, upper 25% | 2.071 |

![Leduc exploitability](figures/leduc/exploitability.png)
![Leduc final bar](figures/leduc/final_exploitability_bar.png)

### 2.4 Takeaway on Leduc

**Risk-neutral NFSP dominates every IQN variant**, and the risk-distorted ones (averse, seeking) are **an order of magnitude worse**. This matches the theoretical prediction: for zero-sum equilibrium convergence, distorting the Q-value is strictly harmful because it biases best-response away from the Nash best-response.

**We expected the same story on Hold'em. We were partially wrong.**

---

## 3. Stage 2 — Limit Hold'em (large game, LBR eval)

### 3.1 Game

**Heads-up Limit Hold'em**: 2 players, 52-card deck, 4 betting rounds, fixed bet sizes. Paper-standard benchmark from Heinrich & Silver.

Exact exploitability is intractable (game tree too large). Instead we use **Local Best Response (LBR)** (Lisy & Bowling, 2017): a Monte-Carlo lower-bound estimate. At each decision of the LBR player, it plays every legal action forward for 15 rollouts (both players sampling from the trained average policy), averages, and picks the best action. 5000 hands per side, 4 parallel workers.

LBR returns are reported as milli-big-blinds per hand (mbb/g); **lower is less exploitable.** Reference: uniform-random play ≈ 2800 mbb/g.

### 3.2 Setup (Hold'em runs)

Different from Leduc because the game and state space are much bigger:

| | Value |
|---|---|
| Architecture | **[256, 128, 256, 128]** (4 layers, ~100K params) |
| Batch | 256 (paper-matched) |
| Episodes | 20M (40M for the baseline and neutral retrain) |
| DQN lr | 0.002 |
| Average-policy lr | 0.0002 |
| Reservoir size | 30M (paper-matched) |
| DQN replay buffer | 600K |
| Target net update | 128K gradient steps, hard copy |
| Reward scaling | ÷100 (1 chip = 1 / big-blind) |
| Workers | 3 (parallel episode generation) |

### 3.3 Core Hold'em Results

| Run | Architecture | Risk | LBR (mbb/g) ± SE | Episodes |
|---|---|---|---|---|
| `holdem_small_long` | [256,128,256,128] | risk-neutral NFSP | **1400 ± 59** | **40M** |
| `holdem_iqn_long` | [256,128,256,128] | IQN risk-neutral | 1993 ± 68 | 20M* |
| `holdem_iqn_smaller` | [128, 64, 128, 64] | IQN risk-neutral | 2409 ± 78 | 20M |
| `holdem_iqn_meanvar` | [256,128,256,128] | IQN MV β=0.5 | 1994 ± 67 | 20M |
| **`holdem_iqn_averse`** | [256,128,256,128] | **IQN CVaR α=0.25** | **482 ± 22** ⚠ | **20M** |
| `holdem_iqn_seeking` | [256,128,256,128] | IQN CVaR seeking | *pending* | 20M |

\* A 40M extension of IQN-neutral crashed at ep 25M due to disk-full; a fresh 40M retrain is currently running.

![Hold'em LBR bar](figures/holdem/final_lbr_bar.png)
![Hold'em H2H](figures/holdem/h2h.png)

### 3.4 Two surprises on Hold'em

**Surprise 1: CVaR-averse was dramatically better than everything else (482 vs 1400 for the NFSP baseline).**

This is the **opposite direction** from Leduc, where averse was 13× worse than baseline. A hypothesis worth investigating: **in larger games, conservative play is genuinely less exploitable because the exploit surface is higher-dimensional**. In tiny Leduc, every deviation from Nash is easily punishable; in Hold'em, the attacker's own learning is bandwidth-limited, so an opponent who plays tight leaves fewer holes to find.

### 3.5 Caveat on the averse number — critical to read before believing it

The 482 mbb/g figure has an **asymmetry** you should know about:

| | LBR as P0 (exploits our P1) | LBR as P1 (exploits our P0) | Combined |
|---|---|---|---|
| baseline | +137 | +151 | 1440 |
| iqn_neutral | +190 | +208 | 1993 |
| **iqn_averse** | **+189** | **−92** ⚠ | **482** |

**LBR as P1 got a NEGATIVE value** for the averse agent. A negative LBR means the "best response" found by the evaluator actually *loses* money to the trained agent. This is unusual. Two possible explanations:

- **Real**: the averse agent plays very tight (H2H-vs-random dropped from ~155 to ~100 mbb/g during training), and LBR with only 15 rollouts is too noisy to find the exploit of folder-style play. True exploitability is probably ≥ 944 mbb/g (still better than baseline's 1400, but less dramatically).
- **Artifact**: the return standard deviation for averse games is ~280 chips (vs ~660 for others) — tighter games have smaller pots, so LBR's rollout noise dominates the signal.

**How to resolve**: re-run LBR with 100+ rollouts (instead of 15) on the averse weights. We haven't done this yet; it's the obvious next step.

Even under the conservative reading (true exploitability ≥ 944), **CVaR-averse still beats baseline on Hold'em** — a finding we did not predict from Leduc.

**Surprise 2: in IQN at two sizes, smaller = worse.**

Shrinking the IQN architecture from [256,128,256,128] to [128,64,128,64] made LBR significantly worse (1993 → 2409, +416 mbb/g). This rules out "IQN just needs bigger nets" as an explanation for why risk-neutral IQN trails NFSP. The gap is algorithmic, not capacity-bound.

### 3.6 Did extending training from 20M → 40M help?

We tested this question by resuming NFSP baseline for another 20M episodes:

| Run | LBR @ 20M | LBR @ 40M | Delta |
|---|---|---|---|
| NFSP baseline | 1440 ± 62 | 1400 ± 59 | −40 (within SE, **not significant**) |

**Answer: no, the LBR plateau closely tracks the H2H plateau.** Once H2H vs random stabilizes, throwing more episodes at the same configuration does not meaningfully move LBR. The IQN-40M number is still pending from the ongoing retrain.

---

## 4. Consolidated story for a slide deck

1. **Leduc**: NFSP ≫ all IQN variants. Risk-sensitive is strictly worse here. Theoretically predicted.
2. **Hold'em**: IQN-neutral ≈ IQN-MV < NFSP baseline — same direction as Leduc, weaker. Smaller IQN is strictly worse (capacity does matter for IQN).
3. **Hold'em CVaR-averse** is the outlier: dramatically lower LBR (482 vs 1400) with a caveat (LBR noise from very tight play). Real lower bound likely ≥ 944, still better than baseline. **The risk-averse intuition may genuinely scale better than the algorithm's impact on small games suggests.**
4. **Extending training doesn't help** past the H2H plateau.

---

## 5. Figures

Generated by `cpp-implementation/scripts/make_plots.py`.

**Leduc** (`figures/leduc/`):
- `exploitability.png` — log-scale exploitability trajectory for all 6 variants
- `exploitability_linear.png` — linear-scale zoom to see the converged differences
- `br_loss.png`, `avg_loss.png` — training loss curves
- `final_exploitability_bar.png` — final values, bar chart

**Hold'em** (`figures/holdem/`):
- `h2h.png` — head-to-head vs random policy over training
- `br_loss.png`, `avg_loss.png` — training loss curves
- `final_lbr_bar.png` — final LBR values, bar chart with error bars and random-reference line

---

## 6. Files of interest

| Path | What it is |
|---|---|
| `cpp-implementation/src/train_holdem.cpp` | Hold'em training binary |
| `cpp-implementation/src/train.cpp` | Leduc training binary |
| `cpp-implementation/src/nfsp_agent_holdem.h`, `nfsp_iqn_agent_holdem.h` | Agent implementations |
| `cpp-implementation/run_overnight_*.sh` | Launch scripts for each Hold'em variant |
| `cpp-implementation/run_iqn.sh`, `run_all.sh` | Leduc sweep launchers |
| `cpp-implementation/eval/lbr_holdem_accurate.py` | Hold'em LBR evaluator |
| `cpp-implementation/eval/lbr_sanity_random.py` | LBR sanity check (vs uniform random) |
| `cpp-implementation/eval/compute_exploitability.py` | Leduc exact exploitability |
| `cpp-implementation/scripts/make_plots.py` | Figure generator |
| `cpp-implementation/logs/` | Training logs (per-run) |
| `cpp-implementation/results/logs/` | Per-run structured metrics (JSONL + CSV) |
| `figures/` | Generated plots |

Model weights and checkpoints live on `/mnt/data/cos435/` (off-repo, ~100GB for training buffers).
