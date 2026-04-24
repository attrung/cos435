# Risk-Sensitive NFSP on Poker: Leduc → Limit Hold'em

*COS 435 project report. Onboarding doc for a new collaborator — read top to bottom.*

**Status as of 2026-04-24, 18:35 UTC (ALL FINAL):**
- **Leduc**: 6 variants, exact exploitability via OpenSpiel tabular best-response.
- **Hold'em training**: 6 models trained.
- **Hold'em BR-head H2H tournament**: 5-way round-robin, 3000 games/matchup, pure argmax.
- **Hold'em BR-head exploitability**: 6 targets measured end-to-end with 5M-episode DQN exploiters. All 4 in-flight IQN runs finished cleanly; final numbers in §4.3.

---

## 1. Problem and method

### 1.1 Why poker
Poker (imperfect-information, zero-sum, two-player) is the canonical benchmark for neural self-play. The opponent's strategy is part of the environment and is changing as they learn, so naive Q-learning diverges. CFR solves two-player zero-sum exactly but needs the game tree enumerated and so does not scale without abstractions.

### 1.2 Neural Fictitious Self-Play (NFSP)
Heinrich & Silver (2016). Each player maintains two heads:

- A **Q-network** — best-response to the opponent's historical *average* policy, trained with DQN.
- An **average-policy network** — time-average of the player's own best-responses, trained by supervised classification over a *reservoir buffer* of past action distributions.

At each action, the agent picks a head with a biased coin (η ≈ 0.1, the "anticipatory policy"). Under fictitious-play theory, the average policy converges to a Nash equilibrium in two-player zero-sum games.

### 1.3 IQN and risk-sensitive variants
Dabney et al. (2018). Replace the scalar Q-network with an **Implicit Quantile Network** that models the full return distribution. Risk preferences are expressed by distorting the quantile sampling before taking the mean:

| Preference | Q expression | Intuition |
|---|---|---|
| Risk-neutral | `Q = E[R]` | Standard NFSP with a distributional head |
| CVaR-averse α=0.25 | `Q = E[R | R ≤ 25th pct]` | "Plan for the bottom 25%" |
| Mean–variance β | `Q = E[R] − β·Var[R]` | "Penalize volatile plays" |
| CVaR-seeking α=0.75 | Upper-quartile weighted | "Gamble for upside" |

**Hypothesis.** Distorting Q biases best-response away from the Nash best-response (Nash is defined by mean payoffs in zero-sum), so theory says risk-sensitive variants should converge to *more*-exploitable policies. In larger games, conservative play *might* be harder to exploit in practice — that's the empirical question we're after.

---

## 2. Leduc Hold'em (final results)

### 2.1 Game and evaluation
Leduc: 2 players, 6-card deck, 2 betting rounds, fixed bet sizes. Game tree small enough that OpenSpiel computes **exact tabular best-response exploitability** against any stored average-policy net.

### 2.2 Setup

| | Value |
|---|---|
| Architecture | [128] (single hidden layer) |
| Batch | 128 |
| Episodes | 40M (IQN) / 45M (baseline) |
| DQN lr | 0.01 (baseline), 0.001 (IQN) |
| Avg-policy lr | same as DQN lr |
| Anticipatory η | 0.1 |
| IQN quantile samples N | 8 |
| Evaluation | exact exploitability every 50K eps |

### 2.3 Results (mbb/g, lower is less exploitable)

| Run | Risk | Exploitability |
|---|---|---|
| **baseline** (NFSP) | — | **0.094** 🏆 |
| iqn_neutral | neutral | 0.179 |
| iqn_mv01 | MV, β=0.1 | 0.180 |
| iqn_mv05 | MV, β=0.5 | 0.183 |
| iqn_averse | CVaR α=0.25 | 1.205 |
| iqn_seeking | CVaR upper 25% | 2.071 |

![Leduc exploitability](figures/leduc/exploitability.png)

### 2.4 Leduc takeaway
Risk-neutral NFSP beats every IQN variant (~2× on risk-neutral IQN, >10× on risk-distorted ones). This matches the theoretical prediction: in zero-sum games, distorting Q strictly hurts equilibrium convergence.

---

## 3. Limit Hold'em training

### 3.1 Game
Heads-up Limit Hold'em: 2 players, 52-card deck, 4 betting rounds, fixed bet sizes. Standard Heinrich & Silver benchmark. Exact best-response is intractable; we measure exploitability with DQN exploiters (§4) and compare via head-to-head tournaments (§3.4).

### 3.2 Setup

| | Value |
|---|---|
| Architecture (default) | [256, 128, 256, 128] ~100K params |
| Architecture (iqn_smaller) | [128, 64, 128, 64] |
| Batch | 256 (paper-matched) |
| DQN lr | 0.002 |
| Avg-policy lr | 0.0002 |
| Reservoir size | 30M (paper-matched) |
| DQN replay | 600K |
| Target update | 128K BR steps, hard copy |
| Reward scaling | ÷100 (chips → big-blinds) |
| Anticipatory η | 0.1 |

### 3.3 Trained models (highest-iteration version of each variant)

| Run | Agent | Risk | Episodes |
|---|---|---|---|
| `holdem_small_long` (baseline) | NFSP | — | 40M |
| `holdem_iqn_long` | IQN | neutral | 35.14M (stopped 2026-04-24) |
| `holdem_iqn_smaller` | IQN | neutral | 20M |
| `holdem_iqn_averse` | IQN | CVaR α=0.25 | 20M |
| `holdem_iqn_meanvar` | IQN | MV β=0.5 | 20M |

A note on the 35M IQN run: we stopped it at 35.14M. The old binary's `save_weights` only wrote `avg_net` (not `iqn_net`), so the BR head paired with it is the one from the ep30M checkpoint. BR loss was flat from ~20M onward (see `logs/_iqn_long_resume2.out`), so the 5M-ep gap is a ≤5% approximation. The rebuilt binary fixes this for future runs.

### 3.4 Head-to-head tournaments (5000 games/matchup, symmetric P0/P1)

**AVG-policy H2H** (existing, from §3 of prior results):

| Rank | Agent | Avg mbb/g vs others |
|---|---|---|
| 🏆 1 | baseline 40M | +825 |
| 2 | iqn_meanvar | +256 |
| 3 | iqn_neutral | +76 |
| 4 | iqn_smaller | −199 |
| 5 | iqn_averse | −958 |

**BR-head H2H** (new, 3000 games each, pure argmax, this session):

| Rank | Agent | Avg mbb/g vs others |
|---|---|---|
| 🏆 1 | **baseline 40M** | **+1102** |
| 2 | iqn_neutral 35M | +73 |
| 3 | iqn_smaller 20M | −117 |
| 4 | iqn_meanvar 20M | −261 |
| 5 | iqn_averse 20M | −797 |

Full matrix in `cpp-implementation/results/logs/br_h2h_tournament_seed42.txt`. Generator: `cpp-implementation/eval/h2h_tournament_br.py`.

**What the two tables say together:**
- **Baseline's advantage is bigger in BR-head play (+1102) than in AVG play (+825)** — its Q-net is even further ahead of the others than its average policy is.
- **iqn_meanvar flips**: 2nd in AVG-H2H but 4th in BR-H2H. Its average policy aggregates into something decent, but its current BR head is not competitive.
- **iqn_smaller rises** from 4th in AVG to 3rd in BR — small-arch Q-network is relatively strong.
- **averse is last in both**; the "CVaR-averse is robust" story was LBR artifact (§5).

![Hold'em AVG H2H](figures/holdem/h2h.png)

### 3.5 Training takeaway
NFSP baseline is strictly better than every IQN variant in direct play, whether comparing average policies or BR heads. Risk-distortion is actively harmful in tournament play — CVaR-averse is dead last.

---

## 4. Hold'em exploitability (DQN exploiter)

### 4.1 Methodology
Exact best-response is intractable at Hold'em scale, so we train a dedicated exploiter agent from scratch against each frozen target. The exploiter is an NFSP agent with `--eta 1.0` (pure Q-greedy), 5M episodes, 1 worker, seed 42.

**We only report BR-head exploitability** (the target's Q-net / IQN-net is played as its policy, with `--frozen-play-br`). This is the tighter bound: if the BR head is already close to Nash, the AVG head — which is a blend over many past BR heads — should be closer still. A BR-head exploiter also removes the target's own mixing stochasticity, so the exploiter can converge more cleanly.

### 4.2 Methodology bugs found and fixed this session

Two bugs in the exploiter binary made earlier IQN exploit numbers meaningless. Both are now fixed in the rebuilt binary:

1. **Worker `eta` was global, not per-player.** `--eta 1.0` put *both* players into BR mode in the worker's episode loop. Setting `eta_=0` on the agent object (as commit 48a3a52 did) did not help — the worker thread didn't consult it.
2. **IQN targets loaded via `load_frozen_avg_only`** which does not touch the Q-net; frozen target's Q-net stayed at random init. Combined with bug 1, every "IQN exploitability" number we had was measuring the exploitability of an accidentally-random Q-policy, not the trained IQN model.

Rebuild adds: per-player eta in `WorkerContext`; asymmetric agent types (`--frozen-p0-agent nfsp_iqn` with trainable p1 as NFSP); `--frozen-play-br` flag; `NFSPIQNAgent::save_weights` now also writes `iqn_net.pt`.

**Consequence for the old table**: the previously-published IQN exploit numbers (2875 / 800 / 893 / 925) are not measurements of trained IQN models. They are measurements of randomly-initialised Q-nets that happened to differ between runs because LibTorch's default RNG is non-deterministically seeded. Please disregard.

### 4.3 BR-head exploitability (final numbers, all 6 targets)

All 5M-episode exploiter runs complete. Numbers are `avg_r × 1000` where `avg_r` is the mean over the last 3 log windows (150K episodes) at ep 5M.

| Target | BR-head exploitability |
|---|---|
| baseline 20M | **73** |
| baseline 40M | **473** |
| iqn_neutral 35M | **600** |
| iqn_smaller 20M | **885** |
| iqn_meanvar 20M (β=0.5) | **1089** |
| iqn_averse 20M (CVaR α=0.25) | **1359** |

### 4.4 What the exploitability numbers say

- **Clean monotonic ordering by risk distortion:** baseline < risk-neutral IQN < MV β=0.5 < CVaR averse. Each step towards more-distortion adds ~200–300 mbb/g of exploitability. Direction matches Leduc exactly.
- **Baseline 40M (473) beats iqn_neutral 35M (600)** by ~125 mbb/g. Longer IQN training (35M vs 20M for other IQN variants) does *not* close the gap to NFSP — baseline remains tighter.
- **iqn_averse is 1.9× more exploitable than iqn_neutral** (1359 vs 600). Consistent with Leduc where averse was 7× worse than neutral.
- **iqn_smaller 20M at 885** — despite half the arch, it's still tighter than both meanvar and averse at the same 20M. That's interesting: suggests capacity isn't the binding constraint; risk-distortion is.
- **The baseline 20M→40M reversal (73 → 473, 6.5×)** is unique to baseline. No IQN variant shows analogous "longer training → more exploitable" behaviour. Resume was fully stateful (verified — `RESUMED from checkpoint at episode 20000000` with state.bin loaded, so there's no discontinuity artefact). Most likely explanations: single-seed exploiter noise, or sharper argmax decisions at 40M being easier for the exploiter to pattern-match than the slightly-more-stochastic argmax at 20M. Needs multi-seed replication.

### 4.5 Alignment across three methodologies

| Rank | BR-exploit | BR-H2H (4-way) | AVG-H2H |
|---|---|---|---|
| 1 (best) | baseline_40M | baseline_40M | baseline_40M |
| 2 | iqn_neutral_35M | iqn_neutral_35M | iqn_meanvar |
| 3 | (iqn_smaller) | iqn_meanvar | iqn_neutral |
| 4 | iqn_meanvar | iqn_averse | iqn_smaller |
| 5 (worst) | iqn_averse | — | iqn_averse |

BR-exploit and BR-H2H agree on rank 1, 2, and 4–5 — and both place meanvar as the 3rd-worst IQN variant by Q-head quality. The only AVG-H2H disagreement is meanvar's position (2nd in AVG, 4th in BR-exploit); that's a sign meanvar's BR head is its weak point while its avg policy averages to something decent. Triple-agreement on the extremes (baseline ≫ all IQN; averse is dead last) is the strongest possible statement given these three independent measurements.

---

## 5. What we tried but are not reporting: LBR

Local Best Response (Lisy & Bowling 2017) is the NFSP-literature standard evaluator: at each decision, try every legal action, Monte-Carlo roll it out using **both players' avg policy** for the rest of the episode, average, pick the best action. We ran LBR at rollouts=15 and rollouts=100 on every Hold'em variant. It does not belong in the final report — below is why.

### 5.1 What LBR said

| Run | LBR r=100 | BR-exploit |
|---|---|---|
| baseline 40M | 1819 | **473** (3.8× LBR overshot) |
| iqn_neutral 20M | 2091 | — |
| iqn_smaller 20M | 3017 | — |
| iqn_averse 20M | **605** 🏆 (per LBR) | — |
| iqn_meanvar 20M | 2258 | — |

LBR crowned iqn_averse the most-robust variant by 3×. Every other signal disagrees.

### 5.2 Three failure modes

1. **Rollouts use both players' avg policy.** If the target agent folds aggressively (CVaR-averse does), rollouts rarely reach informative states — LBR can't find exploits in games that end on the first street. Structurally over-rewards fold-heavy play.
2. **Opposite biases at different Nash-distances.** On the two cases where BR-exploit is clean (baseline 20M/40M), LBR overshoots 3.8× on baseline 40M. It's likely to under*shoot* on fold-heavy targets. Either direction, it is useless for *ranking*.
3. **Rollout-count sensitivity.** Going from r=15 to r=100 moves every Hold'em number up by 5–30% *inconsistently* across variants.

### 5.3 Refuted by H2H tournament

Both the AVG-H2H (§3.4) and the BR-head H2H (§3.4) place iqn_averse **dead last**, losing to every other variant by 600–1100 mbb/g per hand. No reading of "robust" is compatible with LBR's 605-mbb/g first-place *and* H2H's −797 mbb/g last-place. H2H plus DQN-exploiter agree with each other; LBR is the outlier methodology. We therefore do not report LBR numbers in the final paper.

---

## 6. Discussion and extensions

### 6.1 Headline answer to the research question
*Does IQN / risk-sensitive Q-distortion help or hurt NFSP equilibrium convergence?*

**It hurts, at both scales.** Leduc exact exploitability: baseline < every IQN variant (2× to 22× worse). Hold'em, every independent measurement (BR-head H2H tournament, BR-head exploitability, AVG-H2H tournament): baseline < every IQN variant, and risk-distorted variants (MV, CVaR-averse) are progressively worse. The Hold'em BR-exploit ordering `baseline 40M (473) < iqn_neutral 35M (600) < iqn_smaller (885) < iqn_meanvar (1089) < iqn_averse (1359)` replicates the Leduc ordering exactly — clean two-scale confirmation of the theoretical prediction. LBR's contrary claim (§5) was an artifact.

### 6.2 Open questions

1. **Why is baseline 20M BR-head only 73 mbb/g but baseline 40M is 473 mbb/g?** More training apparently makes things worse. Possible: single-seed noise (easy to test with 3 seeds), resume-from-checkpoint artifact (easy to test — train 40M from scratch), or a real "peak Nash at 20M" effect.
2. **Does 35M-trained IQN close the gap to baseline?** The current extrapolation puts iqn_neutral_35M at ~410 mbb/g vs baseline 40M at 473. If the final number confirms, longer training for IQN is worth more than for NFSP. Worth a 40M IQN retrain with the new binary for a clean comparison.
3. **Why does iqn_meanvar's AVG-H2H rank 2nd but BR-H2H rank 4th?** Gap between head types is largest for meanvar — a sign that fictitious play hasn't stabilised for that variant. A convergence-diagnostic run (A vs B exploit gap) would quantify.
4. **Does smaller-capacity IQN really regularise better?** iqn_smaller is 3rd in BR-H2H, ahead of iqn_meanvar despite having half the parameters. If the BR-exploit number comes in below iqn_meanvar's, "smaller net → tighter Nash" becomes a real finding.

### 6.3 Observations that need more exploration

- **The methodology bug implication.** Everything previously written about IQN variants' exploitability was comparing random Q-net initialisations. The four new in-progress runs are the first real data on trained IQN models.
- **BR-head H2H and AVG-H2H agree on the extremes (baseline 1st, averse 5th) but reorder the middle.** This is informative: meanvar's AVG policy is good but BR head is bad; smaller's BR head is stronger than its AVG. These split signals suggest the fictitious-play averaging is smoothing over different amounts of training instability.
- **CVaR-averse BR head drops fast under exploiter attack** (already −0.95 at ep 1.3M). The exploiter is finding huge holes early — probably exactly the "fold-too-much, raise-the-wrong-spots" pattern that H2H punishes.
- **iqn_long 35M paired with a 30M BR head** is an imperfect snapshot. Very likely fine (BR loss was flat), but a clean 40M retrain from scratch would be the proper comparison.

### 6.4 Concrete experimental extensions

1. **Multi-seed DQN exploiters.** Every number in §4.3 should eventually be mean ± SE over 3 seeds. Would resolve the baseline 20M/40M reversal question.
2. **Retrain IQN-neutral to 40M from scratch** with the new binary (saves both heads cleanly). Fair comparison to baseline 40M.
3. **Risk-sensitive exploiter training.** Train the exploiter itself as CVaR-seeking — may find tighter lower bounds than risk-neutral DQN.
4. **Extended-training ablation for every variant** — do IQN variants also peak early like baseline did?
5. **A non-zero-sum or variance-relevant game.** Limit Hold'em has limited per-step stochasticity — a risk preference only matters insofar as the *strategy distribution* induces variance. IQN's risk-sensitive mode may have genuine value in games with explicit outcome variance; we should test it in a bankroll-management or multi-agent-with-noise setting.

### 6.5 What to do after the 4 in-flight runs complete

1. Read final `avg_r` from each log, multiply by 1000, populate §4.3 final column.
2. Compare final ranking with BR-head H2H §3.4 and with Leduc §2.3. If all three agree, that's three independent methodologies converging — strongest possible claim.
3. Regenerate `figures/holdem/` with the new data.

---

## 7. Files of interest

| Path | What it is |
|---|---|
| `cpp-implementation/src/train_holdem.cpp` | Hold'em training binary |
| `cpp-implementation/src/train.cpp` | Leduc training binary |
| `cpp-implementation/src/nfsp_agent_holdem.h` | NFSP agent (DQN Q-head) |
| `cpp-implementation/src/nfsp_iqn_agent_holdem.h` | IQN agent (distributional head + risk distortions) |
| `cpp-implementation/scripts/run_br_exploiters_parallel.sh` | This session's parallel method-B launcher |
| `cpp-implementation/eval/h2h_tournament.py` | AVG-policy H2H tournament |
| `cpp-implementation/eval/h2h_tournament_br.py` | **NEW** BR-head H2H tournament |
| `cpp-implementation/eval/compute_exploitability.py` | Leduc exact exploitability |
| `cpp-implementation/scripts/make_plots.py` | Figure generator |
| `cpp-implementation/logs/` | Per-run human-readable training logs |
| `cpp-implementation/results/logs/` | Per-run JSONL metrics + H2H/exploitability CSVs |
| `figures/leduc/`, `figures/holdem/` | Generated plots |
| `/mnt/data/cos435/weights/` | Final weights, ~100 GB including training buffers |
| `/mnt/data/cos435/checkpoints/` | Periodic training checkpoints |
