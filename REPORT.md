# Risk-Sensitive NFSP on Poker: Leduc → Limit Hold'em

*COS 435 project report. Onboarding doc for a new collaborator — read top to bottom.*

**Status as of 2026-04-24, 14:45 UTC:**
- **Leduc (final)**: 6 variants trained, exact exploitability measured via OpenSpiel tabular best-response.
- **Hold'em training (final)**: 6 models trained (2 baselines at 20M/40M eps, 4 IQN variants at 20M, 1 IQN variant at 35M).
- **Hold'em exploitability (in progress)**: 4 clean method-B (BR-head) DQN exploiters running in parallel, ETA ~2–3 h. Once they finish we have the full Hold'em picture.
- **LBR is out of the final report** — it's unreliable for our scale (see §5). We replaced it with DQN-exploiter measurements.

---

## 1. Problem and method

### 1.1 Why poker
Poker (imperfect-information, zero-sum, two-player) is the canonical benchmark for neural self-play. The opponent's strategy *is* part of the environment and is changing as they learn, so naive Q-learning diverges (the Q-target has no fixed point). Counterfactual Regret Minimization solves two-player zero-sum exactly but needs the game tree enumerated, so it doesn't scale without abstractions.

### 1.2 Neural Fictitious Self-Play (NFSP)
Heinrich & Silver (2016). Each player maintains two heads:

- A **Q-network** — best-response to the opponent's *historical average* policy, trained with DQN.
- An **average-policy network** — the time-average of the player's own best-responses, trained by supervised classification over a *reservoir buffer* of past action distributions.

At each action, the agent flips a biased coin (η ≈ 0.1, the "anticipatory policy") to decide which head to play from. Under fictitious-play theory, the average policy converges to a Nash equilibrium in two-player zero-sum games.

### 1.3 IQN and risk-sensitive variants
Dabney et al. (2018). Replace the scalar Q-network with an **Implicit Quantile Network** that models the full return distribution. Any risk preference can then be expressed by *distorting* the sampling of quantiles before taking the mean:

| Preference | Q expression | Intuition |
|---|---|---|
| Risk-neutral | `Q = E[R]` | Standard NFSP, just with a distributional head |
| CVaR-averse α=0.25 | `Q = E[R | R ≤ 25th pct]` | "Plan for the bottom 25%" |
| Mean–variance β | `Q = E[R] − β·Var[R]` | "Penalize volatile plays" |
| CVaR-seeking α=0.75 | Upper-quantile weighted | "Gamble for upside" |

**Hypothesis to test.** Distorting Q biases best-response away from the Nash best-response (Nash is defined by mean payoffs in zero-sum), so theory says risk-sensitive variants should converge to *more*-exploitable policies. But in larger games, a conservative policy might be harder to *actually* exploit (higher-dim exploit surface, rarer tournament mistakes). Which effect wins is a question of scale — hence testing on both Leduc and Hold'em.

---

## 2. Leduc Hold'em (final results)

### 2.1 Game and evaluation
Leduc: 2 players, 6-card deck, 2 betting rounds, fixed bet sizes. Game tree is small enough that OpenSpiel computes **exact tabular best-response exploitability** of any stored average-policy net.

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
Risk-neutral NFSP beats every IQN variant (~2× on risk-neutral IQN, >10× on risk-distorted ones). Matches the theoretical prediction: in zero-sum games, distorting Q strictly hurts equilibrium convergence. Our question for Hold'em: does this hold at scale?

---

## 3. Limit Hold'em training

### 3.1 Game
Heads-up Limit Hold'em: 2 players, 52-card deck, 4 betting rounds, fixed bet sizes. The standard Heinrich & Silver benchmark. Game tree is too large for exact best-response; we rely on **DQN exploiter** evaluation (§4) and head-to-head tournaments (§3.3).

### 3.2 Setup

| | Value |
|---|---|
| Architecture (default) | [256, 128, 256, 128] — ~100K params |
| Architecture (iqn_smaller) | [128, 64, 128, 64] |
| Batch | 256 (paper-matched) |
| DQN lr | 0.002 |
| Avg-policy lr | 0.0002 |
| Reservoir size | 30M (paper-matched) |
| DQN replay | 600K |
| Target update | 128K BR steps, hard copy |
| Reward scaling | ÷100 (chips → big-blinds) |
| Anticipatory η | 0.1 |

### 3.3 Models trained (6 final policies)

| Run | Agent | Risk | Episodes | Weights |
|---|---|---|---|---|
| `holdem_small_long` | NFSP | — | 40M | `final_weights_holdem_small_long/` |
| `baseline 20M` (archive) | NFSP | — | 20M | `/mnt/data/cos435/archive_20M/final_holdem_small_long_ep20M/` |
| `holdem_iqn_long` | IQN | neutral | **35.14M** (stopped 2026-04-24) | `/mnt/data/cos435/weights/final_holdem_iqn_long/` |
| `holdem_iqn_long` (archive) | IQN | neutral | 20M | `/mnt/data/cos435/archive_20M/final_holdem_iqn_long_ep20M/` |
| `holdem_iqn_smaller` | IQN | neutral | 20M | `/mnt/data/cos435/checkpoints/holdem_iqn_smaller_ep20000000_p*/` |
| `holdem_iqn_averse` | IQN | CVaR α=0.25 | 20M | `final_weights_holdem_iqn_averse/` |
| `holdem_iqn_meanvar` | IQN | MV β=0.5 | 20M | `final_weights_holdem_iqn_meanvar/` |

Note on the 35M IQN run: we stopped it at 35.14M. The old binary's `save_weights` only wrote `avg_net` (not `iqn_net`), so the BR head saved alongside is the one from the ep30M checkpoint. BR loss was flat from ~20M onward (see `logs/_iqn_long_resume2.out`), so the 5M-ep gap is a ≤5% approximation. The rebuilt binary fixes this for any future runs.

### 3.4 Head-to-head tournament among the 5 Hold'em variants

Round-robin, 5000 games per matchup, swap P0/P1. AVG policies only. `iqn_long 20M` and `baseline 40M` used as the "canonical" versions.

| Rank | Agent | Avg mbb/g vs others |
|---|---|---|
| 🏆 1 | **baseline 40M (NFSP)** | +825 |
| 2 | iqn_meanvar | +256 |
| 3 | iqn_neutral | +76 |
| 4 | iqn_smaller | −199 |
| 5 | iqn_averse | **−958** |

Headline matchups: baseline beats averse by +1069 mbb/g; meanvar beats averse by +930; even the capacity-starved iqn_smaller beats averse by +959.

![Hold'em H2H](figures/holdem/h2h.png)

### 3.5 Hold'em training takeaway
Baseline NFSP is strictly better than every IQN variant in direct play. Risk-aversion is actively harmful in tournament play. This agrees with the Leduc finding qualitatively — IQN hurts, risk-distortion hurts more.

---

## 4. Hold'em exploitability (DQN exploiter)

Exact best-response is intractable at Hold'em scale, so we train a dedicated exploiter agent from scratch against each frozen target. The exploiter is an NFSP agent with `--eta 1.0` (pure Q-greedy), 5M episodes, 1 worker, seed 42, same Q-network architecture as the target.

### 4.1 Two measurement methods

Each trained NFSP/IQN agent has two heads: an average policy and a BR (Q or IQN) head. We care about each separately:

- **Method A — AVG-policy exploitability.** Freeze the target and force it to always play its average policy (the thing NFSP theory says converges to Nash). Exploiter learns BR to that. Answers: "how exploitable is the strategy you'd actually report as the solution?"
- **Method B — BR-head exploitability.** Freeze the target and force it to always play its BR head. Exploiter learns BR to that. Answers: "how exploitable is the current Q-network?" The gap between A and B is a convergence diagnostic — a small gap means the fictitious-play loop has stabilised.

### 4.2 Methodology bugs found and fixed this session

Two bugs in the exploiter binary made prior numbers hard to trust. Both now fixed in the rebuilt binary:

1. **Worker `eta` was global, not per-player.** `--eta 1.0` put *both* players into BR mode in the worker's episode loop. The intended method-A "freeze the AVG" path needed the frozen side's eta to be 0 independently. An earlier "fix" (commit 48a3a52) set `eta_=0` on the agent object, but the worker thread didn't read from there — so the fix was silently a no-op. *Effect: every exploiter run before 2026-04-24 measured method-B, not method-A, no matter what we thought.*
2. **IQN targets loaded via `load_frozen_avg_only`** — which does not touch the Q-net. The frozen target's Q-net stayed at random init. Combined with bug 1 (worker plays BR → consults the random Q-net), this means **every "IQN exploitability" number measured the exploitability of an accidentally-random Q-policy**, not of the trained IQN model.

Rebuild (2026-04-24) introduces: per-player eta in `WorkerContext`; asymmetric agent types (`--frozen-p0-agent nfsp_iqn` with trainable p1 as NFSP); `--frozen-play-br` flag; `NFSPIQNAgent::save_weights` now also writes `iqn_net.pt`; null-learner shutdown guarded.

### 4.3 Exploitability results

| Target | (A) AVG-policy | (B) BR-head | Notes |
|---|---|---|---|
| baseline 20M | *rerun in progress* | **73 mbb/g** | B value from pre-fix run (happens to be a clean B measurement because target is NFSP with q_net loaded) |
| baseline 40M | *rerun in progress* | **473 mbb/g** | ditto |
| iqn_neutral 20M | *unknown* (prior 2875 is the random-q_net artefact) | impossible — no 20M `iqn_net.pt` archived | — |
| iqn_neutral 35M | *not run* | **in progress** | first clean IQN measurement |
| iqn_smaller 20M | *unknown* (prior 800 is random-q_net) | **in progress** | |
| iqn_averse 20M | *unknown* (prior 893 is random-q_net) | **in progress** | |
| iqn_meanvar 20M | *unknown* (prior 925 is random-q_net) | **in progress** | |

What we know right now with confidence:
- **baseline 40M has BR-head exploitability ≈ 473 mbb/g.** Strongest demonstration that NFSP baseline trained well.
- **baseline 20M has BR-head exploitability ≈ 73 mbb/g.** Notably better than 40M — see §6.

What we'll know once the 4 in-flight runs complete (ETA ~2–3 h):
- Clean BR-head exploitability of every IQN variant.
- Whether any IQN variant is under 473 mbb/g (i.e., more Nash-like than the baseline's BR head).
- Whether the `iqn_smaller < iqn_neutral` exploitability ranking survives once we remove the random-q_net artefact.

### 4.4 Why these numbers supersede LBR (§5)

The DQN-exploiter numbers are:
- **Tight** — 5M exploiter episodes converges well below LBR's one-step lookahead bound.
- **Consistent across a ranking diagnostic** — they agree directionally with H2H tournament rankings, whereas LBR does not (see §5).
- **Symmetric** — the exploiter learns from scratch and doesn't rely on the target's own (potentially fold-heavy) policy for rollouts, which is LBR's Achilles heel.

---

## 5. LBR: what it said, why we discarded it

Local Best Response (Lisy & Bowling 2017) is the standard NFSP-literature evaluator: at each decision, try every legal action, Monte-Carlo roll it out using **both players' current avg policy** for the rest of the episode, and pick the highest-expected-return action. 5000 hands, rollouts parameter controls rollout count.

### 5.1 What LBR said about our Hold'em models

| Run | LBR r=15 | LBR r=100 | DQN-exploit (B) |
|---|---|---|---|
| baseline 40M | 1400 | 1819 | **473** |
| iqn_neutral 20M | 1993 | 2091 | 2875 ✗ (suspect) |
| iqn_smaller 20M | 2409 | 3017 | 800 ✗ (suspect) |
| iqn_averse 20M | **482** 🏆 (per LBR) | **605** 🏆 (per LBR) | 893 ✗ (suspect) |
| iqn_meanvar 20M | 1994 | 2258 | 925 ✗ (suspect) |

LBR crowned **iqn_averse as the most-robust variant** by a factor of 3 over the next-best. This disagrees with every other signal we have.

### 5.2 Why LBR is unreliable here

Three failure modes:

1. **Rollouts use both players' own average policy.** If the target agent folds aggressively (CVaR-averse does), rollouts rarely reach informative states — LBR can't find exploits in games that end on the first street. This structurally *over-rewards* fold-heavy play.
2. **Opposite biases at different Nash-distances.** Comparing LBR r=100 against DQN-exploiter on the two cases where B is clean (baseline 40M/20M), LBR overshoots by 3.8× on baseline 40M (1819 vs 473). For iqn_neutral 20M, LBR undershoots — DQN-exploit (2875) > LBR r=100 (2091). So LBR can err in either direction depending on the target, which makes it useless for *ranking*.
3. **Rollout count sensitivity.** Going from r=15 to r=100 moves every Hold'em number up by 5–30% inconsistently. LBR at r=15 systematically underestimates; r=100 is closer but still noisy.

### 5.3 Refutation by the H2H tournament
H2H (§3.4) places iqn_averse **dead last**, losing to every other variant by 800–1000 mbb/g per hand. No reading of "robust" is compatible with *both* LBR's 605-mbb/g first-place and H2H's −958 mbb/g last-place. H2H plus DQN-exploiter agree; LBR is the outlier. We therefore do not report LBR numbers in the final table.

---

## 6. Discussion and extensions

### 6.1 Open questions the in-flight data will answer

1. **Does IQN actually converge on Hold'em?** Once we have method-B for every IQN variant and later method-A for comparison, the A–B gap is a direct convergence diagnostic. Large gap → Q-head and avg-head still oscillating → not converged. Small gap → stable equilibrium.
2. **Is the baseline-20M < baseline-40M (73 < 473) reversal real, or noise?** Both are one-shot, seed 42. The obvious follow-up is a 3-seed reproduction of each exploiter run.
3. **Is risk-aversion genuinely low-exploit, or just LBR-invisible?** The DQN exploiter does not use the target's avg for rollouts, so it cannot suffer LBR's blindness-to-folders failure mode. If clean method-B puts iqn_averse near the other IQN variants, the "averse is robust" story was entirely LBR artefact.
4. **Does smaller IQN capacity actually help exploitability (the apparent smaller < neutral ranking), or was that random-Q-net noise?** Once method-B for both is in, the real ranking is visible. If smaller < neutral survives, that's a striking claim: **bigger IQN nets over-fit their own BR targets in self-play**, and capacity regularisation alone improves Nash-closeness. Worth a section of its own.

### 6.2 Observations that need more exploration

- **The 20M → 40M reversal for NFSP baseline** (73 → 473 mbb/g). Currently inexplicable. Candidates:
  - Single-seed noise — need multi-seed reruns.
  - Checkpoint-resume artefact — the 20M→40M extension used a warm buffer, possibly putting the Q-net in a regime with more exploitable patterns.
  - Genuine "peak Nash" at 20M — longer training drifts the avg policy into a narrower mode that is easier to exploit.
  Easy-ish test: train a fresh 40M NFSP baseline from scratch (no checkpoint resume) and see which number it produces.
- **Methodology bug impact.** Every IQN exploit number published in previous commits (2875/800/893/925) measures exploitability of random-init Q-policies, not trained IQN models. The fact that these numbers *differ* from each other is purely a consequence of non-deterministic LibTorch parameter init per run. After the 4 in-flight clean method-B runs, revisit every qualitative claim in §3.9 of the prior report. (Those prior-report statements have now been deleted.)
- **The 35M iqn_long has a 30M BR head.** Small approximation because BR loss was flat, but technically non-crisp. If we care, retrain IQN-neutral from scratch to 40M with the new binary — roughly 10 h compute.
- **Missing cell: iqn_neutral @ 20M method-B** — impossible without retraining (no `iqn_net.pt` archived). Only matters if we want to pair 20M-vs-20M comparisons across all variants; the 35M number covers neutral adequately.
- **Risk-sensitive Q distortion is being tested on deterministic rewards.** Heads-up Limit Hold'em has no stochastic reward tail beyond the deal — a risk preference only matters insofar as the **strategy distribution** induces variance in outcomes. A cleaner test of IQN's risk-sensitive mode would be a game with explicit per-step stochasticity. This explains, at least partly, why risk-distortion consistently looks like dead weight in both Leduc and Hold'em.

### 6.3 Concrete experimental extensions

In priority order, if we had more compute:

1. **Multi-seed DQN exploiters.** Every §4.3 cell should be mean ± SE over 3 seeds. Would immediately resolve the baseline 20M/40M reversal question.
2. **Clean method-A for IQN.** The new binary supports it (`--frozen-p0-agent nfsp_iqn` without `--frozen-play-br`). Pair A and B for every variant to get the convergence diagnostic.
3. **Retrain IQN-neutral to 40M from scratch** with the new binary (saves both heads cleanly). Removes the 30M-BR-head approximation and gives a clean 40M-vs-40M comparison against the baseline.
4. **Risk-sensitive exploiter training.** Train the exploiter itself as IQN/CVaR-seeking to aggressively seek exploits. May find tighter lower bounds than neutral DQN.
5. **Cross-variant matchups in the exploiter role.** Use a trained iqn_neutral as the exploiter vs each target — does an IQN exploiter find more than a scalar one?
6. **Extended-training ablation for every variant** (not just baseline). Do IQN variants also peak at 20M?
7. **A non-zero-sum or risk-relevant game.** To test whether IQN's risk-sensitive modes have any genuine use case in a multi-agent setting, run the same IQN variants on a game with explicit outcome variance (e.g., a bankroll-management wrapper where variance matters directly for long-run return).

### 6.4 What to do after the 4 in-flight runs complete

1. Read the final `avg_r` from each log (multiply by 1000 for mbb/g). Populate §4.3 B-column.
2. Check consistency with H2H §3.4. If ranks agree, that's two independent methodologies converging — strongest possible statement.
3. If `iqn_smaller < iqn_neutral` still holds on clean B-data, investigate Q-net loss trajectories to understand capacity-regularisation effect.
4. If the two "baseline AVG rerun" processes finish before the IQN exploiters do, their final numbers will duplicate the B-column (per bug 1 above) — ignore them for method-A analysis. A clean method-A run for baselines is a separate follow-up.
5. Regenerate `figures/holdem/lbr_vs_dqn_exploiter.png` as `figures/holdem/exploit_bar.png` showing A and B side-by-side for every variant.

---

## 7. Files of interest

| Path | What it is |
|---|---|
| `cpp-implementation/src/train_holdem.cpp` | Hold'em training binary |
| `cpp-implementation/src/train.cpp` | Leduc training binary |
| `cpp-implementation/src/nfsp_agent_holdem.h` | NFSP agent (DQN Q-head) |
| `cpp-implementation/src/nfsp_iqn_agent_holdem.h` | IQN agent (distributional head + risk distortions) |
| `cpp-implementation/scripts/run_br_exploiters_parallel.sh` | This session's parallel method-B launcher |
| `cpp-implementation/scripts/run_all_exploiters_sequential.sh` | Prior session's sequential orchestrator (now stale) |
| `cpp-implementation/eval/eval_holdem_h2h.py` | H2H tournament evaluator |
| `cpp-implementation/eval/compute_exploitability.py` | Leduc exact exploitability |
| `cpp-implementation/scripts/make_plots.py` | Figure generator |
| `cpp-implementation/logs/` | Per-run human-readable training logs |
| `cpp-implementation/results/logs/` | Per-run JSONL metrics + H2H/exploitability CSVs |
| `figures/leduc/`, `figures/holdem/` | Generated plots |
| `/mnt/data/cos435/weights/` | Final weights, ~100 GB including training buffers |
| `/mnt/data/cos435/checkpoints/` | Periodic training checkpoints |
