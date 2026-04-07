# CLAUDE.md — COS 435 Distributional NFSP Project

## Quick Reference

- **Goal:** Reproduce NFSP baseline on Leduc poker, then replace DQN with IQN and test risk-sensitive agents.
- **GPU:** Tesla P100 16GB VRAM — **hard cap at 30% = 4.8GB max**. The LoRA project (COS 484) uses the other 70%.
- **Framework:** OpenSpiel (google-deepmind/open_spiel)
- **Game:** Leduc poker (~936 decision states)
- **Metrics:** Exploitability (NashConv), convergence speed, head-to-head performance

## GPU Memory Constraint (CRITICAL)

This machine is shared with the LoRA fine-tuning project which gets 70%. We get 30%.

**At the top of every training script:**

```python
import torch
# Hard cap: 30% of 16GB = 4.8GB
torch.cuda.set_per_process_memory_fraction(0.30, device=0)
torch.cuda.empty_cache()
```

This is more than enough — NFSP on Leduc poker uses tiny networks (<500MB VRAM). The cap is just a safety measure.

## Project Structure

```
cos435/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── configs/
│   ├── nfsp_baseline.yaml       # Standard NFSP with DQN
│   ├── nfsp_iqn_neutral.yaml    # NFSP with IQN, risk-neutral
│   ├── nfsp_iqn_averse.yaml     # NFSP with IQN, risk-averse
│   └── nfsp_iqn_seeking.yaml    # NFSP with IQN, risk-seeking
├── src/
│   ├── nfsp_agent.py            # NFSP agent (DQN best-response + average policy network)
│   ├── iqn_agent.py             # IQN-based best-response module
│   ├── nfsp_iqn_agent.py        # NFSP agent with IQN replacing DQN
│   ├── networks.py              # Neural network definitions (MLP for DQN, IQN, avg policy)
│   ├── replay_buffer.py         # Experience replay (reservoir sampling for avg policy, circular for BR)
│   ├── train.py                 # Main training loop
│   ├── evaluate.py              # Exploitability computation + head-to-head evaluation
│   ├── utils.py                 # Config, seeding, logging, checkpointing
│   └── risk_distortion.py       # Risk distortion functions (CVaR, CPW, Wang)
├── scripts/
│   ├── run_baseline.sh
│   ├── run_iqn_neutral.sh
│   ├── run_iqn_averse.sh
│   └── run_iqn_seeking.sh
├── results/
│   ├── checkpoints/
│   ├── logs/
│   └── figures/
└── tests/
    └── test_agents.py
```

## Dependencies

```
torch
open-spiel
numpy
pyyaml
matplotlib
tqdm
```

Install with: `pip3 install open-spiel numpy pyyaml matplotlib tqdm`

PyTorch should already be installed. If not: `pip3 install torch --index-url https://download.pytorch.org/whl/cu121`

## Key Papers and Their Hyperparameters

### Heinrich & Silver (2016) — NFSP Baseline

The NFSP algorithm has two components per player:
1. **Best-response network (DQN):** learns Q(s,a) via Q-learning from self-play
2. **Average policy network (supervised learning):** trained on the player's own past behavior stored in a reservoir buffer

**Architecture from the paper (Leduc poker):**
- Best-response (DQN) network: MLP with hidden layers [128] (1 hidden layer, 128 units)
- Average policy network: MLP with hidden layers [128]
- Both use ReLU activations

**Hyperparameters from the paper (Leduc poker):**
- Anticipatory parameter η (eta): 0.1 (probability of playing best response vs average policy during training)
- DQN learning rate: 0.1
- DQN discount factor γ: 1.0 (episodic game, no discounting)
- DQN epsilon-greedy: ε = 0.06
- DQN replay buffer size (circular): 200,000
- DQN batch size: 128
- DQN update target network every: 1000 steps
- Average policy learning rate: 0.01
- Average policy reservoir buffer size: 2,000,000
- Average policy batch size: 128
- Training episodes: 1,000,000+ (until exploitability converges)
- Optimizer: SGD (as in original paper)

**IMPORTANT:** Use SGD, not Adam. The original paper uses SGD. Stick to original hyperparameters.

### Dabney et al. (2018) — IQN

IQN replaces the scalar Q(s,a) with a quantile function Q_τ(s,a) where τ ~ U(0,1).

**Key components:**
- Cosine embedding for τ: φ(τ) = ReLU(Σ cos(i·π·τ) · w_i + b) for i=0..n-1
- Embedding dimension n: 64
- Number of quantile samples for training (N): 8
- Number of quantile samples for action selection (K): 32
- Loss: quantile Huber loss (κ=1.0)

**For action selection (risk-neutral):** sample K quantiles uniformly, average Q_τ(s,a) across them.

**For risk-averse:** oversample low quantiles (e.g., use CVaR distortion: τ' = τ·α where α<1, like α=0.25)
**For risk-seeking:** oversample high quantiles (e.g., use τ' = 1-(1-τ)·α)

## Implementation Details

### Phase 1: NFSP Baseline

**NFSP training loop (per episode):**
```
1. Reset Leduc poker game
2. While game not terminal:
   a. Current player chooses action:
      - With probability η: use best-response (DQN epsilon-greedy)
      - With probability 1-η: use average policy network
   b. Store transition in DQN replay buffer (for best-response learning)
   c. If using best-response: store (state, action) in reservoir buffer (for average policy)
   d. Step the game
3. After episode:
   a. Sample batch from DQN buffer, update Q-network (standard DQN update)
   b. Sample batch from reservoir buffer, update average policy network (supervised cross-entropy loss)
4. Every N episodes: compute exploitability using OpenSpiel's exploitability module
```

**Computing exploitability:**
```python
from open_spiel.python.algorithms import exploitability

# Create a policy object from the average policy network
# Call exploitability.exploitability(game, policy) to get NashConv
```

OpenSpiel provides `exploitability.exploitability()` which returns the sum of each player's incentive to deviate — lower is closer to Nash equilibrium.

### Phase 2: NFSP + IQN (Risk-Neutral)

Replace the DQN best-response network with IQN:
- Instead of Q(s,a) → scalar, learn Q_τ(s,a) where τ ~ U(0,1)
- For action selection: sample K=32 uniform quantiles, compute mean Q_τ(s,a) per action, pick argmax
- For training: sample N=8 quantiles for current and N'=8 for target, compute quantile Huber loss
- Everything else in NFSP stays the same (average policy, reservoir buffer, eta, etc.)

### Phase 3: NFSP + IQN (Risk-Sensitive)

Same as Phase 2, but modify action selection:
- **Risk-averse (CVaR α=0.25):** sample τ ~ U(0, 0.25) instead of U(0,1) — focuses on worst-case outcomes
- **Risk-averse (CVaR α=0.5):** sample τ ~ U(0, 0.5)
- **Risk-seeking:** sample τ ~ U(0.5, 1.0) or U(0.75, 1.0) — focuses on best-case outcomes
- Training still uses uniform τ — only action selection changes

## Evaluation

1. **Exploitability curve:** Plot exploitability (NashConv) vs. training episodes for all variants on the same graph
2. **Convergence speed:** How many episodes to reach exploitability < 0.1 (or some threshold)
3. **Final exploitability:** Mean exploitability over last 10% of training, averaged across seeds
4. **Head-to-head:** Play trained agents against each other (1000+ games) to measure relative performance

## Experiment Execution Order

1. NFSP baseline — seeds 42, 123, 456 — verify exploitability converges toward 0
2. NFSP + IQN risk-neutral — seeds 42, 123, 456 — compare convergence to baseline
3. NFSP + IQN risk-averse (CVaR α=0.25) — seeds 42, 123, 456
4. NFSP + IQN risk-seeking — seeds 42, 123, 456
5. Head-to-head tournament: all agents play each other
6. Generate comparison plots

## Coding Conventions

- Use `python3` for all scripts
- Use OpenSpiel's `pyspiel` for game environment and exploitability computation
- Use PyTorch for neural networks (NOT TensorFlow)
- Set random seeds: `torch.manual_seed`, `np.random.seed`, and OpenSpiel's seed if available
- Log exploitability every 10,000 episodes
- Save checkpoints every 100,000 episodes
- Print progress with tqdm
- Use YAML configs for all hyperparameters — no hardcoded values in training code
- Enforce `torch.cuda.set_per_process_memory_fraction(0.30, device=0)` in every script
- Commit after each major milestone

## Git Workflow

- Commit frequently with descriptive messages
- Push to `origin main` after each completed experiment
- `results/checkpoints/` in .gitignore (too large)
- `results/logs/` and `results/figures/` committed to git

**.gitignore:**
```
results/checkpoints/*.pt
__pycache__/
*.pyc
.DS_Store
```

## Common Commands

```bash
# Check GPU (verify staying under 30%)
nvidia-smi

# Run baseline
python3 src/train.py --config configs/nfsp_baseline.yaml --seed 42

# Run IQN risk-neutral
python3 src/train.py --config configs/nfsp_iqn_neutral.yaml --seed 42

# Evaluate exploitability of a checkpoint
python3 src/evaluate.py --checkpoint results/checkpoints/baseline_best.pt

# Head-to-head tournament
python3 src/evaluate.py --tournament --agents baseline iqn_neutral iqn_averse iqn_seeking
```

## Troubleshooting

- **OpenSpiel import errors:** Make sure `open-spiel` is installed via pip. It may need to build from source if pip fails — check OpenSpiel docs.
- **Exploitability computation slow:** It's tabular on Leduc poker, should be fast (<1 sec). If slow, you may be calling it too frequently — reduce to every 10,000 episodes.
- **Training seems stuck:** NFSP on Leduc poker typically needs 500K-1M+ episodes to converge. Be patient. Check that exploitability is trending downward even if slowly.
- **OOM:** Extremely unlikely with 30% cap on these tiny networks. If it happens, reduce batch size.

## What NOT To Do

- Do NOT extend to Limit Texas Hold'em (out of scope for overnight run)
- Do NOT use Adam optimizer for the baseline — the original paper uses SGD
- Do NOT exceed 30% GPU memory
- Do NOT modify the LoRA project files (that's the other project in /workspace/cos484)
- Do NOT change the Leduc poker game rules or parameters
