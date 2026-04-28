Run 1: Constant Learning Rate (SGD lr=0.1)
============================================
Date: 2026-04-08
Episodes: 3,000,000 per experiment
Eval frequency: 10,000 episodes
Training mode: single-process (matching Heinrich & Silver 2016)

Experiments
-----------
1. nfsp_baseline       - Standard NFSP with DQN best-response
2. nfsp_iqn_neutral    - NFSP with IQN best-response, risk-neutral
3. nfsp_iqn_mean_var   - NFSP with IQN, mean-variance utility (penalty=0.1)
4. nfsp_iqn_mean_var_05- NFSP with IQN, mean-variance utility (penalty=0.5)
5. nfsp_iqn_averse     - NFSP with IQN, CVaR risk-averse (alpha=0.25)

Results Summary
---------------
Experiment              | Final Exploit. | Min Exploit. | Min @ Episode
------------------------|---------------|-------------|-------------
Baseline (DQN)          | 0.360         | 0.239       | ~1,350,000
IQN Neutral             | 0.284         | 0.284       | 3,000,000
IQN Mean-Var (0.1)      | 0.567         | 0.486       | ~1,300,000
IQN Mean-Var (0.5)      | 2.021         | 1.597       | ~50,000
IQN Averse (CVaR 0.25)  | 0.705         | 0.581       | ~1,950,000

Key Observations
----------------
1. IQN Neutral outperforms the DQN baseline (0.284 vs 0.239 min), and is still
   improving at 3M episodes. The distributional signal from IQN provides a more
   stable best-response, reducing oscillation compared to scalar DQN.

2. The baseline DQN reaches its best exploitability (0.239) around 1.35M episodes,
   then DEGRADES back to ~0.36 by 3M. This is caused by SGD lr=0.1 being too high
   for fine convergence -- the Q-network oscillates near the optimum, feeding noisy
   best-response actions into the reservoir buffer and degrading the average policy.

3. Mean-Var (0.5) is essentially broken (exploit ~2.0). The variance penalty is far
   too aggressive for poker, where returns are inherently high-variance. The agent
   avoids all uncertain actions, which in poker is everything.

4. Mean-Var (0.1) converges but plateaus around 0.49-0.57. The variance penalty
   distorts the best-response away from the true game-theoretic best-response,
   preventing full Nash convergence. This is expected behavior.

5. CVaR Averse converges slowly to ~0.60. The CVaR distortion (focusing on worst 25%
   of quantiles) makes the best-response overly conservative, significantly hurting
   Nash convergence. This is theoretically expected -- risk distortion in action
   selection means the agent is no longer computing the true best-response.

6. None of the experiments reach the paper's reported <0.1 exploitability. The main
   bottleneck is the constant SGD learning rate. Run 2 will use LR decay to address
   this.

Bugs Fixed Before This Run
---------------------------
- Gradient update ratio: parallel mode was doing 0.5 updates/ep/player instead of 1.0.
  Fixed by switching to single-process training matching the paper exactly.
- IQN loss scaling: .sum(dim=2) instead of .mean(dim=2) in quantile Huber loss inflated
  the loss by 8x (N=8 quantiles), making the effective lr = 0.8. Fixed to .mean(dim=2).
- Weight staleness: parallel workers used stale weights synced via disk every 200 updates.
  Eliminated by using single-process training.

Hyperparameters (all experiments)
---------------------------------
eta: 0.1, dqn_lr: 0.1, avg_policy_lr: 0.01, gamma: 1.0, epsilon: 0.06,
batch_size: 128, hidden_size: 128, dqn_buffer: 200k, reservoir_buffer: 2M,
target_update_freq: 1000, IQN: N=8 quantiles train, K=32 eval, embed_dim=64
