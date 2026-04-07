# Research Observations — Distributional NFSP

## Baseline NFSP Convergence

*(To be filled after running baseline experiment)*

- Expected behavior: exploitability should decrease toward 0 over 10M episodes
- Heinrich & Silver (2016) report convergence on Leduc poker with these hyperparameters
- Key metric: NashConv (sum of players' incentive to deviate from current strategy)

## IQN vs DQN Comparison (Risk-Neutral)

*(To be filled after running IQN neutral experiment)*

- Hypothesis: IQN should match or improve upon DQN baseline since it learns the full
  return distribution, providing a richer signal for best-response computation.
- Key comparison: convergence speed and final exploitability

## Risk-Sensitive Agent Behavior

### Risk-Averse (CVaR alpha=0.25)

*(To be filled after running risk-averse experiment)*

- CVaR focuses on the worst 25% of quantiles during action selection
- Expected: more conservative play, potentially slower convergence but more robust strategies
- Training uses uniform quantiles — only action selection is distorted

### Risk-Seeking (top 25% quantiles)

*(To be filled after running risk-seeking experiment)*

- Focuses on best-case outcomes (quantiles in [0.75, 1.0])
- Expected: aggressive play, may exploit risk-averse opponents but be exploitable itself
- Key question: does risk-seeking behavior improve or hurt convergence to Nash equilibrium?

## Head-to-Head Analysis

*(To be filled after running tournament)*

- Matchups: baseline vs IQN-neutral, baseline vs risk-averse, baseline vs risk-seeking,
  risk-averse vs risk-seeking, etc.
- In a zero-sum game, Nash equilibrium strategies should be unexploitable regardless of
  opponent — risk distortion may hurt equilibrium quality but change behavioral style.
