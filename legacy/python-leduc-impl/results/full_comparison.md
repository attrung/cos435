# NFSP Experiment Results — Leduc Poker

Generated automatically.

## Exploitability Summary

| Experiment | Final Exploitability | Min Exploitability | Episodes |
|---|---|---|---|
| nfsp_baseline_seed42 | 0.359955 | 0.239097 | 3,000,000 |
| nfsp_iqn_averse_seed42 | 0.704781 | 0.580970 | 3,000,000 |
| nfsp_iqn_mean_var_05_seed42 | 2.020971 | 1.596549 | 3,000,000 |
| nfsp_iqn_mean_var_seed42 | 0.567435 | 0.485925 | 3,000,000 |
| nfsp_iqn_neutral_seed42 | 0.284204 | 0.284204 | 3,000,000 |

## Training Speed

- **nfsp_baseline_seed42**: avg 225 eps/sec, total 230.5 min
- **nfsp_iqn_averse_seed42**: avg 111 eps/sec, total 455.3 min
- **nfsp_iqn_mean_var_05_seed42**: avg 109 eps/sec, total 471.0 min
- **nfsp_iqn_mean_var_seed42**: avg 110 eps/sec, total 462.2 min
- **nfsp_iqn_neutral_seed42**: avg 108 eps/sec, total 476.0 min

## Figures

- `results/figures/exploitability_comparison.png` — log-scale comparison
- `results/figures/exploitability_linear.png` — linear-scale comparison
