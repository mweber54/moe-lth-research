# Router-Age Recovery Experiment Results

Reference seeds: 7

Fixed 80%-magnitude-pruned expert state paired with router
checkpoints `R_t` from t in {0, 10, 20, 40, 60, 80, 100}% of the reference
training trajectory. Shared parameters, pruned expert weights, and the
pruning mask are byte-identical across every router-age condition within a
seed (verified via SHA-256 hashes recorded in each condition's
`metadata.json`). Only the router parameters differ across conditions.

Recovery budget(s): 2500 sparse steps.
Early AUC is trapezoidal validation-loss area over the exact first
1250 steps. Recovery fraction is
`(L_pruned_initial - L_final) / (L_pruned_initial - L_dense)` when the
pruning-induced degradation is positive. Lower loss and lower AUC are better;
reported deltas are condition minus the same-control R100/R0 baseline, so
negative is better. Threshold times are observed on the declared validation
snapshot grid and are recorded as `unreached` otherwise.

## Pruning Summary (computed once per seed, applied to every router age)

| Seed | Total expert params | Prunable weights | Pruned | Surviving prunable | Realized sparsity | Method | Mask hash |
|---:|---:|---:|---:|---:|---:|---|---|
| 7 | 16818176 | 16777216 | 13421760 | 3355456 | 0.79999924 | expert_local_magnitude (top-k retained by magnitude per expert) | 7c679a7c017789c7... |

## Native-Confidence Router-Age Sweep

| Seed | Router age % | Router step | L(0) | Early AUC | L(final) | Delta vs R100 | Delta vs R0 | T(5%) | T(10%) | Tau | Sel. prob initial | Sel. prob final | Entropy final | Agreement w/ R100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0 | 0 | 172.2499 | 7593.8180 | 1.8619 | 0.1494 | 0.0000 | unreached | unreached | 1.0000 | 0.2502 | 0.4503 | 1.3791 | 0.2150 |
| 7 | 10 | 250 | 167.8695 | 7402.2987 | 1.8002 | 0.0877 | -0.0617 | unreached | 2250 | 1.0000 | 0.3559 | 0.4892 | 1.2880 | 0.1706 |
| 7 | 20 | 500 | 167.8504 | 7392.3305 | 1.8046 | 0.0921 | -0.0574 | unreached | 2250 | 1.0000 | 0.3590 | 0.4898 | 1.2837 | 0.1950 |
| 7 | 40 | 1000 | 167.5653 | 7393.8039 | 1.8014 | 0.0889 | -0.0605 | unreached | 2250 | 1.0000 | 0.3711 | 0.4916 | 1.2805 | 0.1996 |
| 7 | 60 | 1000 | 167.5653 | 7396.0639 | 1.7955 | 0.0830 | -0.0664 | unreached | 2250 | 1.0000 | 0.3711 | 0.4921 | 1.2804 | 0.1986 |
| 7 | 80 | 2500 | 166.0567 | 7257.1104 | 1.7094 | -0.0031 | -0.1526 | 2200 | 1900 | 1.0000 | 0.4458 | 0.5308 | 1.1832 | 0.3006 |
| 7 | 100 | 2500 | 166.0567 | 7253.7113 | 1.7125 | 0.0000 | -0.1494 | 2200 | 1900 | 1.0000 | 0.4458 | 0.5273 | 1.1912 | 0.3038 |

## Confidence-Matched Control

This two-pass control obtains each condition's native routes and native
capacity-priority scores, then forces those routes while applying the
temperature-scaled gate probabilities. Metadata verifies assignment and
capacity-decision agreement of 1.0 before versus after calibration.

| Seed | Router age % | Router step | L(0) | Early AUC | L(final) | Delta vs matched R100 | Delta vs matched R0 | T(5%) | T(10%) | Tau | Sel. prob initial | Sel. prob final | Entropy final | Agreement w/ R100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0 | 0 | 172.2499 | 7596.8124 | 1.8611 | 0.1170 | 0.0000 | unreached | unreached | 1.0000 | 0.2502 | 0.4519 | 1.3785 | 0.2221 |
| 7 | 40 | 1000 | 170.8543 | 7504.0563 | 1.8035 | 0.0593 | -0.0577 | unreached | 2250 | 1.8027 | 0.2502 | 0.4520 | 1.3883 | 0.2441 |
| 7 | 80 | 2500 | 171.0183 | 7485.5200 | 1.7345 | -0.0096 | -0.1266 | 2300 | 2000 | 2.6811 | 0.2502 | 0.4602 | 1.3608 | 0.2899 |
| 7 | 100 | 2500 | 171.0183 | 7502.5011 | 1.7441 | 0.0000 | -0.1170 | 2350 | 2050 | 2.6811 | 0.2502 | 0.4525 | 1.3835 | 0.2945 |

## Mean Gradient Norms During Recovery (native-confidence conditions)

| Seed | Router age % | Confidence | Mean expert grad norm | Mean router grad norm | Mean shared grad norm |
|---:|---:|---|---:|---:|---:|
| 7 | 0 | native | 0.0925 | inf | 1.6993 |
| 7 | 10 | native | 0.1206 | inf | 1.7796 |
| 7 | 20 | native | 0.1196 | inf | 1.7592 |
| 7 | 40 | native | 0.1241 | inf | 1.8155 |
| 7 | 60 | native | 0.1234 | inf | 1.8241 |
| 7 | 80 | native | 0.1509 | inf | 1.8893 |
| 7 | 100 | native | 0.1501 | inf | 1.8804 |
| 7 | 0 | matched | 0.0914 | inf | nan |
| 7 | 40 | matched | 0.0989 | inf | 1.7687 |
| 7 | 80 | matched | 0.0988 | 0.0929 | inf |
| 7 | 100 | matched | 0.0988 | 0.0938 | inf |

Raw records: [router_age_recovery_all_records.json](router_age_recovery_all_records.json),
condition table: [router_age_recovery_aggregate.csv](router_age_recovery_aggregate.csv),
paired table: [router_age_recovery_paired.csv](router_age_recovery_paired.csv), and
recovery curves: [router_age_recovery_curves.svg](router_age_recovery_curves.svg).
