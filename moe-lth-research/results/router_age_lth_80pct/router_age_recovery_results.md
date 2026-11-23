# Router-Age Recovery Experiment Results

Reference seeds: 7, 17, 29

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
| 7 | 16818176 | 16777216 | 13421760 | 3355456 | 0.79999924 | expert_local_magnitude (top-k retained by magnitude per expert) | 48a320081e56c8f6... |
| 17 | 16818176 | 16777216 | 13421760 | 3355456 | 0.79999924 | expert_local_magnitude (top-k retained by magnitude per expert) | b9a83c31c1969544... |
| 29 | 16818176 | 16777216 | 13421760 | 3355456 | 0.79999924 | expert_local_magnitude (top-k retained by magnitude per expert) | ba4e6efbbd51f1e1... |

## Native-Confidence Router-Age Sweep

| Seed | Router age % | Router step | L(0) | Early AUC | L(final) | Delta vs R100 | Delta vs R0 | T(5%) | T(10%) | Tau | Sel. prob initial | Sel. prob final | Entropy final | Agreement w/ R100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0 | 0 | 171.8091 | 7503.8289 | 1.6551 | 0.1095 | 0.0000 | unreached | unreached | 1.0000 | 0.2501 | 0.5199 | 1.2240 | 0.2136 |
| 7 | 10 | 250 | 169.0243 | 7400.9794 | 1.6251 | 0.0794 | -0.0301 | unreached | 2500 | 1.0000 | 0.3447 | 0.5398 | 1.1781 | 0.2482 |
| 7 | 20 | 500 | 169.0239 | 7399.0017 | 1.6219 | 0.0763 | -0.0332 | unreached | 2500 | 1.0000 | 0.3464 | 0.5404 | 1.1752 | 0.2253 |
| 7 | 40 | 1000 | 168.7365 | 7340.7654 | 1.5715 | 0.0259 | -0.0836 | unreached | 2250 | 1.0000 | 0.3642 | 0.5568 | 1.1402 | 0.2466 |
| 7 | 60 | 1500 | 168.0362 | 7316.9281 | 1.5808 | 0.0352 | -0.0743 | unreached | 2350 | 1.0000 | 0.3924 | 0.5580 | 1.1365 | 0.2400 |
| 7 | 80 | 2000 | 167.1666 | 7266.1393 | 1.5544 | 0.0088 | -0.1007 | unreached | 2150 | 1.0000 | 0.4198 | 0.5669 | 1.1117 | 0.2517 |
| 7 | 100 | 2500 | 166.3595 | 7222.7239 | 1.5456 | 0.0000 | -0.1095 | 2450 | 2050 | 1.0000 | 0.4517 | 0.5678 | 1.1103 | 0.2722 |
| 17 | 0 | 0 | 168.4364 | 7353.9965 | 1.6671 | 0.1275 | 0.0000 | unreached | 2500 | 1.0000 | 0.2487 | 0.5147 | 1.2395 | 0.1957 |
| 17 | 10 | 250 | 165.9794 | 7189.8867 | 1.5914 | 0.0517 | -0.0758 | 2500 | 2150 | 1.0000 | 0.3334 | 0.5284 | 1.2064 | 0.2541 |
| 17 | 20 | 500 | 165.8247 | 7180.3157 | 1.5815 | 0.0418 | -0.0857 | 2450 | 2100 | 1.0000 | 0.3425 | 0.5334 | 1.1995 | 0.2715 |
| 17 | 40 | 1000 | 164.9360 | 7133.7293 | 1.5928 | 0.0532 | -0.0743 | 2500 | 2150 | 1.0000 | 0.3617 | 0.5349 | 1.1903 | 0.3177 |
| 17 | 60 | 1500 | 163.9554 | 7103.2932 | 1.5867 | 0.0470 | -0.0804 | 2500 | 2150 | 1.0000 | 0.3829 | 0.5342 | 1.1874 | 0.2947 |
| 17 | 80 | 2000 | 162.9966 | 7053.0237 | 1.5453 | 0.0057 | -0.1218 | 2250 | 1900 | 1.0000 | 0.4047 | 0.5541 | 1.1394 | 0.3166 |
| 17 | 100 | 2500 | 162.0939 | 7011.6505 | 1.5397 | 0.0000 | -0.1275 | 2200 | 1850 | 1.0000 | 0.4268 | 0.5475 | 1.1514 | 0.3123 |
| 29 | 0 | 0 | 167.4171 | 7354.6469 | 1.6515 | 0.1159 | 0.0000 | unreached | unreached | 1.0000 | 0.2505 | 0.5272 | 1.2141 | 0.1986 |
| 29 | 10 | 250 | 161.3985 | 7110.0795 | 1.5843 | 0.0488 | -0.0672 | unreached | 2450 | 1.0000 | 0.3526 | 0.5281 | 1.2173 | 0.2588 |
| 29 | 20 | 500 | 161.3302 | 7082.9496 | 1.5450 | 0.0094 | -0.1065 | unreached | 2250 | 1.0000 | 0.3603 | 0.5394 | 1.1968 | 0.3057 |
| 29 | 40 | 1000 | 160.9934 | 7077.9605 | 1.5757 | 0.0401 | -0.0758 | unreached | 2400 | 1.0000 | 0.3801 | 0.5262 | 1.2184 | 0.3212 |
| 29 | 60 | 1500 | 160.4008 | 7017.1570 | 1.5325 | -0.0030 | -0.1190 | unreached | 2150 | 1.0000 | 0.4082 | 0.5478 | 1.1659 | 0.3111 |
| 29 | 80 | 2000 | 159.7072 | 6989.4790 | 1.5337 | -0.0018 | -0.1178 | unreached | 2150 | 1.0000 | 0.4370 | 0.5689 | 1.1182 | 0.3402 |
| 29 | 100 | 2500 | 159.3897 | 6963.4740 | 1.5356 | 0.0000 | -0.1159 | unreached | 2150 | 1.0000 | 0.4564 | 0.5777 | 1.0975 | 0.3455 |

## Confidence-Matched Control

This two-pass control obtains each condition's native routes and native
capacity-priority scores, then forces those routes while applying the
temperature-scaled gate probabilities. Metadata verifies assignment and
capacity-decision agreement of 1.0 before versus after calibration.

| Seed | Router age % | Router step | L(0) | Early AUC | L(final) | Delta vs matched R100 | Delta vs matched R0 | T(5%) | T(10%) | Tau | Sel. prob initial | Sel. prob final | Entropy final | Agreement w/ R100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0 | 0 | 171.8091 | 7497.7727 | 1.6498 | 0.0227 | 0.0000 | unreached | unreached | 1.0000 | 0.2501 | 0.5251 | 1.2144 | 0.2120 |
| 7 | 40 | 1000 | 171.0973 | 7462.9438 | 1.6135 | -0.0135 | -0.0362 | unreached | 2450 | 1.7682 | 0.2501 | 0.5055 | 1.2653 | 0.2548 |
| 7 | 80 | 2000 | 170.9098 | 7456.1004 | 1.6314 | 0.0043 | -0.0184 | unreached | unreached | 2.4318 | 0.2501 | 0.4902 | 1.3099 | 0.2600 |
| 7 | 100 | 2500 | 170.8657 | 7443.4903 | 1.6271 | 0.0000 | -0.0227 | unreached | unreached | 2.7904 | 0.2501 | 0.4823 | 1.3304 | 0.2615 |

## Mean Gradient Norms During Recovery (native-confidence conditions)

| Seed | Router age % | Confidence | Mean expert grad norm | Mean router grad norm | Mean shared grad norm |
|---:|---:|---|---:|---:|---:|
| 7 | 0 | native | 0.0912 | inf | 1.6748 |
| 7 | 10 | native | 0.1166 | inf | 1.7103 |
| 7 | 20 | native | 0.1160 | inf | 1.7340 |
| 7 | 40 | native | 0.1246 | inf | 1.7828 |
| 7 | 60 | native | 0.1320 | inf | 1.8001 |
| 7 | 80 | native | 0.1380 | inf | 1.8133 |
| 7 | 100 | native | 0.1380 | inf | 1.8107 |
| 17 | 0 | native | 0.0874 | 0.1661 | inf |
| 17 | 10 | native | 0.1093 | inf | 1.6558 |
| 17 | 20 | native | 0.1117 | inf | nan |
| 17 | 40 | native | 0.1200 | inf | 1.7373 |
| 17 | 60 | native | 0.1280 | inf | 1.7534 |
| 17 | 80 | native | 0.1316 | inf | 1.7537 |
| 17 | 100 | native | 0.1375 | inf | 1.7441 |
| 29 | 0 | native | 0.1005 | inf | nan |
| 29 | 10 | native | 0.1184 | inf | 1.6226 |
| 29 | 20 | native | 0.1216 | inf | 1.6408 |
| 29 | 40 | native | 0.1228 | inf | 1.6457 |
| 29 | 60 | native | 0.1285 | inf | 1.6598 |
| 29 | 80 | native | 0.1385 | inf | 1.7145 |
| 29 | 100 | native | 0.1436 | inf | 1.7296 |
| 7 | 0 | matched | 0.0916 | inf | 1.6505 |
| 7 | 40 | matched | 0.0981 | 0.1211 | inf |
| 7 | 80 | matched | 0.0979 | 0.0906 | inf |
| 7 | 100 | matched | 0.0990 | 0.0793 | inf |

Raw records: [router_age_recovery_all_records.json](router_age_recovery_all_records.json),
condition table: [router_age_recovery_aggregate.csv](router_age_recovery_aggregate.csv),
paired table: [router_age_recovery_paired.csv](router_age_recovery_paired.csv), and
recovery curves: [router_age_recovery_curves.svg](router_age_recovery_curves.svg).
