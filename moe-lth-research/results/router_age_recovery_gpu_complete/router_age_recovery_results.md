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
| 7 | 0 | 0 | 7.2510 | 2044.2201 | 1.3505 | 0.0053 | 0.0000 | 350 | 200 | 1.0000 | 0.2901 | 0.5900 | 1.0647 | 0.5768 |
| 7 | 10 | 250 | 8.1546 | 2039.2655 | 1.3484 | 0.0032 | -0.0020 | 250 | 150 | 1.0000 | 0.3076 | 0.5971 | 1.0490 | 0.5917 |
| 7 | 20 | 500 | 8.3496 | 2034.8463 | 1.3484 | 0.0031 | -0.0021 | 200 | 150 | 1.0000 | 0.3270 | 0.5985 | 1.0450 | 0.5916 |
| 7 | 40 | 1000 | 7.6975 | 2003.2177 | 1.3478 | 0.0025 | -0.0027 | 200 | 100 | 1.0000 | 0.3550 | 0.6019 | 1.0363 | 0.5928 |
| 7 | 60 | 1500 | 6.8776 | 1975.8177 | 1.3467 | 0.0014 | -0.0038 | 150 | 100 | 1.0000 | 0.3878 | 0.6046 | 1.0288 | 0.5926 |
| 7 | 80 | 2000 | 6.1670 | 1953.6781 | 1.3455 | 0.0002 | -0.0050 | 150 | 100 | 1.0000 | 0.4240 | 0.6041 | 1.0271 | 0.5931 |
| 7 | 100 | 2500 | 5.7422 | 1940.6488 | 1.3452 | 0.0000 | -0.0053 | 150 | 100 | 1.0000 | 0.4514 | 0.6052 | 1.0221 | 0.5926 |
| 17 | 0 | 0 | 5.2493 | 2052.1002 | 1.4196 | 0.0222 | 0.0000 | 300 | 200 | 1.0000 | 0.3191 | 0.5777 | 1.0856 | 0.5283 |
| 17 | 10 | 250 | 4.8504 | 2011.4442 | 1.4078 | 0.0104 | -0.0118 | 250 | 150 | 1.0000 | 0.2890 | 0.5932 | 1.0456 | 0.5359 |
| 17 | 20 | 500 | 4.6362 | 1993.6079 | 1.4035 | 0.0061 | -0.0161 | 200 | 150 | 1.0000 | 0.3041 | 0.5928 | 1.0456 | 0.5449 |
| 17 | 40 | 1000 | 4.2991 | 1970.6593 | 1.3968 | -0.0006 | -0.0228 | 150 | 100 | 1.0000 | 0.3399 | 0.5930 | 1.0435 | 0.5468 |
| 17 | 60 | 1500 | 3.9103 | 1956.0951 | 1.3973 | -0.0001 | -0.0223 | 150 | 100 | 1.0000 | 0.3766 | 0.5949 | 1.0385 | 0.5455 |
| 17 | 80 | 2000 | 3.6399 | 1947.0952 | 1.3972 | -0.0002 | -0.0224 | 150 | 100 | 1.0000 | 0.4066 | 0.5958 | 1.0344 | 0.5453 |
| 17 | 100 | 2500 | 3.5143 | 1943.3388 | 1.3974 | 0.0000 | -0.0222 | 150 | 100 | 1.0000 | 0.4263 | 0.5974 | 1.0298 | 0.5444 |
| 29 | 0 | 0 | 6.6763 | 1981.5290 | 1.3380 | 0.0138 | 0.0000 | 300 | 150 | 1.0000 | 0.3216 | 0.5825 | 1.0795 | 0.6343 |
| 29 | 10 | 250 | 6.8028 | 1979.6045 | 1.3420 | 0.0178 | 0.0040 | 250 | 150 | 1.0000 | 0.3037 | 0.5991 | 1.0428 | 0.6311 |
| 29 | 20 | 500 | 6.8490 | 1959.7461 | 1.3296 | 0.0054 | -0.0083 | 200 | 150 | 1.0000 | 0.3252 | 0.5964 | 1.0491 | 0.6357 |
| 29 | 40 | 1000 | 6.7468 | 1943.0567 | 1.3273 | 0.0030 | -0.0107 | 200 | 100 | 1.0000 | 0.3554 | 0.5961 | 1.0474 | 0.6349 |
| 29 | 60 | 1500 | 6.7043 | 1934.6358 | 1.3263 | 0.0020 | -0.0117 | 150 | 100 | 1.0000 | 0.3943 | 0.6010 | 1.0350 | 0.6367 |
| 29 | 80 | 2000 | 6.1820 | 1917.3799 | 1.3238 | -0.0004 | -0.0142 | 150 | 100 | 1.0000 | 0.4399 | 0.6063 | 1.0218 | 0.6368 |
| 29 | 100 | 2500 | 5.7766 | 1904.7752 | 1.3242 | 0.0000 | -0.0138 | 150 | 100 | 1.0000 | 0.4701 | 0.6098 | 1.0109 | 0.6428 |

## Confidence-Matched Control

This two-pass control obtains each condition's native routes and native
capacity-priority scores, then forces those routes while applying the
temperature-scaled gate probabilities. Metadata verifies assignment and
capacity-decision agreement of 1.0 before versus after calibration.

| Seed | Router age % | Router step | L(0) | Early AUC | L(final) | Delta vs matched R100 | Delta vs matched R0 | T(5%) | T(10%) | Tau | Sel. prob initial | Sel. prob final | Entropy final | Agreement w/ R100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0 | 0 | 7.2510 | 2044.6899 | 1.3513 | -0.0016 | 0.0000 | 350 | 200 | 1.0000 | 0.2901 | 0.5900 | 1.0647 | 0.5768 |
| 7 | 40 | 1000 | 7.7005 | 2015.7847 | 1.3500 | -0.0028 | -0.0013 | 200 | 150 | 1.4920 | 0.2901 | 0.5733 | 1.1080 | 0.5852 |
| 7 | 80 | 2000 | 6.7197 | 1990.6819 | 1.3518 | -0.0011 | 0.0005 | 200 | 150 | 2.2599 | 0.2901 | 0.5487 | 1.1678 | 0.5809 |
| 7 | 100 | 2500 | 6.5538 | 1988.4616 | 1.3529 | 0.0000 | 0.0016 | 200 | 150 | 2.5816 | 0.2901 | 0.5410 | 1.1882 | 0.5767 |

## Mean Gradient Norms During Recovery (native-confidence conditions)

| Seed | Router age % | Confidence | Mean expert grad norm | Mean router grad norm | Mean shared grad norm |
|---:|---:|---|---:|---:|---:|
| 7 | 0 | native | 0.1016 | 0.0533 | inf |
| 7 | 10 | native | 0.1079 | 0.0512 | inf |
| 7 | 20 | native | 0.1169 | 0.0511 | inf |
| 7 | 40 | native | 0.1172 | 0.0489 | inf |
| 7 | 60 | native | 0.1169 | 0.0462 | inf |
| 7 | 80 | native | 0.1104 | 0.0426 | inf |
| 7 | 100 | native | 0.1103 | 0.0410 | inf |
| 17 | 0 | native | 0.0987 | 0.0526 | 0.9553 |
| 17 | 10 | native | 0.1002 | 0.0491 | 0.9510 |
| 17 | 20 | native | 0.1007 | 0.0482 | 0.9536 |
| 17 | 40 | native | 0.1026 | 0.0448 | 0.9595 |
| 17 | 60 | native | 0.1031 | 0.0427 | 0.9548 |
| 17 | 80 | native | 0.1035 | 0.0417 | 0.9544 |
| 17 | 100 | native | 0.1039 | 0.0409 | 0.9534 |
| 29 | 0 | native | 0.1015 | 0.0538 | inf |
| 29 | 10 | native | 0.1049 | 0.0510 | inf |
| 29 | 20 | native | 0.1062 | 0.0509 | inf |
| 29 | 40 | native | 0.1066 | 0.0484 | inf |
| 29 | 60 | native | 0.1091 | 0.0467 | inf |
| 29 | 80 | native | 0.1086 | 0.0438 | inf |
| 29 | 100 | native | 0.1088 | 0.0426 | inf |
| 7 | 0 | matched | 0.1016 | 0.0534 | inf |
| 7 | 40 | matched | 0.1067 | 0.0369 | inf |
| 7 | 80 | matched | 0.1048 | 0.0268 | inf |
| 7 | 100 | matched | 0.1037 | 0.0245 | inf |

Raw records: [router_age_recovery_all_records.json](router_age_recovery_all_records.json),
condition table: [router_age_recovery_aggregate.csv](router_age_recovery_aggregate.csv),
paired table: [router_age_recovery_paired.csv](router_age_recovery_paired.csv), and
recovery curves: [router_age_recovery_curves.svg](router_age_recovery_curves.svg).
