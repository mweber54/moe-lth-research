# Exploratory outcome comparisons

Each row averages observed layer/expert matrices within one configuration. Standard deviations describe heterogeneity; they are not uncertainty estimates. No weights or experts are treated as independent experimental replicates.

| Case | Matrix | CE gap | Initial mean survivor magnitude | Mean magnitude growth ratio | Mean normalized drift | Mean survivor W1 |
|---|---|---:|---:|---:|---:|---:|
| e32/seed_7/final_s0.80 | router | -0.035600 | 0.0428863 | 1.1676 | 0.3847 | 0.00766046 |
| e32/seed_7/final_s0.80 | fc1 | -0.035600 | 0.0588082 | 1.0296 | 0.3561 | 0.00565856 |
| e32/seed_7/final_s0.80 | fc2 | -0.035600 | 0.0447272 | 1.0877 | 0.4189 | 0.00673574 |
| e32/seed_7/final_s0.85 | router | -0.021205 | 0.0436615 | 1.1506 | 0.3895 | 0.00702346 |
| e32/seed_7/final_s0.85 | fc1 | -0.021205 | 0.0651941 | 1.0247 | 0.3491 | 0.00621585 |
| e32/seed_7/final_s0.85 | fc2 | -0.021205 | 0.0518031 | 1.0850 | 0.3989 | 0.00748658 |
| e32/seed_17/final_s0.80 | router | 0.021117 | 0.0429185 | 1.1410 | 0.3678 | 0.00653548 |
| e32/seed_17/final_s0.80 | fc1 | 0.021117 | 0.0594978 | 1.0255 | 0.3363 | 0.00572775 |
| e32/seed_17/final_s0.80 | fc2 | 0.021117 | 0.0463885 | 1.0776 | 0.3886 | 0.00673793 |
| e16/seed_17/final_s0.80 | router | 0.001161 | 0.0460614 | 1.1550 | 0.3378 | 0.00764871 |
| e16/seed_17/final_s0.80 | fc1 | 0.001161 | 0.0607481 | 1.0293 | 0.3119 | 0.00600917 |
| e16/seed_17/final_s0.80 | fc2 | 0.001161 | 0.0470021 | 1.0844 | 0.3522 | 0.00681799 |

Negative CE gap denotes a sparse endpoint that beats its matched dense endpoint. These are descriptive comparisons of four selected configurations. Architecture, seed and sparsity differ across the outcome groups, so apparent differences in survivor magnitudes, growth, shifts or heterogeneity cannot be attributed to success itself. See the per-layer heterogeneity and distance tables for variation hidden by these averages. Distributional results do not establish geometric alignment.
