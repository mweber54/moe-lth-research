# Geometric findings: dense vs. sparse that matched/beat dense vs. sparse that did not

Source: `results/weight_distribution_analysis/` (`router_comparison.csv`, `exploratory_summary.csv`, `router_total_drift.csv`, `checkpoint_audit.csv`). Each row is one configuration, 5000-step extended continuation, surviving weights only, averaged over layers (and experts). "Matched/beat dense" = sparse val CE <= matched dense val CE at step 5000 (negative CE gap).

Caveat: this is n=4 selected cases with different architecture/seed/sparsity per group, so the differences below are descriptive, not attributable to winning. Per `results/lth_definition_audit/README.md`, none of these is a strict original-LTH winning ticket under the 5% CE screen.

## Weight geometry (weight drift/growth during continuation)

| Group | Case | CE gap (sparse - dense) | Router norm. drift (sparse) | Router norm. drift (matched dense) | Router mag. growth (sparse / dense) | Router frac. grown (sparse / dense) | fc1 drift / growth | fc2 drift / growth |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Matched/beat dense | e32 s7 80% | -0.0356 | 0.385 | 0.340 | 1.168 / 1.135 | 0.604 / 0.568 | 0.356 / 1.030 | 0.419 / 1.088 |
| Matched/beat dense | e32 s7 85% | -0.0212 | 0.390 | 0.340 | 1.151 / 1.135 | 0.598 / 0.568 | 0.349 / 1.025 | 0.399 / 1.085 |
| Sparse, not winning | e32 s17 80% | +0.0211 | 0.368 | 0.380 | 1.141 / 1.152 | 0.591 / 0.573 | 0.336 / 1.026 | 0.389 / 1.078 |
| Sparse, not winning | e16 s17 80% | +0.0012 | 0.338 | 0.353 | 1.155 / 1.162 | 0.624 / 0.579 | 0.312 / 1.029 | 0.352 / 1.084 |

Dense fc1/fc2 drift was not tabulated in the summary (dense reference is router only).

## Router total drift (full router matrix, `router_total_drift.csv`)

| Group | Case | Sparse norm. drift | Dense norm. drift | Sparse - dense |
|---|---|---:|---:|---:|
| Matched/beat dense | e32 s7 80% | 0.408 | 0.388 | +0.020 |
| Matched/beat dense | e32 s7 85% | 0.402 | 0.388 | +0.014 |
| Sparse, not winning | e32 s17 80% | 0.387 | 0.435 | -0.048 |
| Sparse, not winning | e16 s17 80% | 0.347 | 0.407 | -0.060 |

## Takeaways

- Winners moved their routers *more* than matched dense (drift +0.014 to +0.020; growth ratio above dense). Non-winners moved them *less* than dense (-0.048 to -0.060) with lower growth than dense.
- The sign of sparse-minus-dense router drift separates the two groups in all 4 cases, but with n=4 across different seeds/architectures this is a hypothesis, not a result.
- Distributional geometry (magnitude growth, drift, Wasserstein) does not by itself establish alignment (see source `exploratory_findings.md`).

## Related optimization geometry (all sparse vs dense, not winner-split)

`results/cross_arch_optimization_geometry/diagnostic_rerun/final_report.md` (85%, 18 sparse/dense pairs, 3 seeds): forced-route gradient alignment 0.14-0.19, functional similarity 0.32-0.37, update alignment 0.04-0.06. Favorable ticket gaps did not coincide with higher alignment (best-gap ages R20 in 8E were not the highest-alignment ages; R100 had highest gradient alignment in every seed but not the best gap). Not split by winning status, since no run was within the winning threshold.
