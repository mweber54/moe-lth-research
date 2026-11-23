# Post-pruning adaptation factorial findings

## Status and interpretation

This analysis is complete only after every registered branch, endpoint hash,
diagnostic grid, and update accounting check passes. The independent unit is
the source seed (n=5). The 1000–1500 window was selected after prior
trajectories were observed; this is not an overall preregistration.

Freezing contrasts estimate the effect of restricting adaptation under this
specific training policy. They do not identify a mechanism or establish
universality, a pruning algorithm, a routing mechanism, or pre-pruning
recoverability prediction.

## Seed-level endpoint contrasts

Individual values are in `seed_level_factorial_contrasts.csv`; descriptive
seed-level summaries (including sample SD and t intervals) are in
`seed_level_contrast_summary.csv`. Intervals are not token-level uncertainty.

| Contrast | Mean | Median | Range | Sample SD | Descriptive 95% t interval |
|---|---:|---:|---:|---:|---:|
| E_E_sparse_minus_unpruned | -0.004528 | -0.002618 | 0.055419 | 0.020760 | [-0.030305, 0.021248] |
| E_R_sparse_minus_unpruned | 0.016120 | 0.013566 | 0.013383 | 0.005627 | [0.009134, 0.023107] |
| I_sparse_minus_unpruned | 0.003527 | 0.002967 | 0.015419 | 0.005808 | [-0.003684, 0.010738] |
| sparse:I | -0.009477 | -0.006944 | 0.021386 | 0.008461 | [-0.019983, 0.001029] |
| sparse:P_E | 0.163577 | 0.168078 | 0.028785 | 0.012693 | [0.147816, 0.179338] |
| sparse:P_R | 0.036688 | 0.038214 | 0.024942 | 0.009999 | [0.024273, 0.049104] |
| sparse:P_RE | 0.190789 | 0.195293 | 0.039654 | 0.016751 | [0.169989, 0.211588] |
| sparse:experts_frozen:absolute_CE_reduction | 0.545119 | 0.528597 | 0.089990 | 0.038309 | [0.497552, 0.592686] |
| sparse:full:absolute_CE_reduction | 0.708696 | 0.696675 | 0.083148 | 0.036081 | [0.663896, 0.753497] |
| sparse:router_experts_frozen:absolute_CE_reduction | 0.517908 | 0.501382 | 0.090808 | 0.038591 | [0.469991, 0.565825] |
| sparse:router_frozen:absolute_CE_reduction | 0.672008 | 0.648019 | 0.077168 | 0.036728 | [0.626404, 0.717611] |
| unpruned:I | -0.013004 | -0.010344 | 0.015461 | 0.006498 | [-0.021072, -0.004935] |
| unpruned:P_E | 0.168105 | 0.166132 | 0.053131 | 0.021091 | [0.141917, 0.194294] |
| unpruned:P_R | 0.020568 | 0.017870 | 0.018320 | 0.007092 | [0.011762, 0.029374] |
| unpruned:P_RE | 0.175670 | 0.167948 | 0.050971 | 0.020714 | [0.149950, 0.201390] |
| unpruned:experts_frozen:absolute_CE_reduction | 0.117845 | 0.114860 | 0.031440 | 0.014221 | [0.100187, 0.135502] |
| unpruned:full:absolute_CE_reduction | 0.285950 | 0.285346 | 0.039900 | 0.017331 | [0.264431, 0.307469] |
| unpruned:router_experts_frozen:absolute_CE_reduction | 0.110280 | 0.106027 | 0.028820 | 0.011665 | [0.095796, 0.124765] |
| unpruned:router_frozen:absolute_CE_reduction | 0.265382 | 0.267664 | 0.027291 | 0.011292 | [0.251361, 0.279403] |

The factorial interaction is `L11 - L10 - L01 + L00`, where 1 denotes
training and 0 denotes freezing for router and experts, respectively.
The row `P_RE` is the joint-freezing penalty. Sparse-excess penalties are
reported as sparse minus unpruned.

## Trajectory and function diagnostics

Validation trajectories are in `factorial_trajectory_metrics.csv` and
`trajectory_ce.svg`. Trapezoidal integrated excess CE is calculated only
over the observed offsets 0, 25, 100, 250, and 500; see
`trajectory_integrated_excess.csv` and its seed-level summary. Update
norms, clipping, AMP skips, and path-length/net-displacement accounting
are in `optimizer_compute_accounting.csv`. Routing, expert-function,
shared-representation, and diagnostic-overhead tables remain in each
attempt directory.

Movement and routing summaries are descriptive diagnostics; they are not
causal decompositions of validation CE.

## Separate late dense audit

Dense CE rose for 4/5 seeds in the
historical 4500–5000 interval. The complete audit is in
`late_dense_audit.md`; it is separate from the factorial and does not
establish overfitting.

## Bounded follow-up

A single falsifiable follow-up would be a separately authorized
router-only adaptation intervention with the same step-1000 state and
batch stream, to test whether the interaction pattern persists when
expert parameters are held fixed. It is not launched or included here.
