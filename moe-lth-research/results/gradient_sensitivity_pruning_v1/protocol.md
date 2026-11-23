# Gradient sensitivity pruning v1

Prospective intervention motivated by exploratory work, not preregistration of the whole project. The machine-readable registration is authoritative and is sealed before gradients or intervention losses.

The fixed source model is measured on 32 complete past training batches, disjoint 16-batch halves. A is mean absolute weight-times-task-gradient; S is its negative signed mean. Batch variance is neither token gradient variance nor curvature. Native hard routing, capacity, gates, and task-only FP32 scoring are preserved. No exposure normalization.

Reliability is assessed within each tensor and 50 current-magnitude strata. Each tensor must meet all registered rank, overlap, cross-half separation and traffic thresholds; every seed and layer/projection must meet coverage. Failure stops the experiment without changing batches, scores, or thresholds. Full-32 selection is forbidden before a passing review.

If reliability passes, pair within row/sign/magnitude stratum under fixed growth/variation calipers, then select directed column cycles to equalize both row and column degrees. Target 1% per branch; only the registered minimum-dose/coverage reductions apply. All five masks are sealed before any intervention losses. Equal row/column degrees do not determine exact connectivity.

If both gates pass, use a seed17 engineering pilot and all five source seeds, H/L/C shared fresh resets, original steps1001..2500, 1500 attempted FP16 updates, task-only token-weighted FP32 validation on the fixed grid. Audit existing controls before reuse, otherwise run new controls. Failures and skips remain visible. Mask zeros persist via hooks and post-update masking.

Primary endpoint is final validation CE H minus L; immediate calibration Taylor checks and validation changes are separate diagnostics. Report every source seed before descriptive mean/median/range/SD/SE. No independent replication is attributed to weights, experts or batches. No novelty, mechanism, equivalence or general-policy claim is authorized.

On a measurement/matching stop, downstream artifacts explicitly record NOT_RUN, not zero effects. No production result is inferred.
