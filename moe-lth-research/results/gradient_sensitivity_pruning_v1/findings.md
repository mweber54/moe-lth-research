# Gradient sensitivity pruning: stopped at matching gate

Sensitivity passed the registered split-half reliability gate. The registered matching algorithm failed to obtain acceptable feature balance, sensitivity contrast, structural balance and retained dose/coverage jointly in all five sources. No selected coordinates were applied to a model, no intervention or validation losses were measured, no controls were reused, and no production training was launched.

| Source seed | Reliability eligible / 256 | Median stratum rank agreement | Matching included / 256 | Retained dose per branch | Matching passed |
|---|---:|---:|---:|---:|---|
| 7 | 256 / 256 | 0.9702 | 0 / 256 | 0.0000% | False |
| 17 | 256 / 256 | 0.9753 | 0 / 256 | 0.0000% | False |
| 29 | 256 / 256 | 0.9772 | 0 / 256 | 0.0000% | False |
| 43 | 256 / 256 | 0.9712 | 0 / 256 | 0.0000% | False |
| 61 | 256 / 256 | 0.9757 | 0 / 256 | 0.0000% | False |

| Source seed | Median row-matched candidate pairs | Median column-cycle pairs | Column-cycle pair range |
|---|---:|---:|---:|
| 7 | 11152.0 | 4.0 | 0–113 |
| 17 | 11765.5 | 5.0 | 0–292 |
| 29 | 11657.5 | 4.0 | 0–249 |
| 43 | 10800.5 | 4.0 | 0–572 |
| 61 | 11411.0 | 4.5 | 0–566 |

Excluded tensor diagnostics across sources: {'insufficient_cycles': 1280, 'registered_balance_failure': 0}. These are descriptive tensor counts, not independent training replications.

Matching used the frozen bottom-half/50-stratum population, exact paired sign and row identity, fixed magnitude/growth/variation calipers, then directed column cycles. Every reported candidate has exact equal H/L counts in each row and column. This controls degree distributions, not exact connectivity. Candidate coordinate lists, degree arrays and hashes are retained for excluded as well as included tensors. No alternative matcher, dose sweep, relaxed caliper, batch expansion or score change was attempted.

A tensor required at least 656 pairs, absolute SMD <=0.05 and KS <=0.05 for magnitude, recent slope and absolute variation, removed squared-magnitude relative difference <=0.005, and mean high sensitivity at least twice low. Every source required >=80% tensors, >=26/32 per layer/projection and >=0.5% of all expert coordinates removed in each branch. A stop demonstrates a limitation of this registered matching algorithm/design, not impossibility of every conceivable matching procedure.

Reliability statistics, accepted/proposed/dropped traffic and gate summaries are retained. Scores are batchwise mean absolute weight-times-task-gradients in FP32 native routing, accumulated in FP64, with signed means, gradient second moments, sample batch variances and sign consistency. Missing traffic was never equated with unimportance. Second moments are not Hessian diagonals. Descriptive exposure/reliability associations do not establish a routing mechanism.

The native configuration audit independently verified model/data/routing/training identity across all historical checkpoints and the sealed native reference, along with original training/continuation/validation hashes and separate train/validation text files. This is not a control-reuse compatibility claim. The numerical checks verify the measurement preservation contract, tiny native full-versus-streaming gradients, cancellation distinction, coordinate identity, candidate uniqueness/disjointness and reconstructed degree equality. Two first calibration attempts failed an overly exact floating-point Jensen audit; they are preserved. The documented engineering correction only changed comparison tolerance; scientific definitions and gates stayed fixed. Pre-scoring fixture corrections and original sealed code snapshots are also preserved. The first matching attempt hit the process file-descriptor limit while transferring Torch degree tensors; the preserved engineering correction changes only interprocess transport to NumPy arrays. SciPy emitted an exact-p-value fallback warning for some KS calls; only the unchanged empirical KS statistic is used for balance, with no p-values or altered thresholds.

Signed group Taylor checks, individual native finite differences and route-boundary diagnostics, fresh-reset/persistent-mask checks, the seed17 production pilot and interruption/resumption tests are NOT_RUN after the registered matching stop. They are not marked passed. All 15 planned branch conditions and all seed effects remain visible with unavailable outcomes. Annotated outcome figures show NOT RUN, not zero effects.

The final research question remains unanswered: reproducible sensitivity alone does not establish information beyond magnitude/growth/structure about persistent pruning damage. There are zero source-seed pruning replications and no meaningful mean/SD/SE of pruning effects. No equivalence, novelty, causality or general-policy claim follows. A full pruning-policy test is not recommended on this evidence and was not implemented.
