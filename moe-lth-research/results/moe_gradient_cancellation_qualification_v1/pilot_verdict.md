# Pilot verdict: gradient cancellation qualification v1

Cell: `seed7_A`, checkpoint step 250. Both frozen 90-probe halves completed.

## Answers

1. **Does reproducible gradient opposition represent substantial underlying gradient activity?** The discovery primary population contains 2,664,116 coordinates (3.9698% of qualified coordinates) above the within-matrix 90th activity percentile and with `C >= 0.5`. It contains 37.5724% of all qualified absolute cancelled mass `K`. The mean within-matrix Pearson correlation between aggregate-gradient magnitude and `K` is only 0.0655, while the mean correlation between `K` and domain probe second moment is 0.4650. This establishes substantial signal activity for a sparse subset, not importance.

2. **Do high-activity, high-cancellation connections exist among small weights?** Yes, 528,019 primary selections (19.8197% of all primary selections) are in the bottom 20% within their own matrix by current absolute magnitude; 92.7220% retain high cancellation in validation.
That proportion is not enriched relative to the 20% bin population. Across matrices, mean Pearson correlations with magnitude are 0.0003 for cancellation and 0.0192 for activity, and the six magnitude bins show nearly flat high-cancellation prevalence. Thus the pilot demonstrates existence among small weights, not a magnitude-selected mechanism.

3. **Is the structure reproducible across independent probes?** The mean matrix-level validation joint retention is 82.2241%, cancellation-only retention is 90.1621%, and same opposition-direction retention is 99.1993%. These are descriptive matrix summaries; coordinates and matrices are not independent replications.

4. **Are the measurements numerically trustworthy?** Yes for the pilot: numerical validation passes, direct mixed-gradient reconstruction has max absolute error 7.460e-11 and relative error 1.436e-06; cancellation bounds, nonnegative `K`, triangle inequality, task-only objective, shared routing/capacity attribution, and unchanged model/RNG state all pass.

5. **Are existing artifacts sufficient for subsequent temporal and matched-selection analyses?** They are sufficient for this cell's checkpoint-250 selection features and for reuse by future analysis, but not for temporal claims: checkpoint 500 and the remaining source cells have not been replayed with signed means. Signed aggregates, not the original sign-only artifacts, must be used for any matched-selection extension.

## Limitations and interpretation

- The experiment is one source seed, one order, and one checkpoint. No full-study inference follows.
- Cancellation is measured over target-weighted domain probe gradients under a shared forward, not token-level gradient covariance.
- 22 coordinates at or below the `1e-12` activity floor are classified near-zero and excluded from relative cancellation.
- Probe-level second moments are batch/domain sufficient statistics; they are not within-batch token-gradient moments.
- RAMP and activation scores are acceptance-weighted correlates, not causal importance measures.
- Unknown separator targets are reported separately and excluded from the two-domain reconstruction.
- No coordinate is deleted, masked, or retrained. No importance, dispensability, or pruning-safety conclusion is made.
- Gradient cancellation alone is not an ICML-level novelty claim. The qualified signal motivates, but does not establish, a causal or general result.

## Stop condition

Pilot artifacts are complete. The eleven remaining registered cells were not launched.
