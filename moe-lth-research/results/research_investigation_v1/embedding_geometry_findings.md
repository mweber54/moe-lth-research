# Independent embedding geometry findings

All 240 registered rerun checkpoints were read; all fresh original-dtype E hashes match the previous audit. There are 230 unique 256×256 float32 E tensors, converted to CPU float64. Historical duplicates are not additional replication. No forward/backward or optimizer step was performed.

| Reference | Condition | Mean angle (deg) | Off-diagonal cosine RMSE | Top-1 neighbor retained | CKA | Procrustes distance |
|---|---|---:|---:|---:|---:|---:|
| initial | dense | 3.024335 | 0.005121 | 88.5938% | 0.998466766 | 0.040976 |
| initial | 80% | 3.085371 | 0.005170 | 87.2656% | 0.998427342 | 0.041263 |
| initial | 85% | 3.190876 | 0.005402 | 86.6406% | 0.998304477 | 0.042689 |
| same_step_dense | dense | 0.000000 | 0.000000 | 100.0000% | 1.000000000 | 0.000000 |
| same_step_dense | 80% | 1.848615 | 0.003241 | 91.8750% | 0.999340641 | 0.025365 |
| same_step_dense | 85% | 1.762903 | 0.003156 | 92.2656% | 0.999381008 | 0.024571 |

Means give every seed equal weight. The tiny nonzero self-angle/self-Procrustes distances for dense versus itself are floating-point SVD/acos roundoff, not drift. Row angles depend on the fixed feature coordinate system. Token-cosine Gram comparisons, centered CKA, left/token singular subspaces and optimal Procrustes comparisons are invariant to a global right orthogonal transformation; the right/feature subspaces are not.

## Paired endpoint effects and spectral check

- 80%: mean excess initial-relative rotation +0.061036 degrees, seed range -0.000532 to +0.167006, positive in 3/5. At 80%, two seeds have very small negative effects; only 85% has a consistent excess. No multiplicity-adjusted discovery is claimed.
- 85%: mean excess initial-relative rotation +0.166541 degrees, seed range +0.115322 to +0.234391, positive in 5/5. At 80%, two seeds have very small negative effects; only 85% has a consistent excess. No multiplicity-adjusted discovery is claimed.
- Dense: squared-energy entropy rank 154.894254; top-eight energy 0.112141998; stable rank 66.348751; numerical rank 256–256 at relative threshold 1e-6.
- 80%: squared-energy entropy rank 154.891190; top-eight energy 0.112216469; stable rank 66.354427; numerical rank 256–256 at relative threshold 1e-6.
- 85%: squared-energy entropy rank 154.879642; top-eight energy 0.112188493; stable rank 66.308462; numerical rank 256–256 at relative threshold 1e-6.
- Initialization mean entropy rank 154.749105. The 64 independent Gaussian **null** matrices have entropy rank 155.117520 ± 0.502669 sample SD and top-eight energy 0.112526211 ± 0.000818268. The null is not trained-model evidence. The observed ~155/256 entropy rank is compatible with a random broad spectrum, not learned rank collapse.

Normalized spectrum changes are small; more concentrated leading-subspace angles must be interpreted with cutoff gaps. All cutoff gaps and k=8/16/32 principal-angle metrics are exported, using subspaces rather than signed singular-vector dot products. Centering and squared-energy definitions are explicit; centered legacy ranks are not mixed into uncentered values.

## Token IDs, neighborhoods, and function

The endpoint top-five rotation IDs for every seed/condition are in largest_token_rotations.csv. No semantic byte classes were fitted or invented. Full IDs, angles, nearest-neighbor IDs and top-five preservation are in token_rotation_neighbors.csv. Small changes can switch nearly tied nearest neighbors, so neighbor rearrangement is not automatically meaningful token-group reorganization. No reproducible semantic group mechanism is established.

There is real relational change beyond mere uniform contraction: row-normalized Gram differences and nonzero optimally aligned residuals persist. Nevertheless effects are modest, dense models drift similarly, and sparse–dense CKA exceeds .999 on average. Stable spectrum alone cannot rule out change, but neither change nor high CKA proves a functional mechanism. Tied output prediction adds a second role for each row.

All fixed-list exploratory geometry/performance correlations at every noninitial step are exported in geometry_performance_all_tests.csv, separately by sparsity with n=5 and leave-one-seed-out ranges. They use matched-seed, fixed-step changes from identical initial E; angle/Gram metrics remove row-scale effects and Procrustes additionally removes a global rotation. No outcome-selected metric or best correlation is promoted to a predictor. There is no independent held-out seed set, and five seeds cannot support a reliable multi-covariate predictive fit.

## Falsification of the proposed lead

The strong claim that expert pruning induces a large, general embedding-geometry reorganization is not supported: 80% rotation effects have exceptions, the 85% excess is only about .17 degrees, most neighbors are retained, spectra are random-like, and direct E-gradient histories or causal E interventions are absent. The narrower stable-spectrum-with-modest-drift observation is confirmed but is not by itself a compelling novel paper. A global rotation preserves spectrum/Gram by linear algebra; our synthetic orthogonal control explicitly verifies this distinction.

## Temporal counterexample and uncertainty

At initialization all matched E tensors are identical, while sparse-minus-dense validation gaps already range from -5.253417 to -2.583474 nats/token. Expert masks and other downstream computations differ before E evolves. Thus E differences are not necessary for an initial performance gap; this does not rule out a later contribution. The first postinitial geometry observation is step 25, so simultaneous early observations do not establish which process leads. See geometry_validation_gap_timecourse.csv for every observed paired time point. All performance correlations are post hoc exploratory, with 240 metric/step/sparsity tests exported and no claim that a maximum correlation predicts unseen seeds.
