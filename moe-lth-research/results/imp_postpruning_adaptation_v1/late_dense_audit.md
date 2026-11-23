# Separate late-dense trajectory audit

This is an analysis-only audit of already observed 4500–5000 trajectories; it
does not extend training or enter the primary router × expert factorial.
Validation is the common target-token-weighted FP32/TF32-off readout.

* Dense validation CE increased for 4/5 seeds:
  7, 29, 43, 61.
* Sparse-minus-dense CE was positive at step 4500 for
  15/15
  seed-condition comparisons.
* Sparse-minus-dense CE was nonpositive at step 5000 for
  7/15 comparisons.
* Sparse CE decreased from step 4500 to 5000 for
  15/15 comparisons.

These observations do not establish overfitting, a pruning advantage, or a
causal explanation. The full per-seed table is in `late_dense_audit.csv`; the
15 seed-condition contrasts are in `late_sparse_comparisons.csv`.
