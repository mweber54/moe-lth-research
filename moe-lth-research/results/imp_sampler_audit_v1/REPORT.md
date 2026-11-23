# MoE sampler audit v1 — CPU-only decision

No GPU training was launched. Historical artifacts were copied with hashes into `preserved/`.

## Decision

Do not launch connectivity recovery: the validated sampler has no common frozen dose passing every gate in all 18 configurations. Treat this as a feasibility failure and preregister a separate design.

## Confirmed findings

- The suspected sampler defect is disproved: both branches make `x` and `y` exactly the two computed cross-corners. No deterministic counterexample exists in the tested seeds, and the separately named implementation is output-equivalent.
- Exact saved-selection row/column equality holds in 18/18 sealed v2 pairs; step-0 reconstruction hashes match in 18/18 pairs.
- All 12 completed v3 repair trajectories inherit the sealed v2 draw-1 selections (six seed/order pairs × two arms).
- Direct historical IMP-versus-degree-mask recovery, mask degree preservation, and observational sign-persistence results do not depend on this rectangle sampler.
- Rectangle-based v2 and v3 interpretations remain subject to the saved-location and frozen-gate audit, but not to the hypothesized indexing defect.

## Artifacts

See `provenance.json`, `defect_reproduction.json`, `v2_pair_reconstruction.json`, `v2_matrix_row_column_audit.json`, `v3_repair_inheritance.json`, `connectivity_feasibility.json`, and `connectivity_pool_stats.json`.
