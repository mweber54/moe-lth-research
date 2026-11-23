# Functional geometry extension v1: findings

Checkpoint-only analysis. No training, recovery, mask search or fitted connectivity path was run, and no existing artifact was modified. Seed is the replication unit (n=5). All 10 sparse conditions (5 seeds x .80/.85) are kept, and individual seeds come before aggregates. No winner classifier was fit and nothing here is causal. "Matches" means sparse CE <= matched dense CE at 5,000 attempted steps (rerun track). Seeds 7, 43 and 61 match at .80, and seed 7 also matches at .85. Seeds 17 and 29 never match.

Per-condition table: [endpoint_metrics_by_condition.csv](endpoint_metrics_by_condition.csv). Figures: [figures/](figures). Design: [analysis_config.json](analysis_config.json). Coverage: [coverage_and_exclusions.csv](coverage_and_exclusions.csv).

## Provenance of each result
| Class | Content |
|---|---|
| Previously measured, consolidated | coupling, matched-pair contrasts, weight geometry, representations, CE/KL, update budgets, ticket labels (`consolidated_existing_metrics/`, rerun track). Sparse-native expert function is the historical n=128 table. |
| Newly computed from checkpoints | activation geometry, fixed-gate distortion, C->D pruning of the selection state, gradient geometry, mask structure |
| Unavailable | conditional connectivity (see `connectivity/eligibility_report.md`) |

Historical and rerun tracks share bit-identical weights at steps 0, 2500, 3000, 4000 and 5000. They are one trajectory measured twice, so curves use the rerun track only. Fixed-source and dense-native comparisons are reported separately. Diagnostics run in fp32 with TF32 off.

## Q1. Is function preserved at masked initialization?
**No, not at the 80/85% levels.**
- Masked-init vs original-init distortion, relative to the reference norm, is large. With same-step-dense gates and dense-native inputs it averages 3.6 at .80 and 2.3 at .85 across seeds. With fixed-source inputs it varies widely by seed (5.7 to 29.3 at .80).
- Output CKA at step 0 is 0.44 (dense-native) and 0.57 (fixed-source) at .80, and 0.41 / 0.54 at .85.
- Pruning the trained IMP state that selected the mask is a different case. For .80 masks, C->D changes CE by -0.0004 to +0.0009 (KL <= 0.0014). For .85 masks, C is the .80-trained endpoint, and CE rises by +0.04 to +0.16 (KL 0.05 to 0.17). Applying the mask to a trained state is nearly lossless at .80 and clearly lossy at .85.

Magnitude-based selection preserves function at the state it was selected on, and not at the original initialization.

## Q2. Do sparse and dense models diverge during recovery?
**Yes, consistently across all 5 seeds and both sparsities.**
- Output CKA with dense is 0.51 at step 2500 (.80) and 0.49 to 0.52 at step 5000. Output relative error grows from 1.35 at step 2500 to about 2.0 at 5000 (fixed-source). Feature-subspace overlap at r=16 is low (~0.25 to 0.30), and sample-space overlap falls from ~0.56 to ~0.45 to 0.52.
- Fixed-gate distortion at step 5000 is 3 to 8x larger than at step 2500, and widely seed-dependent (1.5 to 11).
- Divergence is larger at .85 than at .80 in all 5 seeds (lower CKA and lower feature overlap). Sparsity is confounded with the ticket label, because .85 is the less-often-matching level.
- Step 2500 is the optimizer reset, and the 2525/2600 rows show an immediate small change in geometry. Whether the reset itself causes divergence is unresolved.

## Q3. Does the opposite coupling trend accompany functional differences?
Sparse coupling rises (initial 0.13 to 0.21, final 0.27 to 0.39) while dense coupling falls (initial about -0.06 to +0.04, final -0.31 to -0.43). This holds in all 10 conditions. The sparse-minus-dense coupling gap widens in every one, from 0.15 to 0.26 at initialization to 0.63 to 0.80 at the end. It is **not** different initial values converging, because the gap grows. The opposite trend does co-occur with the activation and distortion divergence above, but only as a co-occurrence. Dense coupling falls in every seed (including seeds where dense CE is better and where it is worse), and no test here ties coupling to function. Sparse coupling is a sparse-native n=128 quantity and is not directly comparable with fixed-source geometry.

## Q4. Do any properties separate matching from non-matching beyond seed/sparsity?
**No robust separator.** Within .80 (matching: seeds 7, 43, 61; non-matching: 17, 29):
- Output CKA at step 5000: 0.510, 0.521, 0.527 vs 0.547, 0.510. These overlap.
- Distortion at step 5000: 1.9, 3.1, 5.4 vs 2.2, 8.0. These overlap.
- Final sparse coupling: 0.312, 0.271, 0.300 vs 0.292, 0.359. These overlap.
- Gradient energy retention exceeds the random-mask null by at most 0.004 everywhere, with no label dependence.
- Seeds 43 and 61 match at .80 but not at .85. Both have lower CKA and lower feature overlap at .85 than at .80, as do all other seeds, which fits a sparsity effect and not a label effect.
With n=2 and n=3 per group, any separation would be unresolved anyway.

## Gradient geometry (new)
- Energy of the dense gradient retained by the IMP mask is essentially the keep fraction and the random-mask null. For example, fc1 at .80, step 5000 gives 0.323 vs 0.328 keep fraction vs 0.324 random. The IMP mask does not align with dense gradient energy.
- Cosine between the masked dense gradient and the sparse gradient decays from 0.25 (fc1) / 0.77 (fc2) at step 0 to about 0.02 / 0.04 at step 5000. Top-subspace overlap at ranks 1 to 8 falls from 0.05 (fc1) / 0.4 to 0.6 (fc2) to about 0.002 to 0.004. Subspaces are valid at all ranks (16 batches).
- Gradients are computed on training-loader batches by index. All the training data was seen during training, so these are not held-out gradients.

## Mask structure (new)
- The .85 mask lies entirely inside the .80 mask (Jaccard 0.75). This is structural to IMP and not a finding.
- Cross-seed raw coordinate Jaccard is 0.128 to 0.169 vs an independent expectation of 0.081 to 0.111. Expert coordinates are not semantically aligned across seeds, so this is a weak, descriptive signal only.
- Per-expert structure (dead units, Gini, top-decile share) is in `mask_structure/expert_unit_structure.csv` against exact-count random controls.

## Robust / contradictory / unresolved
- **Robust (all 5 seeds):** masked init is not functionally preserved. Sparse/dense geometry diverges during recovery. .85 diverges more than .80. The coupling gap widens. The IMP mask retains no more gradient energy than random.
- **Contradictory or sensitive:** distortion magnitude is seed-dependent by an order of magnitude, so cross-seed means are dominated by a few seeds. Read the per-seed table.
- **Unresolved:** any label separator, the role of the 2500 reset, and whether coupling relates to function.

## What remains unsupported
- Mode connectivity (no distinct same-mask endpoints exist).
- Any causal claim, including that coupling or geometry causes matching.
- Sparse-native activation geometry (only the historical n=128 table is available).
- Generalization to other seeds, architectures (8E/12E excluded) or sparsities, and held-out gradient behavior.
