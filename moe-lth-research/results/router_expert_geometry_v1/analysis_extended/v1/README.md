# Completed post hoc extension v1

700 saved states analyzed: 525 recovery snapshots and 175 source/pruning/initialization/historical states. No training or Dataset B restart. Original outputs and model modules were hash-checked and preserved. Five source seeds are the independent replication units.

Start with [combined findings](combined_findings.md), [definitions and paper provenance](metric_definitions.md), and [existing reference findings](existing_reference_findings.md).

## Main observations

- Endpoint-matched rank-16 learned source overlap decreases in 28/30 runs, correcting the original 30/30 claim. Requiring valid source bases also gives 28/30.
- Source-final full Gram shape departs in 22/30 learned runs and 17/30 random runs. Random rank-16 overlap instead increases in all 30 runs.
- Native proposed-pair Pearson coefficients decrease in all 30 learned runs; fixed all-expert coefficients increase in all 30. This is selection/mode dependence, not a universal coupling-strengthening story. Positive final native proposed association above shuffled controls still coexists with Gram departure in 15/30 learned runs.
- Overall interpretation: metric-dependent behavior, with descriptive coupling despite departure in some cases. No causal mechanism or complete nonlinear-function equivalence is established.

## Deliverables

| Purpose | Files |
| --- | --- |
| Identities and coverage | state_coverage_inventory.csv, expected_recovery_coverage.csv, input_audit.json |
| SVD ranks and validity | svd_expert_raw.parquet, existing_svd_expert_raw.parquet, svd_correction.json, svd_inclusion_exclusion.csv |
| Full Gram shape and scale | gram_expert_raw.parquet, gram_seed_trajectories.csv, gram_endpoint_seed_summary.csv |
| Signed GELU activations | activation_expert_raw.parquet, activation_global_pooled_endpoint_statistics.csv, activation_coverage.csv |
| Raw token identity/pairs | cache/**/token_pairs.npz; per-state completed.json and CSV files |
| Shuffled controls | activation_shuffled_seed_trajectories.csv, activation_shuffled_seed_summary.csv |
| Across-expert extension | within_token_association.parquet, within_token_seed_trajectories.csv |
| Joint expert matching | joint_expert_endpoint_effects.parquet, joint_matched_expert_seed_effects.csv, joint_learned_random_seed_summary.csv, joint_router_age_paired_summary.csv |
| Loss and concordance | combined_seed_cases.csv, within_condition_seed_correlations.csv |
| Pruning-only effects | pruning_joint_seed_pairs.csv, pruning_joint_paired_summary.csv |
| Exact coverage limitations | summary_coverage_exceptions.csv, missing_legacy_svd_energy_fields.csv |
| Validation/preservation | validation.json, metric_consistency_validation.json, original_output_preservation.csv, delivery_status.json |
| Figures | figures/ — 89 PNG/PDF pairs, separated by rank/layer/reference/mode/selection |

The attached paper PDF was unavailable; the matching public arXiv v1 was read and its substitution disclosed. Some older SVD tables omit an energy field; weights and new full Gram measurements are available for every state. Sparse routed coverage is explicitly excluded, never filled with zero correlations. Some layer-level/joint summaries have fewer than five eligible seeds; their missing values and small-n limits are documented.

Reproduce using scripts/router_expert_geometry_extended/. Commands appear in combined_findings.md. Completed checkpoint measurements resume by input/config/code hash. After intermediate aggregation is present, `analyze.py --finish` resumes joint comparisons, figures and report. `validation_analysis/` contains isolated subset checks, not additional experimental replications. Intermediate debugging logs are preserved separately; delivery_status.json and the final reports identify completed results.
