# Saved tickets at target sparsity >=80% — audit 2026-10-12

The canonical Dataset A collection contains 100 distinct validated masks and 1,100 validated sparse recoveries. Each target sparsity (80%, 85%, 90%, 95%) has 25 masks and 275 recoveries: five architectures (8/12/16/24/32 experts), five seeds (7/17/29/43/61), eleven router ages (0–100% by 10%). Router-age variants reuse the same mask and surviving initialization weights; they are not independent masks. All 1,100 listed mask files, original initialization checkpoints, and final recovery checkpoints existed at audit time. Existing canonical integrity validation was relied upon; model tensor hashes were not recomputed by this catalog.

Broader local metadata plus canonical rows expose 118 unique mask hashes: the 100 canonical masks, 13 additional historical masks, and five explicitly random control masks. Broader observations have heterogeneous protocols and are not admitted to the canonical collection by this audit. The scan skips diagnostic/output directories and does not recursively follow symlinks; canonical rows cover the active Dataset A symlink. These counts describe the identified saved collection, not every possible unindexed file or remote-host artifact.

## Restart protocol

For canonical Dataset A, masks are derived from trained source step-2500 expert weights by expert-local magnitude pruning. Surviving expert values and shared weights are restored from the ORIGINAL source step-0 checkpoint. Router weights are loaded from the selected source-age checkpoint and remain trainable during recovery. AdamW starts fresh. Thus R0 is a full initialization restart; R10–R90 are hybrid starts with a developed router; R100 uses the final source router. They are not full-model early-weight rewinds. There are 100 R0, 900 intermediate-router, and 100 R100 recovery conditions.

Evidence: results/final_dataset_a/reports/dataset_a_final_analysis.md (protocol validation); scripts/fixed_budget_dataset/finalize.py (mask manifest and rewind checkpoint); src/moe_lth/experiments/run_router_age_recovery.py (fixed expert/shared initialization and LTH isolation audit).

The older results/project_summary_so_far.md describes separate historical full-model rewind suites: typical 80% results required 10% early rewind, with a long-budget initialization-rewind exception. Those historical claims must not be applied to Dataset A. Several historical suite directories referenced by that summary are absent locally.

## Performance and selection

These are sparse ticket candidates, not 100 proven dense-matching lottery tickets. Relative gap = 100 * (sparse final validation CE - matched dense final validation CE) / dense final CE. Canonical endpoints use step 2500, not best-validation checkpoint selection. No canonical >=80% recovery beats dense or comes within 2%. Eight recoveries (four distinct masks) are within 5%, all at target 80%. 193 recoveries are within 10%. Five percent is a descriptive selection threshold here, not a newly established lottery-ticket criterion; no random-control superiority is asserted.

| Target sparsity | Distinct masks | Recovery conditions | Within 5% of matched dense |
| --- | --- | --- | --- |
| 80% | 25 | 275 | 8 |
| 85% | 25 | 275 | 0 |
| 90% | 25 | 275 | 0 |
| 95% | 25 | 275 | 0 |

Target sparsity refers to prunable expert weight matrices, excluding biases, router, and shared weights. Actual sparsity is recorded separately and can differ slightly due to integer keep-count rounding (nominal 80% can be 79.9999%). This is not 80% whole-model parameter sparsity.

## Files for further experiments

- canonical_recoveries.csv: 1,100 conditions with masks, original initialization, selected router and final checkpoint paths, availability flags, dense comparators, and loss gaps.
- distinct_masks_best_router.csv: all 100 masks, selecting minimum matched relative loss gap across router ages for each mask.
- shortlist_80pct_25_masks.csv: balanced 25-mask 80% subset, one per architecture/seed, using each mask's best observed router age. This is exploratory selection on validation outcomes; use held-out evaluation for new performance claims.
- within_5pct_dense.csv: eight strongest near-dense conditions, representing four masks.
- broader_local_observations.csv: historical observations and control masks; not canonical validation.
- audit_counts.json: count and availability checks.
- build_catalog.py: rerun from the repository root with python3 to regenerate CSV catalogs.

To rerun a lottery-ticket experiment, use the stored mask with source step-0 expert/shared weights and the selected router checkpoint. final_recovered.pt contains already retrained weights and is appropriate for continuation or inference, not a fresh lottery-ticket restart. Preserve matching model configs and data identities. The files are catalogs referencing existing artifacts; hundreds of GB of model files were not copied.
