# Dataset A data dictionary

NA means genuinely unavailable diagnostics or an explicitly unavailable derived summary. The candidate table contains validated sparse observations only; the completion audit retains all 1650 expected sparse keys plus dense/source keys. Every excluded key has a reason. A blocked candidate is not frozen.

Source exposure is full-trajectory and repeated by condition; do not treat repetitions as replicates. Developmental sampled exposure is age-cumulative proposed exposure. Entropy and specialization are unnormalized nats, margin is probability difference, load CV dimensionless, agreement a fraction, JS retains the historical functional-JS convention. Summaries use sample SD across seeds.

## Canonical condition table

| Column | Units | Source | Calculation / interpretation |
|---|---|---|---|
| condition_id | identifier | Cartesian grid | sparse_eE_seedS_RA_sSP; unique condition |
| architecture | label | grid | E plus E suffix |
| num_experts | experts/layer | checkpoint model config | one of 8,12,16,24,32 |
| seed | identifier | config | one of 7,17,29,43,61 |
| router_age | percent | grid | one of 0,10,…,100 |
| router_age_percent | percent | grid | alias of router_age |
| source_checkpoint_step | attempts | source checkpoint manifest | 25 × router age |
| sparsity | fraction | grid | target removed expert-weight fraction |
| retained_fraction | fraction | grid | 1 − target sparsity |
| protocol_version | identifier | canonical README | historical-native-fixed-budget-v1 |
| source_config_id | SHA-256 digest | checkpoint/config/metadata or manifest | file, state-dictionary or canonical sorted-JSON hash as named; provenance records file hashes separately |
| recovery_config_id | SHA-256 digest | checkpoint/config/metadata or manifest | file, state-dictionary or canonical sorted-JSON hash as named; provenance records file hashes separately |
| source_attempt_budget | attempts | validated source config | 2500 |
| recovery_attempt_budget | attempts | validated recovery config | 2500 |
| batch_size | examples | validated config | 128 |
| sequence_length | tokens/example | validated config | 128 |
| top_k | experts/token/layer | validated config | 1 |
| capacity_factor | multiplier | validated config | 1.25 |
| aux_loss_weight | coefficient | validated config | 0.1 |
| source_artifact_path | path | selected artifact inventory | absolute location; physical backup relocation preserved |
| source_checkpoint_path | path | selected artifact inventory | absolute location; physical backup relocation preserved |
| source_checkpoint_hash | SHA-256 file digest | source checkpoint | hash of serialized checkpoint |
| router_checkpoint_hash | SHA-256 component digest | source_checkpoint_manifest.csv | router state_dict_hash; distinct from whole checkpoint file digest |
| mask_artifact_path | path | selected artifact inventory | absolute location; physical backup relocation preserved |
| mask_hash | SHA-256 digest | checkpoint/config/metadata or manifest | file, state-dictionary or canonical sorted-JSON hash as named; provenance records file hashes separately |
| dense_artifact_path | path | selected artifact inventory | absolute location; physical backup relocation preserved |
| sparse_artifact_path | path | selected artifact inventory | absolute location; physical backup relocation preserved |
| dense_condition_id | identifier | Cartesian grid | one same E/seed/age comparator shared across sparsities |
| dense_config_hash | SHA-256 digest | checkpoint/config/metadata or manifest | file, state-dictionary or canonical sorted-JSON hash as named; provenance records file hashes separately |
| sparse_config_hash | SHA-256 digest | checkpoint/config/metadata or manifest | file, state-dictionary or canonical sorted-JSON hash as named; provenance records file hashes separately |
| data_hash | SHA-256 batch-sequence digest | metadata.json | training materialized batch sequence hash, not text-file hash |
| validation_hash | SHA-256 batch-sequence digest | metadata.json | validation materialized batch sequence hash |
| dense_final_checkpoint_sha256 | fraction or dimensionless | full exposure logs | acceptance rate=accepted/proposed; drop rate=1−acceptance; load CV=population SD/mean across layer-experts; applied_steps counts successful updates |
| sparse_final_checkpoint_sha256 | fraction or dimensionless | full exposure logs | acceptance rate=accepted/proposed; drop rate=1−acceptance; load CV=population SD/mean across layer-experts; applied_steps counts successful updates |
| dense_initial_validation_loss | nats/token | raw recovery metrics.jsonl | step 0 for initial/postprune; step 2500 for final |
| sparse_preprune_validation_loss | nats/token | matched dense metrics.jsonl step 0 | equivalent unmasked rewound initial model |
| sparse_postprune_validation_loss | nats/token | raw recovery metrics.jsonl | step 0 for initial/postprune; step 2500 for final |
| pruning_shock | nats/token | metrics.jsonl step 0 | sparse initial minus dense initial |
| dense_early_auc | nats/token × attempts | raw recovery validation curve | existing trapezoidal integral 0–1250, every 50 attempts, no normalization |
| sparse_early_auc | nats/token × attempts | raw recovery validation curve | existing trapezoidal integral 0–1250, every 50 attempts, no normalization |
| matched_early_auc_gap | nats/token × attempts | metrics.jsonl steps 0–1250 | sparse trapezoidal AUC minus dense AUC |
| dense_recovery_improvement | nats/token | raw recovery curve | initial minus final loss |
| sparse_recovery_improvement | nats/token | raw recovery curve | initial minus final loss |
| dense_final_validation_loss | nats/token | raw recovery metrics.jsonl | step 0 for initial/postprune; step 2500 for final |
| sparse_final_validation_loss | nats/token | raw recovery metrics.jsonl | step 0 for initial/postprune; step 2500 for final |
| ticket_gap | nats/token | sparse and dense metrics.jsonl at step 2500 | sparse final minus matched dense final |
| source_valid | boolean | completion/protocol/gap audits | all required corresponding checks passed |
| dense_valid | boolean | completion/protocol/gap audits | all required corresponding checks passed |
| sparse_valid | boolean | completion/protocol/gap audits | all required corresponding checks passed |
| mask_valid | boolean | completion/protocol/gap audits | all required corresponding checks passed |
| comparator_valid | boolean | completion/protocol/gap audits | all required corresponding checks passed |
| ticket_gap_valid | boolean | completion/protocol/gap audits | all required corresponding checks passed |
| condition_valid | boolean | completion/protocol/gap audits | all required corresponding checks passed |
| development_artifact | path | selected artifact inventory | absolute location; physical backup relocation preserved |
| actual_sparsity | fraction | stored mask | pruned count / mask parameter count |
| mask_keep_count | parameters | mask manifest | number of mask ones |
| mask_pruned_count | parameters | mask manifest | number of mask zeros |
| router_entropy | see developmental units above | identity-validated existing source-probe table | preserved existing measurement, no recovery-derived substitution or interpolation |
| router_margin | see developmental units above | identity-validated existing source-probe table | preserved existing measurement, no recovery-derived substitution or interpolation |
| router_load_cv | see developmental units above | identity-validated existing source-probe table | preserved existing measurement, no recovery-derived substitution or interpolation |
| router_token_id_specialization | see developmental units above | identity-validated existing source-probe table | preserved existing measurement, no recovery-derived substitution or interpolation |
| router_assignment_agreement_to_r100 | see developmental units above | identity-validated existing source-probe table | preserved existing measurement, no recovery-derived substitution or interpolation |
| router_js_distance_to_r100 | see developmental units above | identity-validated existing source-probe table | preserved existing measurement, no recovery-derived substitution or interpolation |
| router_cumulative_sampled_exposure | see developmental units above | identity-validated existing source-probe table | preserved existing measurement, no recovery-derived substitution or interpolation |
| source_proposed_assignments | layer/expert assignment occurrences | full source or recovery exposure log | sum steps 1–2500; per-expert mean divides by 4 × E; unavailable logs yield NA |
| source_accepted_assignments | layer/expert assignment occurrences | full source or recovery exposure log | sum steps 1–2500; per-expert mean divides by 4 × E; unavailable logs yield NA |
| source_dropped_assignments | layer/expert assignment occurrences | full source or recovery exposure log | sum steps 1–2500; per-expert mean divides by 4 × E; unavailable logs yield NA |
| source_acceptance_rate | fraction or dimensionless | full exposure logs | acceptance rate=accepted/proposed; drop rate=1−acceptance; load CV=population SD/mean across layer-experts; applied_steps counts successful updates |
| source_mean_accepted_per_expert | layer/expert assignment occurrences | full source or recovery exposure log | sum steps 1–2500; per-expert mean divides by 4 × E; unavailable logs yield NA |
| source_mean_active_attempt_steps_per_expert | attempts or applied updates per expert | full exposure logs | number with accepted count >0, averaged over 4 × E; applied variant additionally requires optimizer update flag |
| source_mean_active_applied_steps_per_expert | attempts or applied updates per expert | full exposure logs | number with accepted count >0, averaged over 4 × E; applied variant additionally requires optimizer update flag |
| dense_proposed_assignments | layer/expert assignment occurrences | full source or recovery exposure log | sum steps 1–2500; per-expert mean divides by 4 × E; unavailable logs yield NA |
| dense_accepted_assignments | layer/expert assignment occurrences | full source or recovery exposure log | sum steps 1–2500; per-expert mean divides by 4 × E; unavailable logs yield NA |
| dense_dropped_assignments | layer/expert assignment occurrences | full source or recovery exposure log | sum steps 1–2500; per-expert mean divides by 4 × E; unavailable logs yield NA |
| dense_acceptance_rate | fraction or dimensionless | full exposure logs | acceptance rate=accepted/proposed; drop rate=1−acceptance; load CV=population SD/mean across layer-experts; applied_steps counts successful updates |
| dense_mean_accepted_per_expert | layer/expert assignment occurrences | full source or recovery exposure log | sum steps 1–2500; per-expert mean divides by 4 × E; unavailable logs yield NA |
| dense_mean_active_attempt_steps_per_expert | attempts or applied updates per expert | full exposure logs | number with accepted count >0, averaged over 4 × E; applied variant additionally requires optimizer update flag |
| dense_mean_active_applied_steps_per_expert | attempts or applied updates per expert | full exposure logs | number with accepted count >0, averaged over 4 × E; applied variant additionally requires optimizer update flag |
| dense_drop_rate | fraction or dimensionless | full exposure logs | acceptance rate=accepted/proposed; drop rate=1−acceptance; load CV=population SD/mean across layer-experts; applied_steps counts successful updates |
| dense_load_cv | fraction or dimensionless | full exposure logs | acceptance rate=accepted/proposed; drop rate=1−acceptance; load CV=population SD/mean across layer-experts; applied_steps counts successful updates |
| dense_applied_steps | fraction or dimensionless | full exposure logs | acceptance rate=accepted/proposed; drop rate=1−acceptance; load CV=population SD/mean across layer-experts; applied_steps counts successful updates |
| dense_optimizer_steps_applied | updates or skips | recovery metadata.json | reported separately; applied + AMP skipped = 2500 when both recorded |
| dense_amp_skipped_updates | updates or skips | recovery metadata.json | reported separately; applied + AMP skipped = 2500 when both recorded |
| sparse_proposed_assignments | layer/expert assignment occurrences | full source or recovery exposure log | sum steps 1–2500; per-expert mean divides by 4 × E; unavailable logs yield NA |
| sparse_accepted_assignments | layer/expert assignment occurrences | full source or recovery exposure log | sum steps 1–2500; per-expert mean divides by 4 × E; unavailable logs yield NA |
| sparse_dropped_assignments | layer/expert assignment occurrences | full source or recovery exposure log | sum steps 1–2500; per-expert mean divides by 4 × E; unavailable logs yield NA |
| sparse_acceptance_rate | fraction or dimensionless | full exposure logs | acceptance rate=accepted/proposed; drop rate=1−acceptance; load CV=population SD/mean across layer-experts; applied_steps counts successful updates |
| sparse_mean_accepted_per_expert | layer/expert assignment occurrences | full source or recovery exposure log | sum steps 1–2500; per-expert mean divides by 4 × E; unavailable logs yield NA |
| sparse_mean_active_attempt_steps_per_expert | attempts or applied updates per expert | full exposure logs | number with accepted count >0, averaged over 4 × E; applied variant additionally requires optimizer update flag |
| sparse_mean_active_applied_steps_per_expert | attempts or applied updates per expert | full exposure logs | number with accepted count >0, averaged over 4 × E; applied variant additionally requires optimizer update flag |
| sparse_drop_rate | fraction or dimensionless | full exposure logs | acceptance rate=accepted/proposed; drop rate=1−acceptance; load CV=population SD/mean across layer-experts; applied_steps counts successful updates |
| sparse_load_cv | fraction or dimensionless | full exposure logs | acceptance rate=accepted/proposed; drop rate=1−acceptance; load CV=population SD/mean across layer-experts; applied_steps counts successful updates |
| sparse_applied_steps | fraction or dimensionless | full exposure logs | acceptance rate=accepted/proposed; drop rate=1−acceptance; load CV=population SD/mean across layer-experts; applied_steps counts successful updates |
| sparse_optimizer_steps_applied | updates or skips | recovery metadata.json | reported separately; applied + AMP skipped = 2500 when both recorded |
| sparse_amp_skipped_updates | updates or skips | recovery metadata.json | reported separately; applied + AMP skipped = 2500 when both recorded |

## Long trajectories

One row per actual logged recovery attempt. Dense conditions occur once, sparsity=0. Validation appears at step 0 and every 50 attempts; intermediate validation NAs are unmeasured, not missing scheduled points. Raw float values are retained at binary64 precision in Parquet and round-trip decimal CSV.

| Column | Units | Source / calculation |
|---|---|---|
| condition_id | identity / label | raw selected recovery condition identity; model_type=dense or sparse; path is original artifact |
| architecture | identity / label | raw selected recovery condition identity; model_type=dense or sparse; path is original artifact |
| seed | identity / label | raw selected recovery condition identity; model_type=dense or sparse; path is original artifact |
| router_age | identity / label | raw selected recovery condition identity; model_type=dense or sparse; path is original artifact |
| sparsity | identity / label | raw selected recovery condition identity; model_type=dense or sparse; path is original artifact |
| model_type | identity / label | raw selected recovery condition identity; model_type=dense or sparse; path is original artifact |
| recovery_step | attempts | raw step, not optimizer applied-update count |
| validation_loss | nats/token | metrics.jsonl loss; NA at unevaluated attempts |
| training_loss | nats/token + weighted auxiliary | gradient_stats.jsonl train_loss; exact logged total optimization loss; not language CE alone |
| auxiliary_loss | unavailable | not separately recorded in raw gradient logs; NA; never obtained by subtracting unrelated diagnostics |
| learning_rate | optimizer rate | validated recovery checkpoint config; constant 0.0003; no scheduler |
| optimizer_steps_applied | updates | gradient_stats.jsonl optimizer_step_applied; cumulative true flags through this attempt; NA if flags absent |
| artifact_path | identity / label | raw selected recovery condition identity; model_type=dense or sparse; path is original artifact |

## Summary and selection conventions

All 330 expected architecture/age/sparsity groups are present with n_seeds, including zero-coverage groups. Master and seed-winner tables preserve all 30 and 150 expected keys; unobserved winners are NA. Summary min/median/max and sample SD use available independent seeds; complete_five_seed_curve is required for complete-grid interpretation. Partial winner comparisons are descriptive only. Positive intermediate advantage means lower intermediate gap. Ties use 1e-12. Basin width is max age minus min age in percentage points using the connected paired-SD rule from the September 21 interim script. No confidence intervals or hypothesis tests are implied.
