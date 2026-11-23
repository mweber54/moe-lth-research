# Phase 0 audit review

Committed IMP stages cover E8/E16/E32 and seeds 7,17,29,43,61. The original README is stale: it says not launched, while committed artifacts and hashes demonstrate completed runs. E12/E24 have no committed IMP masks and cannot enter the pilot.

Verified 135/135 masks against exact checkpoint architecture, stored mask hash and keep counts. Dense checkpoint identities verified: 150/150 recovery rows. Source initialization and sparse final checkpoint hashes were independently checked in the primary audit.

All primary IMP recoveries rewind the full model to its exact step-0 checkpoint. Router and shared weights are trainable. Fresh AdamW/scaler and constant LR restart each recovery; no scheduler. Canonical Dataset A uses torch 2.7.0, FP16, 2500 attempted updates, fixed seed-specific training batch hashes, and 32 validation batches. Current canonical recovery code remains available; numerical reproduction is pending and is a hard pilot gate.

Masks are separate bool tensor dictionaries. The stage selecting mask k is not recovery evidence for mask k; its subsequent training stage is. Discovery stages with zero sparsity are excluded from sparse eligibility. Existing 80/85 finals and lower-stage recoveries are preserved, including negative controls.

The ticket catalog and lower-sparsity sweep use source-final one-shot magnitude masks, not IMP, and are inventoried separately without IMP eligibility. Router-age mapping records distinct router donors and trainability conditions; these must not be silently substituted for original IMP R0. Extended IMP uses 2500 original recovery attempts plus 2500 fresh-optimizer attempts on the next batch stream; matched 5000-step dense comparisons are separate from the primary protocol.

Near-dense and sparse-better screens are descriptive; no formal winning-ticket or statistical equivalence conclusion is justified from endpoints alone. All candidate directory JSON/CSV/Markdown artifacts are indexed in audit_details.json. secondary_catalog.csv retains source-recorded validity rather than asserting new byte-level validation of non-IMP artifacts.

Architecture summary:

- E8: 50 eligible sparse recoveries; seeds [7, 17, 29, 43, 61]; status counts {'nonrecovering_above_5pct_descriptive': 15, 'outperforms_matched_dense_descriptive': 14, 'near_dense_within_5pct_descriptive': 21}
- E16: 50 eligible sparse recoveries; seeds [7, 17, 29, 43, 61]; status counts {'outperforms_matched_dense_descriptive': 20, 'near_dense_within_5pct_descriptive': 16, 'nonrecovering_above_5pct_descriptive': 14}
- E32: 50 eligible sparse recoveries; seeds [7, 17, 29, 43, 61]; status counts {'outperforms_matched_dense_descriptive': 33, 'near_dense_within_5pct_descriptive': 16, 'nonrecovering_above_5pct_descriptive': 1}
