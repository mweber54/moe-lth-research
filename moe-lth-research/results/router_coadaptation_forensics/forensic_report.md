# Router Coadaptation Forensics

## Direct Answer

**The existing results are insufficient to identify the mechanism.** The strongest supported statement is: **during approximately steps 500-1000, trainable R20 undergoes coupled router adaptation (nonzero drift, aggregate assignment change, increased margin, reduced entropy, and changing expert gradients) while frozen R20 cannot update its parameters.** However, R0 also moves and reorganizes at least as much on several aggregate metrics, so existing results do not establish what R20 does differently at the token or layer-function level.

## Reconstructed Performance

Final mean 85% ticket gaps are trainable R0/R20/R100 = 0.2857/0.1691/0.1891 and frozen R0/R20/R100 = 0.2322/0.2474/0.2120. Trainable R20 is best for seeds 7 (0.1178) and 29 (0.1464), but not seed 17 (0.2431 versus R100 0.2010). Its mean advantage becomes visible by step 1000: trainable R0/R20/R100 = 0.1999/0.1764/0.2339, and strengthens by 1500: 0.2507/0.2002/0.2394. At step 500 the ordering is not yet R20-favorable (0.0512/0.0721/0.1154).

## What Is Established

- All 36 dense/sparse entries have matched training and validation sequence hashes within family, seed, and router initialization. Initial sparse losses are not directly comparable to dense initialization because pruning creates an immediate loss discontinuity.
- Frozen router initial/final hashes are equal and normalized drift is zero. Trainable final normalized drift averages R0/R20/R100 = 1.0525/0.9906/0.6718.
- R20 trainable versus frozen: entropy change -0.5285 versus -0.1879; margin change +0.2176 versus +0.0642; aggregate assignment churn 0.7767 versus 0.7949. Churn therefore does not distinguish the conditions.
- R20 early expert-gradient AUC is 78.8 trainable versus 59.1 frozen, but R100 is higher at 112.0. This is insufficient to call R20 gradients healthier.
- Routing distributions are layerwise aggregate counts, not token identities. Gradient records are per 10 steps; two expert-gradient records and 20 router-gradient records are non-finite across the cohort and are retained as invalid rather than interpreted.

## Scientific Compatibility

The studies are protocol-compatible for loss and shared-probe routing summaries. Frozen records omit gradient validity/status and aggregate capacity fields, so those missing fields are not treated as zeros. `matched_85pct_runs.csv` records every hash and match status; `available_diagnostics.md` records availability and data-quality qualifications.

The minimum additional diagnostic run is specified in `missing_evidence.md`; it instruments an existing condition rather than proposing a new experimental condition.
