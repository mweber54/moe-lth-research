# Original-LTH Definition Audit (2026-10-12)

## Conclusion

The frozen Dataset A collection contains 100 distinct >=80% expert masks and 1100 router-age recovery conditions. Raw-curve recomputation finds 8 conditions within 5% of their matched dense validation CE, representing 4 masks; all use nonzero-age routers. Dataset A has 100 R0 recoveries with verified step-0 expert/shared/router initialization.

No recovery meets a defensible, evidence-complete operational definition of an original-initialization winning ticket at >=80%. The eight near-dense cases are hybrid router-rewind recoveries, not original LTH tickets. R0 is an original-initialization candidate protocol, but no R0 endpoint is within 5% of matched dense validation CE.

## Definition and Scope

Frankle & Carbin (2019) define a winning ticket as a sparse subnetwork that, when trained in isolation from its original initialization, reaches the original dense network's test accuracy in a comparable number of iterations. This audit uses validation cross-entropy (CE) only as an explicitly limited language-modeling proxy, not as test accuracy. A percent CE gap is a descriptive screen, not an equivalence margin or proof. The 2%, 5%, and 10% bands are reported only as requested screens.

The primary count is the frozen Dataset A grid in `results/final_dataset_a/canonical/dataset_a_canonical.csv`, checked against sparse and paired dense raw `metrics.jsonl` curves. The grid has 5 expert counts (8, 12, 16, 24, 32), 5 seeds (7, 17, 29, 43, 61), 4 target sparsities (80%, 85%, 90%, 95%), and 11 router ages. That is 100 distinct masks and 1,100 recovery conditions. Target sparsity applies only to expert weight matrices; it is not whole-network sparsity.

A separate local raw-metadata scan found 535 additional >=80% recovery artifacts, with 50 recorded mask hashes. Their union with the canonical hashes is 118 distinct recorded hashes, including 18 raw-only hashes. These additional runs are not mixed into dense-paired threshold counts: datasets/treatments vary and pairing or initialization provenance may be unavailable. Unknown evidence is marked `unknown`. For these additional rows, `validated=True` means the raw metadata integrity flag is true, a final recovery checkpoint exists, and the metrics curve has a parseable finite endpoint; it is not an independent replay of the full protocol or proof of dense pairing.

All 18 raw-only mask hashes were matched to locally saved mask tensors by recomputing the same SHA-256 state-dictionary hash; their reload paths are recorded in `all_candidates.csv`. 5 are explicitly labeled random-mask controls, not learned-mask LTH candidates. For the remaining legacy masks, source-dense checkpoint provenance is still unverified.

The repository's project summary reports a separate long-budget multi-domain 80% initialization-rewind result close to or better than dense. Its referenced raw result directory and checkpoints are absent locally, so mask provenance, full initialization restoration, exact pairing, and budgets cannot be independently verified. The summary also says its larger validation mixture included WikiText test data, so it is not evidence from an untouched test-only evaluation. It is not counted as a confirmed ticket or candidate run here.

## Checkpoint and Mask Logic

Canonical masks are expert-local magnitude masks from the trained source step-2500 model. `scripts/fixed_budget_dataset/run.py` loads the step-0 state, derives masks from step 2500, masks the step-0 base, and calls `run_router_age_recovery._run_recovery_condition`. `src/moe_lth/pruning/router_age.py` builds the fixed base from the rewind checkpoint, swaps only state tensors classified as router from the selected checkpoint, then reapplies masks. Its component classifier treats all non-expert, non-router state (including embeddings and normalization) as shared.

For Dataset A, R0 selects the step-0 source checkpoint for both router and expert/shared base. Thus R0 restores all model components to the original initialization before applying the trained-dense-derived expert mask. R10-R100 retain step-0 expert/shared state and mask but use a nonzero-age router: hybrid router rewind, not full-network early-weight rewind. AdamW is fresh; router parameters are trainable. Gradient hooks zero masked-coordinate gradients and masks are reapplied after optimizer steps. Finalization checks masked checkpoint coordinates remain zero.

Code: [router_age.py](../../src/moe_lth/pruning/router_age.py), [run_router_age_recovery.py](../../src/moe_lth/experiments/run_router_age_recovery.py), [masks.py](../../src/moe_lth/pruning/masks.py), [rewind.py](../../src/moe_lth/pruning/rewind.py), [magnitude_prune.py](../../src/moe_lth/pruning/magnitude_prune.py), [fixed_budget_dataset/run.py](../../scripts/fixed_budget_dataset/run.py).

Dense and sparse use 2,500 attempted steps and same-seed train/validation batch-sequence hashes; endpoints are at step 2,500. AMP may skip a small number of updates, so applied updates can differ slightly and are recorded separately. Sparse training uses a fresh optimizer. This is expert-only pruning. Validation CE is token-weighted and excludes the auxiliary term per the frozen protocol audit.

## Recomputed Performance

| Population | Runs | Within 2% | Within 5% | Within 10% |
| --- | ---: | ---: | ---: | ---: |
| All canonical >=80% | 1100 | 0 | 8 | 193 |
| Canonical R0 | 100 | 0 | 0 | 19 |

The 19 R0 runs within 10% represent 19 masks. No R0 case is within 2% or 5%; all-age within-10% count is 193.

The eight conditions within 5% are: E16 seeds 17/29 at 80%, R60 (3.35%, 3.45%); E24 seed 17 at 80%, R50 (4.01%) and R40 (4.91%); E16 seed 29 at 80%, R70 (4.29%) and R80 (4.40%); and E8 seed 29 at 80%, R20 (4.83%). These are eight recovery conditions but four masks. Exact losses, gaps, paths, budgets, and matched-step results are in `near_dense_candidates.csv` and `all_recovery_runs.csv`.

The audit compares loss curves only at shared logged steps. All 8 match/exceed dense at step 0; 7 also do so at one or more positive logged steps, while 1 match only at step 0. Step-zero performance is before sparse retraining and is not LTH success. At step 2,500, each is 3.35%-4.91% worse than its dense comparator.

Across each selected run's five-seed architecture/sparsity/router-age cell, the mean gap is 6.78%-9.10% and sample SD is 1.67-3.80 percentage points. Selected individual gaps lie 1.05-1.52 SD below their cell means. 0 of 220 five-seed cell means are within 5%. These are descriptive seed-variation statistics, not equivalence or significance tests.

## Status by Requested Category

- **Strict original-LTH evidence:** 0 verified >=80% winners. R0 satisfies mask-from-trained-dense, original initialization, isolated sparse retraining, and equal attempted-step budget; none is dense-equivalent even under the 5% screen. Test accuracy was not measured.
- **Near-dense initialization candidates:** R0 counts are 0/0/19 within 2%/5%/10% of matched dense validation CE. The 10% band is screening only.
- **Router-rewind recovery candidates:** 8 canonical conditions within 5%, all R20-R80, representing 4 masks at 80% sparsity.
- **Unsuccessful or inconclusive:** other R0 runs miss the 10% screen; higher-age runs outside the screens are not near-dense. Additional raw runs without verified full initialization or dense pairing remain inconclusive, not failures. Missing historical experiments are unavailable, not failures.

Five seeds exist per canonical architecture/sparsity cell, but the eight cases were selected on the same validation outcomes summarized here. No equivalence test or significance claim is made; the selected cases do not establish seed-population equivalence.

## Collection for Geometry Work

Preserve and use all 100 canonical masks, not just the successes: the set spans all five expert counts, five seeds, and 80/85/90/95% sparsities, with R0-R100 router-age recoveries under the same mask and step-0 expert/shared state. This matched collection retains successful and unsuccessful controls and isolates router age. Prioritize the eight <5% hybrid endpoints and 19 R0 <10% screens for initial inspection, but keep same-mask failing ages and high-gap masks as negative controls. Avoid selecting only on validation outcomes when studying geometry/performance associations.

Exact mask, initialization, router, and recovery checkpoint paths are in `checkpoint_manifest.csv`; files are referenced, not copied. The four masks behind the strongest hybrid runs and their age-matched R0/nonzero-age alternatives are listed by hash in the CSV inventory.

## Inventory Files and Limitations

- `all_candidates.csv`: distinct >=80% hashes; canonical provenance or explicit unknowns for raw-only hashes.
- `all_recovery_runs.csv`: all 1,100 canonical conditions plus additional local raw metadata artifacts; only exact canonical dense pairings contribute threshold labels.
- `r0_candidates.csv`: all R0 entries with verified and unknown component restoration kept distinct.
- `near_dense_candidates.csv`: paired runs within any 2%, 5%, or 10% screen.
- `winning_ticket_evidence.csv`: header-only because no candidate meets the conservative evidence criteria.
- `checkpoint_manifest.csv`: reload paths, configuration IDs, and batch hashes; unknown paths are explicit.
- `build_audit.py`: read-only builder; run `python3 results/lth_definition_audit/build_audit.py` from the repository root.

The metadata scan does not follow symlinks. Canonical paths come from the frozen Dataset A manifest. All raw-only mask tensors were located and rehashed, but this does not independently establish which trained dense checkpoint generated a legacy magnitude mask or verify every legacy initialization. The inventory covers local files only, not remote/unindexed artifacts. No experiments were launched and no existing results were modified.

## Final Answers

1. Distinct >=80% masks: 118 recorded hashes across canonical plus local raw metadata (100 canonical; 18 additional raw-only hashes).
2. Dataset A recoveries verified to use original initialization for all relevant weights: 100 R0 conditions. There are 91 other raw R0 records, but their expert/shared initialization and/or mask provenance is unverified and they are excluded from this verified count.
3. Canonical R0 within 2% / 5% / 10%: 0 / 0 / 19.
4. Defensible original-LTH winners at >=80%: 0 verified; existing evidence supports candidates and router-rewind recoveries only.
5. Near-dense (<5% CE gap) canonical conditions requiring a nonzero-age router: 8 of 8; four masks.
6. Existing local raw evidence is insufficient to substantiate the original LTH claim at >=80%; the historical long-budget claim is not locally reproducible because raw artifacts are absent.
7. Preserve the complete canonical 100-mask × 11-age collection, including strong candidates and matched negative controls; paths are in `checkpoint_manifest.csv`.
