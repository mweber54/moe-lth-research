# Corrected findings — functional geometry audit v1

This report supersedes affected statements in [the original extension findings](../findings.md) without modifying that file. Results remain exploratory and descriptive. Source seed is the replication unit (`n=5`).

## 1. Coupling reconciliation

The paired `fixed_source` / `all_expert` Pearson result reproduces:

| Sparsity | Sparse change | Dense change | Difference-in-changes |
|---:|---:|---:|---:|
| .80 | +0.1396 | -0.3652 | +0.5048 |
| .85 | +0.1712 | -0.3652 | +0.5363 |

All five sparse seeds increase at both sparsities; all five matched dense controls decrease. The sparse-minus-dense gap widens in all ten conditions on the common four-endpoint population of 128 valid layer-expert units.

An independent recomputation from saved `raw.npz` fixed-source score, activation, and all-expert selection arrays reproduces all ten primary paired rows. The common four-endpoint population is 128 units in every row; the maximum absolute discrepancy from the paired report is `1.11e-16`.

These coupling measurements use the **same source-final dense hidden inputs** for sparse and dense models. They correlate raw router projection scores (not probabilities) with signed mean post-GELU `fc1` activations (bias included; not absolute values). The all-expert counterfactual uses 128 token positions per expert. Thus `n=128` in this context is the token-position count, while the separately defined layer-expert population is 4 layers x 32 experts.

The extension's “sparse-native n=128” wording is a reporting-label error: the numeric changes correspond to fixed-source/all-expert coupling. The similarly named expert-function export is a distinct relative-output-error/cosine metric and contains both `fixed_source` and `native` input modes. Native/fixed-initial changes are a separate paired result (mean sparse change `+0.1658/+0.2022`, dense change `-0.3190`). Native/current-accepted changes differ in sign (mean sparse `-0.1567/-0.1558`, dense `-0.0366/-0.0393`) and depend on route-accepted populations. Those comparisons must remain separate.

The per-seed table preserves track, condition, checkpoint and reference hashes, probe/token/position hashes, sample counts, pair definition, valid layer-expert IDs, coefficient, aggregation, and capacity-selection semantics: [coupling_reconciliation.csv](./coupling_reconciliation.csv).

**Retain:** the fixed-source opposite-change result and widening gap with the probe/population definition attached. **Narrow:** native results to their particular input and selection mode. **Withdraw:** the “sparse-native” label for the fixed-source values and any causal interpretation.

## 2. Recovery geometry

The broad claim that sparse/dense geometry diverges consistently does not survive across probe modes.

- **Fixed-source output CKA vs same-step dense:** decreases from step 0 to 5000 in every seed at both sparsities. Cross-seed means are `.570 -> .523` at `.80` and `.544 -> .487` at `.85`. Intermediate snapshots show the trajectories are not monotonic.
- **Dense-native output CKA vs same-step dense:** increases from step 0 to 5000 in all five seeds. Means are `.442 -> .489` at `.80` and `.413 -> .456` at `.85`; step 2500 is generally higher than the final value. The sparse and dense experts receive the same input within a checkpoint, but that input moves with the dense model across checkpoints.

CKA is a similarity, not a distance. Fixed-source and dense-native comparisons answer different questions; the former holds expert inputs fixed, while the latter follows changing dense-native representations. Feature-space and sample-space subspace overlaps are reported separately at ranks 8, 16, and 32. Relative output error is the unsquared `||sparse-dense||/||dense||`; cosine is the mean per-row cosine with zero rows excluded. Existing weight-Gram and subspace metrics retain their fixed valid populations and references separately. In particular, `same_step_dense` is a moving reference, while `fixed_dense_final` is fixed. See [geometry_interval_changes.csv](./geometry_interval_changes.csv).

### Fixed-gate distortion

`D` is the mean **squared** norm of the aggregate routed expert-output difference under frozen reference gates. Its denominator is the mean squared reference MoE-output norm under the same gates. Full-probe and accepted-only populations are separate; dropped positions contribute zero to full-probe values.

For fixed-source inputs with same-step dense gates, the equal-layer mean per-layer ratios rise for every seed between steps 2500 and 5000. Their cross-seed means rise from `1.045` to `4.108` at `.80`, and `1.191` to `4.978` at `.85`. But the mean numerator `D` falls from `475.2` to `338.0` at `.80` and `490.2` to `341.2` at `.85`; reference energy falls from `597.7` to `306.0`. The ratio of aggregate energies is different from the mean of per-layer ratios (`.796 -> 1.101` at `.80`, `.821 -> 1.113` at `.85`). The denominator therefore materially affects the reported ratio. The reference is moving; these values do not show movement away from a fixed geometry. Numerators, denominators, and both aggregation definitions appear in [geometry_interval_changes.csv](./geometry_interval_changes.csv).

## 3. Mask counts and gradient accounting

`nested_expert_local_masks` forms each target count over one expert's concatenated `fc1` and `fc2` weights. It does not impose the nominal keep rate separately on every tensor, nor one joint count across all experts. At `.80`, `fc1` may retain about `.328` while `fc2` retains about `.072`; together they yield the intended expert-weight keep fraction. Actual cohort counts are:

- `.80`: `13,421,824 / 67,108,864 = 0.2000008` kept.
- `.85`: `10,066,304 / 67,108,864 = 0.1499996` kept.

Counts are reported by tensor, expert, layer, matrix, and aggregate in [mask_count_reconciliation.csv](./mask_count_reconciliation.csv). The unique model has 68,461,056 parameters (tied embedding/LM head counted once): 67,108,864 prunable expert weights, 163,840 unpruned expert biases, and 1,188,352 other unpruned parameters.

Gradient energy is `||M*g_dense||^2 / ||g_dense||^2`; the dense denominator covers all coordinates in the layer/matrix expert group, while sparse gradients are compared on the same masked coordinates. Task and auxiliary gradients are separate. Paired checkpoints use the same 16 training-loader batch indices (0, 20, ..., 300), mean-token CE, fp32 with TF32 disabled, and no scaler or optimizer step. These are training batches already seen during optimization, not held-out gradients.

At step 5000, task-gradient actual-minus-random-mean ranges are:

| Matrix / sparsity | Actual minus random mean |
|---|---:|
| fc1 / .80 | -0.00243 to +0.00085 |
| fc1 / .85 | +0.00277 to +0.00370 |
| fc2 / .80 | +0.00305 to +0.00533 |
| fc2 / .85 | +0.00173 to +0.00330 |

For fc1/.85 and fc2 at both sparsities, all five seed-level values exceed all ten stored exact-count random draws. fc1/.80 is near its null. Random controls were not generated for auxiliary gradients. This withdraws “no more energy than random everywhere,” but does not establish a large or practically useful advantage, and is not an equivalence test.

Gradient subspace overlap uses aligned named parameter coordinates and uncentered 16x16 within/cross Gram matrices. The existing focused test compares this computation with direct SVD; numerical rank uses a relative eigenvalue threshold. No parameter-by-parameter covariance is formed. No random-subspace baseline or saved cutoff-degeneracy diagnostic is available, so small absolute overlaps are not calibrated against a null. Per-seed energies, null ranges, cosines, ranks, and statuses are in [gradient_audit.csv](./gradient_audit.csv).

## 4. Direct C-to-D selection-state pruning

`C` is the direct trained IMP state that selected the next mask: round_07 with mask_07 already present for the `.80` target, and round_08 with mask_08 already present for the `.85` target. `D` is an isolated copy of `C` with only the next mask applied to expert weights. Across the cohort, `C` has no nonzero expert weight outside its existing mask.

- `.7902848 -> .80`: 651,904 mask positions removed from 14,073,728 remaining (`4.6321%`).
- `.80 -> .85`: 3,355,520 positions removed from 13,421,824 remaining (`25.0005%`).

The C checkpoint, prior-mask and target-mask hashes, plus per-tensor, per-expert, per-layer, and aggregate counts are in [pruning_transition_audit.csv](./pruning_transition_audit.csv). C/D evaluations use identical validation data and precision. Only expert weights are masked; router and other parameters are copied unchanged. Whole-model CE/KL uses native routing with fixed router parameters, so route assignments are recomputed rather than frozen.

For `.80`, paired probe CE changes range from `-0.000408` to `+0.000907`, with KL(C||D) from `.000322` to `.001385`. For `.85`, CE rises by `+.041234` to `+.162638`, with KL from `.051657` to `.169837`. These transitions remove very different fractions of remaining weights, so they do not isolate target sparsity alone. The general claim “magnitude selection preserves function” is narrowed to these measured transitions.

## 5. Performance labels and candidate conclusion

The strict label remains sparse validation CE <= matched same-seed dense validation CE at 5,000 attempted steps on the two-segment rerun, with no tolerance. Four conditions match: seed 7 at `.80` and `.85`, seed 43 at `.80`, and seed 61 at `.80`. Six do not. Each sparse condition is paired with its same-seed dense control. Historical/rerun copies of identical weights are not independent replications.

For each matching condition, [claim_evidence_table.csv](./claim_evidence_table.csv) gives CE gap, output CKA/error and feature overlap, plus fc1/fc2 task-gradient cosines. Thus the proposed conclusion is supported only as a **descriptive co-occurrence** in this 32E IMP cohort. Replace “substantial” with numerical magnitudes; there was no predefined threshold. Gradient comparisons are native-state measurements on previously seen training batches and are affected by model-state, representation, and routing differences. The evidence does not establish causality, a unique winner signature, universal divergence, generalization, or publication novelty.

## 6. Reporting problems, measurement problems, and blockers

- **Reporting problems:** the extension's expert-output export name incorrectly implied sparse-native-only data although it contained both modes; the coupling figure mislabeled router-score/activation correlation as expert-output coupling. The consolidation code and figure title are corrected; the original report is retained.
- **Measurement problems:** none identified in the stored coupling, activation, distortion, gradient-energy/subspace, mask-count, or C-to-D calculations. No raw measurement output was invalidated or recomputed.
- **Numerical/interpretive revisions:** fixed-source coupling magnitudes survive with corrected labels; recovery geometry is probe-specific; task-gradient energy has small tensor-specific excess over null; fc1 keep rate follows joint pruning; C-to-D evidence is limited to unequal transitions.
- **Unresolved:** held-out gradient behavior, random-subspace calibration, causal relationships, and conditional connectivity remain unsupported or unavailable.
