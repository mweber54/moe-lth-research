# RAMP–IMP correspondence and sparse recoverability

Research date: 2026-11-13. Experiment `ramp_imp_correspondence_v1`. Status: retrospective CPU analysis complete; no recovery training launched. Pre-implementation audit: [ramp_imp_audit.md](ramp_imp_audit.md).

## Measured findings

Across architectures/sparsities, discovery-final within-expert RAMP–retention Spearman means are 0.024–0.049: weak alignment. Initialization RAMP means are 0.140–0.179. Initial magnitude alone reaches 0.280–0.316 at IMP80, exceeding the discovery-final RAMP alignment. Recovery prediction is unestablished with five independent seeds per stratum; no superiority over simple baselines is demonstrated. Degree-preserving controls show a concrete limitation of neuron-density summaries despite matched original-state RAMP scores.

The final-state pattern is closer to Outcome C than a strong-support Outcome A: weak RAMP alignment coexists with some IMP recoveries reaching paired dense validation CE. This is not proof that RAMP has zero recovery information, nor confirmation of Outcome B; its early scores and the limited calibration/sample size need separate interpretation.

## Scope and interpretation

Analyzed 30 historical IMP recoveries: E8/E16/E32, seeds 7/17/29/43/61, 80%/85% sparsity. The two masks within a seed are dependent. Five seeds per architecture/sparsity are the replication units; architectures reuse seed labels and are not pooled. All results, including failures to reach dense loss, remain included. Loss references are exact protocol-matched dense recoveries, not the dense checkpoint chosen for profiling. Recovery gap is (sparse CE − paired dense CE) / paired dense CE.

RAMP–retention correlations are descriptive associations. The H2 sample is too small for a defensible multivariable predictor; no fitted predictive model, feature selection, or neuron-level recovery regression is used. Leave-one-seed-out correlation ranges measure sensitivity, not generalization accuracy. The report does not establish that RAMP predicts recovery beyond simpler baselines.

A sharper limitation is directly testable: historical degree-preserving randomized supports have exactly the same up/down neuron retention as IMP, hence identical RAMP-weighted retention for **any fixed original-state RAMP profile**. Yet their final recovery losses differ. This means these neuron-retention summaries cannot explain all connection-level recoverability. It does not imply that profiling the altered networks would yield identical RAMP.

## Architecture and faithful score

The existing ExpertFFN is conventional: `fc2(dropout(GELU(fc1(x))))`, with bias terms. No neuron gate projection exists; the MoE router is separate. Adapt [Zhou et al., Eq. (3)](https://arxiv.org/html/2611.13975v1#S3.SS2) by taking actual GELU(fc1(x)) as intermediate activation in eval mode. RAMP = sqrt(sum (accepted routing coefficient × activation)^2) × norm(fc2 column). Biases are unpruned and output bias is outside the individual neuron score. No shared experts exist. Native top-k routing uses selected raw softmax coefficients without renormalization, including after capacity truncation. Rejected edges contribute zero. Secondary routes are included. Actual native traces and layer inputs drive a bounded eval replay, validated against actual fc2-input hooks and a direct calculation. Primary RAMP is unnormalized; per-dispatch and per-routing-energy scores are separately named baselines.

FP32 CPU native activations and float64 multiplication/reductions are used. Historical recovery used CUDA FP16; fresh CPU RAMP is a measurement on the saved state, not reproduction of historical FP16 dispatch bit for bit. Every profile records its checkpoint hash, config, calibration hash, source hashes, and software version. No historical mask or result is changed. Unsupported expert architectures fail explicitly; the generic accumulator accepts gated activations but only this repository's conventional FFN adapter is validated for full-model profiling.

## Calibration

Fixed selection seed 20261113: 64 sequences × 128 tokens = 8192 input tokens, equal exact-source TinyStories/WikiText quotas, batch size 32. Sampling is from canonical training-split blocks using `load_token_splits` and `TokenBlockDataset`; it is not an independent held-out calibration corpus. Every chosen 128-token sequence is rejected if it appears anywhere in the entire canonical validation byte stream. This verifies exact sequence disjointness, not absence of shorter phrases or document-level source overlap. Unique source-substring membership supplies domain labels; ambiguous/cross-boundary blocks are excluded. Exact token IDs, source paths/byte offsets, block indices, tokenizer implementation hash, file hashes, and token hash are in `results/ramp_imp_correspondence_v1/calibration_manifest.json`.

All checkpoints use identical token sequences and batch partitions. Source subsets accumulate contributions under the same pooled native capacity competition. Subset results are not equivalent to standalone source-only dispatch. This small 8192-token calibration can leave rarely routed experts unexposed; undefined within-expert correlations are omitted with valid-group and unexposed-expert counts retained. There is no domain-specialization claim or outcome-based calibration tuning.

## H1: support correspondence

Unpruned discovery round_00 is the directly ancestral dense state, separately named `discovery_dense_final`. It is compared with the eventual nested IMP masks without claiming that its dense magnitude ranking alone generated those later-round masks. Each group uses exact fc1 rows and fc2 columns. Pearson correlation of average ranks gives Spearman correlation with ties. Constant score/retention groups return undefined, not zero. Group means are formed per seed, then descriptive 95% Student-t intervals across five seeds. These intervals are not multiplicity-adjusted confirmatory tests.

| architecture | sparsity | scope | independent_seeds | mean | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- |
| E16 | 0.8000 | within_expert | 5 | 0.0241 | 0.0124 | 0.0358 |
| E16 | 0.8000 | within_layer | 5 | 0.0241 | 0.0105 | 0.0377 |
| E16 | 0.8500 | within_expert | 5 | 0.0361 | 0.0202 | 0.0519 |
| E16 | 0.8500 | within_layer | 5 | 0.0343 | 0.0206 | 0.0481 |
| E32 | 0.8000 | within_expert | 5 | 0.0354 | 0.0162 | 0.0545 |
| E32 | 0.8000 | within_layer | 5 | 0.0307 | 0.0102 | 0.0511 |
| E32 | 0.8500 | within_expert | 5 | 0.0489 | 0.0302 | 0.0675 |
| E32 | 0.8500 | within_layer | 5 | 0.0443 | 0.0281 | 0.0604 |
| E8 | 0.8000 | within_expert | 5 | 0.0375 | 0.0257 | 0.0493 |
| E8 | 0.8000 | within_layer | 5 | 0.0422 | 0.0283 | 0.0561 |
| E8 | 0.8500 | within_expert | 5 | 0.0437 | 0.0299 | 0.0575 |
| E8 | 0.8500 | within_layer | 5 | 0.0450 | 0.0295 | 0.0605 |

Within-expert baseline comparison at IMP80 on the same discovery dense states:

| architecture | feature | independent_seeds | mean | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- |
| E16 | activation_only | 5 | 0.0526 | 0.0369 | 0.0683 |
| E16 | checkpoint_magnitude | 5 | 0.0555 | 0.0440 | 0.0671 |
| E16 | down_norm | 5 | -0.0827 | -0.0923 | -0.0730 |
| E16 | initial_magnitude | 5 | 0.2971 | 0.2875 | 0.3067 |
| E16 | ramp | 5 | 0.0241 | 0.0124 | 0.0358 |
| E16 | ramp_per_dispatch | 5 | 0.0241 | 0.0124 | 0.0358 |
| E16 | ramp_per_route_energy | 5 | 0.0241 | 0.0124 | 0.0358 |
| E16 | random | 5 | -0.0009 | -0.0076 | 0.0057 |
| E16 | routing_activation | 5 | 0.0495 | 0.0361 | 0.0629 |
| E16 | utilization | 0 | undefined | undefined | undefined |
| E32 | activation_only | 5 | 0.0571 | 0.0473 | 0.0668 |
| E32 | checkpoint_magnitude | 5 | 0.0634 | 0.0176 | 0.1092 |
| E32 | down_norm | 5 | -0.0630 | -0.1054 | -0.0206 |
| E32 | initial_magnitude | 5 | 0.2798 | 0.2651 | 0.2946 |
| E32 | ramp | 5 | 0.0354 | 0.0162 | 0.0545 |
| E32 | ramp_per_dispatch | 5 | 0.0354 | 0.0162 | 0.0545 |
| E32 | ramp_per_route_energy | 5 | 0.0354 | 0.0162 | 0.0545 |
| E32 | random | 5 | 0.0002 | -0.0037 | 0.0041 |
| E32 | routing_activation | 5 | 0.0536 | 0.0429 | 0.0643 |
| E32 | utilization | 0 | undefined | undefined | undefined |
| E8 | activation_only | 5 | 0.0761 | 0.0644 | 0.0877 |
| E8 | checkpoint_magnitude | 5 | 0.0644 | 0.0275 | 0.1013 |
| E8 | down_norm | 5 | -0.0720 | -0.0973 | -0.0467 |
| E8 | initial_magnitude | 5 | 0.3157 | 0.3001 | 0.3312 |
| E8 | ramp | 5 | 0.0375 | 0.0257 | 0.0493 |
| E8 | ramp_per_dispatch | 5 | 0.0375 | 0.0257 | 0.0493 |
| E8 | ramp_per_route_energy | 5 | 0.0375 | 0.0257 | 0.0493 |
| E8 | random | 5 | -0.0014 | -0.0046 | 0.0018 |
| E8 | routing_activation | 5 | 0.0728 | 0.0624 | 0.0832 |
| E8 | utilization | 0 | undefined | undefined | undefined |

Full up/down correlations, within-layer comparisons, paired baseline differences, retention deciles, top-decile enrichment, AUC and average precision are in `h1_seed_summary.csv`, `h1_baseline_differences.csv`, and `h1_groups.parquet`. High-retention labels use retention >= within-layer 90th percentile, retaining tied labels; prevalence is saved. Average-rank deciles leave tied sets together and may leave empty bins. Random rankings use a fixed independent generator; utilization is constant within expert and cannot produce a within-expert rank correlation. Exposure-adjusted baselines distinguish frequently selected experts from per-dispatch contribution; adjustment is not substituted for Eq. (3).

## H2: sparse recovery

| architecture | sparsity | independent_seeds | mean_relative_gap | min_gap | max_gap |
| --- | --- | --- | --- | --- | --- |
| E16 | 0.8000 | 5 | 0.0456 | -0.0143 | 0.1025 |
| E16 | 0.8500 | 5 | 0.0690 | 0.0254 | 0.1025 |
| E32 | 0.8000 | 5 | 0.0016 | -0.0056 | 0.0081 |
| E32 | 0.8500 | 5 | 0.0345 | 0.0096 | 0.0556 |
| E8 | 0.8000 | 5 | 0.0541 | 0.0244 | 0.1022 |
| E8 | 0.8500 | 5 | 0.0898 | 0.0518 | 0.1149 |

Prespecified initial-state RAMP-weighted retention versus relative recovery gap, separately by architecture and sparsity:

| architecture | sparsity | independent_seeds | spearman_rho | loo_rho_min | loo_rho_max |
| --- | --- | --- | --- | --- | --- |
| E16 | 0.8000 | 5 | 0.0000 | -0.4000 | 0.4000 |
| E16 | 0.8500 | 5 | 0.8000 | 0.6000 | 0.8000 |
| E32 | 0.8000 | 5 | -0.1000 | -0.4000 | 0.8000 |
| E32 | 0.8500 | 5 | 0.2000 | -0.2000 | 0.8000 |
| E8 | 0.8000 | 5 | 0.5000 | 0.2000 | 0.6000 |
| E8 | 0.8500 | 5 | 0.1000 | -0.4000 | 0.6000 |

`h2_descriptive_associations.csv` contains the same information-budget univariate summaries for random, utilization, activation, down-norm, routing-weighted activation, and initial/current magnitude baselines. Positive correlation means larger score associates with worse recovery (positive gap). There are only five independent seeds per stratum and many exploratory summaries; a high sample correlation is not evidence of a reliable predictor or incremental value. No across-neuron train/test split is used.

Mask summaries are descriptive: importance-weighted coordinate density, retained-weight-weighted mean importance, importance share in neurons with >=50% retention, and mean within-expert/layer alignment. Individual contribution magnitudes can add in squared norm, while neuron output vectors can cancel or compensate; these summaries are not fractions of preserved function. Strong-retention shares can be zero or nearly constant at these sparsities. All H2 scores are conditioned on historical masks, so even initialization-state summaries do not constitute a deployable early mask-selection predictor.

## Timing and mechanistic integration

Actual reference trajectory steps 0,250,1250,2500 and unpruned discovery round_00 final are profiled. Rewind is step 0 in the eligible recovery cohort. The reference trajectory and IMP ancestor are distinct trajectories even when both are at step 2500. No unavailable state is fabricated. Measured early-to-final support alignment at IMP80:

| architecture | profile_step | independent_seeds | mean | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- |
| E16 | 0 | 5 | 0.1656 | 0.1440 | 0.1872 |
| E16 | 250 | 5 | 0.1391 | 0.1226 | 0.1556 |
| E16 | 1250 | 5 | 0.1171 | 0.1113 | 0.1230 |
| E16 | 2500 | 5 | 0.0241 | 0.0124 | 0.0358 |
| E32 | 0 | 5 | 0.1477 | 0.1309 | 0.1645 |
| E32 | 250 | 5 | 0.1334 | 0.1167 | 0.1501 |
| E32 | 1250 | 5 | 0.1065 | 0.0904 | 0.1226 |
| E32 | 2500 | 5 | 0.0354 | 0.0162 | 0.0545 |
| E8 | 0 | 5 | 0.1785 | 0.1529 | 0.2041 |
| E8 | 250 | 5 | 0.1682 | 0.1384 | 0.1980 |
| E8 | 1250 | 5 | 0.1413 | 0.1262 | 0.1564 |
| E8 | 2500 | 5 | 0.0375 | 0.0257 | 0.0493 |

Reference early RAMP uses only weights and activations at that state; current-magnitude features never use later weights. Initial magnitude is available at all timings. Later checkpoints are retrospective explanation, not evidence for pruning during training. `checkpoint_rank_stability.csv` reports rank correlation to the actual final reference state; Figure 5 compares early RAMP with eventual IMP support. Earliest sampled step is not the true onset of predictability.

Historical E16 IMP80 degree controls, averaged across five nested randomized masks per seed:

| seed | nested_masks | all_neuron_retentions_preserved | ramp_score_difference | control_minus_imp_ce |
| --- | --- | --- | --- | --- |
| 7 | 5 | True | 0.0000 | 0.0951 |
| 17 | 5 | True | 0.0000 | 0.2221 |
| 29 | 5 | True | 0.0000 | 0.1825 |
| 43 | 5 | True | 0.0000 | 0.2047 |
| 61 | 5 | True | 0.0000 | 0.0251 |

All saved control masks are reloaded, tensor hashes checked, and exact up/down neuron retentions verified. `historical_mask_controls.csv` preserves every mask and curve path; the separate seed summary prevents treating random draws as independent training seeds. The original-profile neuron-retention score cannot distinguish these degree controls. Matrix-random controls are also retained. Their recovery protocol is kept separate from primary 2500-step endpoints; a historical dense value is labeled as such. No causal preservation of function is inferred from matched scores.

The existing E32 adaptation factorial is joined separately in `adaptation_context_separate_protocol.csv`, using condition-matched dense endpoints at its own 5000-step budget. Fixed initial RAMP summaries are identical across changes to trainable parameter groups, while later recovery can differ. Thus an initial functional contribution metric alone does not encode the allowed optimization/adaptation process. Frozen router parameters do not freeze token exposure as upstream states change. New router/exposure perturbations were not introduced.

The completed initial-value-placement study (`results/imp_initial_value_placement_v1/findings.md`, `source_level_summary.csv`, `endpoint_losses.csv`) reports that shuffling surviving values harms original IMP more than sampled random supports across five sources. That intervention combines sign, magnitude, and placement. A distinct completed sign-only Stage1 cohort (`results/imp_sign_recovery_stage1_v1/completion.json`, 54 runs, 3 source seeds) reports IMP-minus-random sign-shuffling interactions at step2000 of 0.1744, 0.1785 and 0.1947 CE for seeds7/17/29. Its fixed-order/draw uncertainty is conditional on sources; `sign_only_context_separate_protocol.csv` preserves those historical summaries without pooling protocols. This experiment has not separately profiled those transformed states or matched neurons by RAMP to identify sign effects; no claim that RAMP explains or removes the historical sign-specific effect is justified. `endpoint_alignment_groups.csv` additionally separates at-or-better-dense endpoints (gap<=0) from positive-gap endpoints without calling either group a certified winning ticket; E8/85 and other strata can lack one group, and the small groups preclude a reliable group-effect estimate. Primary comparisons use continuous relative gaps, with no outcome-selected seeds or threshold tuning. No individual final-mask sign statistic is treated as evidence of causal sign importance.

## Artifacts, figures and reproducibility

Primary artifact paths and hashes are in `verified_recovery_pairs.csv`; per-checkpoint manifests identify exact states. `e*_seed*/*/neurons_s*.parquet` is the consolidated partitioned pooled neuron/checkpoint/mask dataset (one row per pairing); profile files contain source-subset RAMP/exposure and pairing files contain initial/current surviving magnitude/sign statistics. Stable seed, initialization, mask, discovery, rewind, recovery, and checkpoint identifiers are included. `mask_run_results.csv` has one row per mask/checkpoint/domain. `h1_groups.parquet` and `retention_deciles.parquet` preserve group and decile calculations. The calibration sample is registered before any outcomes enter analysis. Partial reruns require matching registration, input hashes, and completed artifact hashes. `engineering_amendment.json` preserves the original profiling source/registration and records a correction confined to the unexecuted H3 apportionment helper; the profiling implementation, calibration and H1/H2 choices are unchanged.

Figures 1a/1b: scatter and retention-decile curves; Figure 2: seed7/layer0 RAMP-rank and up/down structure; Figure 3: independent-seed alignment points against baselines; Figure 4: mask-level initial score versus relative recovery gap; Figures 5/5b: actual dense-checkpoint timing and rank stability. PDF vector exports and 240-dpi PNGs are under `results/ramp_imp_correspondence_v1/figures`. Exact source runs and underlying data paths are listed per figure in `figures/figure_manifest.json`. Supplement: existing mask controls, not H3. Figure 6 is deliberately absent because controlled new recoveries were not executed.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/ramp_imp_correspondence_v1/run.py --output results/ramp_imp_correspondence_v1
.venv/bin/python -m pytest tests/test_ramp_imp_correspondence.py tests/test_ramp_imp_pipeline.py -q
# Final sparse-weight and canonical-stream validation supplement:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/ramp_imp_correspondence_v1/supplement.py --output results/ramp_imp_correspondence_v1
# Validate final bundle hashes/counts:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/ramp_imp_correspondence_v1/validate_results.py --output results/ramp_imp_correspondence_v1
# Rebuild summaries and figures from validated saved profiles:
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/ramp_imp_correspondence_v1/run.py --output results/ramp_imp_correspondence_v1 --summarize-only
```

`recovered_neuron_statistics/*.parquet` supplies separately timed final sparse surviving-weight statistics; join by architecture/seed/sparsity/mask/layer/expert/neuron. These measurements are excluded from initial/early features. `reconstruction_validation.json` verifies canonical train/validation stream reconstruction and profile payload steps/configs; `supplement_manifest.json` records all supplemental artifact hashes.

The analysis uses pandas, installed in the repository environment through existing dataset dependencies; no packages or datasets were downloaded. Inference uses current source including pre-existing local architecture edits; source hashes preserve that version. Unit/integration test results are recorded separately in `test_validation.txt`. Calibration, masks, checkpoint payloads, historical sparse endpoints, curve values and data identities have explicit validation checks. A missing/mismatched artifact halts rather than substituting a new model. `completion.json` records completion only after all summary/control/plot stages succeed.

## Validation and remaining scope

The completed run passes 33 unit/integration/regression tests. `validation_summary.json` verifies all75 profile bundles, 11,468,800 pooled neuron/checkpoint/mask rows, 30 final sparse supports, all15 reconstructed canonical train/validation streams, checkpoint payload/config identities, and original dense reference hashes. All25 historical degree-control masks preserve neuron retention exactly. No unresolved numerical or provenance check failed.

Intermediate discovery masks01–07 are inventoried and supported by the generic mapping utility but are outside this report’s primary 80%/85% final-recovery analysis. Perturbed sign/router endpoint profiles are not claimed; their historical findings are contextual only.

## Decision and limitations

Interpret H1 using the displayed effect sizes and baseline comparisons, without treating correlation as causal importance. H2 remains descriptive; five seeds cannot demonstrate robust predictive value or superiority over magnitude/utilization baselines. The degree-control evidence directly rules out a complete account of recoverability based solely on original-profile importance and neuron connection density. The requested A/B/C outcomes therefore need not be exhaustive: support alignment and failure to capture connection topology can coexist. A strong positive H1 with inconclusive H2 is not yet a confirmed Outcome B (absence of predictive value).

Limitations include one conventional FFN family varied by expert count, one mixed dataset and protocol, small in-training calibration, capacity/batch dependence, CPU/FP16 numerical differences, limited seeds, dependence of nested masks, source seeds varying both initialization and training order, no independently repeated discoveries per seed, no untouched test or equivalence criterion, and no intervention-specific RAMP measurement. No new recovery experiment, causal claim, domain-specialization claim, or winning-ticket certification follows automatically.

A narrowly controlled H3 is scientifically reasonable to distinguish ranking value from connection topology, but the current associations do not establish expected benefit. The prepared design in `scripts/ramp_imp_correspondence_v1/H3_design.md` preserves exact expert-local unstructured budgets and all recovery controls. It is not launched. No conclusion should force the observations to support routing-conditioned optimization; the measured limitations indicate which connection-level and adaptation information RAMP alone lacks.
