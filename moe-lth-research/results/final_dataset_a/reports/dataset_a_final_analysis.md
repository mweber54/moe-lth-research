# Dataset A final analysis attempt — PASS

## 1. Executive summary

**FINALIZATION PASS.** This is an existing-artifact diagnostic analysis. The canonical-named table is a partial candidate unless the finalization gate passes; it must not be cited as a complete frozen dataset.

1. 30/30 architecture/sparsity curves have all five seeds and eleven ages.
2. Among complete curves, 16 have intermediate mean winners, 10 R0 winners and 4 R100 winners.
3. All five seed winner sets overlap the mean winner in 1/30 complete curves.
4. At 95% pruning, complete-curve mean winners are 8E: R0, 12E: R0, 16E: R0, 24E: R0, 32E: R0.
5. Ticket gaps independently recomputed for 1650 rows; 207 stored comparisons match and 0 mismatch.

## 2. Dataset completeness

| architecture | dense | source | sparse |
| --- | --- | --- | --- |
| 12E | 55 | 5 | 330 |
| 16E | 55 | 5 | 330 |
| 24E | 55 | 5 | 330 |
| 32E | 55 | 5 | 330 |
| 8E | 55 | 5 | 330 |


Expected 25 source trajectories, 275 dense and 1650 sparse recoveries. Observed validated counts: 25, 275, 1650. Exact Cartesian products, not manifest counts, drive the gate. 0 expected conditions fail. See `../audits/dataset_a_completion_audit.csv` for every cell, original status and reason; `dataset_a_candidate_audit.csv` retains relocated copies and rejected candidates. Source artifacts were moved: the active directory is a host-local /data symlink, and earlier recoveries survive under HOME_BACKUP. Selection uses original manifest paths first and then stable candidate path order, never endpoint quality. Byte-identical backup copies are provenance duplicates, not extra replicates. Conflicting valid checkpoints block admission.

The local manifest has stale PENDING/RUNNING statuses; completion logs do not substitute for absent checkpoint artifacts. The adder and viper logs report shard completion but the adder-owned groups are not all present on this host. Remote access was not established (SSH host-key verification failed). No statuses or raw artifacts were changed.

## 3. Protocol validation

Canonical protocol: source and recovery each use 2500 attempted updates, batch 128, sequence length 128; source global input budget is 40,960,000 token occurrences. Model vocabulary 256, width 256, four transformer layers, four heads, expert width 1024, native top-1 learned routing, capacity factor 1.25, dropout zero, auxiliary coefficient 0.1. Recovery creates fresh AdamW with zero initial optimizer state, LR 0.0003, weight decay 0.01, clip norm 1, fp16 GradScaler, no scheduler. Applied optimizer steps and AMP skips are retained separately when recorded.

Source YAMLs use `common.protocol_errors`; recovery checkpoint configs are loaded directly and checked with that same protocol validator. Source checkpoint SHA-256 values are checked against original checkpoint manifests; recovery checkpoint SHA-256 values against validation attestations or the accepted historical audit. Every admitted recovery checkpoint is checked for finite tensors and every sparse checkpoint for masked zeros. Initial router, shared rewind, sparse rewound expert and mask component hashes are checked. Sparse rewind hashes are recomputed from R0 and the existing stored mask; masks are never regenerated. All 150 stored masks are independently checked against their source-R100 weights: within each expert, the smallest retained magnitude is at least the largest pruned magnitude, the rounded keep count matches the existing pruning definition, and biases are excluded. Existing origin attestations are preserved; no masks are regenerated and no new pruning is performed. Data files and per-seed batch-sequence hashes are preserved and metadata-linked. Historical missing native-router labels are allowed only under the original default trainable mode, changing router hashes and accepted provenance; unavailable historical update counters remain NA.

Source evaluation cadence is 250 attempts; checkpoint-only saves do not change it. Recovery curves must contain exactly steps 0,50,…,2500 in order with finite validation losses and an endpoint matching the summary. The final endpoint is step 2500, never best-validation selection. Validation is token-weighted cross entropy excluding the auxiliary term. Training loss in gradient logs includes the weighted auxiliary term; the separate auxiliary scalar was not logged and remains NA.

Protocol audit and candidate rejection tables expose all checks and exceptions. Historical incompatible runs remain excluded; a raw label match alone never admits a run. Hash validation supports artifact integrity and linkage; it does not independently replay training or prove the historical data generator's entire execution.

## 4. Canonical metric definitions

Ticket gap = raw sparse step-2500 validation CE − same-architecture/seed/age raw dense step-2500 CE. Dense is reused across six sparsities. Gap comparison tolerance is absolute 1e-10. Missing historical stored gaps remain NA with NO_STORED_GAP, not failures. Pruning shock = sparse step-zero CE − matched dense step-zero CE; the preprune field denotes that equivalent unmasked rewound model. Early AUC uses the existing finalizer's trapezoidal integral over 0–1250 at 50-step spacing, without normalization. Recovery improvement = initial − final CE. Positive intermediate advantage = boundary mean gap − best intermediate mean gap. Numerical ties use absolute 1e-12 and preserve all tied ages. Seeds are independent replicates; SD uses ddof=1.

## 5. Full router-age results

OBSERVATION: the following are available-data mean winners. Partial architecture rows are provisional, with changing seed coverage; use the full summary table and balanced-seed sensitivity before interpreting them.

| architecture | 70% | 75% | 80% | 85% | 90% | 95% |
| --- | --- | --- | --- | --- | --- | --- |
| 8E | R70 | R80 | R90 | R80 | R80 | R0 |
| 12E | R50 | R50 | R100 | R70 | R100 | R0 |
| 16E | R60 | R80 | R60 | R60 | R60 | R0 |
| 24E | R100 | R50 | R50 | R50 | R0 | R0 |
| 32E | R0 | R100 | R0 | R0 | R0 | R0 |


| architecture | sparsity | mean_winning_ages | complete_five_seed_curve | min_seeds_per_age | max_seeds_per_age | minimum_mean_ticket_gap | intermediate_advantage_vs_r0 | intermediate_advantage_vs_r100 | descriptive_basin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8E | 0.70000 | R70 | True | 5 | 5 | 0.03715 | 0.02089 | 0.00627 | R70;R80;R90;R100 |
| 8E | 0.75000 | R80 | True | 5 | 5 | 0.06627 | 0.04786 | 0.00856 | R70;R80;R90;R100 |
| 8E | 0.80000 | R90 | True | 5 | 5 | 0.12009 | 0.04657 | 0.00615 | R70;R80;R90;R100 |
| 8E | 0.85000 | R80 | True | 5 | 5 | 0.19903 | 0.05680 | 0.00869 | R10;R20;R30;R40;R50;R60;R70;R80;R90;R100 |
| 8E | 0.90000 | R80 | True | 5 | 5 | 0.31363 | 0.00855 | 0.00252 | R60;R70;R80;R90;R100 |
| 8E | 0.95000 | R0 | True | 5 | 5 | 0.41964 | -0.02272 | 0.03153 | R0;R10;R20;R30 |
| 12E | 0.70000 | R50 | True | 5 | 5 | 0.04371 | 0.01220 | 0.00452 | R30;R40;R50;R60;R70;R80;R90;R100 |
| 12E | 0.75000 | R50 | True | 5 | 5 | 0.07352 | 0.02844 | 0.01984 | R40;R50;R60;R70;R80;R90;R100 |
| 12E | 0.80000 | R100 | True | 5 | 5 | 0.12424 | 0.04439 | -0.00101 | R90;R100 |
| 12E | 0.85000 | R70 | True | 5 | 5 | 0.20214 | 0.03064 | 0.00402 | R20;R30;R40;R50;R60;R70;R80;R90;R100 |
| 12E | 0.90000 | R100 | True | 5 | 5 | 0.29884 | 0.00574 | -0.00239 | R50;R60;R70;R80;R90;R100 |
| 12E | 0.95000 | R0 | True | 5 | 5 | 0.39005 | -0.04543 | 0.01358 | R0 |
| 16E | 0.70000 | R60 | True | 5 | 5 | 0.02613 | 0.04191 | 0.00960 | R40;R50;R60;R70;R80;R90;R100 |
| 16E | 0.75000 | R80 | True | 5 | 5 | 0.04782 | 0.06790 | 0.01688 | R50;R60;R70;R80;R90;R100 |
| 16E | 0.80000 | R60 | True | 5 | 5 | 0.09517 | 0.05947 | 0.01978 | R40;R50;R60;R70;R80;R90;R100 |
| 16E | 0.85000 | R60 | True | 5 | 5 | 0.16604 | 0.04028 | 0.02382 | R30;R40;R50;R60;R70;R80;R90;R100 |
| 16E | 0.90000 | R60 | True | 5 | 5 | 0.26317 | 0.00091 | 0.01987 | R30;R40;R50;R60;R70;R80;R90;R100 |
| 16E | 0.95000 | R0 | True | 5 | 5 | 0.36586 | -0.03074 | 0.02749 | R0 |
| 24E | 0.70000 | R100 | True | 5 | 5 | 0.03937 | 0.00989 | -0.00227 | R100 |
| 24E | 0.75000 | R50 | True | 5 | 5 | 0.05939 | 0.02612 | 0.00816 | R10;R20;R30;R40;R50;R60;R70;R80;R90;R100 |
| 24E | 0.80000 | R50 | True | 5 | 5 | 0.09757 | 0.01474 | 0.01166 | R40;R50;R60;R70;R80;R90;R100 |
| 24E | 0.85000 | R50 | True | 5 | 5 | 0.16494 | 0.00761 | 0.00498 | R40;R50;R60;R70;R80;R90;R100 |
| 24E | 0.90000 | R0 | True | 5 | 5 | 0.21980 | -0.01750 | 0.00625 | R0;R10;R20 |
| 24E | 0.95000 | R0 | True | 5 | 5 | 0.31323 | -0.03579 | 0.01452 | R0 |
| 32E | 0.70000 | R0 | True | 5 | 5 | 0.04329 | -0.01209 | -0.00810 | R0;R10;R20;R30;R40;R50;R60;R70;R80;R90;R100 |
| 32E | 0.75000 | R100 | True | 5 | 5 | 0.06927 | -0.00332 | -0.00849 | R50;R60;R70;R80;R90;R100 |
| 32E | 0.80000 | R0 | True | 5 | 5 | 0.11076 | -0.01440 | -0.00460 | R0 |
| 32E | 0.85000 | R0 | True | 5 | 5 | 0.15263 | -0.03505 | -0.00688 | R0 |
| 32E | 0.90000 | R0 | True | 5 | 5 | 0.22502 | -0.02686 | -0.00176 | R0 |
| 32E | 0.95000 | R0 | True | 5 | 5 | 0.31130 | -0.06782 | 0.00207 | R0 |


The connected descriptive basin reuses September 21's rule: adjacent measured ages belong while their paired difference from the representative mean winner is within one paired seed SD. Missing ages break connectivity. This descriptive measure is not a confidence interval, equivalence test, or established optimum. Full differences, runner-up gaps and seed SD are retained.

## 6. Sparsity dependence

OBSERVATION: 16/30 complete architecture/sparsity curves have an intermediate mean below both boundaries. The best intermediate is computed separately even when R0 or R100 wins overall. Extreme 95% observations are listed in the executive summary. Seed-specific boundary contrasts remain paired and descriptive. No monotone shift in age or universal intermediate preference is assumed.

## 7. Architecture dependence

OBSERVATION: inspect the architecture × sparsity table for varying mean winners, gaps and basin widths. Comparisons include equal global source/recovery attempts, not equal exposure or FLOPs. Partial 24E and 32E results cannot resolve five-seed architecture effects. Mean curves never pool grid cells as independent seeds.

## 8. Five-seed reproducibility

OBSERVATION: 1/30 complete curves have all five seeds' numerical winner sets overlapping the mean winner. Individual curves and the 150-row seed table preserve heterogeneous winners and absent seed/age coverage. Per-seed best-intermediate comparisons are selected on that same seed and are not independent confirmation. Mean-winner boundary contrasts are paired by seed; selection remains descriptive.

| architecture | sparsity | mean_winning_ages | seeds_agreeing_with_mean_winner | seed_specific_winning_ages | gap_to_second_lowest | median_seed_sd |
| --- | --- | --- | --- | --- | --- | --- |
| 8E | 0.70000 | R70 | 0 | {"7": [20], "17": [0], "29": [20], "43": [0], "61": [90]} | 0.00268 | 0.02158 |
| 8E | 0.75000 | R80 | 1 | {"7": [20], "17": [0], "29": [20], "43": [80], "61": [90]} | 0.00097 | 0.02884 |
| 8E | 0.80000 | R90 | 1 | {"7": [20], "17": [100], "29": [20], "43": [90], "61": [20]} | 0.00525 | 0.02595 |
| 8E | 0.85000 | R80 | 1 | {"7": [20], "17": [60], "29": [30], "43": [80], "61": [40]} | 0.00247 | 0.02546 |
| 8E | 0.90000 | R80 | 0 | {"7": [20], "17": [100], "29": [20], "43": [90], "61": [100]} | 0.00252 | 0.01812 |
| 8E | 0.95000 | R0 | 3 | {"7": [20], "17": [0], "29": [30], "43": [0], "61": [0]} | 0.02272 | 0.02233 |
| 12E | 0.70000 | R50 | 0 | {"7": [0], "17": [10], "29": [60], "43": [70], "61": [0]} | 0.00452 | 0.02564 |
| 12E | 0.75000 | R50 | 1 | {"7": [0], "17": [50], "29": [60], "43": [80], "61": [90]} | 0.00247 | 0.02900 |
| 12E | 0.80000 | R100 | 1 | {"7": [90], "17": [10], "29": [100], "43": [70], "61": [90]} | 0.00101 | 0.03373 |
| 12E | 0.85000 | R70 | 1 | {"7": [0], "17": [70], "29": [80], "43": [10], "61": [100]} | 0.00053 | 0.03945 |
| 12E | 0.90000 | R100 | 0 | {"7": [0], "17": [70], "29": [60], "43": [10], "61": [0]} | 0.00239 | 0.02971 |
| 12E | 0.95000 | R0 | 3 | {"7": [0], "17": [50], "29": [60], "43": [0], "61": [0]} | 0.04543 | 0.01948 |
| 16E | 0.70000 | R60 | 1 | {"7": [50], "17": [90], "29": [60], "43": [80], "61": [80]} | 0.00020 | 0.02799 |
| 16E | 0.75000 | R80 | 2 | {"7": [50], "17": [90], "29": [70], "43": [80], "61": [80]} | 0.00222 | 0.02619 |
| 16E | 0.80000 | R60 | 2 | {"7": [90], "17": [60], "29": [60], "43": [80], "61": [70]} | 0.00162 | 0.02622 |
| 16E | 0.85000 | R60 | 2 | {"7": [100], "17": [60], "29": [60], "43": [80], "61": [70]} | 0.00529 | 0.02000 |
| 16E | 0.90000 | R60 | 2 | {"7": [0], "17": [60], "29": [60], "43": [80], "61": [0]} | 0.00091 | 0.02823 |
| 16E | 0.95000 | R0 | 3 | {"7": [0], "17": [60], "29": [30], "43": [0], "61": [0]} | 0.03074 | 0.02998 |
| 24E | 0.70000 | R100 | 2 | {"7": [0], "17": [30], "29": [100], "43": [100], "61": [70]} | 0.00227 | 0.02014 |
| 24E | 0.75000 | R50 | 1 | {"7": [20], "17": [30], "29": [100], "43": [50], "61": [70]} | 0.00368 | 0.02116 |
| 24E | 0.80000 | R50 | 1 | {"7": [20], "17": [50], "29": [60], "43": [40], "61": [60]} | 0.00921 | 0.02454 |
| 24E | 0.85000 | R50 | 1 | {"7": [20], "17": [40], "29": [70], "43": [100], "61": [50]} | 0.00141 | 0.02500 |
| 24E | 0.90000 | R0 | 3 | {"7": [0], "17": [0], "29": [100], "43": [0], "61": [70]} | 0.01750 | 0.03221 |
| 24E | 0.95000 | R0 | 3 | {"7": [0], "17": [30], "29": [100], "43": [0], "61": [0]} | 0.03579 | 0.03647 |
| 32E | 0.70000 | R0 | 3 | {"7": [0], "17": [100], "29": [30], "43": [0], "61": [0]} | 0.00398 | 0.01708 |
| 32E | 0.75000 | R100 | 2 | {"7": [30], "17": [100], "29": [80], "43": [90], "61": [100]} | 0.00517 | 0.02682 |
| 32E | 0.80000 | R0 | 2 | {"7": [0], "17": [0], "29": [50], "43": [90], "61": [100]} | 0.00980 | 0.03223 |
| 32E | 0.85000 | R0 | 4 | {"7": [0], "17": [0], "29": [0], "43": [80], "61": [0]} | 0.02817 | 0.04473 |
| 32E | 0.90000 | R0 | 2 | {"7": [30], "17": [10], "29": [0], "43": [80], "61": [0]} | 0.02510 | 0.03607 |
| 32E | 0.95000 | R0 | 5 | {"7": [0], "17": [0], "29": [0], "43": [0], "61": [0]} | 0.06782 | 0.02880 |


## 9. Recovery-trajectory analysis

OBSERVATION: the long table preserves every logged training attempt and validation point, with dense trajectories stored once per dense condition. Missing training or auxiliary measurements remain NA. Validation endpoints and cadence are checked independently. Thirty PDF/PNG panel sets show all eleven ages for each architecture/sparsity, sparse and dense means with seed SD; log CE axes keep large initial losses visible.

`dataset_a_early_auc_decomposition.csv` separates the 0–50 interval from 50–1250, since a large step-zero shock can dominate early AUC. Across available rows, the median initial matched loss gap is -3.38016, at step 50 0.95760, step 1250 0.19057, and step 2500 0.16405. These medians describe grid coverage, not independent replication or inferential estimates. Per-age/seed traces, rather than these pooled medians, support timing comparisons.

## 10. Metric disagreement

OBSERVATION: among 30 complete curves, disjoint winner-set counts versus ticket gap are {'sparse_final_validation_loss': 23, 'dense_final_validation_loss': 30, 'pruning_shock': 22, 'matched_early_auc_gap': 18, 'sparse_early_auc': 26}. Raw sparse CE and matched ticket gap have different denominators/comparators; pruning shock and early AUC measure different recovery phases. See the disagreement table and paired sparse/dense boundary decomposition rather than treating every minimum as the same outcome.

## 11. Router developmental statistics

OBSERVATION: only existing identity-validated source probes are used. Availability across 275 source seed/age rows: {"router_entropy": 105, "router_margin": 105, "router_load_cv": 105, "router_token_id_specialization": 105, "router_assignment_agreement_to_r100": 105, "router_js_distance_to_r100": 105, "router_cumulative_sampled_exposure": 275}. Entropy, margin, load CV, token-ID specialization, R100 agreement and historical functional JS are never inferred from recovery diagnostics. Missing probes remain gaps in figures. Sampled cumulative proposed exposure is distinct from complete accepted-assignment counters. Historical JS is retained under its existing functional-distance definition, not silently square-root transformed.

INTERPRETATION: normalized age does not guarantee equal developmental state across architectures. Probe coverage is too sparse to establish a full five-seed developmental equivalence; age-matched plots show only the measured subset. No interpolation or new probes were run.

## 12. Exposure scaling

OBSERVATION: source accepted exposure is computed over the complete 2500-attempt log when available and divided by four layers × expert count. It is not multiplied by router age. Available-seed statistics are:

| num_experts | mean | std | count |
| --- | --- | --- | --- |
| 8 | 4893650.38125 | 8369.55880 | 5 |
| 12 | 3236539.81250 | 6004.68914 | 5 |
| 16 | 2416012.33594 | 1655.17125 | 2 |
| 24 | 1585824.77083 | 4249.87380 | 2 |
| 32 | 1171930.65625 | 4756.65051 | 5 |


Missing historical source counters remain NA. Counts and SD refer to source seeds once each, never to repetitions of the source value across 66 sparse cells. These measurements describe exposure decline or heterogeneity; no scaling law is fitted. Applied expert steps and attempted active steps remain separate.

## 13. Comparison against September 21 interim observations

| architecture | sparsity | september_21_winning_ages | mean_winning_ages | winner_changed | complete_five_seed_curve |
| --- | --- | --- | --- | --- | --- |
| 8E | 0.70000 | R70 | R70 | False | True |
| 8E | 0.75000 | R80 | R80 | False | True |
| 8E | 0.80000 | R90 | R90 | False | True |
| 8E | 0.85000 | R80 | R80 | False | True |
| 8E | 0.90000 | R80 | R80 | False | True |
| 8E | 0.95000 | R0 | R0 | False | True |
| 12E | 0.70000 | R50 | R50 | False | True |
| 12E | 0.75000 | R50 | R50 | False | True |
| 12E | 0.80000 | R100 | R100 | False | True |
| 12E | 0.85000 | R70 | R70 | False | True |
| 12E | 0.90000 | R100 | R100 | False | True |
| 12E | 0.95000 | R0 | R0 | False | True |
| 16E | 0.70000 | R60 | R60 | False | True |
| 16E | 0.75000 | R80 | R80 | False | True |
| 16E | 0.80000 | R60 | R60 | False | True |
| 16E | 0.85000 | R60 | R60 | False | True |
| 16E | 0.90000 | R60 | R60 | False | True |
| 16E | 0.95000 | R0 | R0 | False | True |
| 24E | 0.70000 | R30 | R100 | True | True |
| 24E | 0.75000 | R30 | R50 | True | True |
| 24E | 0.80000 | R50 | R50 | False | True |
| 24E | 0.85000 | R40 | R50 | True | True |
| 24E | 0.90000 | R0 | R0 | False | True |
| 24E | 0.95000 | R0 | R0 | False | True |
| 32E | 0.70000 | NA | R0 | True | True |
| 32E | 0.75000 | NA | R100 | True | True |
| 32E | 0.80000 | NA | R0 | True | True |
| 32E | 0.85000 | R0 | R0 | False | True |
| 32E | 0.90000 | NA | R0 | True | True |
| 32E | 0.95000 | NA | R0 | True | True |


OBSERVATION: completed 8E/12E/16E results can be compared directly; 24E/32E still require full artifact coverage before earlier provisional findings become final. The 32E/85% R0 preference must be read with its current coverage flag, not advertised as five-seed confirmation. Balanced full-age seed sensitivity is saved separately. No interim conclusion is silently promoted to final.

## 14. Negative and contradictory results

Boundary winners, disagreement between metric minima, and heterogeneous seed curves are retained equally. Intermediate wins do not imply all seeds favor the same age. Missing remote artifacts are an integrity limitation, not unfavorable outcomes to omit. A selected exact mean minimum is not a statistically established optimum.

## 15. Limitations

The full-grid finalization gate is PASS. Five seeds limit ranking precision. Same data select and describe minima; no p-values or significance claims are made. Equal global budgets do not establish equal per-expert exposure, equal development, equal convergence, identical FLOPs, or causal router mechanisms. Historical missing diagnostics are explicit. Source/mask history relies partly on pre-existing generation attestations; raw training is not replayed.

## 16. Implications for Dataset B

REQUIRES FURTHER EXPERIMENT: a separately specified study could assess whether observed associations persist after controlling expert exposure or changing source development budgets. Dataset B was not started and its protocol was not changed or generated here.

## 17. Candidate mechanism hypotheses

HYPOTHESIS: router development may covary with recoverability through assignment specialization, accepted-load balance, or dense-baseline changes. These competing explanations are not identified by Dataset A. INTERPRETATION: paired endpoint decompositions help locate which measured loss moves; they do not isolate a causal mechanism. REQUIRES FURTHER EXPERIMENT: controlled interventions and an independently specified confirmation design.

## 18. Conclusions supported by Dataset A alone

Only the validated observed cells support descriptive associations under equal global budgets. Full Dataset A completion and freezing require every required source, dense and sparse cell to pass integrity checks. The status manifest and automated tests are authoritative; no frozen marker is written when a required check fails.

No new training, recovery, masks, or Dataset B runs were started. Raw experimental artifacts were opened read-only and were not modified.

## Additional measured diagnostics

OBSERVATION: the current 24E/32E provisional winners and coverage are explicitly compared below. Neither changing nor unchanged available-seed winners establish completion.

| architecture | sparsity | september_21_winning_ages | mean_winning_ages | complete_five_seed_curve | min_seeds_per_age | max_seeds_per_age |
| --- | --- | --- | --- | --- | --- | --- |
| 24E | 0.70000 | R30 | R100 | True | 5 | 5 |
| 24E | 0.75000 | R30 | R50 | True | 5 | 5 |
| 24E | 0.80000 | R50 | R50 | True | 5 | 5 |
| 24E | 0.85000 | R40 | R50 | True | 5 | 5 |
| 24E | 0.90000 | R0 | R0 | True | 5 | 5 |
| 24E | 0.95000 | R0 | R0 | True | 5 | 5 |
| 32E | 0.70000 | NA | R0 | True | 5 | 5 |
| 32E | 0.75000 | NA | R100 | True | 5 | 5 |
| 32E | 0.80000 | NA | R0 | True | 5 | 5 |
| 32E | 0.85000 | R0 | R0 | True | 5 | 5 |
| 32E | 0.90000 | NA | R0 | True | 5 | 5 |
| 32E | 0.95000 | NA | R0 | True | 5 | 5 |


OBSERVATION: accepted source exposure ratios are paired to 8E within the same seed. Missing source counters are not replaced with sampled proposed exposure.

| num_experts | n_paired_source_seeds | mean_paired_exposure_ratio_vs_8e | sd_paired_exposure_ratio_vs_8e |
| --- | --- | --- | --- |
| 8 | 5 | 1.00000 | 0.00000 |
| 12 | 5 | 0.66138 | 0.00107 |
| 16 | 2 | 0.49374 | 0.00074 |
| 24 | 2 | 0.32408 | 0.00016 |
| 32 | 5 | 0.23948 | 0.00075 |


OBSERVATION: measured R100−R0 source developmental changes (paired source seeds only):

| num_experts | metric | n_paired_seeds | r100_minus_r0_mean | r100_minus_r0_sample_sd |
| --- | --- | --- | --- | --- |
| 8 | router_entropy | 3 | -0.56947 | 0.00759 |
| 8 | router_margin | 3 | 0.26034 | 0.00951 |
| 8 | router_load_cv | 3 | -0.22214 | 0.03095 |
| 8 | router_token_id_specialization | 3 | 0.32254 | 0.03191 |
| 8 | router_assignment_agreement_to_r100 | 3 | 0.80397 | 0.01363 |
| 8 | router_js_distance_to_r100 | 3 | -0.18633 | 0.00401 |
| 12 | router_entropy | 3 | -0.63856 | 0.04449 |
| 12 | router_margin | 3 | 0.24836 | 0.02021 |
| 12 | router_load_cv | 3 | -0.26657 | 0.01679 |
| 12 | router_token_id_specialization | 3 | 0.30900 | 0.01924 |
| 12 | router_assignment_agreement_to_r100 | 3 | 0.85591 | 0.01341 |
| 12 | router_js_distance_to_r100 | 3 | -0.19310 | 0.00973 |
| 16 | router_entropy | 3 | -0.64860 | 0.02906 |
| 16 | router_margin | 3 | 0.21935 | 0.01210 |
| 16 | router_load_cv | 3 | -0.31493 | 0.07545 |
| 16 | router_token_id_specialization | 3 | 0.33185 | 0.02712 |
| 16 | router_assignment_agreement_to_r100 | 3 | 0.87398 | 0.00914 |
| 16 | router_js_distance_to_r100 | 3 | -0.19277 | 0.00227 |
| 24 | router_entropy | 3 | -0.62054 | 0.07893 |
| 24 | router_margin | 3 | 0.18118 | 0.01556 |
| 24 | router_load_cv | 3 | -0.31958 | 0.04465 |
| 24 | router_token_id_specialization | 3 | 0.34063 | 0.01824 |
| 24 | router_assignment_agreement_to_r100 | 3 | 0.90768 | 0.00365 |
| 24 | router_js_distance_to_r100 | 3 | -0.17732 | 0.01539 |
| 32 | router_entropy | 3 | -0.60474 | 0.04696 |
| 32 | router_margin | 3 | 0.15555 | 0.01178 |
| 32 | router_load_cv | 3 | -0.37054 | 0.06566 |
| 32 | router_token_id_specialization | 3 | 0.30765 | 0.05527 |
| 32 | router_assignment_agreement_to_r100 | 3 | 0.92228 | 0.00636 |
| 32 | router_js_distance_to_r100 | 3 | -0.17139 | 0.01241 |


At mean-winning ages, source entropy is measured in 66/150 architecture/sparsity/seed rows. The remaining winner-linked developmental values stay NA. This prevents a complete developmental characterization of favorable regions. `dataset_a_development_at_mean_winners.csv` retains every selected winning age and available source probe. These duplicated architecture/sparsity links are not independent developmental observations.

The preserved September 21 normalization and pairwise-distance tables allow reproducible state comparisons without redefining the metric. Same-age cross-architecture measured-state distance has median 0.14904 and maximum 0.40517 over 156 available pairs; these pairs are descriptive, not independent replicates. Normalized chronological age is therefore not an identity of measured developmental state. No interpolation or extrapolation is used.

Raw gradient-log applied-update counts matched metadata for 1924 distinct recoveries with both available; 1 lack an available comparison. See the update audit for AMP accounting and the manifest reconciliation for the final observed step of incomplete runs.
