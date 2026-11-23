ARCHITECTURE SCALING AUDIT

24E VALID UNDER FIXED-COMPUTE INTERPRETATION ONLY

Audit only: no training, recoveries or mechanism experiments launched; historical artifacts preserved.

1. **Were total training tokens held constant across expert counts?**

Yes: 40,960,000 input-token occurrences per source trajectory (2500 × 128 × 128), across all seeds. Four layers generate 163,840,000 proposed top-1 assignments. Tokens include repeated data; they are not unique examples.

2. **Were optimizer steps held constant?**

Attempted steps: yes, 2500. Applied source optimizer steps: NA because GradScaler skip history was not persisted. Recovery applied counts are measured separately; fp16 overflows make attempts and successful updates differ.

3. **Were accepted expert assignments per expert held approximately constant?**

No in sampled source logs and complete recovery logs. Exact complete source totals are NA. Active steps must be considered separately from accepted tokens.

4. **What is actual relative per-expert source exposure for 8E/12E/16E/24E?**

Full historical accepted ratios: NA. Best available sampled accepted ratios averaged across seeds: 8E 1.0000, 12E 0.6608, 16E 0.4905, 24E 0.3245. These are explicitly samples, not extrapolated cumulative totals.

5. **Does exposure approximately scale as 1/E?**

Compare measured sampled source ratios with 1, 2/3, 1/2, 1/3 in source_relative_exposure.csv. Recovery common-age ratios are independently measured below. Proposed source sample means follow 1/E exactly; capacity acceptance need not.

6. **What exactly does the historical cumulative-exposure metric measure?**

It sums proposed top-k assignment counts over full logged training batches, then averages over 4E physical experts. It logs step 1 and each multiple of 25. It is not first-row-only, accepted counts, optimizer updates, or checkpoint samples. The first-validation-batch caveat concerns a separate maturity probe.

7. **Why do 16E R100 ≈103424 and 24E R100 ≈68949 scale inversely?**

101 × 16384 = 1,654,784 proposed assignments per layer in the logged batches. Divide by 16 → 103424 exactly; divide by 24 → 68949.333333. The four-layer factor cancels in the mean. This identity follows from logging and cannot demonstrate undertraining.

8. **Are those complete accepted exposure or sampled logging exposure?**

Sampled logging exposure, before capacity. See sampled_source_exposure.csv for acceptance-corrected counts on exactly those batches.

9. **Were active expert update opportunities comparable?**

Complete source history: HISTORICAL ACTIVE-STEP EXPOSURE UNAVAILABLE. Recovery per-expert active-attempt and active-applied-update counts are available in recovery_expert_exposure.csv and summarized below. Lower tokens per expert does not automatically imply proportionally fewer active updates.

10. **Were source models comparably converged by R100?**

No architecture is shown to have converged. All improve materially near R100. Their late slopes are comparable; R100 is a normalized budget label, not evidence of maturity.

11. **Was 24E still materially improving near R100?**

Yes: mean R80→R100 improvement 0.107613 (6.79%). This is comparable to 8E/12E/16E, not a 24E-specific lag.

12. **Did 24E experts move less from initialization?**

No; 24E mean displacement is larger. Mean per-expert absolute displacement: 8E 11.492731, 24E 12.231183; relative displacement: 0.555479 versus 0.591227. Full age/seed/component distributions are saved. Displacement is descriptive and does not alone diagnose insufficient training.

13. **Does capacity behavior change materially with expert count?**

Measured sampled source drop rates and full recovery acceptance ratios are tabulated. The formula scales capacity inversely with E with ceil rounding. No 24E-specific capacity bug was found; observed drop differences modify the simple 1/E relationship.

14. **Does router auxiliary-loss scaling change with expert count?**

Balanced baseline remains 1 per layer before weight 0.1. No accidental E normalization error found. Router dimensionality, full-softmax gate amplitudes and distribution-dependent gradients still change with E; see router_scaling_audit.md.

15. **Are there architecture-dependent implementation bugs?**

No 24E-specific bug found in inspected paths and artifacts. Same-seed initialization is not cross-architecture identical because RNG consumption changes with E. This is expected, not an accidental seed mismatch.

16. **Are checkpoint ages correctly mapped?**

Yes for all 27 used 24E seed-age combinations: payload step=25×age, correct seed/E, fresh file and component hashes match historical manifest. Additional saved 5%-grid checkpoints were also checked. No duplicated component hashes within a seed trajectory.

17. **Are masks correct?**

Yes. All 27 runs match independently regenerated source-R100 expert-local magnitude masks. Actual weight sparsity is approximately 85%, with rounded per-expert keep counts; biases are excluded. One mask per seed is intentionally reused across router ages. R0 rewind hashes and final masked zeros pass.

18. **Are matched dense comparators correct?**

Yes: all 27 24E sparse runs point to their same-seed/same-age dense recovery endpoint, with matched source router, data hashes, protocol and fresh-optimizer metadata. No universal R100 comparator was substituted.

19. **Were 24E ticket gaps recomputed correctly?**

Yes: 27 raw metrics.jsonl final losses independently cross-check summaries; sparse minus dense reproduces published means/sample SD to numerical precision, without using the existing summary script.

20. **Is the flat 24E curve real according to raw artifacts?**

Yes as a descriptive fixed-budget observation. Winners remain seed 7 R20, seed 17 R40, seed 29 R100; mean winner R50. Near ties and three seeds limit precision; “flat” is not a formal equivalence test.

21. **Is source training exposure a confound?**

For equal-development claims, E and per-expert sampled exposure are confounded, and maturation equality is unestablished. For a fixed-global-token question, reduced expert exposure is part of the architectural change. No material 24E-specific developmental failure is demonstrated.

22. **Is recovery exposure a separate confound?**

Yes for exposure-matched recoverability claims: full recovery counters directly show different accepted examples per expert. Fixed-step recoverability remains a valid different question. This does not by itself establish that 2500 attempts are inadequate.

23. **Are sparse and dense recovery exposures comparable within matched pairs?**

They share input data/attempt schedules but native routing does not ensure equal acceptance. Across 24E pairs the sparse/dense total accepted ratio ranges 0.998318–1.016490, mean 1.007979. Per-expert allocations may differ more; all distributions are retained. Interpret the ticket gap as total native recovery performance, not an exposure-controlled contrast.

24. **Is 24E scientifically valid under fixed-global-compute interpretation?**

Yes under the precise fixed-global-token/attempt-budget interpretation supported by the manuscript and localization protocol. FLOPs, wall time and exact successful source update counts are not proven matched.

25. **Is it valid for a developmental cross-architecture comparison?**

Valid for descriptive router-development statistics under the specified budget; insufficient to claim equally developed experts or a budget-invariant developmental mechanism. Neither source nor recovery was accepted-exposure matched.

26. **Does 24E need to be rerun?**

24E VALID UNDER FIXED-COMPUTE INTERPRETATION ONLY. No mandatory rerun is supported for the current fixed-budget claims. Stronger developmental claims require a separate design, not a 24E-only repair.

27. **If yes, exactly what must be rerun?**

No rerun mandated. If the estimand changes, rerun scope is specified in rerun_recommendation.md: source changes propagate to source/masks/both recoveries; recovery-only changes require both paired recoveries, across all affected architectures.

28. **Does the issue invalidate 8E/12E/16E comparisons?**

No fixed-budget results are invalidated. The same unequal-exposure/nonconvergence limits apply to developmental comparisons across all four architectures. Original/expanded 12E model states match at all reused ages.

29. **Should 32E proceed under the current protocol?**

32E CLEARED UNDER CURRENT PROTOCOL. Only for the current explicitly fixed-global-budget descriptive question. No training was started.

30. **What protocol should be used going forward?**

Protocol A for continuity with existing claims; separately preregister B (accepted exposure matched, calibrated separately for source/recovery) or C (convergence matched) for a stronger developmental question. Do not select a protocol or stopping rule to obtain a preferred ticket curve.

## Primary source-exposure table

NA is intentional: complete accepted source history is unavailable.

| Experts | Total tokens | Total accepted assignments | Mean accepted/expert | Relative exposure vs 8E | Active steps/expert | Expert params | Dense R100 loss |
|---|---|---|---|---|---|---|---|
| 8 | 40960000 | NA | NA | NA | NA | 16818176 | 1.48767 |
| 12 | 40960000 | NA | NA | NA | NA | 25227264 | 1.47118 |
| 16 | 40960000 | NA | NA | NA | NA | 33636352 | 1.4624 |
| 24 | 40960000 | NA | NA | NA | NA | 50454528 | 1.47704 |

## Theoretical baseline (one physical expert in one layer)

| experts | total_training_tokens | top_k | potential_expert_assignments | ideal_assignments_per_expert | relative_to_8E |
|---|---|---|---|---|---|
| 8 | 40960000 | 1 | 163840000 | 5.12e+06 | 1 |
| 12 | 40960000 | 1 | 163840000 | 3.41333e+06 | 0.666667 |
| 16 | 40960000 | 1 | 163840000 | 2.56e+06 | 0.5 |
| 24 | 40960000 | 1 | 163840000 | 1.70667e+06 | 0.333333 |

## Measured source samples and capacity

| experts | EXPECTED_FROM_SIMPLE_BALANCE | ACTUAL_OR_BEST_AVAILABLE | DEVIATION |
|---|---|---|---|
| 8 | 1 | 1 | 0 |
| 12 | 0.666667 | 0.660819 | -0.00584744 |
| 16 | 0.5 | 0.490456 | -0.00954414 |
| 24 | 0.333333 | 0.324479 | -0.00885401 |

| experts | capacity_per_expert | capacity_fraction_batch_tokens | sampled_source_drop_rate |
|---|---|---|---|
| 8 | 2560 | 0.15625 | 0.043081 |
| 12 | 1707 | 0.104187 | 0.0514743 |
| 16 | 1280 | 0.078125 | 0.0613469 |
| 24 | 854 | 0.052124 | 0.0684987 |

## Full recovery exposure, common R0/R20/R100 grid

Equal seed/age weighting. Accepted assignments include overflow forwards; active applied steps exclude skipped updates.

| experts | kind | accepted_per_expert | accepted_assignments | active_applied_steps | loss | relative_to_8E | exposure_matched_budget_multiplier |
|---|---|---|---|---|---|---|---|
| 8 | dense | 4.93177e+06 | 1.57817e+08 | 2457 | 1.46421 | 1 | 1 |
| 8 | sparse | 4.95072e+06 | 1.58423e+08 | 2466.76 | 1.6747 | 1 | 1 |
| 12 | dense | 3.27364e+06 | 1.57135e+08 | 2452.61 | 1.41927 | 0.663786 | 1.50651 |
| 12 | sparse | 3.27902e+06 | 1.57393e+08 | 2463.78 | 1.63366 | 0.662332 | 1.50982 |
| 16 | dense | 2.42771e+06 | 1.55373e+08 | 2444.31 | 1.4254 | 0.492259 | 2.03145 |
| 16 | sparse | 2.44207e+06 | 1.56293e+08 | 2459.3 | 1.62577 | 0.493277 | 2.02726 |
| 24 | dense | 1.60503e+06 | 1.54083e+08 | 2440.42 | 1.44089 | 0.325447 | 3.07269 |
| 24 | sparse | 1.6151e+06 | 1.55049e+08 | 2453.81 | 1.60512 | 0.326235 | 3.06527 |

## Source late development

R80→R100 source validation improvement by architecture/seed:

| experts | seed | start_loss | end_loss | absolute_improvement | relative_improvement |
|---|---|---|---|---|---|
| 8 | 7 | 1.61178 | 1.48812 | 0.123663 | 0.0767247 |
| 8 | 17 | 1.62934 | 1.52503 | 0.10431 | 0.0640197 |
| 8 | 29 | 1.54835 | 1.44988 | 0.098467 | 0.0635949 |
| 12 | 7 | 1.60618 | 1.51255 | 0.0936374 | 0.058298 |
| 12 | 17 | 1.54988 | 1.42672 | 0.12316 | 0.079464 |
| 12 | 29 | 1.5756 | 1.47428 | 0.101329 | 0.0643113 |
| 16 | 7 | 1.60106 | 1.47922 | 0.12184 | 0.0760993 |
| 16 | 17 | 1.58967 | 1.47936 | 0.110307 | 0.0693897 |
| 16 | 29 | 1.52741 | 1.42863 | 0.0987865 | 0.0646756 |
| 24 | 7 | 1.62035 | 1.50488 | 0.115474 | 0.0712648 |
| 24 | 17 | 1.59345 | 1.49477 | 0.0986759 | 0.0619261 |
| 24 | 29 | 1.54015 | 1.43146 | 0.108689 | 0.0705705 |

Expert R100 displacement (equal-seed mean of physical-expert means):

| experts | absolute_mean | relative_mean |
|---|---|---|
| 8 | 11.4927 | 0.555479 |
| 12 | 11.8972 | 0.575026 |
| 16 | 12.1452 | 0.58703 |
| 24 | 12.2312 | 0.591227 |

## Independent 24E endpoints

| router_age | mean_ticket_gap | sd_ticket_gap |
|---|---|---|
| 0 | 0.159538 | 0.0281531 |
| 10 | 0.168149 | 0.0197934 |
| 20 | 0.157216 | 0.0243863 |
| 30 | 0.167997 | 0.031674 |
| 40 | 0.160072 | 0.048763 |
| 50 | 0.157097 | 0.0324828 |
| 60 | 0.17267 | 0.0121718 |
| 80 | 0.184755 | 0.00886171 |
| 100 | 0.175942 | 0.00491684 |

| seed | winner | R50_minus_R0 | R50_minus_R100 |
|---|---|---|---|
| 7 | 20 | 0.0208302 | -0.00379337 |
| 17 | 40 | -0.0195994 | -0.0601032 |
| 29 | 100 | -0.00855415 | 0.00736028 |

Mean contrasts: R50−R0 = -0.002441130419061105; R50−R100 = -0.018845426200653215.

Sparse/dense covariance is descriptive, with ages not independent replicates:

| seed | sparse_dense_correlation | dense_range | sparse_range | gap_range |
|---|---|---|---|---|
| 7 | 0.884623 | 0.115156 | 0.0814063 | 0.0627159 |
| 17 | 0.733349 | 0.1068 | 0.0812569 | 0.0895805 |
| 29 | 0.958547 | 0.0447856 | 0.0569238 | 0.0159144 |

## Competing explanations

| Hypothesis | Explanation | Assessment | Evidence |
|---|---|---|---|
| H1 | Source expert undertraining | Unproven | Sampled accepted exposure falls with E and expert displacement differs, but 24E late validation improvement is comparable to other architectures. Complete source accepted history unavailable; no unique source developmental failure established. |
| H2 | Recovery undertraining | Budget dependence measured; inadequacy unproven | Complete accepted exposure falls with E, while active applied-step coverage remains high. All endpoints are fixed-budget recoverability measurements; no longer-budget counterfactual is available. |
| H3 | Router scaling issue | Architecture dependence present; no normalization/capacity bug found | E-normalized load loss has balanced baseline 1. Full-softmax gate amplitudes and expert competition change with E. Capacity drops measured separately. |
| H4 | Comparator effect | Descriptive possibility quantified | Matched sparse and dense final losses, within-seed covariance and ranges are reported in comparator_covariation.csv. Ticket gap is a difference, with no denominator. Covariation cannot by itself establish why the curve is flat. |
| H5 | Genuine seed-conditioned phenomenon | Supported as a fixed-budget observation | Raw endpoints reproduce all 27 gaps and published seed winners; no protocol error explains the observed flat mean curve. Mechanistic or budget-invariant interpretation is not established. |
| H6 | Implementation/configuration error | No 24E-specific error found | Checkpoint, masks, rewind, dense comparator and final hashes pass. Expert count and save/refinement grids intentionally differ. Current code and historical metadata do not establish bitwise hardware/runtime equivalence. |
| H7 | Logging/analysis error | Ticket calculation passes; exposure interpretation corrected | Published gaps and sample SD agree with raw final losses. Historical cumulative exposure is sampled proposed counts, not complete accepted training exposure. |

Best-supported interpretation: (1) a real seed-conditioned fixed-budget observation, (2) native sparse/dense recovery and comparator behavior, (3) architecture-dependent accepted exposure and gate weighting as plausible contributors. Historical data do not causally rank those mechanisms. H1/H2 are not established diagnoses.

## Evidence and reproducibility

See code_path_manifest.md for file/function provenance; architecture_config_diff.csv and architecture_parameter_counts.csv for full configuration/counts; historical_data_availability.md for coverage; router_scaling_audit.md for formulas; rerun_recommendation.md for the three candidate protocols and full blast radius. audit.py performs CPU checkpoint/log arithmetic; finalize_audit.py independently checks raw final-loss logs and historical manifests; write_report.py assembles this report. All CSV missing quantities use NA. No paper/result files outside this audit directory were edited.

32E CLEARED UNDER CURRENT PROTOCOL
