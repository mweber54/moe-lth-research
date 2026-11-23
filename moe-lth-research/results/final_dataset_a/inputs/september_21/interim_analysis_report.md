# Interim Analysis — Fixed-Budget Router-Conditioned Sparse Recoverability

Snapshot: **20260921T133833582668Z UTC**; selected manifest export: **2026-09-21T12:49:12Z**. Exploratory analysis of the pre-specified Dataset A grid, using existing artifacts only. This is a snapshot, not a frozen final dataset. No significance tests, causal conclusions, scaling-law fits, or experiment changes were made.

## 1. Dataset status

| architecture | source_seeds_complete | valid_sparse_cells | expected_sparse_cells | valid_dense_cells | expected_dense_cells | completion_fraction | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8E | 5 | 330 | 330 | 55 | 55 | 1.0000 | COMPLETE |
| 12E | 5 | 330 | 330 | 55 | 55 | 1.0000 | COMPLETE |
| 16E | 5 | 330 | 330 | 55 | 55 | 1.0000 | COMPLETE |
| 24E | 5 | 239 | 330 | 40 | 55 | 0.7247 | PARTIAL |
| 32E | 5 | 21 | 330 | 21 | 55 | 0.1091 | PARTIAL |

All architectures have five complete source trajectories (seeds 7, 17, 29, 43, 61), and 11 source checkpoints per seed. Completion fraction above counts recovery cells: (valid sparse + valid dense)/385. Source completion does not imply recovery completion.

8E/12E/16E each contain all five seeds × 11 ages × six sparsities. 24E has complete recovery grids for seeds 7/17/29, and seed 43 through R50 plus R60 at 70–90%; its R60 95% cell was RUNNING, with the pre-existing runner process detected. Seed 43 still lacks sparse R60/95%, sparse R70–R100 at all sparsities and dense R70–R100; seed 61 lacks all recoveries. 24E is missing 91 sparse and 15 dense cells.

32E has seeds 7/17/29 at R0/R10/R20/R40/R60/R80/R100, only at 85% sparsity, with 21 matched dense recoveries. It lacks 309 sparse and 34 dense cells. Its new recoveries were PENDING, not RUNNING, at snapshot. Thus **32E is PARTIAL, not NOT_STARTED**. No seed 43/61 recoveries or R30/R50/R70/R90 recoveries exist in that snapshot.

[Exact missing cells](interim_missing_conditions.csv), [seed completion](interim_seed_status.csv), and [status table](interim_dataset_status.csv) provide the full inventory. The September 11 canonical audit is a historical reuse audit, not a current completion count. The active runner may advance after this snapshot; this task neither paused nor changed it.

## 2. Protocol validation

**1250 sparse and 226 dense recovery conditions passed the interim checks; 0 failed checks.** Source YAMLs, raw recovery metadata/curves/summaries, canonical data hashes, source checkpoint identities, mask identities, and historical validation attestations agree. Exact selected artifact paths are preserved in the condition table. Incompatible historical candidates remain excluded according to the frozen `historical_recovery_audit.csv` (copied under inputs); no legacy result was admitted merely because architecture/seed/age labels matched.

The intended protocol is equal GLOBAL attempt/token budgets:

- Source: 2,500 attempts × 128 examples × 128 input tokens = **40,960,000 global input-token occurrences**; four MoE layers, top-1 routing imply **163,840,000 proposed layer/expert assignments**. Gradient accumulation is one.
- Recovery: 2,500 attempts, batch 128, sequence length 128, fresh AdamW (zero optimizer state at start), learning rate 0.0003, weight decay 0.01, gradient norm clip 1, no scheduler/warmup, fp16 autocast with GradScaler. Successful optimizer updates can differ because of AMP overflows.
- Shared architecture: vocabulary 256, d_model 256, four layers, four heads, expert hidden size 1,024, top_k 1, capacity_factor 1.25, dropout 0; expert count is the intended architectural difference. Learned native trainable routing; auxiliary router loss weight 0.1.
- Data: canonical per-seed training sequence hashes and shared validation sequence hashes checked for every recovery. Byte tokenizer; fixed validation blocks 32; source evaluation every 250 steps, recovery validation every 50 steps. The reported validation loss is token-weighted cross entropy, excluding the auxiliary term. Save-only checkpoint differences are not changes in evaluation cadence.
- Masks: two expert weight matrices jointly ranked **within each expert** by source-R100 magnitude; rounded retained count, biases excluded; one mask per architecture/seed/sparsity reused across ages. Surviving experts and shared weights rewind to R0; only the router initializes from the requested age. Masks remain enforced through gradient hooks and post-update masking. Existing validation attests final masked zeros.
- Early AUC: historical trapezoidal integral over steps 0–1,250 at 50-step spacing, **not divided by 1,250**. `sparse_preprune_validation_loss` is the equivalent unmasked rewound model's matched dense initial loss, not a new R100 prepruning evaluation. Negative pruning shock is possible and observed.

Validation scope: we independently reopen every selected raw curve and summary and check source YAML and hash linkage. We use existing checkpoint/config/masked-zero validation attestations; we do not reread/hash every multi-megabyte recovery checkpoint tensor. The historical audit and runner validation code were inspected; this scope is recorded in `analysis_context.json`. [Detailed audit](interim_protocol_audit.csv) and [input hashes](input_hashes.json) make that distinction explicit.

Ticket gap is independently recomputed as sparse final CE minus **same architecture, seed, router age and protocol** dense final CE. All 1250 raw recomputations match the difference of independently stored manifest endpoints within 1e-10. A genuinely stored historical ticket gap was located with matching dense/sparse endpoints for **207** rows; all of these pass. Remaining stored-gap values and literal stored-gap `pass` fields are NA, not fabricated. 0 historical candidate rows had nonmatching selected endpoints and are separately documented in [historical candidates](interim_historical_gap_candidates.csv); they are not treated as arithmetic failures of the current comparator. See [ticket-gap validation](interim_ticket_gap_validation.csv).

## 3. Main router-age results

Among the **18 completed architecture × sparsity curves**, mean minima are intermediate in **13**, R0 in **3**, and R100 in **2**. Intermediate mean winners beat both endpoint means in those 13 cells, but this is not evidence of a unique population optimum.

| architecture | 70% | 75% | 80% | 85% | 90% | 95% |
| --- | --- | --- | --- | --- | --- | --- |
| 8E | R70 | R80 | R90 | R80 | R80 | R0 |
| 12E | R50 | R50 | R100 | R70 | R100 | R0 |
| 16E | R60 | R80 | R60 | R60 | R60 | R0 |
| 24E | R30 | R30 | R50 | R40 | R0 | R0 |
| 32E | NA | NA | NA | R0 | NA | NA |

24E and 32E rows are **provisional**. Available-seed 24E means have three versus four seeds at different ages; its 85% winner is R40 with all available rows but R50 using only full-grid seeds 7/17/29. Both are shown, without deleting partial observations, in [balanced-seed sensitivity](interim_balanced_seed_sensitivity.csv). 32E's R0 minimum is only over its seven observed ages and three observed seeds at 85%.

Most minima are broad or weak relative to seed variation. For example, 8E/85% has a connected descriptive basin R10–R100; 12E/85% R20–R100; 16E/80% R40–R100. The historical descriptive basin rule starts at the mean winner and includes adjacent observed 10-point ages while their paired mean difference is no larger in magnitude than one paired sample SD. This is a descriptive yardstick, **not a confidence interval or a statistical test**. Missing grid ages break connectivity; no ages are interpolated. Tables retain exact differences to every age, the second-lowest age and gap, neighborhood spread and all seed values; no scientific near-tie threshold is used to alter the numerical winner.

| architecture | sparsity | winning_ages | second_lowest_age | gap_to_second | neighbor_mean_spread | mean_winner_seed_sd | descriptive_basin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8E | 0.7000 | R70 | 90 | 0.0027 | 0.0242 | 0.0104 | R70;R80;R90;R100 |
| 8E | 0.7500 | R80 | 20 | 0.0010 | 0.0037 | 0.0134 | R70;R80;R90;R100 |
| 8E | 0.8000 | R90 | 80 | 0.0052 | 0.0062 | 0.0259 | R70;R80;R90;R100 |
| 8E | 0.8500 | R80 | 70 | 0.0025 | 0.0098 | 0.0209 | R10;R20;R30;R40;R50;R60;R70;R80;R90;R100 |
| 8E | 0.9000 | R80 | 100 | 0.0025 | 0.0215 | 0.0126 | R60;R70;R80;R90;R100 |
| 8E | 0.9500 | R0 | 20 | 0.0227 | 0.0375 | 0.0223 | R0;R10;R20;R30 |
| 12E | 0.7000 | R50 | 100 | 0.0045 | 0.0116 | 0.0253 | R30;R40;R50;R60;R70;R80;R90;R100 |
| 12E | 0.7500 | R50 | 60 | 0.0025 | 0.0025 | 0.0290 | R40;R50;R60;R70;R80;R90;R100 |
| 12E | 0.8000 | R100 | 90 | 0.0010 | 0.0010 | 0.0242 | R90;R100 |
| 12E | 0.8500 | R70 | 90 | 0.0005 | 0.0140 | 0.0332 | R20;R30;R40;R50;R60;R70;R80;R90;R100 |
| 12E | 0.9000 | R100 | 90 | 0.0024 | 0.0024 | 0.0107 | R50;R60;R70;R80;R90;R100 |
| 12E | 0.9500 | R0 | 40 | 0.0454 | 0.0598 | 0.0506 | R0 |
| 16E | 0.7000 | R60 | 80 | 0.0002 | 0.0093 | 0.0363 | R40;R50;R60;R70;R80;R90;R100 |
| 16E | 0.7500 | R80 | 70 | 0.0022 | 0.0060 | 0.0262 | R50;R60;R70;R80;R90;R100 |
| 16E | 0.8000 | R60 | 70 | 0.0016 | 0.0042 | 0.0418 | R40;R50;R60;R70;R80;R90;R100 |
| 16E | 0.8500 | R60 | 70 | 0.0053 | 0.0189 | 0.0410 | R30;R40;R50;R60;R70;R80;R90;R100 |
| 16E | 0.9000 | R60 | 0 | 0.0009 | 0.0125 | 0.0542 | R30;R40;R50;R60;R70;R80;R90;R100 |
| 16E | 0.9500 | R0 | 60 | 0.0307 | 0.0414 | 0.0327 | R0 |

[Seed-trace figures and heatmaps](figure_index.md) display mean ± sample SD and individual values. All 11 grid positions remain visible; missing values are unfilled.

## 4. Sparsity dependence

The winner sequences across 70, 75, 80, 85, 90, 95% sparsity are 8E: R70/R80/R90/R80/R80/R0; 12E: R50/R50/R100/R70/R100/R0; 16E: R60/R80/R60/R60/R60/R0. There is no monotone movement of the winning age with sparsity.

At 90%, improvement over R0 is weak: best intermediate minus R0 advantages are only 0.00855 (8E), 0.00574 (12E; R100 wins overall), and 0.00091 (16E). At 95%, **R0 wins in all completed architectures**, and the best intermediate is worse than R0 by 0.02272, 0.04543, and 0.03074 CE respectively. The minimum mean gaps increase from 90% to 95% by about 0.1060, 0.0912, and 0.1027 CE. Absolute minimum gaps also rise with sparsity in the completed grids, but the router-age pattern is heterogeneous and nonmonotone.

| architecture | sparsity | minimum_mean_ticket_gap | r0_ticket_gap | r100_ticket_gap | intermediate_advantage_vs_r0 | intermediate_advantage_vs_r100 |
| --- | --- | --- | --- | --- | --- | --- |
| 8E | 0.7000 | 0.0371 | 0.0580 | 0.0434 | 0.0209 | 0.0063 |
| 8E | 0.7500 | 0.0663 | 0.1141 | 0.0748 | 0.0479 | 0.0086 |
| 8E | 0.8000 | 0.1201 | 0.1667 | 0.1262 | 0.0466 | 0.0062 |
| 8E | 0.8500 | 0.1990 | 0.2558 | 0.2077 | 0.0568 | 0.0087 |
| 8E | 0.9000 | 0.3136 | 0.3222 | 0.3161 | 0.0086 | 0.0025 |
| 8E | 0.9500 | 0.4196 | 0.4196 | 0.4739 | -0.0227 | 0.0315 |
| 12E | 0.7000 | 0.0437 | 0.0559 | 0.0482 | 0.0122 | 0.0045 |
| 12E | 0.7500 | 0.0735 | 0.1020 | 0.0934 | 0.0284 | 0.0198 |
| 12E | 0.8000 | 0.1242 | 0.1696 | 0.1242 | 0.0444 | -0.0010 |
| 12E | 0.8500 | 0.2021 | 0.2328 | 0.2062 | 0.0306 | 0.0040 |
| 12E | 0.9000 | 0.2988 | 0.3070 | 0.2988 | 0.0057 | -0.0024 |
| 12E | 0.9500 | 0.3900 | 0.3900 | 0.4491 | -0.0454 | 0.0136 |
| 16E | 0.7000 | 0.0261 | 0.0680 | 0.0357 | 0.0419 | 0.0096 |
| 16E | 0.7500 | 0.0478 | 0.1157 | 0.0647 | 0.0679 | 0.0169 |
| 16E | 0.8000 | 0.0952 | 0.1546 | 0.1149 | 0.0595 | 0.0198 |
| 16E | 0.8500 | 0.1660 | 0.2063 | 0.1899 | 0.0403 | 0.0238 |
| 16E | 0.9000 | 0.2632 | 0.2641 | 0.2830 | 0.0009 | 0.0199 |
| 16E | 0.9500 | 0.3659 | 0.3659 | 0.4241 | -0.0307 | 0.0275 |

Positive intermediate advantage means boundary mean gap minus best intermediate mean gap. The best intermediate need not be the overall winner.

## 5. Seed reproducibility

No completed architecture/sparsity has all five seeds selecting its exact mean-winning age. Exact agreement ranges from 0/5 to 3/5. 8E/70%, 8E/90%, 12E/70%, and 12E/90% have **no seed selecting the mean winner**, illustrating how heterogeneous curves can produce a mean minimum that no individual seed selects.

All five seeds have intermediate minima for 16E at 70–80% and 8E at 85%; they do not agree on one age. At 95%, seeds 7/43/61 favor R0 in 12E/16E, while 8E seeds 17/43/61 favor R0; two seeds in each architecture still favor intermediate ages. The recurring R0 mean preference at extreme sparsity is more reproducible across completed architectures than a specific intermediate winning age. Every available seed is retained.

| architecture | sparsity | 7 | 17 | 29 | 43 | 61 |
| --- | --- | --- | --- | --- | --- | --- |
| 8E | 0.7000 | R20 | R0 | R20 | R0 | R90 |
| 8E | 0.7500 | R20 | R0 | R20 | R80 | R90 |
| 8E | 0.8000 | R20 | R100 | R20 | R90 | R20 |
| 8E | 0.8500 | R20 | R60 | R30 | R80 | R40 |
| 8E | 0.9000 | R20 | R100 | R20 | R90 | R100 |
| 8E | 0.9500 | R20 | R0 | R30 | R0 | R0 |
| 12E | 0.7000 | R0 | R10 | R60 | R70 | R0 |
| 12E | 0.7500 | R0 | R50 | R60 | R80 | R90 |
| 12E | 0.8000 | R90 | R10 | R100 | R70 | R90 |
| 12E | 0.8500 | R0 | R70 | R80 | R10 | R100 |
| 12E | 0.9000 | R0 | R70 | R60 | R10 | R0 |
| 12E | 0.9500 | R0 | R50 | R60 | R0 | R0 |
| 16E | 0.7000 | R50 | R90 | R60 | R80 | R80 |
| 16E | 0.7500 | R50 | R90 | R70 | R80 | R80 |
| 16E | 0.8000 | R90 | R60 | R60 | R80 | R70 |
| 16E | 0.8500 | R100 | R60 | R60 | R80 | R70 |
| 16E | 0.9000 | R0 | R60 | R60 | R80 | R0 |
| 16E | 0.9500 | R0 | R60 | R30 | R0 | R0 |
| 24E | 0.7000 | R0 | R30 | R100 | R0 | NA |
| 24E | 0.7500 | R20 | R30 | R100 | R50 | NA |
| 24E | 0.8000 | R20 | R50 | R60 | R40 | NA |
| 24E | 0.8500 | R20 | R40 | R70 | R40 | NA |
| 24E | 0.9000 | R0 | R0 | R100 | R0 | NA |
| 24E | 0.9500 | R0 | R30 | R100 | R0 | NA |
| 32E | 0.8500 | R0 | R0 | R0 | NA | NA |

24E seed 43 and all 32E winners are minima over incomplete age grids, flagged in [seed winners](interim_seed_winners.csv). [Winner map](interim_architecture_sparsity_winner_map.csv) gives counts at boundaries/intermediate ages, ranges, seed spread, and minimum/runner-up distances. Numerical ties within 1e-12 retain all ages; unique-age fields are NA if tied.

## 6. Architecture dependence

The completed architectures show different mean-winning ages and advantages. At 85%, the advantage of the best intermediate over R0 is 0.05680 → 0.03065 → 0.04028 from 8E → 12E → 16E; over R100 it is 0.00869 → 0.00402 → 0.02382. This is not a monotonic architecture trend. At 75%, the R0 advantages are 0.04785 → 0.02844 → 0.06790. Minimum ticket gaps at 85% are 0.19903, 0.20214, and 0.16604.

**24E provisional:** the available-seed 85% advantage is 0.00937 versus R0 and 0.01468 versus R100, with a 0.16126 minimum; full-grid seeds instead give 0.00244, 0.01885, and 0.15710. R0 wins provisionally at 90% and 95%. Its 85% curve is especially weak/flat relative to seed variability (mean range/median seed SD ≈0.91). These are not five-seed architecture conclusions.

**32E provisional:** at observed 85% cells, all three seeds favor R0; its mean gap is 0.13485 versus 0.17540 at R100, and the best observed intermediate is worse than R0 by 0.03143. Missing ages and two missing seeds preclude a complete-architecture claim. This partial counterpattern is retained openly. No scaling law or universal intermediate optimum is inferred.

## 7. Metric disagreement

For completed curves, sparse final loss and ticket gap choose the same age in **5/18** cells; pruning shock agrees in **4/18**, and matched early AUC gap in **5/18**. Therefore neither a lower absolute sparse loss nor a smaller early integral can be substituted for better matched recoverability.

| architecture | sparsity | ticket_gap_best_age | sparse_loss_best_age | pruning_shock_best_age | early_auc_gap_best_age |
| --- | --- | --- | --- | --- | --- |
| 8E | 0.7000 | R70 | R100 | R100 | R100 |
| 8E | 0.7500 | R80 | R90 | R100 | R100 |
| 8E | 0.8000 | R90 | R90 | R100 | R100 |
| 8E | 0.8500 | R80 | R80 | R100 | R0 |
| 8E | 0.9000 | R80 | R100 | R0 | R0 |
| 8E | 0.9500 | R0 | R70 | R0 | R0 |
| 12E | 0.7000 | R50 | R100 | R100 | R100 |
| 12E | 0.7500 | R50 | R60 | R100 | R100 |
| 12E | 0.8000 | R100 | R100 | R100 | R100 |
| 12E | 0.8500 | R70 | R70 | R100 | R0 |
| 12E | 0.9000 | R100 | R100 | R20 | R0 |
| 12E | 0.9500 | R0 | R100 | R0 | R0 |
| 16E | 0.7000 | R60 | R100 | R100 | R100 |
| 16E | 0.7500 | R80 | R90 | R100 | R100 |
| 16E | 0.8000 | R60 | R90 | R100 | R60 |
| 16E | 0.8500 | R60 | R100 | R100 | R0 |
| 16E | 0.9000 | R60 | R90 | R0 | R0 |
| 16E | 0.9500 | R0 | R90 | R0 | R0 |
| 24E | 0.7000 | R30 | R100 | R100 | R100 |
| 24E | 0.7500 | R30 | R90 | R100 | R60 |
| 24E | 0.8000 | R50 | R70 | R100 | R60 |
| 24E | 0.8500 | R40 | R60 | R100 | R0 |
| 24E | 0.9000 | R0 | R100 | R20 | R0 |
| 24E | 0.9500 | R0 | R60 | R0 | R0 |
| 32E | 0.8500 | R0 | R60 | R100 | R0 |

Concrete early/final contrasts:

- 8E/85%: R0 has the better early gap (148.524 versus R80's 194.389 CE·attempts), yet R80 finishes closer to its dense comparator (0.19903 versus 0.25583). R100 has slightly more favorable shock (−2.756 versus −2.630 at R80) but a worse final gap (0.20773).
- 16E/85%: shocks at R60 and R100 are similar (−1.821 versus −2.009); R60 has lower early gap (194.972 versus 210.311) and final gap (0.16604 versus 0.18986). Its final advantage does not arise from having the lowest initial shock.
- 16E/80%: R60 wins both early gap and final gap (−25.924, 0.09517), while R100 minimizes shock (−6.805 versus −6.233 at R60). Thus early/final agreement exists in some cells.
- 8E/80%: R20 and R100 have early gaps −52.536 and −71.280 but close final gaps 0.13202 and 0.12624. Larger early separation need not produce an equally large endpoint separation.
- At 95%, shock, early gap, and final gap all favor R0 in completed architectures, while absolute sparse loss favors later routers.

Initial rewound validation losses are large, and the first trapezoid can strongly influence AUC. [AUC decomposition](interim_early_auc_decomposition.csv) separates steps 0–50 from 50–1,250 and records the initial-loss contribution (25 × pruning shock). AUC measures integrated loss, not a direct convergence-rate estimate. These are descriptive associations; no predictive model or significance test was fitted.

## 8. Router development

Historical definitions from `analyze_r20_8_vs_r60_12.py`, localization analyses, and Dataset A finalization:

| Statistic | Exact interpretation |
| --- | --- |
| Entropy | Mean −Σ p log(p), natural logs, over tokens/layers in the first eight deterministic validation batches of the **full source model**. |
| Margin | Mean largest minus second-largest routing probability on those same batches. |
| Load CV | Population SD/mean of proposed top-1 expert counts within each layer, averaged over four layers; not accepted-load CV. |
| Token-ID specialization | Empirical mutual information I(expert; byte token ID), natural-log units, averaged over layers; this is not semantic specialization. |
| Assignment agreement to R100 | Fraction of identical top-1 expert IDs on the **first validation batch per layer**, averaged across layers. |
| JS distance to R100 | Mean Jensen–Shannon **divergence**, not square-root JS metric, of probability vectors on the first batch per layer, natural logs. Historical field name retains “distance.” |
| Cumulative sampled exposure | Sum of logged proposed assignment counts through source step 25×age, averaged over 4×E layer/expert instances; sampled at step 1 and every 25 steps. No multiplication to estimate unlogged steps. |

The first four full-probe metrics use eight batches (131,072 input occurrences per layer); agreement and JS preserve the historical first-batch definition. Full-source developmental statistics are distinct from recovery-initial diagnostics in a rewound sparse/dense model; those cannot fill missing source probes.

Verified full-probe coverage:

- 8E: seeds 7, 17, 29 at R0, R20, R100
- 12E: seeds 7, 17, 29 at R0, R10, R20, R40, R60, R80, R100
- 16E: seeds 7, 17, 29 at R0, R20, R40, R50, R60, R70, R80, R90, R100
- 24E: seeds 7, 17, 29 at R0, R10, R20, R30, R40, R50, R60, R80, R100
- 32E: seeds 7, 17, 29 at R0, R10, R20, R40, R60, R80, R100

No new probes were run. Missing probe ages/seeds remain NA; cumulative sampled counters were additionally read from existing source logs for all 275 source states. Raw entropy/specialization were recovered from historically normalized columns by multiplication by log(E); this is unit conversion, not an estimated measurement.

Entropy generally declines after the early phase and margin increases; 12E/24E/32E have an initial entropy increase at R10, so strict monotonicity is not assumed. Specialization **is not monotonically increasing**: its observed mean peaks at R20 in every architecture and then falls. For 16E it rises 0.8734 → 1.5737 from R0 to R20, then falls to 1.2053 at R100. Load CV generally falls but has local reversals (16E R50→R60: 0.2206→0.2214; 24E R20→R30: 0.3227→0.3342).

At 16E R60, the mean-winning age at 70/80/85/90%, historical seeds 7/17/29 have entropy 2.2245, margin 0.1806, load CV 0.2214, specialization 1.2811, R100 assignment agreement **0.6265**, and JS **0.0408**. Thus this recoverability region precedes full R100 assignment similarity; it is not already fully stabilized. At R80, agreement is 0.7776 and JS 0.0113. There is no evidence here that the ticket minimum consistently coincides with the fastest developmental change; the early specialization maximum at R20 differs from most full-grid mean winners. 8E's historical probe grid lacks R70/R80/R90, and 12E lacks R50/R70, so their precise mean-winning developmental states cannot be measured from existing probes.

Same normalized age is not the same measured state: at R20, assignment agreement to R100 falls from 0.4462 (8E) to 0.3836, 0.3378, 0.2750, and 0.2409 (32E). This is descriptive and based on three historical seeds, not all five recovery seeds.

Developmental comparison uses the **existing frozen historical 8E/12E feature ranges**, not a normalization selected from ticket gaps. Entropy and MI are divided by log(E); each of the seven historical features is range-scaled using the original 30 historical rows; distance is equal-weight RMS of scaled differences. Scales are frozen even for 16E/24E/32E; coordinates can exceed [0,1]. [Normalization](interim_development_normalization.csv), [all cross-architecture distances](interim_developmental_distances.csv), and [nearest observed states](interim_development_nearest_observed_states.csv) retain exact values. No missing features or ages are imputed.

| seed | 8E R20 to 12E R20 | 8E R20 to 12E R60 |
| --- | --- | --- |
| 7 | 0.1040 | 0.3110 |
| 17 | 0.1241 | 0.2615 |
| 29 | 0.1592 | 0.2835 |

For every historical seed, 12E R20 is closer to 8E R20 than 12E R60. This metric-space resemblance does not establish equal developmental maturity or explain ticket gaps causally. Nearest observed states depend on the uneven available probe grid.

## 9. Exposure scaling

Global source token/attempt budgets are fixed, but accepted assignments per layer-specific expert decline strongly. Source exposure statistics below count each independent source seed once; recovery means first average available conditions within seed, then summarize seeds. All available full source counters were independently recomputed and matched manifests.

| architecture | accepted_millions_per_expert | mean_accepted_per_expert_n_seeds | acceptance_rate_mean | mean_seed_paired_exposure_ratio_vs_8e |
| --- | --- | --- | --- | --- |
| 8E | 4.8937 | 5 | 0.9558 | 1.0000 |
| 12E | 3.2365 | 5 | 0.9482 | 0.6614 |
| 16E | 2.4160 | 2 | 0.9438 | 0.4937 |
| 24E | 1.5858 | 2 | 0.9292 | 0.3241 |
| 32E | 1.1719 | 5 | 0.9156 | 0.2395 |

Relative to same-seed 8E, average accepted source exposure is about **66.14%, 49.37%, 32.41%, and 23.95%** at 12E/16E/24E/32E: decreases of **33.86%, 50.63%, 67.59%, and 76.05%**. Means at 16E/24E have only seeds 43/61; historical full source counters for seeds 7/17/29 are unavailable. This missingness is not repaired with dense-recovery counters or sampled logs.

Acceptance drops from 95.58% at 8E to 91.56% at 32E (drop rates 4.42% and 8.44%). Different seed coverage limits intermediate-architecture comparisons; [seed-level exposure](interim_exposure_by_seed.csv) makes it visible. Active attempt steps per expert stay near 2,430–2,447 despite much lower accepted assignment counts, so an active-step count is not equivalent to token exposure. Applied active steps exclude AMP-skipped optimizer steps where known.

[Exposure summary](interim_exposure_by_architecture.csv) includes proposed, accepted, dropped, acceptance, mean accepted, active attempted/applied steps, sample SD, median, min, max and metric-specific n. Recovery accepted exposure also falls with expert count, but partial 24E/32E recovery grids have unequal age/sparsity composition and are not matched architecture estimates. Theoretical global proposed assignments are 163,840,000 per trajectory; measured source means are NA where historical counters are absent.

Dataset A is **not exposure matched**. Architecture differences are plausibly entangled with unequal per-expert input assignments, capacity acceptance and update histories under fixed global budgets. This is an alternative explanation/confound, not a claim that exposure causes the observed recoverability differences. Fixed attempts/tokens also do not establish exactly equal measured FLOPs or wall-clock cost across architectures.

## 10. What 32E needs to tell us

Under the unchanged pre-specified grid, the remaining 32E measurements need to resolve whether its currently observed R0 preference at 85% persists across all five seeds and all 11 ages; how its six sparsity curves compare; whether metric disagreements and broad versus localized minima persist; and what complete router/exposure records are actually available. Completion also permits architecture comparisons with consistent seed/age/sparsity coverage. These are pre-specified descriptive questions, not predictions. This analysis does not remove/add ages, change sparsities, rerun/select seeds, alter 32E, or terminate it.

## 11. Questions for Dataset B

After final Dataset A, questions may include whether architecture patterns persist under a separately specified comparison of per-expert exposure, how source versus recovery exposure relate to recoverability, and whether global attempts or accepted assignments best summarize the remaining heterogeneity. No exposure-scaling rule or Dataset B protocol is defined here. **Dataset B was not started.**

## 12. Current limitations

24E and 32E recovery grids are incomplete. 24E available-seed means mix three/four seeds across ages; 32E covers only seven ages and 85% sparsity. Developmental probes cover only historical seeds and subsets of ages; no new probes or mechanisms were run. Six source trajectories lack full accepted-exposure counters. Some literal historical stored ticket gaps are absent or tied to different selected endpoints; NA is explicit. Full tensor checkpoint rehashing/masked-zero validation was not repeated; existing validated provenance was reused and raw metrics independently checked. Historical flatness/basin labels are descriptive relative-to-SD summaries, not statistical establishment of an optimum. Seeds are the independent replicates; grid points, experts, tokens and steps are not additional independent samples. The live runner can advance beyond the snapshot without altering this report.

### Direct answers to the 15 requested questions

1. **Do intermediate ages generally beat both boundaries?** In 13/18 completed mean curves, yes; R0 wins three and R100 two. This is heterogeneous, not universal.
2. **Narrow optimum or broad basin?** Usually broad/weak; many adjacent ages differ by less than one paired seed SD. Exact means are not established optima.
3. **Consistency across five seeds?** No exact mean-winning age is unanimous. All seeds favor some intermediate age in 16E 70–80% and 8E 85%; other cells include boundaries.
4. **Sparsity dependence?** Strong: the advantage over R0 weakens by 90%, and R0 wins all completed 95% curves.
5. **Does the winner move?** Yes, nonmonotonically; full sequences are in section 4.
6. **Advantage from 8E→12E→16E→24E?** At 85%, advantage over R0 is 0.05680→0.03065→0.04028→0.00937 (last provisional; balanced three-seed 24E 0.00244). No monotone completed-architecture trend.
7. **Where does R0 win?** Completed: 8E/12E/16E at 95%. Provisional: 24E at 90/95%, 32E at observed 85%.
8. **Where does R100 win?** Completed: 12E at 80% and 90%; individual seeds also have endpoint minima elsewhere.
9. **Effectively flat relative to variability?** 12E/85% has full mean range/median seed SD ≈0.98; 24E/85% ≈0.91 and 24E/90% ≈0.94 are provisional examples. Broad local basins occur elsewhere. This is a scale comparison, not a test of equality.
10. **Does sparse-loss winner usually match ticket-gap winner?** No: only 5/18 completed cells.
11. **Does pruning shock predict the final pattern?** It is not a reliable descriptive proxy: only 4/18 winners agree; no prediction model was tested.
12. **Does early recovery predict the final pattern?** Agreement is mixed (5/18 winners); some early advantages reverse or converge. Integrated AUC is not pure recovery speed.
13. **Router statistics around minima?** Available 16E R60 probes remain partly dissimilar to R100 (agreement 0.6265); entropy/JS generally fall, margin rises, load CV generally falls with reversals, specialization peaks early then falls. Missing 8E/12E winning-age probes prevent exact matching there.
14. **How far does per-expert source exposure fall?** Same-seed ratios to 8E are 0.6614, 0.4937, 0.3241, 0.2395; 16E/24E ratios use two counter-observed seeds.
15. **Unequal exposure entangled with architecture?** Yes, it is a plausible confound/alternative explanation in this fixed-global-budget design; no causal conclusion follows.

### Outputs and reproduction

[Condition table](interim_condition_table.csv), [summary statistics and individual seed values](interim_summary_by_architecture_age_sparsity.csv), [winner map](interim_architecture_sparsity_winner_map.csv), [figures](figure_index.md), [protocol audit](interim_protocol_audit.csv), and [input identities](input_hashes.json) accompany this report. PDF and 240-dpi PNG figures are provided for each architecture, with partial labels and no missing-condition interpolation. Run `analyze.py` from this analysis directory with the existing analysis Python environment to create a **new timestamp directory**; it never invokes training/finalization/probe routines. The script keeps canonical outputs untouched. `report.py <snapshot-directory>` builds this report from the selected snapshot and linked existing raw artifacts.

**DATASET B WAS NOT STARTED. NO NEW TRAINING WAS STARTED. NO EXISTING EXPERIMENT ARTIFACTS WERE MODIFIED BY THIS TASK.**
