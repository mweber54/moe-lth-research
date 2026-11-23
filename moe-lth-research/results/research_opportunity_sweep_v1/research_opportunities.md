# Research-opportunity sweep

## Executive verdict

**Complete a descriptive contribution and stop expansion.**

The best usable result is already paid for: a completed E32, five-seed, 50-65% expert-local mask-control extension. From saved validation outputs, the learned mask has lower endpoint cross-entropy than its per-tensor-count-matched random mask in **all 40 matched comparisons** (five seeds × four sparsities × R0/R100). That is enough to write a narrow, explicitly exploratory descriptive result; it is not enough to justify fresh training, to claim generalization, or to claim sparse tickets beat dense models. Most learned sparse means remain above their matched dense controls.

The broad router-age and lower-sparsity maps are already complete, but selected optima are weakly reproducible and no strict all-five-seed dense match appears in the full 50-65% grid. Dataset B’s budget factorial is incomplete and cannot be interpreted as a factorial result. The requested gate, geometry, embedding, and high-sparsity recovery negative conclusions remain closed. **Do not launch training, recovery, mask search, or interventions from this sweep.**

## Decision table

| Question | Existing evidence | Main confound | Missing control | Reuse potential | Next-step cost | Verdict |
|---|---|---|---|---|---|---|
| At 32E/50-65%, do learned masks beat random masks with the same per-tensor keep counts? | All five seeds favor learned masks at each of four sparsities and both R0/R100 ages; mean difference ranges from -0.0590 to -0.0927 nats/token at R100 and -0.0608 to -0.0803 at most R0 cells (the .50 R0 value is -0.0682). | Reused seeds, validation-based selection/evaluation, one random draw per seed/sparsity, and native routing/accepted exposure can differ. | Fresh seeds, multiple random draws, and a proven untouched evaluation split for a confirmatory claim. | 100 completed sparse recoveries, 15 matched dense references, masks, full validation curves, hashes, and attempt/update logs. | **Zero new runs.** A standard-library CPU join of saved artifacts is complete. | **Continue only as a bounded descriptive analysis; no new experiment.** Learned beats random, usually not dense. |
| Is the mean router-age optimum reliable across E8-32 and 70-95%? | 30 complete five-seed curves; mean minima are intermediate in 16, R0 in 10, R100 in 4; all five seed-wise winners agree with the mean winner in only 1/30 curves. | The same validation surface selects ages; source exposure/maturity vary across architecture. | Fresh seeds and held-out validation would be needed for transportability; no basis for another grid. | 1650 sparse and 275 dense recoveries plus complete trajectories and seed-winner tables. | Zero for description; a fresh full sweep would be 1650 sparse + 275 dense recoveries. | **Already answered descriptively. Do not expand the grid.** |
| Do source and recovery budgets separately change relative sparse performance? | Dataset B design is staged A-D, but only 63/180 new jobs are complete; 2 interrupted and 115 pending, with 45 reused references. B/D and long sources are incomplete. | Sequential completion; source budget changes source model, router age and selected mask; optimizer state is fresh at boundaries. | Complete all arms and then review matched cells; fresh seeds/held-out testing for confirmation. | Exact prefix exposure and batch hashes, source/mask/recovery artifacts, and existing A/C pairs. | Remaining cost is uncertain; the original full new workload estimate was 66-74 GPU-hours, not a valid estimate of remaining work. | **Do not resume based on partial results. No current factorial conclusion.** |

## Strongest bounded opportunity: report the existing learned/random result

The source is [dataset_b_lower_sparsity/manifests/run_manifest.csv](../dataset_b_lower_sparsity/manifests/run_manifest.csv), with artifact-level outcomes under [dataset_b_lower_sparsity/recoveries/](../dataset_b_lower_sparsity/recoveries/). The registered mask generator in [pilot_common.py](../../scripts/nominal_exposure_dataset/pilot_common.py) samples each tensor independently using the learned mask’s retained count; [worker.py](../../scripts/nominal_exposure_dataset/worker.py) asserts exact per-tensor counts and validates the recovery. The data joins are rechecked by [audit_existing_masks.py](./audit_existing_masks.py); the full per-seed condition table is [verified_observations.csv](./verified_observations.csv).

The manifest reports 100 sparse jobs `COMPLETE` and 15 dense references `REUSED`. Its `execution_status=NOT_LAUNCHED` field is stale/inconsistent with the outcome status. Rather than trust that field or status alone, the audit required all 115 `validated_result.json` integrity flags, final-checkpoint hashes, 51-point 0:50:2500 metric curves, and exact shared training/validation sequence hashes for each learned/random pair and its same-age dense reference. No model weights were loaded and no forward pass was run.

The endpoint is the registered full-validation token-weighted CE. Negative learned-minus-random means lower CE for the learned mask. Each stratum has five paired source seeds; the same seeds recur across sparsities and ages, so those repeated strata are not eight independent replication groups.

| Target sparsity | Router age | Mean learned CE | Mean random CE | Mean matched dense CE | Learned − random | Learned − dense | Random − dense | Seeds favoring learned |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50% | R0 | 1.4870 | 1.5552 | 1.4951 | -0.0682 | -0.0081 | +0.0601 | 5/5 |
| 50% | R100 | 1.4066 | 1.4656 | 1.3954 | -0.0590 | +0.0112 | +0.0702 | 5/5 |
| 55% | R0 | 1.5071 | 1.5679 | 1.4951 | -0.0608 | +0.0120 | +0.0728 | 5/5 |
| 55% | R100 | 1.4115 | 1.4792 | 1.3954 | -0.0676 | +0.0161 | +0.0837 | 5/5 |
| 60% | R0 | 1.5086 | 1.5801 | 1.4951 | -0.0714 | +0.0135 | +0.0849 | 5/5 |
| 60% | R100 | 1.4236 | 1.5058 | 1.3954 | -0.0822 | +0.0282 | +0.1103 | 5/5 |
| 65% | R0 | 1.5220 | 1.6023 | 1.4951 | -0.0803 | +0.0269 | +0.1071 | 5/5 |
| 65% | R100 | 1.4327 | 1.5254 | 1.3954 | -0.0927 | +0.0373 | +0.1300 | 5/5 |

This table has two distinct readings:

1. **Mask selection matters relative to the matched random coordinates** in this saved E32 validation setup. The sign is consistent across every seed/condition pair, not just a pooled correlation.
2. **That does not establish dense-equivalent sparse performance.** At all four R100 sparsities, learned-mask mean CE is higher than the age-matched dense CE. At R0 only the 50% learned mean is slightly below dense; seed-level outcomes vary. Dense absolute CE itself changes with router age (mean 1.4951 at R0 versus 1.3954 at R100), so gaps must remain age-matched and absolute losses must remain visible.

The next action is just to preserve the paired table and write this result accurately. Do not run more metrics until one is specified in advance. No training is needed to complete this descriptive contribution. For any future confirmatory reopening, a prospective stopping rule would require (before data collection) a single endpoint, fresh source seeds, a demonstrably untouched evaluation set, all four sparsity-age strata retained, and simultaneous paired uncertainty intervals favoring learned masks in each stratum; otherwise stop or narrow the claim. That prospective rule does not convert this retrospective result into preregistered evidence.

## Other candidate assessments

### Router-age optimum selection: already answered, and unstable

The authoritative Dataset A table contains five complete seeds at every one of 30 architecture/sparsity curves and 11 router ages. The mean curve selects an intermediate age in 16 curves, R0 in 10, and R100 in 4, but all five seed-specific winner sets include that mean winner in only one curve. At 95% sparsity the mean winner is R0 for all five expert counts. This challenges an across-seed “optimal router age” claim rather than motivating a new search. Reusing the same validation sweep to select another age would amplify winner’s curse. A new 85%-only localization suite exists but its E16 seed-7 repeat is not an independent seed; E24/E32 use only three seeds.

The full lower grid is also descriptive, not a new confirmation: 212/1100 seed-level endpoint gaps are nonpositive, but **zero** of the 220 architecture-age-sparsity settings has all five seeds nonpositive; only 13 setting means are negative, and eight reverse sign for at least one leave-one-seed-out mean. Every architecture has a positive marginal mean gap. A selected best cell is not a replicated strict ticket.

### Budget/exposure factorial: incomplete, not a result

The live Dataset B pilot manifest contains 225 jobs: 45 reused references and 180 planned new jobs. Its current queue status is `DEFERRED_BY_USER`: 63 new jobs completed, two were interrupted by the user, and 115 remain pending. The completed new jobs are concentrated in A random controls and C recoveries; B/D and all five long-source runs are not complete. Do not interpret arm contrasts from this sequential partial population. Moreover, 10k source runs change source development, router age, and mask selection simultaneously; all recoveries use fresh optimizer state, so they are not exact continuations. The audit can inform a future cost/estimand decision, but does not justify resuming the full workload now.

### IMP versus one-shot or capacity allocation: no comparable current experiment

One-shot OMP masks and iterative IMP masks are different discovery protocols. The 8/16/32E, 80/85% IMP implementation is prepared but not launched; its resource estimate is hundreds of GPU-hours for discovery and recovery before adding a matched capacity-allocation baseline. A legacy 8E IMP experiment did complete and underperformed the one-shot/rewind result in that separate protocol. Neither fact verifies a motivating advantage for another IMP run. An exact IMP-versus-allocation question needs a fixed architecture, total attempt budget, matching source/data, masks, and dense controls; that design is absent.

The separately saved extended-recovery comparison is not a complete or self-consistent substitute for that planned IMP grid: `sparse_dense_comparison.csv` has 18 endpoint rows for E8/E16/E32, seeds 29/43/61, and 80/85%, while its README describes seeds 7/17 (12 cells for the stated architecture × seed × sparsity design). Treat the rows as represented observations only; reconcile their provenance before using them as a complete extension cohort. The distinct E32 five-seed high-sparsity trajectory analysis is accounted for separately in the recovery-advantage conclusion below.

## Rejected directions: preserve these stops

| Direction | Existing evidence to preserve | Decision |
|---|---|---|
| Gate factorial mechanism / natural gate mediation | Predeclared promotion criterion failed. Effect depends on imposed donor readout, reverses under native and assignment-only readouts, and dense-control changes contribute. Jacobian-only and interaction effects are not stable. | **No-go.** Do not reopen by selecting a favorable readout. |
| Fixed-source geometry as a winner-specific mechanism | Router-score/post-GELU association rises in sparse models and falls in dense controls, but the pattern occurs in matching and nonmatching cases. Functional geometry finds no robust match/nonmatch separator. | **Rejected as an explanation.** Association is not a mechanism. |
| Embedding sparsity/geometry mechanism | E remains dense; shrinkage is modest and spectral structure nearly unchanged. | **Do not promote small descriptive changes.** |
| Recovery advantage as broad token-level wins | Four of ten E32 IMP conditions have nonpositive gap at step 5000; only two demonstrate sustained observed tails. Token evidence is a fixed 2048-position validation-batch probe, not full validation. No untouched test set is established. | **Do not expand on probe gains or claim generalization.** |
| R20-specific router adaptation mechanism | Frozen/trainable aggregate changes do not identify a mechanism; R0 adapts substantially. Seed-7 token diagnostics find R20 is not uniquely better than both R0 and R100. | **Stop the current mechanism claim.** |
| Cross-architecture exposure or convergence law | Fixed global attempts/tokens are documented, but accepted per-expert source exposure and applied source updates are not complete across all architectures; 24E is valid only under the fixed-compute interpretation. | **No causal scaling/exposure claim from pooled groups.** |
| “Higher sparsity means faster inference” | These masks are unstructured; no corresponding sparse-kernel execution or latency measurement establishes a speedup. | **Do not claim efficiency from parameter counts alone.** |
| Semantic token/domain gains | Verified semantic domain labels are unavailable; probe positions are not labeled domains. | **Do not invent domain explanations.** |

## Asset inventory and protocol matching

The detailed family-by-family inventory is [asset_inventory.csv](./asset_inventory.csv). The direct comparability distinctions are in [protocol_compatibility_matrix.csv](./protocol_compatibility_matrix.csv). In particular:

- One-shot pruning is not IMP discovery.
- Source training, recovery, and an IMP discovery stage are different budgets.
- A step-2500 model continued with fresh AdamW/GradScaler is not an exact optimizer continuation.
- Router-age recovery compares a developed router with original expert/shared weights; it is not a whole-model checkpoint rewind.
- A gate-replay loss is conditional on its imposed assignment/gate readout, not native validation behavior.
- A full-validation CE curve, a 2048-position diagnostic token probe, a training-batch gradient probe, and a test metric are different populations and cannot substitute for one another.
- Seeds—not experts, token positions, layers, or checkpoints—are the replication units. Many named sweeps reuse the same five seeds.
- Current Dataset A and B protocols use train/validation data. No untouched test split with verified provenance was found for the candidate comparisons in this sweep.

## Scope and reproducibility

No repository `AGENTS.md` was present. The root `README.md`, experiment protocol, manifests, implementation notes, audits, completion markers, and numerical tables were read; historical narrative summaries were not treated as evidence when current tables or completion state differed. The read-only audit script reads JSON/CSV/JSONL metadata and metrics, verifies curves and hashes, and exports per-seed rows. It does not load tensors, run model forwards, inspect test data, or change existing inputs/reports. The only new files are in this requested output directory.

## Recommendation

Write up the existing E32 learned-versus-random comparison as a narrow validation result with explicit selection and split limitations; include the unstable age-optimum map and negative mechanisms as scope boundaries. Then stop expansion unless an independent dataset and fresh-seed confirmatory design is approved. **No additional experiment is strong enough, on current evidence, to justify another training or recovery run.**
