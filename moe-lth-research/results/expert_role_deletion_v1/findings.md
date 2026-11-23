# Stage 1 expert-role deletion-sensitivity pilot

## A. What is measurable from existing artifacts

Verified seed 17 32E IMP80 and matched unpruned recovery snapshots exist at 25,250,1000,2500; no substitution or training was required. The 2500 snapshot is pre-reset. Seed17 is a **nonmatching engineering case**, retained regardless of outcome. Historical/rerun endpoint aliases have identical verified model hashes and count once. Native training-only complete batches give contextual routing, fixed-context-group differentiation and direct reversible deletion damage. No old raw geometry, sensitivity or replay loss measurements were reused; their numerical/probe/population provenance differs. See artifact_inventory.csv and checkpoint_and_probe_hashes.json.

Probe budget: 2 complete 128 × 128 batches per A/B; these may have been seen during training. B was excluded from decisions and clustering fit; it is not an independent generalization test. K = 16 fixed clustering uses earliest unpruned A only, with frozen reference labels for both trajectories. Small clusters remain explicit in cluster_counts.csv.

## B. What the pilot actually shows

Primary tolerance of 0.001 nats/token is exploratory. Continuous signed damage is primary. The following fractions use all 32 sampled groups, not all possible connections. Each group contains floor(1% of actual surviving fc1 coordinates), and the unpruned comparison removes identical coordinates. Persistent selection at 250 uses 25/250 A; at 1000 it uses 250/1000 A. Future outcomes use next-checkpoint B.

| Recovery | Decision → future | Persistent groups | Selected coordinate fraction of panel | False-safe rate | Selected future mean damage |
|---|---|---:|---:|---:|---:|
| sparse | 250 → 1000 | 32/32 | 100.000% | 0 | -1.12137e-06 |
| sparse | 1000 → 2500 | 32/32 | 100.000% | 0 | -4.27295e-06 |
| unpruned | 250 → 1000 | 32/32 | 100.000% | 0 | 4.0194e-06 |
| unpruned | 1000 → 2500 | 32/32 | 100.000% | 0 | -2.72563e-06 |

At the primary tolerance, all 32/32 groups qualify at both persistence decisions in both trajectories, and none exceeds the threshold on next-checkpoint B. This is a ceiling result: the sparse pilot does **not** show an increasing persistently low-damage fraction. At 0.0001, sparse qualification also stays 32/32; unpruned persistent qualification rises 16/32 to 32/32. These fixed sensitivity results are descriptive, not a chosen replacement primary outcome. Whole-probe group damages range from -4.43331e-05 to 0.000427993 nats/token. Low whole-probe damage must be read with conditional effects and sparse traffic: 24 of 128 expert/time/probe/population divergence rows lack the declared minimum of 32 positions. No low-traffic expert or group was removed.

Damage distribution by checkpoint and probe:

| Recovery | Step | Probe | Mean damage | Minimum | Maximum |
|---|---:|---|---:|---:|---:|
| sparse | 25 | A | -1.0684e-06 | -6.24297e-06 | 0 |
| sparse | 25 | B | -5.91964e-07 | -3.53792e-06 | 3.07492e-07 |
| sparse | 250 | A | 1.51904e-06 | -2.39254e-06 | 8.8379e-06 |
| sparse | 250 | B | 7.88618e-07 | -8.93111e-06 | 9.37578e-06 |
| sparse | 1000 | A | -3.52423e-08 | -7.38716e-06 | 6.64339e-06 |
| sparse | 1000 | B | -1.12137e-06 | -4.90237e-06 | 1.54409e-06 |
| sparse | 2500 | A | 3.54264e-06 | -2.02848e-05 | 4.28137e-05 |
| sparse | 2500 | B | -4.27295e-06 | -4.43331e-05 | 1.91464e-05 |
| unpruned | 25 | A | 0.00012781 | 0 | 0.000378805 |
| unpruned | 25 | B | 0.000135156 | 0 | 0.000427993 |
| unpruned | 250 | A | 6.58053e-06 | -3.38502e-06 | 1.51521e-05 |
| unpruned | 250 | B | -3.0887e-06 | -2.84008e-05 | 1.70197e-05 |
| unpruned | 1000 | A | 1.87533e-06 | -6.75918e-06 | 9.9703e-06 |
| unpruned | 1000 | B | 4.0194e-06 | -3.23909e-06 | 1.06964e-05 |
| unpruned | 2500 | A | -8.43157e-07 | -2.04526e-05 | 1.03758e-05 |
| unpruned | 2500 | B | -2.72563e-06 | -1.87277e-05 | 1.96216e-05 |

H1 remains a descriptive engineering observation with only two persistence decision times. Increasing qualifying fractions would concern the sampled panel only. More qualification and more reliable identification are separate: compare forecast_summary.csv false-safe rates and continuous future damage, including all fixed tolerance sensitivities. No retrospective all-checkpoint filter was applied.

H2 is unresolved. expert_level_descriptive_comparisons.csv aligns step, proposed/accepted load and gates, magnitudes/movement, past damage and separate role measures against future B damage. descriptive_associations.csv reports expert-level rank associations, separately by time and pooled with explicit time confounding. No incremental predictive model was fitted: four experts in one seed do not support reliable role-augmented prediction. Eight groups do not create eight independent expert-role observations.

Combined-deletion diagnostics: 48 checkpoint/probe/selection rows; cap 134,218 coordinates (1% of all surviving expert fc1+fc2 weights across all layers). Whole groups are included in canonical order; selection is past-only. Union damage and sum of individual damage are compared in combined_deletion_checks.csv. Each model is restored afterward.

Numerical checks passed for repeat losses, zero dose, post-ablation restoration, state hashes, gradients, RNG and sparse-mask integrity. Paired per-token and per-sequence damages and baseline assignments/gates/router inputs are cached with checkpoint/probe/group/configuration/implementation identities. Tokens are not treated as independent source replications. Conditional subsets use baseline accepted positions; attention and later layers can spread effects beyond them.
Combined damage is not additive: maximum absolute deviation from the sum of singles is 0.00024346 nats/token. Sparse unions reach 0.000223249 nats/token, exceeding the strict 0.0001 convention despite individually qualifying groups. Accepted-position conditional group damage reaches 0.182201 nats/token; small affected populations can dilute whole-probe damage. These are baseline-conditioned task losses, not isolated expert losses.


## C. What cannot be concluded

These contextual clusters are not established semantic domains. Persistence and differentiation are separate; stable loads alone do not establish stable assignments. Capacity changes affect accepted sets; proposed metrics remain separate. Low traffic is unavailable where coverage is inadequate, not evidence of dispensability. There is no independent-seed uncertainty estimate or reliable H2 learning relationship. Deletion effects are checkpoint interventions; role associations are observational and do not prove a causal specialization mechanism, permanent-pruning recoverability or future-training dispensability. IMP mask membership defines this cohort only and cannot be assumed by a future dense-training algorithm.

## D. Cost of confirming the same analysis across five seeds

Measured main measurement pass: 65.6s on NVIDIA H100 NVL; baseline profile 0.462s, one-group profile 0.022s. Linear five-seed main-pass estimate 0.091 GPU-host hours; this is an engineering estimate, not a reserved budget. Full-stage wall times (including unions) are recorded separately in compute_accounting.json when available. Storage scales to about 3.089GB using this pilot cache. CPU/IO, checkpoint availability, per-host contention and union selection can change costs. Keep all five seeds including matching/nonmatching, unchanged sampling, seed-held-out preprocessing and one simple regularized baseline versus role-augmented model if prediction is later authorized. Five seeds remain exploratory. No expansion launched.
Including union diagnostics, the observed full pilot runtime is bounded by 106.3–148.1 seconds from process-start/heartbeat/completion timing. A linear five-seed estimate is 8.9–12.3 minutes on the same host. This excludes the initial inventory/preparation pass, whose wall time was not instrumented; it is not a total-project runtime guarantee.


## E. Whether a later permanent-pruning intervention is warranted

This one-seed diagnostic does not warrant launching Stage2. First confirm the fixed analysis across existing seeds and evaluate incremental held-out predictive value, coverage and nonadditivity. A later permanent-pruning study needs separate authorization and matched-budget recovery endpoints. No training, permanent pruning, recovery, optimizer updates or Stage2 launch occurred.
