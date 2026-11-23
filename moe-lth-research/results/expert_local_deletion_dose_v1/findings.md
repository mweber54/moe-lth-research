# Expert-local deletion-dose and reliability check

Seed 17 is the same **nonmatching 32E IMP80 engineering case**. Both trajectories use steps 25, 250, 1000 and pre-reset 2500, layer 1, experts 0/8/16/24, original fc1 groups and complete A/B batches. Both A and B have already been examined. No training, optimizer update, permanent pruning, predictor, clustering refit, extra seed or Stage 2 ran.

Four damage-independent group permutations generate nested 0/1/2/4/8-group doses. Identical physical unions are evaluated once; ordering aliases remain in ordering_aliases.csv. The prior global union cap is absent. Manifest fractions distinguish sparse surviving-matrix dose from the smaller coordinate-matched unpruned fraction. Output-zero controls are separate interventions, not extra weight-deletion doses.

## Verdict

**Numerical measurement is reproducible.** Repeated same-input interventions have maximum token-loss disagreement 0 nats; zero dose, Stage 1 cached baselines/single-group checks, restoration, full hashes, native layer-1 assignments/gates, RNG, gradients and sparse masks pass. Across positive weight doses, whole-probe damage spans -0.00012026 to 0.0030185 nats/token. Output RMS changes and positive controls establish whether a small whole-probe effect accompanies a real function change; a zero-traffic expert has undefined output RMS rather than fabricated zero dependence.

**Equal-dose A/B rankings are a separate question.** Among adequate-coverage within-expert panels at 1/2/4 groups, 78/96 CE rank correlations are defined; their median is 0.4, range -1 to 1. These combine neither doses nor duplicate orderings and are descriptive nested observations, not independent source replications. Stratified values below and ab_reliability.csv take precedence over a pooled impression. Exact-repeat reliability does not establish cross-population ranking reliability.

**Whole-probe averaging can conceal conditional effects.** Baseline-accepted and complement effects, accepted-token counts and per-sequence paired deltas are explicit. The conditional subset stays fixed before deletion; it is still a task-loss subset, not an isolated expert loss. Read low-coverage effects with their token and sequence counts.

**Dependence changes over recovery must be read expert by expert.** temporal_dependence.csv holds adjacent-time contrasts for the same eight-group coordinate set, together with traffic. The tables and per-expert plots below show signed changes rather than assuming monotonic dose response or a common recovery trend. Early high loss or absent traffic can dominate a pooled comparison. No specialization interpretation follows.

**Next step:** this check can justify cross-seed confirmation of the *measurement method*, but does not justify a specialization predictor or Stage 2. Improve or explicitly retain unresolved low-traffic coverage and verify equal-dose A/B rankings across seeds before making dispensability forecasts. No additional seed is launched here.

## Equal-dose CE agreement

Across experts, unique physical unions only; adequate coverage in both probes. Correlations across different doses are deliberately absent. Empty/constant or fewer-than-three-point rank panels are undefined. Within-expert versions, functional-output agreement, sign agreement, pairwise concordance and error scales are in ab_reliability.csv.

| Trajectory | Step | Groups | A/B pairs | Spearman | MAE (nats/token) | Pair order concordance |
|---|---:|---:|---:|---:|---:|---:|
| sparse | 25 | 1 | 0 | undefined | undefined | undefined |
| sparse | 25 | 2 | 0 | undefined | undefined | undefined |
| sparse | 25 | 4 | 0 | undefined | undefined | undefined |
| sparse | 25 | 8 | 0 | undefined | undefined | undefined |
| sparse | 250 | 1 | 12 | 0.41259 | 2.801e-06 | 0.62121 |
| sparse | 250 | 2 | 16 | 0.21765 | 3.4568e-06 | 0.51667 |
| sparse | 250 | 4 | 16 | 0.44412 | 2.8534e-06 | 0.625 |
| sparse | 250 | 8 | 4 | 0.8 | 2.3534e-06 | 0.83333 |
| sparse | 1000 | 1 | 12 | -0.11888 | 3.1417e-06 | 0.45455 |
| sparse | 1000 | 2 | 16 | 0.24706 | 3.0601e-06 | 0.56667 |
| sparse | 1000 | 4 | 16 | 0.45882 | 3.9143e-06 | 0.66667 |
| sparse | 1000 | 8 | 4 | -0.2 | 1.1978e-05 | 0.5 |
| sparse | 2500 | 1 | 12 | 0.11888 | 1.9991e-05 | 0.56061 |
| sparse | 2500 | 2 | 16 | 0.63824 | 2.6583e-05 | 0.70833 |
| sparse | 2500 | 4 | 16 | 0.42353 | 4.3567e-05 | 0.64167 |
| sparse | 2500 | 8 | 4 | 0.4 | 4.9766e-05 | 0.66667 |
| unpruned | 25 | 1 | 6 | 1 | 2.0666e-05 | 1 |
| unpruned | 25 | 2 | 8 | 1 | 5.0322e-05 | 1 |
| unpruned | 25 | 4 | 8 | 1 | 9.7149e-05 | 1 |
| unpruned | 25 | 8 | 2 | undefined | 0.00018027 | 1 |
| unpruned | 250 | 1 | 12 | 0.062937 | 1.2494e-05 | 0.51515 |
| unpruned | 250 | 2 | 16 | 0.24706 | 1.3176e-05 | 0.58333 |
| unpruned | 250 | 4 | 16 | 0.88235 | 1.22e-05 | 0.85833 |
| unpruned | 250 | 8 | 4 | 0.8 | 2.643e-05 | 0.83333 |
| unpruned | 1000 | 1 | 12 | -0.23776 | 5.7944e-06 | 0.40909 |
| unpruned | 1000 | 2 | 16 | -0.69706 | 7.7029e-06 | 0.23333 |
| unpruned | 1000 | 4 | 16 | -0.22941 | 1.0348e-05 | 0.44167 |
| unpruned | 1000 | 8 | 4 | 0 | 2.1067e-05 | 0.5 |
| unpruned | 2500 | 1 | 12 | 0.2028 | 8.9173e-06 | 0.57576 |
| unpruned | 2500 | 2 | 16 | -0.26765 | 1.6051e-05 | 0.41667 |
| unpruned | 2500 | 4 | 16 | -0.52353 | 2.974e-05 | 0.3 |
| unpruned | 2500 | 8 | 4 | -0.4 | 3.1197e-05 | 0.33333 |

## Fixed eight-group dose and output-zero control

Mean and range across the four sampled experts, not a population estimate; conditional means omit undefined empty populations but the count of defined cases remains explicit in dose_summary.csv. Low traffic is retained.

| Trajectory | Step | Probe | Intervention | Whole CE mean | Accepted CE mean | Accepted CE range |
|---|---:|---|---|---:|---:|---:|
| sparse | 25 | A | output zero (separate control) | -8.8785e-05 | -0.73205 | -0.73205 … -0.73205 |
| sparse | 25 | A | 8 groups | -8.5477e-06 | -0.066178 | -0.066178 … -0.066178 |
| sparse | 25 | B | output zero (separate control) | -5.3535e-05 | -0.49477 | -0.49477 … -0.49477 |
| sparse | 25 | B | 8 groups | -4.7152e-06 | -0.040201 | -0.040201 … -0.040201 |
| sparse | 250 | A | output zero (separate control) | 0.0001266 | 0.0034085 | -0.00030265 … 0.0093491 |
| sparse | 250 | A | 8 groups | 8.4947e-06 | 0.0003497 | 5.1109e-05 … 0.00073223 |
| sparse | 250 | B | output zero (separate control) | 0.00016159 | 0.0046869 | 0.00054016 … 0.012021 |
| sparse | 250 | B | 8 groups | 9.1572e-06 | 0.00026879 | -7.2469e-06 … 0.00063084 |
| sparse | 1000 | A | output zero (separate control) | 0.00037058 | 0.0097083 | 0.0015618 … 0.028412 |
| sparse | 1000 | A | 8 groups | 1.2214e-05 | 0.00035159 | -6.2663e-05 … 0.0007722 |
| sparse | 1000 | B | output zero (separate control) | 0.0003886 | 0.01012 | 0.0011588 … 0.032994 |
| sparse | 1000 | B | 8 groups | 3.5475e-06 | 0.00018022 | -0.0001602 … 0.00069223 |
| sparse | 2500 | A | output zero (separate control) | 0.0045862 | 0.12753 | 0.035782 … 0.25012 |
| sparse | 2500 | A | 8 groups | 5.3872e-05 | 0.0013913 | -0.00052806 … 0.0026907 |
| sparse | 2500 | B | output zero (separate control) | 0.0046159 | 0.11809 | 0.019361 … 0.20125 |
| sparse | 2500 | B | 8 groups | 2.2614e-05 | 0.00039965 | -0.003212 … 0.0026666 |
| unpruned | 25 | A | output zero (separate control) | 0.013051 | 8.7374 | 3.213 … 14.262 |
| unpruned | 25 | A | 8 groups | 0.0010425 | 0.71739 | 0.28009 … 1.1547 |
| unpruned | 25 | B | output zero (separate control) | 0.012536 | 8.987 | 2.6557 … 15.318 |
| unpruned | 25 | B | 8 groups | 0.0010883 | 0.76653 | 0.24812 … 1.2849 |
| unpruned | 250 | A | output zero (separate control) | 0.00021727 | 0.0054969 | -0.0015716 … 0.013014 |
| unpruned | 250 | A | 8 groups | 2.3383e-05 | 0.00027219 | -0.00024687 … 0.0009466 |
| unpruned | 250 | B | output zero (separate control) | 0.00023706 | 0.007515 | 0.00074547 … 0.024194 |
| unpruned | 250 | B | 8 groups | -3.0468e-06 | 9.9801e-05 | -0.00047356 … 0.00062117 |
| unpruned | 1000 | A | output zero (separate control) | 0.00010668 | 0.014445 | 0.00057207 … 0.042275 |
| unpruned | 1000 | A | 8 groups | 1.4818e-05 | 0.0011045 | 0.00037801 … 0.0021376 |
| unpruned | 1000 | B | output zero (separate control) | 0.00019699 | 0.017908 | -0.00029401 … 0.058545 |
| unpruned | 1000 | B | 8 groups | -6.1098e-06 | -0.00015492 | -0.00061371 … 0.00036747 |
| unpruned | 2500 | A | output zero (separate control) | 0.0026279 | 0.08758 | 0.0090692 … 0.19623 |
| unpruned | 2500 | A | 8 groups | -1.7138e-05 | -0.00070874 | -0.0020084 … 0.00079721 |
| unpruned | 2500 | B | output zero (separate control) | 0.0026054 | 0.084847 | -0.0005484 … 0.19764 |
| unpruned | 2500 | B | 8 groups | -3.6965e-05 | -0.0011725 | -0.0020041 … -0.00042401 |

## Nonadditivity, resolution and limitations

Maximum absolute union-minus-sum difference is 0.00013279 nats/token whole-probe and 0.025608 on accepted positions. This is the directly measured difference, not assumed additivity. Nonadditivity is undefined for output-zero controls. No threshold is used to choose groups or create a trend; 0.001 and 0.0001 are descriptive plot reference lines only.

Raw expert-output RMS is measured on baseline-accepted native-dispatch positions and all output channels. Relative RMS uses the corresponding baseline expert RMS; values at or below 1e-12 are undefined. Gate-weighted RMS is the RMS of baseline gate times output difference. Baseline expert RMS and baseline gate-weighted RMS are also stored. The expert-only replay for cached CE doses never reroutes isolated tokens. Positive-control routing/gates remain bitwise unchanged at the affected layer; later layers may change.

A/B disagreement reflects distinct, previously examined diagnostic contexts as well as traffic. Exact repeatability measures numerical stability, not sampling uncertainty or independent generalization. Eight groups, four orderings, token positions, sequences and checkpoints are not independent seed replications. An unpruned comparison deletes the same coordinates at a different active-matrix fraction. No causal specialization or permanent-pruning recoverability claim is supported.

## Runtime and storage

Preparation 0.79s; complete measurement and table serialization 103.09s on NVIDIA H100 NVL; report/figures before sealing 12.95s. 1888 native complete-batch forwards; 192 single-group CE measurements reused. CUDA event evaluation intervals total 50.59s (includes orchestration/transfers between device events, not a kernel-only bill). New cache 351.41 MB; total outputs before final seal 387.21 MB. Source Stage 1 caches and model snapshots are referenced, not copied. Actual byte counts and stage timing remain in compute_accounting.json.
