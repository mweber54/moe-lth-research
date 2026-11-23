# Recovery advantage analysis v1

Checkpoint-only descriptive analysis of all ten 32-expert sparse conditions against five unique same-seed dense controls. The rerun trajectory is primary; source seed is the replication unit (n=5), and .80/.85 within a seed are dependent. No new training, recovery, mask search, optimizer update, model forward, or causal intervention was run. No source model tensor was loaded or modified.

## Interpretation in brief

Every condition is already nonpositive at the step-0 masked-initialization observation; therefore no sparse condition first crosses from a positive gap during recovery. The step-0 difference is not an effect of recovery. All ten later change sign at least once in the dense 50-attempt validation trajectory, and only four finish at a nonpositive gap. Distinguish this initial status from the later regained/persistent endpoint advantage.

The sparse--dense fixed-source/all-expert Pearson coupling divergence is generic to matching and nonmatching conditions. It is not a success explanation. The available comparisons do not isolate a predictive mechanism: observed performance tails vary by seed and sparsity, native accepted-routing and fixed-source probes answer different questions, and geometry/distortion move against moving dense references. No coherent causal explanation emerges.

## Endpoint comparisons and observed persistence

| Seed | Sparsity | Initial gap | First nonpositive | Persistent nonpositive tail begins | Supporting evaluations | Final gap | Endpoint label |
|---:|---:|---:|---|---:|---:|---:|---|
| 7 | 0.80 | -4.738764 | step 0 (already nonpositive at initialization) | 4750 | 6 | -0.035739 | matches/beats |
| 7 | 0.85 | -2.643991 | step 0 (already nonpositive at initialization) | none | 0 | -0.021304 | matches/beats |
| 17 | 0.80 | -4.660313 | step 0 (already nonpositive at initialization) | none | 0 | +0.021199 | does not match |
| 17 | 0.85 | -2.664101 | step 0 (already nonpositive at initialization) | none | 0 | +0.036323 | does not match |
| 29 | 0.80 | -4.835612 | step 0 (already nonpositive at initialization) | none | 0 | +0.003709 | does not match |
| 29 | 0.85 | -2.589152 | step 0 (already nonpositive at initialization) | none | 0 | +0.024770 | does not match |
| 43 | 0.80 | -5.253417 | step 0 (already nonpositive at initialization) | none | 0 | -0.001772 | matches/beats |
| 43 | 0.85 | -3.267041 | step 0 (already nonpositive at initialization) | none | 0 | +0.013132 | does not match |
| 61 | 0.80 | -4.515657 | step 0 (already nonpositive at initialization) | 4550 | 10 | -0.026802 | matches/beats |
| 61 | 0.85 | -2.583474 | step 0 (already nonpositive at initialization) | none | 0 | +0.029708 | does not match |

“Persistent tail” means at least two observed validation evaluations from the reported start through step 5,000 remain <= 0; it is not a claim between observations. The exact time of any crossing is unknown. Every 50-step validation observation is a scalar evaluation, not a saved checkpoint; only the explicitly listed snapshot steps have model checkpoints. At step 2,500, the optimizer/scaler reset leaves model weights unchanged.

See [crossover_summary.csv](./crossover_summary.csv) for later positive/nonpositive reversals and complete observed intervals; [interval_gap_decomposition.csv](./interval_gap_decomposition.csv) decomposes every adjacent change as `Δgap = Δsparse CE - Δdense CE`.

## Token gains and losses

Per-token results are only the fixed 2,048-position probe from validation batch 0, stored for each sparse condition and the same-seed dense checkpoint at the 16 saved snapshots. Full 128x128 batches were retained in the underlying forward caches for MoE capacity behavior. Token deltas use protected FP32 diagnostic arrays with TF32 disabled; full-validation scalar CE uses the separate rerun setting (FP32, TF32 enabled). The token probe is not full validation. Each position receives equal target-token weight; its mean delta reconciles with the sparse-minus-dense probe CE. The full 32-batch validation total is not substituted by this probe.

Token fractions, quantiles, signed contribution totals, and top-tail concentration appear in [token_gain_loss_summary.csv](./token_gain_loss_summary.csv); top-k shares use k% of all 2,048 positions, separately for each sign. Fixed-position arrays and checkpoint/cache hashes are in [token_arrays/](./token_arrays) and [token_array_provenance.csv](./token_array_provenance.csv). Tokens/positions are descriptive observations, not independent replicates; no token-level significance tests are used. Verified domain labels are unavailable.

For the four endpoint matches, the token probe is mixed rather than uniformly favorable:

| Seed | Sparsity | Probe mean delta | Full-validation gap | Improved positions | Worsened positions | Median delta |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.80 | -0.01778 | -0.03574 | 39.3% | 60.5% | +0.0036 |
| 7 | 0.85 | -0.03079 | -0.02130 | 42.3% | 57.5% | +0.0017 |
| 43 | 0.80 | -0.05901 | -0.00177 | 38.0% | 61.9% | +0.0079 |
| 61 | 0.80 | -0.07950 | -0.02680 | 51.0% | 48.9% | -0.0000 |

In three of the four matching conditions, more evaluated probe positions worsen than improve, while the mean delta is favorable; the net probe gain is therefore carried by larger-magnitude improvements on a subset, not a majority-token win. Seed 61/.80 is approximately evenly split. The top-k contribution shares in the CSV use k% of all 2,048 probe positions, separately for positive and negative mass. Some nonmatching conditions also have a favorable probe mean (notably seed 29/.80), while seed 43/.80's probe mean is much more favorable than its near-zero full-validation gap; do not generalize probe outcomes to full validation.

## Internal measurements aligned to observed steps

The long-form [aligned_internal_trajectories.csv](./aligned_internal_trajectories.csv) joins only existing measurements at exact seed, sparsity, and checkpoint step. It retains each metric's probe/reference, layer, precision/population definition, and source. Fixed-source/all-expert coupling is the same-input raw-router-score/post-GELU-activation Pearson probe; it increases in all sparse conditions, including all six endpoint nonmatches, while its matched dense comparator decreases. This generic pattern cannot be sufficient for an endpoint match.

Native routing entropy/margin and probe capacity acceptance/load are on native hidden inputs and selected probe positions with the complete batch capacity preserved. They are also recorded for each unique dense control; the same dense rows are explicitly reused for both sparsity comparisons, not counted twice. Fixed-source output errors are the identical-input expert-output comparison; native sparse/dense vectors are not identical inputs. Layerwise representation CKA uses a same-step dense, moving reference. Fixed-gate distortion reports the squared numerator D and squared reference-output energy separately; a rising normalized ratio is not itself an absolute increase. Gradient geometry exists only at 0/2,500/5,000 on previously observed training batches, not between those points. Update-event measurements are sparse selected steps; applied updates and AMP skips are exact at attempted-step resolution.

## Competing explanations and limits

Four strict endpoint matches are seed 7 at both sparsities, seed 43 at .80, and seed 61 at .80. Seeds 43 and 61 show the requested paired-sparsity contrast: .80 matches, .85 does not; that contrast is sparsity-confounded and within-seed, not independent replication. The .80 seed-43 endpoint margin is especially small and is retained without a post-hoc tolerance. TF32 off/on endpoint gap signs agree; sensitivity is tabulated separately in [precision_sensitivity.csv](./precision_sensitivity.csv).

TF32-off/on endpoint gap signs agree, and the maximum endpoint gap shift is reported in [validation_checks.json](./validation_checks.json). The closest endpoint margin is not reversed by this setting change. Intermediate near-zero gaps were not directly reevaluated under both TF32 settings; the curve uses strict zero and has no post-hoc tolerance.

All recovery runs deliberately reset AdamW/scaler at step 2,500; model weights match before and after the reset. There is no no-reset control, so the reset cannot be assigned causal responsibility. Data stream and validation batch hashes match within each seed's sparse/dense pair. Training CE and auxiliary routing loss are reported separately as recent training-window means; they are not substituted for validation CE. Routing capacity effects are probe-specific and must not be interpreted as whole-validation token routing.

The frozen configurations declare train and validation inputs, but no explicit test path or untouched-test provenance. No test split was inspected or evaluated, and validation is not relabeled as test. Historical measurements were excluded from the main trajectory because the TF32 setting differs; they are not counted as independent replications. No novel, causal, or beyond-cohort conclusion is supported.

## Recommendation

No focused intervention is recommended from these data: the generic coupling trend occurs in winners and nonwinners, and the endpoint/persistence pattern is seed-specific and sparsity-confounded. There is no specific causal mechanism sufficiently discriminated to justify selecting a manipulation without cherry-picking.

## Figures

- [A. Absolute full-validation CE](./figures/A_absolute_validation_ce.svg)
- [B. All ten validation gaps](./figures/B_validation_gap.svg)
- [C. Probe token gains/losses and concentration](./figures/C_probe_gain_loss.svg)
- [D. Layerwise internal trajectories](./figures/D_layerwise_internal.svg)

## Checks

Focused joins, unique dense reuse, checkpoint-cache/full-log precision agreement, strict crossover/reversal calculation, gap decomposition, token-position alignment, token-to-probe-CE reconciliation, and source-tensor non-mutation are recorded in [validation_checks.json](./validation_checks.json).
See the separate [competing-explanations matrix](./competing_explanations.md) for candidate status and counterevidence.
