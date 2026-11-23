# Result schema and timing

`e*_seed*/<profile>/neurons_s*.parquet` is one consolidated partitioned dataset. A unique row key is `(architecture, seed, mask_id, profile_name, layer, expert, neuron)`. Sparsity, initialization_id, discovery_run, rewind_step, recovery_run, profile_step and profile_checkpoint identify lineage. All these pairing files are for pooled calibration; source subsets are in `profiles.parquet` and corresponding H1/H2 summaries.

- `ramp`: unnormalized Eq.3 with conventional GELU adaptation.
- `activation_only`: L2 over actual dispatched intermediate activations without routing coefficient or output norm.
- `down_norm`: fc2 output column L2 norm.
- `routing_activation`: L2 over routing-weighted dispatched activations.
- `ramp_per_dispatch`: RAMP/sqrt(accepted assignment count); zero on unexposed experts.
- `ramp_per_route_energy`: RAMP/sqrt(sum of accepted routing coefficient squared); zero on zero routed contributions. Separately named exposure control, never replaces primary RAMP.
- `utilization`: accepted assignment count, repeated per neuron in the expert. `routing_exposure`: sum accepted coefficients. `route_energy`: sum accepted squared coefficients.
- `up_retention`, `down_retention`: exact Boolean mask row/column mean. `retention`: combined eligible-coordinate fraction; in these equal-width matrices this equals their average. `surviving_weights`: exact combined count.
- `initial_magnitude`: mean absolute full unmasked coordinates at original initialization.
- `checkpoint_magnitude`: mean absolute full coordinates at the explicitly named profiling checkpoint.
- `initial_surviving_abs_mean`, `_rms`, `_positive_fraction`: statistics on historically surviving coordinates at original initialization.
- `final_surviving_abs_mean`, `_rms`, `_positive_fraction` in pairing files refer to the **supplied profiling checkpoint**, as specified by `neuron_mask_statistics`; they do not necessarily refer to final training and are not post-checkpoint features. The name is inherited from the mapping function's initial/final arguments. Interpret with `profile_name` and `profile_step`.
- No retained-coordinate statistic enters the current H2 predictor set except explicitly identified mask-level score summaries. No later-state magnitude enters an early profile.

`recovered_neuron_statistics/*.parquet` is a separate table. Its `recovered_*` statistics use the actual final sparse recovered checkpoint, after 2500 attempted updates. Join on architecture/seed/sparsity/mask_id/layer/expert/neuron. They are never joined into early H1/H2 features. This provides actual final retained-value measurements without ambiguous timing.

`mask_run_results.csv` has one row per mask/profile/domain. Recovery outcome is present once per such mask/run summary, not used to regress independent neuron rows. This layout contains repeated views of the same recovery; do not pool timing, source subsets or sparsities as independent replicates. Each H2 stratum has five source seeds.

`h1_groups.parquet` contains descriptive within-expert and within-layer correlations. `h1_seed_results.csv` averages groups per run. Only `h1_seed_summary.csv` uses seed uncertainty; no thousands-of-neurons pseudo-replication. Constant columns produce undefined correlations. Invalid/unexposed group counts remain in mask/run summaries.

`h2_descriptive_associations.csv` contains prespecified univariate Spearman associations and the range when each source seed is left out. These ranges are sensitivity summaries, not confidence intervals or test-set prediction errors. Many associations are exploratory and no feature is selected from them.

`historical_mask_controls*.csv`, `adaptation_context_separate_protocol.csv`, and `sign_only_context_separate_protocol.csv` retain their source protocol labels; do not pool them into the primary recovery strata. Randomized mask draws are nested within source seed, not additional independent sources.
