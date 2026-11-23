E32/IMP80 matched-distance recovery stopped after all 108 registered runs (270,000 attempted updates). Each arm preserved the same masks, with fixed original controls. ([run_accounting.csv](run_accounting.csv): `attempted_updates`, `second_segment_started`.)

|Seed|Magnitude reopening R_M|Matched-sign reopening R_Bm|Bm−M contrast|95% paired interval|Direct criterion|
|---|---|---|---|---|---|
|7|-0.016352|0.033644|0.049996|[-0.009065, 0.109056]|False|
|17|0.003827|0.021600|0.017773|[0.010250, 0.025296]|False|
|29|0.011242|0.047150|0.035908|[0.011676, 0.060140]|True|

The prespecified seed-consistency rule did not pass. This compares arms directly; three-draw intervals are conditional on each tested seed, masks and orders. Failure does not prove no difference. ([per_seed_results.csv](per_seed_results.csv): `C_mean`, `C_CI_low`, `C_CI_high`, `direct_pass`, `arm_outcome`, `M_near_zero`; [seed_consistency.csv](seed_consistency.csv): `consistency_pass`.)

These data do not establish a consistent practically larger sign effect at this matched dose; inspect per-seed intervals and arm outcomes before interpreting similarity or imprecision.

All individual paired draws are reported in [individual_paired_draw_values.csv](individual_paired_draw_values.csv): `C`, `R_M`, `R_Bm`, `I_250_M`, `I_250_Bm`, `L_M`, `L_Bm`. Absolute IMP and distinct random-mask losses and penalties, by seed/order/arm/checkpoint, are in [per_seed_order_arm_step.csv](per_seed_order_arm_step.csv): `CE_imp`, `CE_random1`, `CE_random2`, `original_CE_imp`, `original_CE_random1`, `original_CE_random2`, `penalty_imp`, `penalty_random1`, `penalty_random2`, `I`. Matching and flips are in [matrix_invariants.csv](matrix_invariants.csv): `target_distance`, `realized_distance`, `absolute_distance_mismatch`, `frac_sign_flipped`. Actual costs are in [actual_resources.csv](actual_resources.csv).

The three tested seeds and fixed masks constrain generalization. The high-CE original initialization regime remains a caveat from the prior audit. This experiment does not establish which subnetworks ultimately beat dense, whether sign alignment causes IMP selection, or what happens beyond step 2500. No extension is enabled.
