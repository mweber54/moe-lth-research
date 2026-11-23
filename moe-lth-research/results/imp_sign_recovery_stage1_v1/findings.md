Stage 1 stopped after all 54 approved sign-only runs; no second segment or other arm was run. ([run_accounting.csv](run_accounting.csv): `attempted_updates`, `second_segment_started`.)

Per-seed joint decision uses three paired draws and the committed amendment. These intervals are conditional on the tested source seed, masks and fixed orders. ([per_seed_gate.csv](per_seed_gate.csv): `L`, `R`, `L_CI_low`, `R_CI_low`, `outcome`, `interval_scope`.)

|Seed|L = I_S(2000)|R = I_S(2000)-I_S(250)|Outcome|
|---|---|---|---|
|7|0.17440024|0.15492979|joint_pass|
|17|0.17850366|0.15920807|joint_pass|
|29|0.19470674|0.15901088|joint_pass|

The joint interim gate passed; seeds passing both criteria: [7, 17, 29]. Failure to pass is not proof of no effect. The gate does not authorize further training. ([stage1_gate.csv](stage1_gate.csv): `stage1_gate_pass`, `same_seeds_passing_both`, `training_continuation_authorized`.)

Per-seed/order absolute IMP and separate random-mask losses, separate shuffle penalties, both interaction levels and their within-run difference are in [per_seed_order_summary.csv](per_seed_order_summary.csv); all individual draws are in [per_seed_order_draw.csv](per_seed_order_draw.csv) and [individual_paired_draw_values.csv](individual_paired_draw_values.csv). Actual attempts, storage and elapsed slot-hours are in [actual_resources.csv](actual_resources.csv): `attempted_updates`, `logical_retained_bytes`, `production_worker_slot_hours`.
