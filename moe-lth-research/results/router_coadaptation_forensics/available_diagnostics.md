# Available Diagnostics

| Metric | Runs | Steps | Invalid scalar records | Usable | Direct comparison |
|---|---:|---|---:|---|---|
| router_parameter_drift_normalized | 18 | 0-2500 | 0 | yes | frozen value is structurally zero/not applicable |
| router_grad_norm | 36 | 1-2500 | 20 | qualified/no | frozen value is structurally zero/not applicable |
| expert_grad_norm | 36 | 1-2500 | 3 | qualified/no | yes |
| routing_entropy | 36 | 0-2500 | 0 | yes | yes |
| mean_selected_probability | 36 | 0-2500 | 0 | yes | yes |
| top1_top2_margin | 36 | 0-2500 | 0 | yes | yes |
| expert_utilization | 36 | 0-2500 | 0 | yes | yes |
| dropped_fraction_by_layer | 36 | 0-2500 | 0 | yes | yes |
| assignment_agreement_with_initial_recovery_router | 36 | 0-2500 | 0 | yes | yes |

Frozen records lack `gradient_valid`, `measurement_status`, and capacity aggregate fields; absence is not treated as zero.
