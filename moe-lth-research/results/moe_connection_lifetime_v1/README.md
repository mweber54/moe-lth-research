# MoE connection lifetime v1

Run commands from the inner moe-lth-research project directory on Viper. The native archive matches the .venv-highres Python environment. The two-host workflow uses the V2 admission rule on Viper and the existing shared_gpu_worker policy on Adder.

```sh
.venv-highres/bin/python results/moe_connection_lifetime_v1/supervise.py
OMP_NUM_THREADS=1 .venv-highres/bin/python results/moe_connection_lifetime_v1/report.py
```

The supervisor takes an exclusive NFS lock, runs the engineering checks, freezes the protocol, seals the prospective forecasts, and then runs the continuations. A stage with failed checks stops the dependent work and keeps the attempt log. Rerunning the supervisor reuses verified receipts and saved continuation checkpoints. A launch or checkpoint alone does not contribute to endpoint counts. The reporter opens the final checkpoint, checks its SHA-256 and serialized counters, and verifies every union-masked weight and optimizer moment before counting it.

| Evidence | Location |
|---|---|
| Registered scientific design and engineering choices | registered_request.md, registration.sha256, registration.md |
| Post-engineering protocol freeze | freeze.json, freeze.sha256, created by the freeze stage |
| Loaded sources, architecture, runtime, data and roster | native_provenance.json, validation_domains.json, split_audit.json, seed_roster_audit.json |
| Analytical/native gradients and full-state continuation oracle | gradient_fixture.json, gradient_native.json, smoke_7_A_isolated/native251.json |
| Two-cut restart, recipient isolation and sign checks | smoke_7_A_isolated/verified.json, smoke_7_A_isolated/swapped_labels.json |
| Dose, stratum checks and recipient coverage | groups/*/sets.pt, audit.json and independent_audit.json in each cell directory |
| Per-pair forecasts, matched-coordinate drift and raw gate telemetry | groups/*/forecast*.json, coordinates*.json and routes*.pt |
| Forecast seals and prospective starting states | seals/, prefixes/ |
| Native trajectories, per-sequence loss sums and final checkpoints | branches/ and random_control/ |
| Execution attempts, current counts and summary tables | jobs/, *_attempt*.log, status.json, runs.csv, paired.csv, seed_means.csv, summary.csv |
| Incremental gains, coverage and uncertainty | incremental_gains.csv, coverage.csv, trajectories.csv, domains.csv |
| Resource records and final interpretation | timing/, job_timing.csv, verdict.md |

The engineering archive retains the first incompatible-environment oracle, GPU admission and lock failures, and the original donor-replay failure. Failed_counter_alias outputs, stored under failed_counter_alias/, used mutable AdamW counters. Their forecasts are excluded from the valid seal directory. Study-specific modules use lifetime_engineering.py to distinguish them from the archived V2 engineering module.
