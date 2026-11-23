# Bounded E16/80% replication and conditional connectivity controls

Frozen scope: seeds 7/17/29/43/61, 2500 attempted updates, exact source initialization, native trainable router/shared/expert parameters. Six compatible seed17 endpoints are reused. Twenty-four new Phase 1 recoveries run on one detached worker per Viper/Adder. Twenty-five degree controls are conditional on every seed gap strictly exceeding 0.02 and all registered mixing checks passing. No other expansion exists.

Preparation has already completed. Do not rerun preparation or replace the frozen manifest. Read `results/imp_mask_information_v2/protocol.md`, `registration.json`, `manifest.json`, `seal.json`, `reuse_checks.json` and `validation.json`.

```bash
.venv-highres/bin/python scripts/imp_mask_information_v2/cluster.py --seal
.venv-highres/bin/python scripts/imp_mask_information_v2/cluster.py --launch
.venv-highres/bin/python scripts/imp_mask_information_v2/report.py
```

The first two commands refuse duplicate seals/launches. Launch verifies both hosts and all five data sequences before detaching workers. Locks were tested across the shared filesystem. Workers admit jobs only with 24 GiB free GPU memory, preserve fresh attempts, heartbeat every 15 seconds and halt further dispatch after failure. No processes are killed and no failed/uncertain attempts are retried. A manually rerun worker can finish pending jobs only if existing claims and code seals remain valid; it never resumes or overwrites an existing attempt.

Inspect `launch.json`, `workers/*.heartbeat.json`, `job_status.csv`, `status/*.json`, `completion.json`, `logs/*.log`, and per-attempt checks/results. `findings.md`, `results.csv`, `paired_effects.csv`, `aggregate_effects.json`, `recovery_curves.csv` and `figures/` update after each recovery. `phase2_gate.json` records the bounded execution decision. Admitted finite swap chains record diagnostics under `degree_diagnostics.json`, `degree_masks/`, and `degree_sampler_runtime.json`; no uniform or stationary-independence claim is made.

The existing trainer and forward pass remain unchanged. Safe checkpoint loading is forced. Kernel compilation is local C++17 using a fixed-proposal double-edge-swap implementation; its source and runtime are recorded. Diagnostic and production degree masks must all pass before any degree recovery begins.
