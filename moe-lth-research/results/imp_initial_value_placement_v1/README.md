# Surviving initial-value placement experiment

Run from the repository root using `.venv-highres/bin/python`. The exact request
is preserved in user_request.txt. The immutable registration fixes 60 new jobs
and ten reused IMP controls. No additional training grid is authorized.

Preparation and gates:

```bash
.venv-highres/bin/python scripts/imp_initial_value_placement_v1/run.py prepare
.venv-highres/bin/python scripts/imp_initial_value_placement_v1/run.py engineering
.venv-highres/bin/python scripts/imp_initial_value_placement_v1/run.py benchmark
.venv-highres/bin/python scripts/imp_initial_value_placement_v1/run.py verify-host
ssh -i ~/.ssh/dataset_b_cluster adder.cs.kent.edu 'cd /users/kent/student/mweber54/moe-lth-research/moe-lth-research && .venv-highres/bin/python scripts/imp_initial_value_placement_v1/run.py verify-host'
.venv-highres/bin/python scripts/imp_initial_value_placement_v1/run.py seal
```

One worker per host/GPU initially. Existing external GPU processes cause waiting;
workers never kill or signal other processes. Viper runs seeds 7/17/29 and Adder
43/61. Before cohort dispatch, Viper must finish the three fixed seed-17/order-A
pilot jobs; the gate checks provenance, numerical integrity, accounting, snapshots
and resources, without consulting scientific effect signs.

Launch locally on each host after sealing:

```bash
.venv-highres/bin/python -u scripts/imp_initial_value_placement_v1/run.py worker
```

Monitor:

```bash
.venv-highres/bin/python scripts/imp_initial_value_placement_v1/run.py status
cat results/imp_initial_value_placement_v1/workers/*.json
find results/imp_initial_value_placement_v1/jobs -name heartbeat.json -exec cat {} \;
cat results/imp_initial_value_placement_v1/pilot_gate.json
```

After interruption, run on the same host:

```bash
.venv-highres/bin/python -u scripts/imp_initial_value_placement_v1/run.py worker --resume
```

Workers hold per-GPU and per-job file locks. Atomic statuses, unique attempts,
and native optimizer/scaler/Python/NumPy/Torch/CUDA RNG checkpoints preserve
resumption; completed steps are not repeated in the persisted update logs.
Uncommitted updates since the last interruption checkpoint may be recomputed.
Model-only snapshots are explicitly nonresumable. All failed attempts remain.

Analysis refreshes automatically after completed jobs and when the cohort is
complete. It can also be regenerated without training:

```bash
.venv-highres/bin/python scripts/imp_initial_value_placement_v1/run.py analyze
```

Missing results stay explicit in completion.json. Figures remain labeled pending
until their required observations exist. Never infer completion from a worker PID.

The transport helper verifies both hosts and launches detached persistent workers:

```bash
.venv-highres/bin/python scripts/imp_initial_value_placement_v1/cluster.py check-hosts
.venv-highres/bin/python scripts/imp_initial_value_placement_v1/run.py seal
.venv-highres/bin/python scripts/imp_initial_value_placement_v1/cluster.py launch
```

`launch.json` records worker PIDs and logs; launch refuses a second invocation.
Preparation attempt 1 failed the exact starting-state hash because of signed
zeros. Its artifacts/code/log are retained under failed_preparations/attempt_1.
Attempt 2 preserves native signed-zero bits and does not alter retained values.
