# Monitoring and resuming the recovery-order study

Run all commands from the repository root with `.venv-highres` (PyTorch 2.7.0).
Preparation, engineering, throughput profiling, and historical evaluation are
non-production gates:

```bash
.venv-highres/bin/python scripts/imp_recovery_order_stability_v1/run.py prepare
.venv-highres/bin/python scripts/imp_recovery_order_stability_v1/run.py engineering
.venv-highres/bin/python scripts/imp_recovery_order_stability_v1/run.py benchmark
.venv-highres/bin/python scripts/imp_recovery_order_stability_v1/run.py evaluate-historical
.venv-highres/bin/python scripts/imp_recovery_order_stability_v1/run.py verify-host
.venv-highres/bin/python scripts/imp_recovery_order_stability_v1/run.py seal
```

Run `verify-host` on both verified hosts, then run `seal`. The shared worker
manifest assigns Viper seeds 7/17/29 and Adder seeds 43/61. Start one worker per
available GPU only after checking that external GPU workloads are not active:

```bash
.venv-highres/bin/python scripts/imp_recovery_order_stability_v1/run.py worker
```

The worker runs the preselected seed-17/order-A sparse and unpruned pilot first,
then dispatches remaining jobs only after the outcome-independent pilot gate
passes. Inspect status and heartbeats with:

```bash
.venv-highres/bin/python scripts/imp_recovery_order_stability_v1/run.py status
find results/imp_recovery_order_stability_v1/status -maxdepth 1 -type f -print
find results/imp_recovery_order_stability_v1/jobs -name heartbeat.json -print
```

After an interruption, relaunch the same host's worker with:

```bash
.venv-highres/bin/python scripts/imp_recovery_order_stability_v1/run.py worker --resume
```

Completed updates are restored from the native interruption checkpoint with
optimizer, scaler, RNG, model, and batch position. The original permutation
arrays and identities remain fixed. Failed attempts are retained. After all
registered jobs complete, produce summaries and figures with:

```bash
.venv-highres/bin/python scripts/imp_recovery_order_stability_v1/run.py analyze
```
