# Controlled known-mask transitions

The only authorized cohort is five seeds×dense/fixed/abrupt/gradual, IMP80,5000
attempts with the native2500 two-call reset. The mask is privileged diagnostic
information. Never change dispatch conditions based on CE/winning status.

Preparation/audit: `prepare.py`, `audit_supplement.py`; CPU tests:
`.venv/bin/python -m pytest tests/test_imp_mask_transition.py -q`.
Run `validate_gpu.py` once on each admitted Viper/Adder host before sealing and
launching. The launcher refuses a duplicate launch or an existing failure stop.

```sh
.venv-highres/bin/python scripts/imp_mask_transition_v1/cluster.py --seal
.venv-highres/bin/python scripts/imp_mask_transition_v1/cluster.py --launch
cat results/imp_mask_transition_v1/completion.json
cat results/imp_mask_transition_v1/workers/*.heartbeat.json
cat results/imp_mask_transition_v1/status/seed17_abrupt.json
cat results/imp_mask_transition_v1/status/seed17_gradual.json
tail -n 20 results/imp_mask_transition_v1/workers/viper.cs.kent.edu.log
tail -n 20 results/imp_mask_transition_v1/workers/adder.cs.kent.edu.log
.venv-highres/bin/python scripts/imp_mask_transition_v1/report.py
```

For a claimed job, its status JSON names an immutable attempt directory. Inspect
`attempt.log`, `heartbeat.json`, `segment_*/record.json`, `event_*.json`, and
`grid_*.json` there. Worker heartbeats update every15 seconds. The shared dispatch
lock prevents duplicate claims; host locks prevent multiple study workers per host.
Seed17 must complete numerical endpoint/reset/mask checks before other seeds can
claim work. Scientific direction is not a dispatch input.

Resumable checkpoints: `attempts/<job>/<attempt>/segment_*/interruption.pt`, atomic
every100 attempted updates. They retain optimizer/scaler/all RNGs/model, materialized
sampler position, exact event index/current mask hash, identity and log sizes.
Normal dispatch never automatically retries or resumes a failed/uncertain process.
First verify its old host/PID is dead and inspect the technical failure. Resume only
that explicitly identified existing attempt through `engine.execute` with the same
sealed job/variant. Preserve STOP.json and failed records under versioned names when
documenting an engineering correction; never silently restart or change the design.

Reused references supply only observed snapshots under the common readout, not a
fabricated full trajectory or optimizer continuation. Raw historical TF32 CE is not
the common endpoint. Missing diagnostic/update fields remain unavailable. Read
`coverage_and_exclusions.csv` and `reference_compatibility.csv` before interpreting
curves/AUC. Accepted-token-weight-coordinate work is a diagnostic count proxy,
not measured FLOPs or wall-time savings. IMP discovery cost is an excluded oracle cost.

`completion.json.scientifically_complete` is true only after all20 verified endpoint
records exist. Findings remain explicitly provisional until then. No follow-up is
launched by this workflow.
