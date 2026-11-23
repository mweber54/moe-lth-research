# Dataset B 32E pilot

Root: `results/dataset_b_pilot/` (home-backed; separate from Dataset A and `/data`).

The authorized pilot has 225 registered jobs: 45 reused A references and 180 new jobs. Read `PROTOCOL.md` before interpreting results. Source and recovery budget arms, random-mask allocation restrictions, primary contrasts, schedule, restart limitations, and diagnostic units are fixed there and in `configs/design.json`.

## Current status and monitor

From `/users/kent/student/mweber54/moe-lth-research/moe-lth-research`:

```bash
cat results/dataset_b_pilot/queue_identity.json
cat results/dataset_b_pilot/queue_status.json
screen -ls
tail -f results/dataset_b_pilot/logs/supervisor.log
```

`WAITING_GPU` means the durable process is alive but **production training has not started**. The supervisor waits for all external GPU-0 compute processes to exit; it does not kill them. `RUNNING` reports the actual worker PID, condition ID, and per-attempt log path. The queue identity records PID/host/start time. The live manifest is `manifests/run_manifest.csv`; `initial_manifest.csv` is immutable. Jobs execute sequentially in stage order: A random controls → C → new long sources → B/D and D random controls.

## Launch / resume the durable queue

After inspecting the existing session and status, the command used is:

```bash
screen -DmS dataset-b-32e-v1 bash -lc 'cd /users/kent/student/mweber54/moe-lth-research/moe-lth-research && exec .venv-highres/bin/python -u scripts/nominal_exposure_dataset/supervisor.py >> results/dataset_b_pilot/logs/supervisor.log 2>&1'
```

A process flock prevents a duplicate pilot supervisor. Do not stop other users' GPU processes. If a supervisor has exited, the same command resumes its queue: completed/reused jobs stay intact; an existing live worker is detected rather than duplicated; an interrupted trajectory restarts from the prescribed initialization into a new numbered attempt. This is **not** model-only checkpoint continuation. Failed scientific checks enter `FAILED_NEEDS_DIAGNOSIS`; do not simply flip their status and retry. No source, recovery or mask artifact is overwritten.

Preflight seals the actual code and static evidence. Changes to validated code require rerunning tests and preflight before resumption; immutable scientific configurations and masks must not be silently edited. Runtime analysis outputs are not sealed as immutable preflight inputs.

## Reproduce checks / analyses

Preparation has already run; `prepare.py` intentionally refuses to overwrite the initial manifest. Do not rerun it on this root.

```bash
OMP_NUM_THREADS=2 .venv-highres/bin/python -u scripts/nominal_exposure_dataset/test_pilot.py
OMP_NUM_THREADS=4 .venv-highres/bin/python -u scripts/nominal_exposure_dataset/verify_initial_loss.py
OMP_NUM_THREADS=2 .venv-highres/bin/python -u scripts/nominal_exposure_dataset/evidence.py
.venv-highres/bin/python -u scripts/nominal_exposure_dataset/preflight.py
.venv-highres/bin/python -u scripts/nominal_exposure_dataset/analyze.py
```

The real-initialization check uses CPU to avoid competing with an occupied GPU. The synthetic smoke test is segregated under `smoke/` and is never a production observation. `analyze.py` can be run during the experiment; absent future-arm endpoints remain explicitly unavailable, not imputed. Paired primary contrasts, reference intervals, per-seed signs and R50's descriptive role are documented in the protocol.

## Outputs and validation

- `audits/preflight.json`: actual checks, resource state, code hashes, static evidence hashes.
- `audits/initial_loss_verification.json`: real reconstructed high initial loss and independent CE reduction.
- `audits/resource_estimate.json`: historical timing samples, runtime uncertainty and retention estimate.
- `analysis/dataset_a_*`: raw existing curves, all-sparsity paired age changes, tail improvements and report.
- `analysis/approximately_equal_retained_parameter_comparisons.csv`: actual mask-count matching at declared 1% tolerance.
- `analysis/pilot_observed_endpoints.csv`, `pilot_paired_contrasts.csv`, `pilot_contrast_summary.csv`: observed results and all prespecified contrasts.
- `configs/`, `manifests/`, `validated_code/`: immutable design, reuse/source provenance and exact code.
- `sources/<source_id>/attempt_NNNN/`: long source checkpoints, raw counts and exact checkpoint-prefix exposure.
- `recoveries/<condition_id>/attempt_NNNN/`: metrics, raw counts, probes, mask checks, final checkpoint, metadata and validated completion marker.
- `masks/`: one random mask per source/seed/sparsity, shared across router ages; long learned/random masks are created only after their valid long source completes.

A job is complete only after finite endpoints/checkpoints, exact cadence, mask persistence, zero pruned optimizer moments, batch hashes, and attempted/applied/AMP accounting pass. Recovery probes are at 0/2500/final. `checkpoint_exposure.json` contains genuine prefix counters, not repeated whole-run totals. No huge token-level training dumps are enabled.

Expected additional storage is about 75 GiB plus a 30 GiB reserve. Historical mtime-derived training time is roughly 66–74 GPU-hours (median 68.3), excluding queue waits and setup uncertainty. The queue checks free storage before each job and preserves interrupted artifacts. No Dataset A files or checkpoints are deleted to make room.
