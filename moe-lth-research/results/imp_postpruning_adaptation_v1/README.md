# Post-pruning adaptation study operations

This directory contains the isolated protocol, inputs, and outputs for the
1000–1500 router × expert adaptation study. The implementation never modifies
the source checkpoints, masks, or prior experiment directories.

## Before launch

The study requires the sealed probe manifest and registered code/input hashes:

```bash
python3 scripts/imp_postpruning_adaptation_v1/prepare.py --seal-probes
```

The worker requires CUDA, at least 24 GiB free GPU memory, at least 200 GiB free
output-filesystem space, and no more than 90% GPU utilization. It takes an
exclusive host claim and per-job shared claim, and runs one worker per host.
Do not kill unrelated processes to satisfy admission.

## Exact launch sequence

Run only the fixed seed-17 engineering pilot first:

```bash
python3 scripts/imp_postpruning_adaptation_v1/worker.py --pilot
```

The initial pilot attempt reached step 1500 for
`seed17_unpruned_full` but failed while saving its final router artifact because
the resumed native output directory lacked `component_checkpoints/`. Its
checkpoint and logs were preserved. Engineering registration revision 3 records
the directory-setup repair; revision 4 corrects the finalizer's diagnostic-only
optimizer/scaler clone; revision 5 makes frozen-state tensor comparison
device-independent after the next cell stopped before its first update. These
revisions do not change the scientific design. Finalize the saved checkpoint
without re-entering training, then resume the pilot:

```bash
python3 scripts/imp_postpruning_adaptation_v1/prepare.py --engineering-revision-5
python3 scripts/imp_postpruning_adaptation_v1/finalize_step1500.py \
  --attempt results/imp_postpruning_adaptation_v1/attempts/seed17_unpruned_full/attempt_1793026405891041730
python3 scripts/imp_postpruning_adaptation_v1/worker.py --pilot
```

The finalizer requires a verified step-1500 checkpoint and complete,
nonduplicated update/diagnostic rows; it performs no optimizer updates and
consumes no training batches. It records the one missing offset-zero diagnostic
reconstruction separately and preserves the original attempt logs and
checkpoint.

The pilot reconstructs and validates seed 17's common 1000-update prefix, then
runs all eight seed-17 model-type × intervention branches. It must write
`pilot_validation.json` with status `PASS`; scientific effect direction is not a
gate.

Only after the pilot passes, launch the remaining four seeds:

```bash
python3 scripts/imp_postpruning_adaptation_v1/worker.py --cohort
```

The cohort runner processes one job at a time on its host and reconstructs each
seed's prefix before its eight branches. Alternatively, run an explicitly
selected seed-17 branch with `--job 17 MODEL_TYPE CONDITION`; non-pilot seeds
remain blocked until the pilot passes.

## Resume and monitoring

Re-run the same `--pilot`, `--cohort`, or `--job` command after an interruption.
The worker reuses a single incomplete attempt and resumes from its latest native
interruption checkpoint; completed jobs are accepted only after final checkpoint
hash verification. If more than one incomplete attempt exists for a job, resolve
the ambiguity manually rather than selecting one implicitly.

Monitor atomic status and heartbeat files under `status/`, the attempt's native
`interruption.pt`, and the host GPU with:

```bash
find results/imp_postpruning_adaptation_v1/status -maxdepth 1 -type f -print
nvidia-smi
```

If a claim remains from a crashed process on the same host, a retry will reclaim
it only after verifying that its recorded PID is no longer alive. Claims owned
by another host are not stolen automatically.

## Analysis

After every registered branch is `COMPLETE` and its outputs pass hash/grid
validation, run:

```bash
python3 scripts/imp_postpruning_adaptation_v1/prepare.py --engineering-revision-6
python3 scripts/imp_postpruning_adaptation_v1/analyze.py --analyze
```

Revision 6 registers the analysis-only correction for flattening the
tuple-keyed trajectory map in the SVG renderer. It is required for this cohort
because the first analysis attempt exposed a renderer shape mismatch after all
branch outputs had completed; it does not dispatch training.

The separate historical late-trajectory audit can be generated without
dispatching training:

```bash
python3 scripts/imp_postpruning_adaptation_v1/analyze.py --late-audit
```

No training beyond global step 1500, new mask search, or follow-up experiment is
authorized by this protocol.
