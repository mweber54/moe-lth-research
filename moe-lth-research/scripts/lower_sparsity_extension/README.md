# Lower-sparsity extension

320 new sparse recoveries: 32 experts, five seeds, R0/R50/R100,
50/55/60/65% sparsity, arms A/B/C/D, plus random controls at R0/R100
in A/D. Reuses 60 dense records and all parent sources. No new source training.

Install only this new scripts/lower_sparsity_extension directory into the existing
repository. Do not overwrite scripts/nominal_exposure_dataset or rerun its prepare.
The previously prepared results/dataset_b_lower_sparsity must already exist.

Run from repository root on viper:

    .venv-highres/bin/python scripts/lower_sparsity_extension/test_extension.py
    .venv-highres/bin/python scripts/lower_sparsity_extension/launch.py --check
    screen -dmS dataset-b-lower bash -c 'exec .venv-highres/bin/python -u scripts/lower_sparsity_extension/launch.py >> results/dataset_b_lower_sparsity/handoff.log 2>&1'

Only launch after tests/check pass. launch.py waits for the parent's queue.lock,
then verifies all 225 parent rows are COMPLETE/REUSED. A crashed/incomplete parent
blocks handoff. The lock remains held throughout extension execution, excluding
both old supervisors. A separate handoff lock excludes duplicate launchers.

After parent completion, preparation verifies code/input hashes, generates missing
long-source masks, and creates a separate runtime manifest with 320 PENDING sparse
rows and 60 REUSED dense rows. It verifies inputs on both hosts before launching.
All output remains under results/dataset_b_lower_sparsity. Existing preparation
files are never overwritten. The extension has its own immutable execution seal.

Memory admission: 24 GiB free per host, one sparse worker per host. This is a
conservative threshold, not reserved memory; other users may increase consumption.
Expected work: 2,000,000 recovery attempts, approximately 104 historical GPU-hours
at 0.1874 seconds/attempt, before contention/I/O. Actual wall time is uncertain.
The coordinator retains a 30 GiB free-disk gate, not a user quota guarantee.

Monitor:

    tail -F results/dataset_b_lower_sparsity/handoff.log
    cat results/dataset_b_lower_sparsity/handoff_status.json
    cat results/dataset_b_lower_sparsity/queue_status.json

queue_status.json appears only after extension dispatch begins. Worker log paths
are listed there. Closing SSH is safe after launch inside screen.

Failure/restart: do not automatically reset RUNNING or FAILED_UNCERTAIN rows.
Resolve worker identity on both hosts first. On coordinator failure, remote workers
can survive; the extension refuses to relaunch uncertain rows. Completed jobs are
preserved. Partial trajectories are not resumed: the original worker restarts from
initialization on explicitly authorized retry. Interrupted unsealed preparation
fails closed if a runtime manifest already exists; inspect rather than delete it.
Do not run this coordinator directly: use launch.py to retain the parent lock.

Validation: ten standard-library tests pass, including exact 320-row plan match,
parent-completion guards, exclusive locking, tamper detection, worker target.
All scripts compile. Actual CUDA training was not executed in the development
workspace (PyTorch unavailable). Scientific recovery is delegated to the unchanged
parent worker, including its runtime assertions for data hashes, persistent masks,
optimizer step accounting, finite losses, and final checkpoint validation.

The extension does not choose a dense-matching tolerance. Report paired final
sparse-minus-dense losses and learned/random differences across seeds. Choose any
approximate-recovery tolerance before viewing extension results. These runs use
32E only and do not establish a universal architecture result.
