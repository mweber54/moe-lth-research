# Full Dataset A lower-sparsity extension

Exact learned-mask grid: E={8,12,16,24,32}, seeds={7,17,29,43,61},
ages={0,10,20,30,40,50,60,70,80,90,100}, sparsities={.50,.55,.60,.65}.
1100 conditions; 60 validated completed 32E learned conditions reused;
1040 new recoveries. No random-mask jobs, new sources, or new dense training.
All source/recovery budgets 2500 attempted steps. Router remains trainable.

Install this new directory only. Parent and completed subset are untouched.
Run on viper from repository root:

    .venv-highres/bin/python scripts/full_lower_sparsity/test_grid.py
    OMP_NUM_THREADS=2 .venv-highres/bin/python -u scripts/full_lower_sparsity/prepare.py
    .venv-highres/bin/python -u scripts/full_lower_sparsity/launch.py --check
    screen -dmS dataset-a-full-lower bash -c 'exec .venv-highres/bin/python -u scripts/full_lower_sparsity/launch.py >> results/dataset_a_full_lower_sparsity/launch.log 2>&1'

Run sequentially; stop if any command fails. Preparation may take several minutes
for checkpoint hashing and CPU mask generation. It writes missing_checkpoints.json
and stops if required source files are absent. No hash guards may be bypassed.
Remote verification may expose missing host-local /data inputs; do not launch until
those are materialized and hash-verified. No automatic transfer is performed.

Dense references are explicitly table-backed endpoints from the provided finalized
Dataset A dense CSV. Each reference has provenance.json with original row and
inventory digest. No nonexistent dense checkpoint is synthesized or claimed verified.
Original dense checkpoint files need not remain on disk. Sparse training does not
load dense weights; the comparator is the matched dense final validation loss.

Reused 32E conditions must match source checkpoint identities, generated mask state,
seed, router step, recovery steps, router mode and dense comparator, and must have
validated completion/final-checkpoint hashes. Failure blocks preparation rather than
silently rerunning or accepting a mismatch.

New output: results/dataset_a_full_lower_sparsity. Successful final counts:
COMPLETE=1040, REUSED=335 (60 sparse + 275 dense references). Separate job_type when
summarizing. Two hosts, one worker each; 24 GiB free admission per host, 30 GiB disk
gate. Neither is a resource reservation. Expect hundreds of GiB of outputs; inspect
available space/quota before launch. No deletion of old checkpoints is authorized.

Monitor queue_status.json and launch.log in the new output root. Worker log paths
appear in queue status. FAILED_UNCERTAIN requires manual host/process verification
before resetting any job. Model-only checkpoints cannot resume partial trajectories.
All predecessor queue locks stay held; Dataset B never restarts automatically.

Tests: eight pure-Python tests passed locally, including all 1100 exact keys and
both 275-row uploaded inventories. Syntax compilation passed. No GPU or full
PyTorch recovery could be tested in the development environment. --check performs
host input verification and CUDA availability checks, not a training smoke test.
The unchanged existing recovery worker retains its runtime scientific assertions.

Scientific scope: within each source/seed mask is fixed across router ages. Experts
and shared weights initialize at source R0, selected router comes from requested
checkpoint. Learned masks derive from the final 2500-step source. Report seed-paired
loss gaps and uncertainty; this package does not define a post-hoc ticket tolerance.
