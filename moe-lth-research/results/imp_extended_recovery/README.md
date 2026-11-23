# Extended IMP and Dense Recoveries

The matrix covers E8/E16/E32, seeds 7/17, and both 80% and 85% IMP sparsity.
Each sparse final recovery and its matched dense baseline extend from 2,500 to
5,000 total attempted steps. Existing 2,500-step masks and checkpoints are
reused. If a requested IMP mask or final recovery is missing, the runner calls
the existing IMP pipeline for only that architecture-seed group with
`--resume`; committed discovery stages are verified and reused, and discovery
stages remain 2,500 steps. All extension outputs stay under this directory.

## Continuation Method

All sparse and dense continuations use **fresh AdamW and GradScaler state** from
their corresponding 2,500-step model weights. They are not exact optimizer
continuations. The data stream is reconstructed and its original 2,500-batch
prefix and validation sequence are hash-checked; the next 2,500 batches drive
each extension. Sparse masks stay fixed and routers remain fully trainable.

The original optimizer hyperparameters, FP16 protocol, auxiliary routing loss,
batch construction, and validation procedure are retained. Validation CE is
written every 50 absolute steps from 2,500 through 5,000. Full training-state
checkpoints are saved at 3,000, 4,000, and 5,000. Atomic interruption snapshots
are written every 100 extension steps, so rerunning the same command resumes a
compatible partial run; a partial output without a valid snapshot is preserved
and rejected rather than silently restarted.

`sparse_dense_comparison.csv` reports matched sparse-minus-dense validation CE
gaps at both 2,500 and 5,000 steps. A 5,000-step sparse result beating a
2,500-step dense baseline is not treated as a winning-ticket result.

## Commands

Show source availability without starting training:

```bash
.venv-highres/bin/python scripts/imp_extended_recovery/run.py
```

Run all remaining source preparation, sparse extensions, and matched dense
extensions sequentially from the repository root:

```bash
mkdir -p results/imp_extended_recovery
nohup .venv-highres/bin/python scripts/imp_extended_recovery/run.py --execute > results/imp_extended_recovery/launch.log 2>&1 < /dev/null &
echo $!
```