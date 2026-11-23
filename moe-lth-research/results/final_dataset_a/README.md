# Dataset A finalization

Run from the repository root with `.venv-highres/bin/python`:

```
OMP_NUM_THREADS=2 .venv-highres/bin/python -u results/final_dataset_a/scripts/build.py
OMP_NUM_THREADS=2 .venv-highres/bin/python -u results/final_dataset_a/scripts/verify_masks.py
.venv-highres/bin/python -u results/final_dataset_a/scripts/analyze.py
.venv-highres/bin/python -u results/final_dataset_a/scripts/verify.py
```

`verify.py` intentionally exits 1 if the required full Cartesian product is incomplete. A BLOCKED status is not a software success or permission to freeze. All files are generated atomically in this separate output tree. Raw experiment files are read only. No experiment execution entrypoint is called. Required library versions and code hashes are recorded in provenance.

Set `DATASET_A_EXTRA_ROOTS` to colon-separated readable artifact roots to audit additional existing shard outputs. Candidates must still pass original source, mask, config, endpoint and checkpoint hash checks. No training or mask regeneration occurs. The original manifest remains unchanged. If additional valid checkpoint candidates conflict, the condition fails instead of selecting by outcome.

The hash cache stores SHA-256 with size/mtime identity and avoids rereading unchanged checkpoints during reruns. Delete only this output tree's `cache/hashes.json` to force full byte rehashing. Cached diagnostics are invalidated by input size/mtime; exact metric/exposure functions are loaded from the repository's existing finalizer and its code is copied into `inputs/code`.

The canonical-named table is only a candidate while the status is BLOCKED. Audit rows explicitly preserve every missing/invalid expected condition. Summary tables include coverage flags and available-seed counts. Do not cite partial architecture curves as completed five-seed results. No frozen manifest is emitted on a failed integrity gate.
