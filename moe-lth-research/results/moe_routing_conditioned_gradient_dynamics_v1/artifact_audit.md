# Artifact audit

- Checkpoint: `results/moe_connection_lifetime_v1/prefixes/seed7_A/state500.pt`
- Checkpoint SHA-256: `8d19ebd582d5bf630b0c2b60c4bf89a9d5a3a919d89a8663da41ee7d7fdde557`
- Provenance cell: `seed7_A_500`
- Configuration model-source hashes: verified from `native_provenance.json`.
- Frozen 180-batch split: verified independently against the two saved split-half files.
- Existing gradient artifact scope: 32 final-layer `fc2.weight` matrices only.
- Existing sign sums, resolved counts, and absolute normalized-gradient sums cannot recover signed means, probe second moments, or `fc1` values.
- No full per-probe gradient tensor archive was found.
- Existing domain labels, corpus hash, batch indices, route replay, state hash, and deterministic pilot are reused.
- Additional checkpoint replay is required for the 256-matrix pilot and for direct mixed-gradient reconstruction.

## Checkpoint payload

- Payload keys: cuda_rng, model, numpy_rng, optimizer, optimizer_steps_applied, python_rng, scaler, scheduler, step, torch_rng
- Optimizer instantiated: `False`
- Parameter updates: `0`
- Expert coordinate count: 67,108,864
- Matrices: 128 `fc1.weight` and 128 `fc2.weight`

## Reused provenance

- Training corpus SHA-256: `7aad299a6af6f1c85e6ee399ed9d83156ea54af98f1cd5d69a2bf614bde1aed3`
- Domain labels SHA-256: `9a490c5179b9f5d989ed29254c4bc9b5cacb33fc527f71e131d881fed692a37a`
- Batch-index SHA-256: `0ea96efc48b52e7dc84d92e63b8aadcd3873e3126df5b0e88cddd1eb9659baaf`
- Configuration SHA-256: `5b1e44d86ceea81dfbe743860b1f221d855efa372624532f26a10dd9695fed14`

No pre-existing signed aggregate is inferred from sign consistency.
