# Results Revision Summary

This summary records the current state of the revision work in the repository as of 2026-08-15.

## Current Verified Results

### Protocol implementation and compatibility
- The compute-matched rewind protocol is implemented and exposed with explicit names in `src/moe_lth/pruning/train_ticket_v2.py`:
  - `full_budget_early_state_retrain()`
  - `compute_matched_rewind()`
- The richer route-trace schema is implemented and versioned in `src/moe_lth/routing/rich_trace.py`.
- The route history serialization now includes schema metadata, integrity checks, and deterministic hashing.
- The default config now includes the needed fields for rich-route logging and routing-corruption fraction controls.

### Validation evidence
- Fresh verification command:
  - `python -m pytest -q`
- Result:
  - 23 passed in 10.35s

## Interpretation for the manuscript

The current repository state supports the following, cautiously:
- The codebase distinguishes the early full-budget retraining protocol from the compute-matched continuation protocol.
- The richer trace schema is structurally in place and compatible with legacy traces via upgrade logic.
- The scientific intervention sweeps (deconfounded routing, graded corruption, cross-init matrix, larger-scale validation) are implemented or scaffolded, but the expensive experimental runs have not yet been executed in this repository state.

## Current protocol-gate status

The repository is still intentionally blocked at the WikiText reference gate. The most recent evidence from `scripts/seed7_wikitext_gate.py` shows:

- `LEARNED = 129.4652099609375`
- `REPLAY = 129.4652099609375`
- `LEGACY_SHUFFLE = 142.22572326660156`
- `DECONFOUNDED_SHUFFLE = 129.4652099609375`

This is not a valid reproduction of the historical reference run in `results/wikitext103_gpu_suite/normal/suite_summary.json`, which reports validation loss `1.6817417740821838` for the reference configuration. The current gate is therefore treated as a failed reference reproduction, not as a pass.

The root cause is configuration mismatch rather than a route bug: the gate script runs on CPU with `fp32`, `steps = 5`, `eval_interval = 5`, and `validation_blocks = 1`, while the historical reference run uses the exact GPU/FP16/2500-step config in `configs/wikitext103_gpu.yaml` and the saved historical resolved config under `results/wikitext103_gpu_suite/normal/resolved_config.yaml`.

## Expected manuscript impact

- Strengthens the paper: clear separation of protocol semantics and stronger route-trace supporting evidence.
- Requires further evidence before claiming causal strength: deconfounded shuffle, graded corruption, and cross-initialization results remain to be run, and the exact historical reference must be matched before any broadened claim is made.
- No manuscript wording should be finalized until the experimental sweeps above are executed and reviewed.

## Remaining work before submission

The next operating priorities are:
1. Diff the current proxy gate against the exact historical WikiText config and evaluation pipeline.
2. Reproduce the historical learned baseline under the true reference config before any broader claim is considered valid.
3. Prove the deconfounded shuffle is genuinely nontrivial by showing route disagreement while preserving counts, acceptance masks, and gate statistics.
4. Only then proceed to the graded corruption, cross-init, and larger-scale validation pass.
