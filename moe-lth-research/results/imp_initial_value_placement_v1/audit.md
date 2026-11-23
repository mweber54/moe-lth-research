# Implementation and provenance audit

The authoritative sources are the order-study registration, its artifact inventory,
canonical stream manifest, recovery order manifest, and frozen high-sparsity cohort.
All recorded input file hashes are checked during preparation, then checked again
on both hosts. No source paths are inferred from checkpoint filenames.

The original step-0 payload schema is `step`, `model`, and `config`. Source, expert
count, step, full state and mask hashes are checked. Each original mask's JSON
provenance must identify round 8, target sparsity 0.8, its source checkpoint path
and matching initialization hash. Native `validate_masks` requires exact expert
weight names, boolean dtype and model shapes. Every expert projection is a native
`nn.Linear`: fc1 has [hidden_size, d_model] and fc2 [d_model, hidden_size].
The model uses native PyTorch module initialization (Linear, Embedding, LayerNorm);
no replacement state is used. The saved complete source model supplies every
starting tensor. The tied LM head/embedding is retained by strict state loading.

`imp_80_85/run.py` implements nested expert-local IMP with rewind to the original
initialization. `imp_extended_recovery/pipeline.py` and the native
`run_router_age_recovery.py` establish the fresh optimizer/scaler/source-seed reset
at the 2500 boundary. The order-study executor preserves the complete boundary
model, masks gradients, enforces zeros after AdamW and records overflow decisions
and pre-clipping norms. Its complete-batch streams and saved A/B permutations are
copied byte for byte, never regenerated for production.

The v1 mask-information implementation is in scripts/imp_mask_information;
its results are in results/imp_mask_information_v1. The v2 coordinate validator
checks surviving values against W0 and masked zeros. This study adds exact
within-matrix multiset and permutation checks. Random masks preserve per-matrix
counts and therefore the combined per-expert count, without degree matching.

The new job.py is the order-study `_run_one_job` with its source/mask reconstruction
block replaced by loading the sealed treatment. Native segment training, complete
model reset, callbacks, common evaluation and interruption checkpoints are reused.
All variants retain `model_type=imp80` internally to select the same masked native
training path; the public variant/family/placement fields identify treatments.

Common readout uses FP32, TF32 off, full native validation batches, sum reduction
and actual target-token counts. The manifest requires 492416 targets per source;
partial batches are retained as materialized by the native loader. Historical
TF32 losses are not used for scientific readouts. Existing dense A/B controls are
references, with no new dense training authorized.

The engineering suite repeats the native before/after-reset interruption checks,
paired batch identity, model-preserving reset, masking, and RNG-preserving common
evaluation; separate tests check transformation identities and contrast algebra.
The environment has no pytest in .venv-highres, so the standalone test functions are run directly with that verified Python.
The available .venv pytest also runs eight new/existing transformation and order
tests, including the signed-zero regression. The failed pytest invocation
is preserved in unit_tests.log and the direct result in unit_tests_direct.log.

Storage reservation counts eleven model snapshots, one post-reset snapshot,
two native segment endpoints and two three-model-equivalent interruption states
per job. Atomic staging on both hosts, a 3% serialization/log margin and 10 GiB
reserve are added. Existing artifacts are never removed to make room.
