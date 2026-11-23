# Protocol audit

All 18 reconstructed E0/shared/router/ticket/ET-mask component hashes match historical metadata. All materialized training and validation sequence hashes match before training. Within each architecture/seed the three router ages use identical masks, E0, and shared state.

Training calls the existing recovery function unchanged except for an optional callback after step 0 and completed training steps. Historical evaluation every 50 steps and routing diagnostics are retained. The callback saves complete state at nine requested steps; dense checkpoints are then loaded into a separate model for sparse diagnostics. Model parameters/buffers, optimizer, GradScaler and RNG are audited. Diagnostic gradients are cleared before the next training step. No training iterator is passed to diagnostics.

Sparse/dense vectors use identical sorted named parameter order and select surviving mask indices (unmasked expert biases survive in full). Norms and invalid counts are retained. Forced routing shares sparse assignment IDs and capacity priorities, with exact accepted-token equality asserted; native gate values remain model-specific. Functional comparisons evaluate both expert functions on identical sparse-native accepted inputs.

FP32 diagnostic backward uses unit loss scale, so gradients are already unscaled; it never calls the training GradScaler. Training remains fp16. Undefined zero-norm or nonfinite cosines are unavailable, never zero-filled.

## Completion verification

All 18 pairs completed sequentially. All 36 final model states are bit-identical to historical checkpoints; all final losses and ticket gaps match exactly. All 324 snapshot mutation audits passed. Eight representative complete snapshots were reloaded and model/optimizer hashes verified; all snapshot paths exist. All required CSVs and 16 PNG/PDF figure pairs passed artifact checks. Primary gradient measurements contain no nonfinite rows; zero-norm/empty comparisons remain explicitly unavailable. See artifact_integrity.json and full_final_state_reproduction.csv.
