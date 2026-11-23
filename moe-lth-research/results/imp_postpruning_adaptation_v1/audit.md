# Artifact and diagnostic audit

## Provenance and full-state compatibility

The source artifacts, paths, SHA-256 hashes, and sizes are recorded in
`artifact_inventory.csv`. The original initialization checkpoints are step zero
and model-only. The available dense step-1000 checkpoint is a historical
model-only snapshot. The abrupt post-event step-1000 checkpoint is also
model-only. No exact complete branch state at the event boundary is available
(no optimizer, scaler, RNG, or global data cursor in those saved step-1000
snapshots). The initial-optimizer snapshots from older audits do not include the
complete state at step 1000. Therefore production must reconstruct the common
prefix with the original native implementation, init, mask schedule, RNG, and
deterministic stream; compare the reconstructed model and validation loss to
historical checkpoints under the registered tolerances; then save a complete
prefix checkpoint. If it fails, stop production.

For abrupt pre-deletion state, the dense step-1000 snapshot is the candidate
common-prefix reference. The equality between abrupt pre-event and dense is
expected from identical unpruned training but remains to be explicitly
verified; it is not assumed. Abrupt post-deletion state is the retained
step-1000 model snapshot in the transition attempt. Per-tensor IMP80 mask counts
are in `imp80_tensor_counts.csv`. Branch state must additionally include the
native optimizer's scalar step counters and coordinate moments, GradScaler,
Python/NumPy/Torch CPU/CUDA RNG states and `sampler_next_batch=1000`.

Training stream and exact attempted updates 1001–1500 are hashed for each seed
in `next_500_batch_hashes.csv`. Validation hash, batch count, per-batch target
counts and total targets are included in `artifact_inventory.csv` and the input
hash file. The validation set includes the original partial final batch. Existing
complete-native continuation outcomes and the missing factorial grid are
summarized in `measurement_coverage.csv`. The existing full controls have
candidate endpoint data, but do not provide required offset-25/100 model
snapshots or all new diagnostics. Consequently the pre-dispatch estimate
conservatively plans 40 exact 500-attempt continuations (20,000 branch updates),
plus five necessary 1,000-update common-prefix reconstructions (5,000 updates).
Ten historical full-endpoint values are candidates for reuse only after exact
continuation verification; they do not replace the missing trajectories or
diagnostics. This is not a claim that 40 endpoint rows inherently require 40
new jobs; the additional full continuations are planned because of the missing
registered timepoints/diagnostics.

## Existing trajectory review

The historical 1000–1500 window was selected after previous results had been
examined. This experiment is not an overall preregistration. For clarity, the
three quantities below remain distinct:

* Paired pre/post-deletion validation CE change at step 1000: mean
  **0.478721** over five seeds (positive means CE worsened). Per-seed paired
  values are in `historical_transition_contrasts.csv`; they use the same full
  validation batches before and after the abrupt deletion.
* Abrupt-minus-fixed step-1000 trajectory gap: mean **0.498040**. This is a
  comparison of different training trajectories, not immediate pruning damage.
  The approximately 0.498 gap is not described as deletion damage.
* Absolute abrupt CE improvement from global step 1000 to 1500: mean
  **0.708696** (positive means CE fell during continuation).

The paired pre/post measure is also present in the abrupt attempt's `grid_1000_pre`
and post-event `grid_1000`; the event diagnostic separately records a fixed
training-probe damage measure. These two paired readouts are not interchangeable.
Existing routing/gating/capacity and update logs are listed as input artifacts.
Their coverage limitations are explicit: they describe the prior full adaptation
trajectory and do not identify a routing mechanism or the causal contribution of
parameter movement.

## Output and scope controls

The output namespace is isolated at `results/imp_postpruning_adaptation_v1/`.
No source experiment, mask, checkpoint, manifest or report was modified. No new
training has been dispatched by this preparation step. No causal claim about
late sparse outperformance, an online pruning algorithm, a token-routing
mechanism, or a pre-pruning predictor is authorized by these measurements.
