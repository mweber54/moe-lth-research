# Recovery batch-order stability protocol

This prospectively specified follow-up is within an exploratory project; it is
not project-level preregistration. The question is whether relative sparse vs
unpruned recovery outcomes persist across temporal permutations of the same
training batches while each source initialization and IMP80 mask stays fixed.
It does not isolate initialization from IMP discovery, establish a mechanism,
or test a new pruning algorithm.

## Cohort and budget

The fixed cohort is 32 experts and source seeds 7, 17, 29, 43, and 61. Each
source uses its original step-0 model and original IMP80 mask. Exactly 20 new
jobs are registered: five sources x two new orders (A=91001, B=91002) x IMP80
and unpruned. Each job has 5,000 attempted updates. Original-order historical
runs are references only and are reevaluated under the common readout; no
original-order reproduction, extra seed, sparsity, mask, or outcome-dependent
extension is allowed.

The 5,000-batch source-specific training stream is reconstructed by the audited
native loader. Its batches are never regenerated from another seed. Order A/B
permute batch indices independently within 0..2499 and 2500..4999 using
NumPy SeedSequence + PCG64 generators spawned from seeds 91001 and 91002.
Permutation arrays are saved and their hashes are authoritative. Batch tensors,
within-batch order, boundaries, and multiplicities are unchanged. The same
permutation indices are applied to every source, and sparse/unpruned jobs
within a source/order consume identical batches. Existing source streams are
recorded as matching or differing across source seeds, not assumed equal.

## Optimization and reset

Every job begins with the exact source step-0 tensors. IMP80 applies only the
source's original IMP mask; unpruned applies no mask. The native router-age
recovery trainer, configuration, trainability, AdamW, clipping, auxiliary
loss, capacity behavior, precision, autocast, GradScaler, persistent gradient
masking, and post-update mask enforcement are retained. The native two-call
semantics are retained: run 1 consumes the first permuted segment; then the
complete model state is checkpointed, a fresh optimizer/scaler is created,
native RNG reset/model assembly is repeated, and run 2 consumes the second
permuted segment. Parameter/component hashes are checked across reset.
Changed batches may change gradients, routes, clipping, and AMP skips; runs are
not equalized afterward.

Native RNG reset policy is the source seed at the start of each segment, as in
the verified historical runner. Batch-order generation uses local PCG64
generators and does not advance global Python, NumPy, CPU Torch, or CUDA Torch
training RNG state. Dropout is configured off in the audited cohort. Any
remaining stochastic native operations and numerical environment are recorded.

## Evaluation and endpoints

Model snapshots are saved at global steps 0, 250, 500, 1000, 1500, 2000, 2500,
2525, 3000, 4000, and 5000. Common evaluation uses the complete validated
population, FP32, TF32 disabled, eval/inference mode, native capacity, and
target-token-weighted cross entropy. Target counts and partial-batch handling
come from materialized source artifacts. Evaluation is isolated from training
RNG/model/optimizer/scaler state. No geometry or activation-substitution
analysis is performed.

Primary endpoint is step 5000; secondary endpoint is step 2500. For each
source/order, gap = sparse CE - unpruned CE; negative means the sparse model
beats its matched comparator under this protocol, not automatically a strict
original-LTH claim. Historical endpoints are used only when compatibility is
demonstrated and are reevaluated with this same readout. No near-match tolerance
is used.

## Dispatch, resources, and resumption

Engineering checks and a preselected seed-17/order-A sparse+dense pilot precede
the remaining jobs. Pilot dispatch checks are numerical, provenance,
endpoint-completeness, and resource checks only; pilot effect signs/match labels
cannot control dispatch. One worker per available GPU uses shared-home locks,
unique attempts, atomic statuses and heartbeats, hash-verified inputs, and
saved model/optimizer/scaler/RNG/batch position for resumption. Failed attempts
are preserved. No process is killed, no run is extended because of outcome,
and incomplete outcomes are reported.

## Interpretation limits

Orders are nested within each fixed source-initialization/IMP-mask pair. The
five source seeds are the replication units; the orders do not add independent
initializations. One mask per source means initialization, mask identity,
discovery trajectory, and source-specific streams cannot be separated. This
study varies temporal order only, within the two fixed optimizer segments; it
does not vary batch composition, data sampling, discovery order, architecture,
sparsity, or all optimization randomness. It cannot identify why a pair is
good.
