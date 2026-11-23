# Input audit

Verified five complete own initializations, actual nested IMP rounds1..8 and final
joint-per-expert fc1+fc2 keep budgets. Per-tensor allocations are preserved; no
bias/router/shared mask entries. All native parameters remain trainable. Original
component hashes and exact data hashes are recorded. Native top1 selected gates
retain full-softmax probabilities; capacity accepts highest gate priorities, dropped
tokens have zero expert contribution; aux uses detached proposed loads and full
probability means. No routing substitution or router-age sweep is used.

Validation comprises31 batches:30 complete128×128 plus7×128 targets in the last
batch,492416 target tokens total. Historical readouts used TF32; all reused states
require common FP32 TF32-off re-evaluation, never direct historical CE comparison.
Compatible references are ten high-sparsity reruns with the exact two-call reset,
complete initialization and stream identities. Their model-only snapshots support
readout, not optimizer continuation. Missing grid points/diagnostics stay missing.
Each final endpoint is counted once; historical aliases are not replications.

Native clipping occurs after gradient hooks/unscale. Dynamic mask hooks require
explicit updates because native hooks capture a tensor at registration. The new
adapter retains the native training function code and overrides its masking and
readout/serialization globals; engineering equivalence is required before launch.
There are exactly10 new5000-attempt training jobs and10 reused controls.
