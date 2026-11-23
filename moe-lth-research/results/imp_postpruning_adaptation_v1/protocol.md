# Post-pruning router × expert adaptation experiment

## Status and selection

This is a prospective intervention motivated by observed trajectories, not a
preregistered overall study. The global-step 1000–1500 window was selected after
examining prior trajectories; it is therefore not outcome-independent. The mask
cohort is existing, native 32E IMP80, with seeds 7, 17, 29, 43, and 61. Seed 17
is the predetermined engineering pilot. Every seed is retained regardless of
historical or new matching status. No new mask search, source training, sparsity,
architecture, pruning schedule, or training beyond global step 1500 is allowed.

## Conditions and state

At step 1000, compare the abrupt IMP80 post-deletion state with its compatible
unpruned state. Reconstruct the common original native prefix only because saved
step-1000 artifacts do not contain the complete optimizer/scaler/RNG/data state.
Verify reconstructed weights and validation loss against historical checkpoints
using the tolerances below. The abrupt pre-event model is expected to equal the
dense model because both share the same unpruned prefix; test this explicitly.
If this reconstruction cannot be established, stop before production dispatch.

For each seed and model type, all four branches clone one identical complete
branch-start state, including parameters/buffers, AdamW groups/steps/moments,
GradScaler, Python/NumPy/Torch CPU/CUDA RNG, exact next-batch index, event count,
and masks. No optimizer or RNG reset is permitted at this boundary.

| Condition | Router | Experts | Shared |
|---|---|---|---|
| full | train | train | train |
| router_frozen | frozen | train | train |
| experts_frozen | train | frozen | train |
| router_experts_frozen | frozen | frozen | train |

The partition is complete and disjoint. Experts include all routed `fc1`/`fc2`
weights and biases. Tied embedding/output parameters are counted once. All shared
parameters remain trainable. Freezing uses `requires_grad=False`, keeps forward
computation and upstream gradient propagation, preserves optimizer parameter
groups and state, and leaves frozen gradients absent so AdamW neither updates nor
decays those parameters. For sparse models, keep the exact IMP80 mask, masked
gradient hooks, masked-weight enforcement, and native coordinate-shaped moment
zeroing. Native clipping and resulting optimizer responses are part of the
training-policy intervention.

## Schedule and measurements

Each endpoint is exactly 500 attempted updates, global steps 1001–1500. AMP skips
do not extend duration. Keep native learning rate, AdamW betas/epsilon/decay,
gradient clipping, FP16/GradScaler, auxiliary objective, full-softmax/selected
gates, capacity rules, dropout and model modes. All branch starts within a
seed/type receive the same RNG and data position. The exact next 500 original
training batches are used.

Evaluate task-only CE at offsets 0, 25, 100, 250, 500 (global steps 1000, 1025,
1100, 1250, 1500), FP32, TF32 disabled, eval/inference, over identical complete
validation batches including the original partial final batch, target-token
weighted. Require equal branch baseline losses within `atol=1e-06`.
Primary endpoint penalties are `P_R=L_router_frozen-L_full`,
`P_E=L_experts_frozen-L_full`, and `P_RE=L_router_experts_frozen-L_full`, within
each model type. Sparse-excess penalties are `E_R=P_R_sparse-P_R_unpruned` and
`E_E=P_E_sparse-P_E_unpruned`. CE-scale factorial interaction is
`I=L_11-L_10-L_01+L_00`, with 1 indicating adaptation; report sparse, unpruned,
and sparse-minus-unpruned. Also report absolute CE reduction and trapezoidal
integrated excess CE and normalized average excess over the fixed observed grid.
No interpolation beyond linear trapezoidal integration. No fraction-of-pruning-
damage recovery score or primary winner label.

The deterministic diagnostic probe is four complete batches selected by
SHA256 ordering of batch indices from the verified training stream before any
branch outcome is measured. It may overlap training data and is not an
independent test set. Use the same batch structure across times. At every
registered offset record proposed/accepted route loads, dropped assignments,
assignment agreement, accepted-set Jaccard (empty union undefined), selected
gate distributions and gate changes for all tokens, unchanged proposed experts,
and separately proposed/accepted populations. Record expert functions on fixed
branch-start and current native layer-input populations with expert and position
identity; bounded token positions and zero-norm/low-count exclusions are
reported. Record router-input displacement/similarity, component displacement
(attention, normalization, tied embedding/head where applicable), per-update
parameter-change norms by component, unscaled/pre-clip gradient norms, clipping
coefficients, AMP skips/scaler, and path-length versus net displacement.
Diagnostics may not change RNG, modes, gradients, buffers, optimizer/scaler, masks,
or data position. Store streaming summaries and bounded caches, not all gradients
or deltas; report overhead.

## Numerical checks and stopping

Before production verify native continuation equivalence, model restoration,
identical branch-start forwards, bitwise frozen parameters and Adam state,
trainable-state preservation, gradient propagation through frozen modules,
sparse-mask persistence, tied-parameter handling, interruption/resume, batch
sequence and attempted/applied accounting, weighted validation and precision,
and diagnostic side-effect preservation. Run the seed-17 engineering pilot, then
dispatch all remaining seeds after checks pass regardless of scientific effect.
No effect-size or significance gate controls completion. Record model equivalence
as bitwise or tolerance-based; locked tolerances are model allclose
`atol=1e-06, rtol=1e-05` and CE absolute difference `<= 1e-06`. Use one
worker per Viper/Adder host, existing minimum-free-memory admission, no dispatch above 90% GPU utilization, shared
exclusive claims, atomic statuses/heartbeats, unique resumable attempts and
validated completion hashes. Do not kill unrelated jobs.

## Inference

The source seed is the independent unit (`n=5`). Show individual seed effects
first, then mean, median, range, sample SD, and explicitly descriptive seed-level
intervals. No token/expert/batch/checkpoint pseudoreplication, winner classifier,
feature search, or pooled correlation fishing. Freezing estimates the effect of
restricting adaptation under this training policy. Frozen router weights do not
freeze assignments; unchanged assignments do not imply unchanged gates. Component
movement is not a causal contribution to CE. Shared-only improvement does not
show frozen experts are functionally irrelevant. Routing diagnostics do not
isolate assignment, gate, or capacity mechanisms. Results do not establish
pre-pruning recoverability prediction, universality, a pruning algorithm, a
token-routing mechanism, or an explanation for late sparse outperformance.
Late dense deterioration is a separate analysis-only audit. Recommend at most one
falsifiable follow-up; do not launch it.
