# Metric definition audit

## What the source diagnostic calls a dropped token

`RouteTrace.accepted_mask` has shape `(flattened batch positions, top_k)` and
marks accepted token–expert assignments. The source `MoEFeedForward` computes
capacity over the complete flattened input batch (`batch_size * sequence_length`)
at each layer. It does not compute capacity independently per sequence. For each
expert, assignments above capacity are ordered by the selected gate confidence
(unless an explicit `RouteOverride.capacity_scores` is supplied) and only the
highest-confidence `capacity` assignments are accepted. For the registered
top-1 model, a dropped assignment is exactly a token position whose single
selected assignment is rejected. In the shared-only probe implementation,
`dropped_assignment_rate = 1 - accepted_assignments / proposed_assignments`;
`dropped_token_rate = mean(~accepted_mask.all(dim=-1))`, with all flattened
positions as the token denominator. With top-1 these rates coincide. These
CSV fields are counts/rates on the sealed training probe, not validation-set
measurements.

The attached shared-only CSV records those counts and rates, but the columns do
not themselves encode the capacity formula or dispatch boundary; those are
audited against `src/moe_lth/models/moe_layer.py`. A batch is one complete
`model(inputs)` call. Splitting a batch for capacity accounting would change
the acceptance context and is prohibited.

## Dropped token computation

The expert-loop output starts at zero. A rejected assignment makes no selected
expert contribution. The transformer block still adds the pre-MoE `hidden`
residual to the attention output (`hidden + expert_output`); the residual is
not dropped or scaled. No-drop preserves each selected expert and its original
softmax gate, does not redistribute or renormalize, and recomputes subsequent
layers from the changed hidden states.

## Validation and loss

For each policy, record batch summed task NLL and target count. Full-validation
CE is total NLL divided by total targets (492,416), preserving all 31 original
batches, including the final partial batch. Auxiliary loss is reported
separately and is never added to task CE. Prediction JS, top-1 agreement, and
rejected-position contrasts compare NO_DROP_ALL with the same-checkpoint native
baseline. Native rejection masks are fixed before policy comparison. Rejected
subsets can overlap across layers and are not additive loss decompositions.

## Estimands

`G_native = L_sparse,native - L_unpruned,native`;
`G_nodrop = L_sparse,nodrop - L_unpruned,nodrop`;
`B_sparse = L_sparse,native - L_sparse,nodrop`;
`B_unpruned = L_unpruned,native - L_unpruned,nodrop`;
`K = G_native - G_nodrop = B_sparse - B_unpruned`.
`H_native = G_native(0) - G_native(500)`;
`H_nodrop = G_nodrop(0) - G_nodrop(500)`;
`J = H_native - H_nodrop = K(0) - K(500)`.
Layer-local `B`, `K`, and start-to-end change are secondary and must not be
summed across layers. All signs are retained.

## Routing denominators and scopes

Assignment rates use proposed assignments as the denominator; token rates use
flattened token positions. Loads are per expert within one full original batch.
Capacity and maximum proposed load, load CVs, assignment agreement, accepted-set
Jaccard, accepted/rejected gate summaries, and coverage are layer-level values
repeated on expert rows for convenience; the row has an explicit scope field.
Expert proposal/acceptance/rejection counts are expert-level. Undefined
zero-denominator metrics remain blank and have a corresponding flag.
