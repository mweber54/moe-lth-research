# Source artifact audit

## Reused completed artifacts

- `moe_gradient_cancellation_qualification_v1/configuration.json`
- `moe_gradient_cancellation_qualification_v1/coordinate_statistics_half{1,2}.pt`
- `moe_gradient_cancellation_qualification_v1/numerical_validation.json`
- `moe_group_gradient_consistency_v1/indices_seed7_A.pt`
- `moe_group_gradient_consistency_v1/split_half_gradients_seed7_A_250_h{1,2}_original.pt`
- Verified step-250 and step-500 checkpoints in
  `moe_connection_lifetime_v1/prefixes/seed7_A/`

## Available measurements

The completed qualification files provide for each of 256 expert matrices and
67,108,864 coordinates:

- signed domain mean task gradient for each half,
- target-weighted empirical probe-level squared-gradient moment,
- current weight magnitude through checkpoint reload,
- accepted/proposed target counts, contributing batches, and gate statistics
  by domain/layer/expert,
- activation, routing-activation, RAMP, and normalized RAMP neuron statistics.
- A supplemental no-gradient replay adds per-domain requested/accepted router
  entropy, margin, and gate summaries for the identical 180 probes at both
  steps in `routing_entropy_dynamics.csv`.

The earlier gradient-consistency study additionally contains final-layer
`fc2` sign consistency and absolute batch-gradient magnitude sums. It cannot
recover signed means, squared moments, or `fc1` statistics, so those earlier
aggregates are provenance controls only.

## Additional replay required

Step-500 signed means and moments are not saved. One replay is required using
the exact same batch IDs, model, native routing, capacity, gates, task-loss
normalization, and no-update collector.

The source checkpoint family contains step-250 and step-500 states but no
additional intermediate state. No intermediate training was launched.

## Temporal coordinate identity

Both checkpoints report the same 256 names and tensor shapes. Expert and
coordinate identity is tracked by parameter name plus row-major flat index.
The model, optimizer, scaler, scheduler, Python RNG, CPU RNG, CUDA RNG, and
NumPy RNG are loaded only as frozen payloads; no optimizer is instantiated and
no update is performed.

## Attribution limitation

The domain mean is attributed to the target domain of the cross-entropy term.
Because causal Transformer attention lets each target depend on earlier
tokens, accepted routes contributing to its gradient need not share that
target's source domain. Expert target acceptance frequencies therefore cannot
in general define an exact
`acceptance probability × conditional contribution` decomposition of the
target-attributed gradient. The requested factorization is reported as a
reweighting diagnostic and explicitly not interpreted causally.
