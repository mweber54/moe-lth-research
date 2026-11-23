# Missing Evidence

## Can Compute Retrospectively

- Final router parameter distances using saved initial/final router checkpoints, if a checkpoint reader is used; no intermediate recovered router checkpoints were found.
- Loss-gap, drift, routing-distribution, token-drop, and gradient-norm trajectories included in the CSVs.

## Requires Diagnostic Rerun

- Token-level assignment identities on the shared probe, including per-layer agreement with R0/R20/R100 over recovery.
- Intermediate router state checkpoints for parameter trajectories and distances to reference routers.
- Frozen gradient-validity, AMP overflow, and capacity-overflow fields compatible with the trainable diagnostic version.

## Requires Causal Intervention

- Hold router weights fixed while replaying R20's evolving assignments.
- Constrain R0/R100 to R20's layerwise assignment churn, entropy, or utilization trajectory.

Minimum diagnostic run: repeat only the existing 85% R0/R20/R100 sparse and dense recoveries for one representative seed with checkpoint snapshots and deterministic shared-probe token-level routing outputs at steps 0, 50, 100, 250, 500, 1000, 1500, 2000, and 2500.
