# MoE Routing-Conditioned Gradient Dynamics V1

Single-source temporal pilot for source seed 7, order A, tracking the step-250
qualification checkpoint to step 500. The exact 180 canonical probes from the
completed gradient-consistency study are reused at both checkpoints.

This experiment is observational. It performs no training, optimizer update,
masking, deletion, recovery, architecture change, or router intervention.

## Procedure

1. Reuse the validated step-250 coordinate statistics.
2. Replay the identical 90/90 probe split at step 500 with the frozen pilot
   collector implementation.
3. Compare exact matrix coordinates through fixed-matrix and percentile
   thresholds.
4. Audit temporal routing, exposure, and conditional-gradient changes.
5. Assess conventional-baseline redundancy and exploratory grouped
   prediction.
6. Report matching feasibility for future controlled deletion populations.

## Reproduce

From the repository root:

```bash
.venv/bin/python results/moe_connection_lifetime_v1/admitted.py \
  results/moe_routing_conditioned_gradient_dynamics_v1/run_step500.py
.venv/bin/python results/moe_routing_conditioned_gradient_dynamics_v1/analyze.py
```

The analyzer consumes only saved tensors and checkpoints; it does not launch a
forward pass.

## Important limitations

The target-domain loss attribution can depend on accepted context tokens whose
domain differs from the target's domain. The requested acceptance-probability
times conditional-gradient factorization is therefore retained only as an
explicit descriptive reweighting proxy. Its exact identity is reconstructed
algebraically, but it is not an exact causal routing decomposition. A causal
claim would require a frozen-checkpoint router intervention.

Coordinate, expert, and matrix rows are not independent replications. This
single-source study cannot establish cross-seed predictive performance.
