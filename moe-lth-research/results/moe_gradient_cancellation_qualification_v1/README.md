# MoE Gradient Cancellation Qualification V1

This directory is a pilot-only signal-qualification study for `seed7_A`,
checkpoint step 250. It measures domain-conditioned task gradients without
masking, pruning, updating parameters, or launching recovery.

## Scope

The replay uses the exact 180 pairwise-disjoint archived training probes frozen
by `moe_group_gradient_consistency_v1`, divided 90/90 into independent
discovery and validation halves. A single native forward is evaluated for each
probe. TinyStories and WikiText-103 target gradients are attributed from that
shared forward, preserving native top-1 routing, gate weighting, and capacity
acceptance.

All 256 expert `fc1.weight` and `fc2.weight` matrices in the 4-layer, 32-expert
model are measured. Existing sign-only and absolute-magnitude-only artifacts
are retained as provenance controls but are not used to infer signed means or
cancellation.

## Reproduce

From the repository root:

```bash
.venv/bin/python results/moe_connection_lifetime_v1/admitted.py \
  results/moe_gradient_cancellation_qualification_v1/run_pilot.py
.venv/bin/python results/moe_gradient_cancellation_qualification_v1/analyze_pilot.py
```

The replay writes one reusable coordinate-statistics file per half. It stores
domain means and target-weighted probe second moments, not full per-token or
per-probe gradients.
Within each saved matrix tensor, coordinates use PyTorch row-major flattening.
The matching `names` and `shapes` arrays identify every tensor; coordinate `j` is
therefore `(name, flat_index=j)`, convertible with `torch.unravel_index`.

## Core statistics

For half `h`, domain `d`, and coordinate `i`:

- `mu_h,d[i]` is the target-weighted mean task gradient.
- `G_h[i] = p_h,1 mu_h,1[i] + p_h,2 mu_h,2[i]`.
- `A_h[i] = p_h,1 |mu_h,1[i]| + p_h,2 |mu_h,2[i]|`.
- `C_h[i] = 1 - |G_h[i]| / A_h[i]`.
- `K_h[i] = A_h[i] - |G_h[i]|`.
- `probe_second_moment_h,d[i]` is the target-weighted mean of squared
  domain-probe gradients. It is distinct from `mu_h,d[i]^2`.

`p_h,d` is the known-domain target fraction in that half. Unknown separator
targets are reported but excluded from the two-domain reconstruction.

## Required outputs

- `artifact_audit.md`
- `numerical_validation.json`
- `cell_summary.csv`
- `magnitude_bin_summary.csv`
- `split_half_reproducibility.csv`
- `metric_correlations.csv`
- `activity_cancellation_distribution.png`
- `cancellation_by_magnitude.png`
- `pilot_verdict.md`
- `coordinate_statistics_half{1,2}.pt`

The pilot stops after these outputs. It does not authorize the other eleven
cells or any intervention.
