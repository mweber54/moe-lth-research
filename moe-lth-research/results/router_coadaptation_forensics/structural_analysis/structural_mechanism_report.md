# Structural Mechanism Report

## Result

**H7 is the best-supported hypothesis: existing evidence is insufficient to determine whether R20 makes the right token-expert assignments.** The saved routing records are aggregate expert distributions/counts per layer at each checkpoint. They do not contain token identities, per-token assignments, per-token probabilities, or intermediate recovered router checkpoints.

## What Can Be Tested

- Timing is available at 50-step resolution. The parent report establishes an aggregate mean R20 advantage after approximately steps 500-1000, but `seedwise_mechanism_comparison.csv` shows seed 17 does not favor R20 at the final checkpoint; do not assume identical timing across seeds.
- `layer_assignment_churn.csv` reports per-layer distribution TV/JS, utilization-entropy, and drop changes for 0-500, 500-1000, 1000-2500, and 0-2500. These are structural distribution shifts, not assignment retention/churn.
- Aggregate 500-1000 top-1 distribution shifts do not isolate R20: its TV is lower than R0 in layers 0-2; layer 3 is higher than R0/R100 on average, but this repeats in only one seed.
- `expert_token_flow.csv` reports net accepted-token load changes from 500 to 1000. It cannot identify gains/losses/retention of token sets.
- `sparse_dense_structural_difference.csv` isolates whether each layer's aggregate shift is larger in sparse recovery than dense recovery.

## Hypotheses

- H1 Adaptation magnitude: weakened; prior results show R0 drifts/churns at least as much.
- H2 Targeted routing reorganization: untestable without token-level assignments.
- H3 Routing stabilization: untestable without token trajectories through 500, 750, 1000, and 1500.
- H4 Load correction: only net load changes are available; no damage/recoverability proxy linked to expert identity was saved. Insufficient.
- H5 Gradient alignment: only aggregate/per-layer gradients are available; they are descriptive and frozen validity flags are absent. Insufficient.
- H6 Distributed multi-layer effect: aggregate layer shifts do not show a seed-consistent R20-specific layer, and cannot establish coordinated causal behavior.
- H7 Existing evidence insufficient: supported.

**R20 does not recover better because it adapts more; it recovers better because it cannot be determined from the existing artifacts whether it makes targeted, persistent, sparse-specific assignment changes during approximately steps 500-1000.**

Minimum diagnostic rerun, not launched: instrument the existing matched 85% R0/R20/R100 sparse and dense recoveries to save shared-probe token IDs, layerwise top-1 assignments, top-2 probabilities/logits at steps 0, 250, 500, 750, 1000, 1500, and 2500, plus recovered router checkpoints at those steps and valid per-layer expert gradients.
