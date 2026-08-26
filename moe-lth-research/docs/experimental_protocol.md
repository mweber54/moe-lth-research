# Experimental Protocol

## Phase 1: Establish Sparse Expert Structure

1. Train `normal` and `random_every_step` with identical model, data order,
   optimizer, and seed.
2. Compare fixed-validation routing stability, usage entropy, dead experts,
   expert-local loss, and context samples.
3. Extract expert-local magnitude masks at 50%, 70%, 80%, 90%, and 95%.
4. Compare dense, magnitude, random, transferred-expert, and randomly
   reinitialized controls.
5. Compare same-expert masks across checkpoints and conditions using Jaccard.

Evidence at this phase supports sparse expert specialization, not LTH.

## Phase 2: Test Lottery-Ticket Behavior

1. Extract a final trained expert-local mask.
2. Rewind the masked model to initialization, 1%, 5%, and 10%.
3. Retrain with pruned weights permanently fixed at zero.
4. Compare learned mask, random mask, learned mask with random expert
   reinitialization, and learned mask under randomized routing.
5. Report full validation and expert-local loss degradation from dense.

Only successful rewind/retrain conditions support a lottery-ticket claim.

## Phase 3: Test Routing Causality

Run `normal`, `fixed_random`, `random_every_step`, `replay`, `swapped`,
`shuffled_usage`, and `strong_balance`. Keep seed, data order, architecture,
optimizer, and schedule fixed.

The strongest causal pattern is:

- Replay resembles normal.
- Swap makes masks follow swapped histories.
- Usage-preserving shuffle differs from normal.
- Random-every-step weakens stability/specificity.
- Strong balancing preserves the effect.

## Phase 4: Robustness

Use at least three seeds. Vary expert count, depth, load-balancing strength,
dataset text file, and sparsity. Top-2 is intentionally a later extension.

## Metrics and Artifacts

| Claim | Primary artifact |
|---|---|
| Routing settles | `validation_routes_step_*.npz`, routing-stability figure |
| Usage is not the whole effect | usage entropy/CV and shuffled-usage condition |
| Experts specialize | expert-local loss, substitution matrix, token JS divergence |
| Sparse structure matters | pruning curves and random-mask controls |
| Early weights matter | rewind suite and random-reinit control |
| History causes masks | replay/swap pairwise routing agreement vs mask Jaccard |
| Geometry explains masks | router-vector cosine similarity vs mask similarity |

## Reproducibility Rules

- Do not compare conditions with different data order or seeds unless that is
  the named ablation.
- Use the exact same fixed validation set at every checkpoint.
- Preserve `resolved_config.yaml`, checkpoints, and compressed route logs.
- Report all configured seeds and negative results.
- Treat a failed initialization rewind but successful early rewind as practical
  LTH, not strict LTH.

