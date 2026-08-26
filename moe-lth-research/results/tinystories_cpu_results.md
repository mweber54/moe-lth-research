# TinyStories CPU Pilot Results

## Setup

- Local dataset: `data/TinyStories`
- Training sample: 5,000 stories
- Validation sample: 500 stories
- Tokenization: byte-level, vocabulary size 256
- Model: 2 layers, `d_model=64`, 4 heads, 4 Top-1 experts per layer
- Expert hidden size: 128
- Training: 500 steps, batch size 8, sequence length 64
- Pruning: expert-local one-shot magnitude pruning
- Seed: 7

This is a mechanistic pilot, not a final multi-seed result.

## Dense Performance

| Condition | Validation loss | Perplexity |
|---|---:|---:|
| Strong-balanced learned routing | **2.549** | **12.79** |
| Strong-balanced swapped histories | 2.558 | 12.91 |
| Standard learned routing | 2.584 | 13.25 |
| Random routing every step | 2.681 | 14.60 |

Strong-balanced learned routing reduced loss by about 4.9% relative to random
routing. Swapping experts 0 and 1 changed internal structure while preserving
nearly all dense performance.

## Expert Usage

The standard `aux_loss_weight=0.01` run collapsed:

| Layer | Final normalized usage entropy | Dead experts |
|---|---:|---:|
| Standard learned, layer 0 | 0.506 | 1 |
| Standard learned, layer 1 | 0.194 | 1 |

Increasing `aux_loss_weight` to `0.1` recovered balanced routing:

| Layer | Final normalized usage entropy | Usage max/min | Dead experts |
|---|---:|---:|---:|
| Strong-balanced, layer 0 | 0.968 | 2.19 | 0 |
| Strong-balanced, layer 1 | 0.979 | 1.86 | 0 |

All causal conclusions should therefore use the strong-balanced suite.

## Routing and Mask Causality

Strong-balanced suite at 80% sparsity:

| Comparison | Final routing agreement | Mask Jaccard |
|---|---:|---:|
| Learned vs replay | **1.000** | **1.000** |
| Learned vs swapped | 0.445 | 0.606 |
| Learned vs random-every-step | 0.247 | 0.551 |
| Random-every-step vs swapped | 0.256 | 0.537 |

Across all six condition pairs, routing-history similarity and mask similarity
had correlation **0.985**.

This is preliminary evidence that expert masks track routing history. Replay
exactly reproduced the original masks, while swapping histories changed masks
substantially despite a dense-loss change of only 0.009.

## Pruning Results

### Strong-Balanced Learned Routing

| Sparsity | Magnitude mask | Random mask | Other-expert mask | Random reinit |
|---|---:|---:|---:|---:|
| 50% | **2.561** | 2.947 | 2.757 | 3.409 |
| 80% | **2.844** | 3.320 | 3.058 | 3.398 |
| 90% | **3.185** | 3.380 | 3.239 | 3.398 |

At 50% sparsity, the learned magnitude mask degraded dense loss by only 0.49%.
At 80%, it remained much better than random, transferred, and randomly
reinitialized controls, but degraded dense loss by 11.6%. It does not meet the
plan's 2-5% ticket-performance criterion at 80% sparsity.

### Random Routing Every Step

At 80% sparsity:

| Mask | Validation loss |
|---|---:|
| Magnitude | 2.695 |
| Random | 2.707 |
| Other expert | 2.697 |

The small gap between all masks suggests random-routing experts are relatively
redundant and their sparse mask identity matters little. In contrast, mask
identity matters strongly under learned routing.

## Specialization

For the strong-balanced model:

- Expert ID linear-probe accuracy from pre-router hidden states was 97.6% in
  layer 0 and 92.4% in layer 1.
- Token-distribution Jensen-Shannon divergences were substantial across experts.
- In the expert-substitution matrix, the routed expert was the lowest-loss
  choice for 7 of 8 expert distributions; the remaining case differed by only
  about 0.001 loss.
- Layer-0 expert substitutions caused especially large loss increases.

These results support functional specialization, while the low silhouette
scores show that expert regions are linearly separable but not clean,
well-separated geometric clusters.

## Mask Stability

80%-sparse mask Jaccard versus the final checkpoint:

| Condition | Step 50 | Step 100 | Step 250 |
|---|---:|---:|---:|
| Standard learned | 0.740 | 0.811 | **0.885** |
| Random every step | 0.573 | 0.732 | 0.852 |
| Strong-balanced learned | 0.613 | 0.670 | 0.768 |

Standard learned masks stabilize earlier than random-routing masks, but that
run is confounded by collapse. Strong balancing keeps routing and masks more
dynamic while preserving specialization.

## Interpretation

The pilot supports three claims:

1. Learned routing creates more functionally meaningful sparse expert masks
   than random routing.
2. Replaying routing history reproduces the learned masks exactly.
3. Swapping routing histories changes masks even when dense performance changes
   very little.

It does **not** yet establish lottery tickets because rewind/retrain testing has
not been run on the balanced TinyStories model. Results are also limited to one
seed, a small CPU model, 500 steps, and byte-level tokenization.

