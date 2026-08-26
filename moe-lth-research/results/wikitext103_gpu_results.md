# WikiText-103 Subset GPU Results

## Run setup

- Hardware: NVIDIA GeForce RTX 3080 Laptop GPU, 16 GB VRAM
- Config: `configs/wikitext103_gpu.yaml`
- Data: `data/wikitext103_subset/wikitext103_train.txt` and `wikitext103_validation.txt`
- Training corpus: 2,936,883 bytes; validation corpus: 246,391 bytes
- Model: 17,981,952 parameters, 4 layers, 8 experts, top-1 routing
- Training: FP16, batch 128, sequence length 128, 2,500 steps
- Budget: 40.96 million tokens per condition
- Validation: fixed 12-block subset

The full routing/pruning suite took about 32 minutes. Each 16-run rewind suite
took about 138 minutes. Across the routing suite and both rewind suites, the GPU
processed approximately 1.47 billion training tokens.

## Dense routing conditions

| Condition | Validation loss | Perplexity | Delta vs normal |
|---|---:|---:|---:|
| Normal learned routing | 1.6817 | 5.3749 | - |
| Random routing every step | 2.1213 | 8.3421 | +26.14% |
| Replay normal routes | 1.6817 | 5.3749 | 0.00% |
| Swap experts 0 and 1 | 1.7127 | 5.5437 | +1.84% |

Replay exactly reproduces the normal run. Random routing substantially hurts
performance, while swapping a pair of route identities causes a smaller but
measurable penalty.

## Normal-run pruning

| Sparsity | Magnitude mask | Random mask | Other-expert mask | Magnitude mask + random reinit |
|---:|---:|---:|---:|---:|
| 0% dense | 1.6817 | - | - | - |
| 50% | 1.7331 | 4.3020 | 2.0216 | 8.5632 |
| 70% | 2.5098 | 6.6151 | 2.9254 | 8.4978 |
| 80% | 4.8942 | 7.6956 | 5.1422 | 8.5709 |
| 90% | 8.4087 | 8.2572 | 8.5496 | 8.5923 |
| 95% | 8.8405 | 8.3615 | 8.9272 | 8.6298 |

At 50% sparsity, the learned magnitude mask is only 3.06% worse than dense and
dramatically better than every control. Direct pruning is too destructive at
80% and above, making rewind-and-retrain testing necessary.

## Rewind results

### 50% sparsity

| Rewind fraction | Learned mask | Random mask | Random reinit | Randomized routing |
|---:|---:|---:|---:|---:|
| 0% / initialization | 1.6672 | 1.7408 | 1.7499 | 2.1595 |
| 1% | 1.6421 | 1.7149 | 1.7341 | 2.1603 |
| 5% | 1.6150 | 1.7025 | 1.7210 | 2.0898 |
| 10% | **1.6100** | 1.6701 | 1.6799 | 2.0743 |

The 50% learned mask qualifies as a strict lottery ticket: even initialization
rewinding beats the dense final model by 0.87%. Rewinding to 10% training beats
dense by 4.27%.

### 80% sparsity

| Rewind fraction | Learned mask | Random mask | Random reinit | Randomized routing |
|---:|---:|---:|---:|---:|
| 0% / initialization | 1.8329 | 1.9163 | 1.9219 | 2.1915 |
| 1% | 1.8680 | 1.9462 | 1.8943 | 2.1674 |
| 5% | 1.7377 | 1.8800 | 1.8597 | 2.1164 |
| 10% | **1.6953** | 1.8414 | 1.8338 | 2.1021 |

At the plan's target 80% sparsity, the learned mask rewound to 10% training is
only 0.81% worse than dense. The same final mask without retraining has loss
4.8942. Initialization rewind remains 8.99% worse than dense, so this is strong
practical modern-LTH evidence, but not strict initialization-LTH evidence at
80%.

## Routing and mask evidence

- Final normal routing has zero dead experts in every layer.
- Final normalized usage entropy is 0.9870, 0.9975, 0.9994, and 0.9907.
- First-to-final routing agreement is 0.1827 overall; chance is 0.125.
- Normal vs random routing: route agreement 0.1253, mask Jaccard 0.4557.
- Normal vs replay: route agreement 1.0000, mask Jaccard 1.0000.
- Normal vs swapped: route agreement 0.7465, mask Jaccard 0.6024.
- Routing-history agreement and mask similarity correlate at 0.8656.

The 80% normal mask's Jaccard overlap with its final mask rises from 0.5795 at
initialization to 0.6518 at step 1,000, then reaches 1.0 at step 2,500. Under
random routing it stabilizes earlier, reaching 0.7883 by step 1,000.

## Expert specialization

| Layer | Linear probe accuracy | Silhouette score | Own expert is best |
|---:|---:|---:|---:|
| 0 | 0.9960 | 0.0563 | 8 / 8 |
| 1 | 0.9176 | 0.0103 | 8 / 8 |
| 2 | 0.8208 | -0.0516 | 4 / 8 |
| 3 | 0.7936 | -0.0235 | 2 / 8 |

Expert substitution is strongly diagonal in the first two layers and much
weaker in later layers. This supports real expert-specific computation, while
also showing that specialization is depth-dependent.

## Conclusion

The WikiText-103 subset replication supports the execution plan's central
hypothesis:

1. Learned routing produces expert-specific sparse subnetworks that are not
   reproduced by random masks, random reinitialization, or randomized routing.
2. A 50% learned expert-local mask is a strict lottery ticket.
3. An 80% learned mask is a practical early-rewind lottery ticket.
4. Mask similarity strongly follows routing history, and exact routing replay
   exactly reproduces the learned masks and model result.

The main limitations are the single seed, byte-level tokenizer, small 2.94 MB
training subset reused for about 14 passes, and fixed 12-block validation
sample. The checkpoint specialization analysis also emitted logistic-regression
convergence warnings, so its probe accuracies should be treated as approximate.

## Artifacts

- Suite summary: `results/wikitext103_gpu_suite/suite_summary.json`
- Routing/mask analysis: `results/wikitext103_gpu_suite/tables/analysis_report.json`
- Checkpoint specialization: `results/wikitext103_gpu_suite/normal/tables/checkpoint_analysis.json`
- 50% rewind suite: `results/wikitext103_gpu_suite/normal/tables/rewind_suite_sparsity_0.5.json`
- 80% rewind suite: `results/wikitext103_gpu_suite/normal/tables/rewind_suite_sparsity_0.8.json`
- Suite figures: `results/wikitext103_gpu_suite/figures`
- Condition pruning figures: `results/wikitext103_gpu_suite/*/figures/pruning_curves.png`
