# Project Summary So Far

**Project:** Routing-conditioned lottery tickets in Mixture-of-Experts models  
**Report date:** July 2, 2026  
**Detailed report:** [execution_plan_1_results.md](execution_plan_1_results.md)  
**Execution plan:** [../moe_lth_execution_plan.md](../moe_lth_execution_plan.md)

## Executive Summary

We have completed the main MVP, lottery-ticket, causal-control, load-balance,
cross-initialization, architecture-robustness, dataset-robustness,
representative Phase 4 rewind, balanced multi-domain causal-control, broader
swap-intervention, subword-tokenized causal-control, long-budget multi-domain
causal-control, checkpoint-aware long-budget pruning, reduced long-budget
rewind-control suites, representative iterative magnitude pruning, and a
larger-validation long-budget check. The project has moved from "do sparse
expert subnetworks exist?" to a stronger and more nuanced answer:

> Small balanced MoE language models contain expert-local sparse subnetworks
> whose usefulness depends on learned mask structure, routed token histories,
> early training weights, and initialization. Routing history causally shapes
> masks within an initialization, but routing alone does not determine a
> portable coordinate-level mask across independently initialized models.

The strongest current claim is therefore not a pure routing-only law. The best
supported mechanism is:

```text
initialization x routing trajectory -> sparse expert mask
```

The strongest empirical result is at high sparsity after early rewinding.
Across the original TinyStories and WikiText suites, 80% sparse learned masks
rewound to 10% of training recover near-dense performance and beat random-mask,
random-reinit, and randomized-routing controls. The new representative Phase 4
rewind suite extends this result to top-2, deeper, high-capacity, and
multi-domain settings.

## Current Verdicts

| Question | Current verdict | Short answer |
|---|---|---|
| Do MoE experts specialize? | Supported, depth-dependent | Early layers show strong expert-ID separability and diagonal functional specificity; later layers are weaker. |
| Do experts contain sparse subnetworks? | Supported | Magnitude masks beat random masks and other-expert masks across datasets and architecture cells. |
| Are these subnetworks lottery tickets? | Supported with nuance | 50% masks often work from initialization; 80% masks generally need early checkpoint rewinding, though the long-budget best-checkpoint suite produced an 80% initialization ticket. |
| Does routing history matter? | Supported | Random-every-step and shuffled-usage controls damage performance and mask specificity. |
| Are usage counts enough? | Rejected | Shuffled usage exactly preserves expert counts but performs much worse. |
| Is learned router adaptation required? | Not strictly | Fixed-random routing is worse than learned routing, but still produces useful tickets. |
| Does routing alone determine masks? | Not supported | Cross-init replay reproduces route history exactly but does not reproduce source masks. |
| Does the result generalize beyond the first setting? | Supported at current scale | Results survive seeds, TinyStories/WikiText, 4/8/16 experts, top-1/top-2 routing, 4/8 layers, multi-domain validation, multi-domain causal controls, broader swap interventions, and a subword-tokenized setting. The 10k-step suite adds an important checkpoint-selection warning: final checkpoints can mislead, but best-checkpoint analysis restores the normal-routing advantage. |

## Experiments Completed

| Experiment family | Status | Main outcome |
|---|---|---|
| CPU TinyStories pilot | Complete | Found expert collapse under weak load balancing; `aux_loss_weight=0.1` fixed dead experts and became the GPU baseline. |
| Baseline MoE implementation | Complete | Decoder-only byte-level MoE Transformer, routing logs, expert usage logs, pruning, analysis, rewind, replay, swap, fixed-random, shuffled-usage, and plotting pipelines are implemented. |
| TinyStories GPU suite | Complete, 3 seeds | Normal routing strongly outperforms random routing; tickets recover at 50% from initialization and at 80% from 10% rewind. |
| WikiText-103 subset GPU suite | Complete, 3 seeds | Same qualitative pattern as TinyStories, with stronger 50% initialization-ticket behavior. |
| Direct pruning controls | Complete | Learned magnitude masks beat random masks and other-expert masks at 50% and 80% in the main routing conditions. |
| Rewind and lottery-ticket validation | Complete | 50% masks satisfy strict initialization-rewind criteria; 80% masks satisfy practical early-rewind criteria. |
| Replay routing | Complete | Same-initialization replay exactly reproduces normal loss, routes, and masks. |
| Swapped routing | Complete, limited | Expert swaps change masks and mildly harm loss, supporting routing-history dependence. |
| Fixed-random routing | Complete, 3 seeds on TinyStories and WikiText | Frozen random routing is worse than learned routing but still produces useful sparse tickets. |
| Shuffled-usage routing | Complete, 3 seeds on TinyStories and WikiText | Exact expert usage counts do not preserve performance or mask specificity. |
| Cross-initialization replay | Complete on WikiText targets 17 and 29 | Foreign route histories reproduce routes but not source masks, and are worse than matched-data learned routing. |
| Functional cross-init alignment | Complete | Output CKA and neuron-correlation matching barely change cross-init mask Jaccard. |
| Cross-init replay rewind | Complete | Foreign-route masks are rewindable, but matched-data learned routing remains better. |
| Load-balancing sweep | Complete | Stronger balancing improves dense loss and usage entropy; learned masks still beat random masks. |
| Phase 4 architecture grid | Complete | 36 WikiText architecture runs across 4/8/16 experts, top-1/top-2, and 4/8 layers. |
| Multi-domain validation | Complete | Balanced TinyStories/WikiText validation reproduces the basic non-random mask result. |
| Representative Phase 4 rewind suite | Complete | 288/288 retrainings completed across 3 representative cells, 3 seeds, 2 sparsities, 4 rewind points, and 4 controls. |
| Balanced multi-domain causal controls | Complete | Fixed-random, random-every-step, and shuffled-usage controls completed across seeds 7, 17, and 29; shuffled usage exactly matched normal expert counts but was 33.72% worse than normal. |
| Broader swap interventions | Complete | Six balanced multi-domain swap interventions completed across seeds 7, 17, and 29; all swaps worsened dense loss, from +0.65% for layer-3 0/1 swap to +3.77% for a global cyclic expert shift. |
| Subword-tokenized causal controls | Complete | A 1024-ID byte-ngram subword tokenizer replicated the causal-control pattern: fixed-random was +6.97%, random-every-step was +32.83%, and shuffled usage was +33.04% worse than normal. |
| Long-budget multi-domain causal controls | Complete | The 10k-step, 64-validation-block suite completed across all three seeds and four causal conditions. Final-checkpoint normal loss was worse than the controls because normal overfit late. |
| Long-budget best-checkpoint pruning | Complete | Best-saved-checkpoint comparison restored the expected causal pattern: normal routing was best, fixed-random was close but worse, and randomized/shuffled routing were much worse. |
| Reduced long-budget rewind controls | Complete at 0% rewind | 80% normal masks from best saved checkpoints retrained from initialization to `1.2984 +/- 0.0055`, 0.68% better than best-saved dense. They narrowly beat random-mask and random-reinit controls, and strongly beat randomized routing. |
| Representative IMP | Complete | Four-round expert-local IMP on the balanced multi-domain 8E/top-1/4L cell reached `1.5687 +/- 0.0365` at 80% sparsity, 5.99% worse than dense and worse than the previous one-shot 80%/10%-rewind result. |
| Long-budget larger-validation check | Complete | On a larger held-out validation file, the close 80% initialization-rewind comparison still favored the learned mask: `1.3096 +/- 0.0044` versus dense `1.3213`, random mask `1.3242`, and random reinit `1.3225`. |

## Main Quantitative Findings

### TinyStories Multi-Seed Suite

Source: [tinystories_gpu_multiseed/multiseed_results.md](tinystories_gpu_multiseed/multiseed_results.md)

| Result | Value |
|---|---:|
| Normal dense loss | `1.0206 +/- 0.0096` |
| Random-every-step loss | `1.4586 +/- 0.0153`, 42.91% worse than normal |
| Fixed-random loss | `1.1105 +/- 0.0184`, 8.80% worse than normal |
| Shuffled-usage loss | `1.5316 +/- 0.0151`, 50.06% worse than normal |
| Swapped loss | `1.0491 +/- 0.0111`, 2.79% worse than normal |
| Replay loss | exactly equal to normal in all seeds |
| 50% initialization-rewound learned mask | 1.55% worse than dense |
| 80% learned mask at 10% rewind | within 2.03% of dense |
| Routing-history/mask-similarity correlation | `0.7559 +/- 0.0122` |

Interpretation: TinyStories supports routing-conditioned sparse structure, but
at 80% sparsity the ticket claim needs early checkpoint rewinding rather than
strict initialization rewinding.

### WikiText-103 Multi-Seed Suite

Source: [wikitext103_gpu_multiseed/multiseed_results.md](wikitext103_gpu_multiseed/multiseed_results.md)

| Result | Value |
|---|---:|
| Normal dense loss | `1.6859 +/- 0.0160` |
| Random-every-step loss | `2.1121 +/- 0.0080`, 25.28% worse than normal |
| Fixed-random loss | `1.7386 +/- 0.0252`, 3.13% worse than normal |
| Shuffled-usage loss | `2.1337 +/- 0.0191`, 26.56% worse than normal |
| Swapped loss | `1.7081 +/- 0.0151`, 1.32% worse than normal |
| Replay loss | exactly equal to normal in all seeds |
| 50% initialization-rewound learned mask | 1.58% better than dense |
| 80% learned mask at 10% rewind | within 0.13% of dense |
| Routing-history/mask-similarity correlation | `0.7785 +/- 0.0275` |

Interpretation: WikiText gives the cleanest original evidence for true
lottery-ticket behavior: 50% masks beat dense from initialization, and 80%
masks recover near-dense performance after 10% rewinding.

### Direct Pruning and Mask Structure

Magnitude masks consistently beat random masks and transferred masks. In the
main WikiText suite, normal-routing 50% magnitude pruning gives `1.7388`
versus `4.0564` for random masks. In TinyStories, normal-routing 50% magnitude
pruning gives `1.0676` versus `3.2700` for random masks.

This establishes that the learned mask structure matters. It is not just the
sparsity level, the number of active weights, or a generic expert architecture
effect.

### Rewind and Lottery-Ticket Criteria

In the original 3-seed suites:

| Dataset | 50% init ticket | 80% init ticket | 80% 10%-rewind ticket |
|---|---:|---:|---:|
| TinyStories | Yes, within 1.55% of dense | No, 12.60% worse | Yes, within 2.03% |
| WikiText-103 | Yes, 1.58% better than dense | No, clearly worse | Yes, within 0.13% |

Interpretation: moderate-sparsity expert tickets can survive from
initialization. High-sparsity tickets are real in the practical modern LTH
sense, but they need early weights.

### Causal Routing Controls

The causal controls separate routing history from usage count and router
adaptation:

| Control | Finding |
|---|---|
| Random every step | Destroys stable routed-token histories and substantially worsens dense and rewind performance. |
| Replay | Same initialization plus same routes exactly reproduces normal loss, routing, and masks. |
| Swapped routing | Mildly worsens loss and shifts masks, suggesting masks follow routed histories within an initialization. |
| Fixed random | Stable random partitions still train useful tickets, but learned routing is better. |
| Shuffled usage | Preserves expert counts exactly, but nearly removes learned-mask advantage and badly worsens loss. |

The shuffled-usage result is especially important: update counts and load
balance are not enough. Token identity and consistent routed-token
distributions matter.

### Cross-Initialization Replay

Source: [wikitext103_cross_init_replay/cross_init_replay_results.md](wikitext103_cross_init_replay/cross_init_replay_results.md)

Cross-init replay forces seed 7 routes onto target seeds 17 and 29. It exactly
reproduces the source route history, but it is 6.54% worse than matched-data
learned routing. Source-vs-cross-init mask Jaccard is not higher than
source-vs-matched-data Jaccard:

| Sparsity | Source vs matched-data learned | Source vs cross-init replay |
|---:|---:|---:|
| 50% | 0.3747 | 0.3733 |
| 80% | 0.1931 | 0.1874 |

Interpretation: routing history is causal, but it does not act as a portable
mask blueprint across independently initialized coordinate systems.

### Functional Cross-Init Alignment

Source: [wikitext103_functional_alignment/functional_alignment_results.md](wikitext103_functional_alignment/functional_alignment_results.md)

Matching experts by output CKA and neurons by activation correlation changes
Jaccard by less than 0.001 in practice. This weakens the concern that the
cross-init mismatch is simply a trivial expert or neuron permutation.

### Cross-Init Replay Rewind

Source: [wikitext103_cross_init_rewind/cross_init_rewind_results.md](wikitext103_cross_init_rewind/cross_init_rewind_results.md)

Foreign-route masks are still rewindable:

| Condition | Sparsity | Best rewind | Mean loss | Delta vs own dense |
|---|---:|---:|---:|---:|
| cross_init_replay | 50% | 10% | 1.7018 | -4.78% |
| cross_init_replay | 80% | 10% | 1.7977 | +0.59% |
| matched_data_learned | 50% | 10% | 1.5944 | -4.95% |
| matched_data_learned | 80% | 10% | 1.6844 | +0.42% |

Interpretation: a foreign route history can induce internally useful sparse
tickets, but it is worse than the route history that co-adapts with the
target initialization and data stream.

### Fixed-Random Rewind

Source: [wikitext103_fixed_random_rewind/fixed_random_rewind_results.md](wikitext103_fixed_random_rewind/fixed_random_rewind_results.md)

Fixed-random dense WikiText loss is `1.7386 +/- 0.0206`. Yet its learned masks
rewind well:

| Sparsity | Best rewind | Mean loss | Difference from fixed-random dense |
|---:|---:|---:|---:|
| 50% | 10% | 1.6315 | -6.16% |
| 80% | 10% | 1.7165 | -1.27% |

Interpretation: learned router adaptation is helpful but not required for
ticket formation. Stable routing partitions, even random ones, can induce
usable sparse expert subnetworks.

### Load-Balancing Sweep

Source: [load_balance_sweep/load_balance_sweep_results.md](load_balance_sweep/load_balance_sweep_results.md)

Auxiliary load-balancing weights tested: `0`, `0.01`, `0.03`, `0.1`, `0.3`.

| Dataset | Best dense aux weight | Best dense loss | Main conclusion |
|---|---:|---:|---|
| TinyStories | 0.3 | 1.0146 | Stronger balancing improves loss and usage entropy. |
| WikiText-103 | 0.3 | 1.6502 | Stronger balancing improves loss and usage entropy. |

No load balancing causes collapse: TinyStories has 1.08 dead experts/layer and
WikiText has 3.83 dead experts/layer at `aux_loss_weight=0`. Learned masks beat
random masks at every tested weight, so sparse structure is not explained away
by poor load balance. However, direct 80% pruning gets brittle as balancing
gets stronger, reinforcing the need for rewind/retrain at high sparsity.

### Phase 4 Architecture and Dataset Robustness

Source: [phase4_robustness/phase4_results.md](phase4_robustness/phase4_results.md)

The architecture grid covers 36 WikiText runs:

```text
experts: 4, 8, 16
top-k: 1, 2
layers: 4, 8
seeds: 7, 17, 29
```

Main findings:

| Finding | Result |
|---|---|
| Best dense cell | 4 experts, top-2, 8 layers at `1.5606 +/- 0.0102` |
| Top-2 effect | Improved dense loss in all six matched comparisons |
| Depth effect | 8 layers improved dense loss in all six matched comparisons |
| Expert-count effect | Helped 4-layer models more than 8-layer models |
| Dead experts | None in any aggregate cell |
| Mask structure | Magnitude masks beat random masks in every architecture and dataset cell |
| Direct 80% pruning | Often poor without rewinding |

The balanced multi-domain setting also replicated the basic result: dense loss
was `1.4801 +/- 0.0425`, and the 50% magnitude mask scored `1.5360` versus
`3.7565` for a random mask.

### Representative Phase 4 Rewind Suite

Source: [phase4_rewinds/phase4_rewind_results.md](phase4_rewinds/phase4_rewind_results.md)

This suite completed all `288/288` retraining evaluations across:

```text
representatives: best dense, high capacity, balanced multi-domain
seeds: 7, 17, 29
sparsities: 50%, 80%
rewind points: 0%, 1%, 5%, 10%
conditions: learned mask, random mask, random reinit, randomized routing
```

| Representative | Sparsity | Best rewind | Ticket loss | Delta vs dense | Beats controls | Full-loss criterion |
|---|---:|---:|---:|---:|---:|---:|
| Best dense: 4E/top-2/8L | 50% | 1% | `1.5654 +/- 0.0113` | +0.31% | No | Yes |
| Best dense: 4E/top-2/8L | 80% | 10% | `1.6385 +/- 0.0067` | +4.99% | Yes | Yes |
| High capacity: 16E/top-1/8L | 50% | 0% | `1.5846 +/- 0.0175` | -0.43% | No | Yes |
| High capacity: 16E/top-1/8L | 80% | 10% | `1.5710 +/- 0.0085` | -1.28% | Yes | Yes |
| Multi-domain: 8E/top-1/4L | 50% | 10% | `1.4007 +/- 0.0253` | -5.36% | Yes | Yes |
| Multi-domain: 8E/top-1/4L | 80% | 10% | `1.4792 +/- 0.0200` | -0.06% | Yes | Yes |

Interpretation: the high-sparsity result now generalizes beyond the original
8-expert, top-1, 4-layer setting. At 80% sparsity, all representative cells
meet the full-loss criterion, and all beat the random-mask, random-reinit, and
randomized-routing controls. The 50% results preserve dense loss, but two of
the three cells do not beat all controls, so the causal evidence is cleaner at
80% than at 50% in Phase 4.

### Balanced Multi-Domain Causal Controls

Source: [multidomain_causal_controls/multidomain_causal_results.md](multidomain_causal_controls/multidomain_causal_results.md)

This suite replicated the fixed-random, random-every-step, and shuffled-usage
controls on the balanced TinyStories/WikiText corpus across seeds 7, 17, and
29.

| Condition | Mean loss | Std | Delta vs normal |
|---|---:|---:|---:|
| normal | 1.4801 | 0.0425 | - |
| fixed_random | 1.5810 | 0.0207 | +6.82% |
| random_every_step | 1.9564 | 0.0120 | +32.18% |
| shuffled_usage | 1.9791 | 0.0189 | +33.72% |

Shuffled usage exactly matched the normal expert-count logs for all three
seeds: 3,232 records per seed, zero mismatches. Despite that exact count
match, it performed slightly worse than random-every-step routing and nearly
removed the learned-mask advantage:

| Routing condition | 50% magnitude | 50% random | 80% magnitude | 80% random |
|---|---:|---:|---:|---:|
| normal | 1.5360 | 3.7565 | 5.5286 | 7.1896 |
| fixed_random | 1.6244 | 3.1311 | 4.8987 | 6.4320 |
| random_every_step | 1.9788 | 2.3928 | 2.3164 | 2.7988 |
| shuffled_usage | 1.9819 | 2.0373 | 2.0142 | 2.0775 |

Interpretation: this closes the biggest causal gap from the previous summary.
The usage-count result survives the broader multi-domain setting. Matching
expert update counts is not sufficient; coherent routed token identity remains
the important missing ingredient.

### Balanced Multi-Domain Swap Interventions

Source: [multidomain_swap_interventions/swap_interventions_results.md](multidomain_swap_interventions/swap_interventions_results.md)

This suite broadened the earlier limited expert 0/1 swap into six
interventions on the balanced TinyStories/WikiText corpus across seeds 7, 17,
and 29.

| Condition | Mean loss | Delta vs normal | Route agreement to normal | 80% mask Jaccard to normal |
|---|---:|---:|---:|---:|
| normal | 1.4801 | - | 1.0000 | 1.0000 |
| global swap 0/1 | 1.5111 | +2.10% | 0.7461 | 0.5679 |
| global swap 0/4 | 1.5068 | +1.80% | 0.7453 | 0.5405 |
| global swap 2/6 | 1.5027 | +1.53% | 0.7509 | 0.5616 |
| layer-0 swap 0/1 | 1.5026 | +1.52% | 0.9317 | 0.5869 |
| layer-3 swap 0/1 | 1.4897 | +0.65% | 0.9378 | 0.6822 |
| global cyclic shift | 1.5358 | +3.77% | 0.0000 | 0.4267 |

Interpretation: swap coverage now supports the routing-history claim more
directly. Disrupting expert identity consistently hurts performance, global
swaps are more damaging than a late-layer swap, and the fully cyclic
permutation is the strongest intervention. The layer-specific result is also
useful: later-layer swapping is milder, which matches the earlier finding that
functional expert specificity weakens in later layers.

### Subword-Tokenized Causal Controls

Source: [multidomain_subword_causal_controls/subword_tokenized_results.md](multidomain_subword_causal_controls/subword_tokenized_results.md)

This suite repeated the balanced multi-domain causal-control experiment with a
1024-ID deterministic byte-ngram subword tokenizer. The tokenizer reduced the
training stream from 5,857,322 raw byte tokens to 2,659,291 subword tokens, a
0.454 token ratio, while preserving byte fallback and exact text round-trips.

| Condition | Mean loss | Std | Delta vs normal |
|---|---:|---:|---:|
| normal | 3.2753 | 0.0307 | - |
| fixed_random | 3.5037 | 0.0876 | +6.97% |
| random_every_step | 4.3504 | 0.0323 | +32.83% |
| shuffled_usage | 4.3574 | 0.0156 | +33.04% |

Shuffled usage again exactly matched the normal expert-count logs for all
three seeds: 3,232 records per seed, zero mismatches. Despite that exact count
match, shuffled usage and random-every-step routing both lost about one third
relative to normal routing.

| Routing condition | 50% magnitude | 50% random | 80% magnitude | 80% random |
|---|---:|---:|---:|---:|
| normal | 3.3992 | 11.9938 | 14.0545 | 35.6571 |
| fixed_random | 3.6076 | 14.8263 | 17.3239 | 31.3733 |
| random_every_step | 4.3601 | 4.5309 | 4.4818 | 4.7644 |
| shuffled_usage | 4.3623 | 4.4146 | 4.4112 | 4.4794 |

Interpretation: the byte-tokenization confound is now materially reduced. The
same causal pattern survives when the model sees larger learned subword units:
stable coherent routing matters, usage counts alone are not enough, and
normal/fixed-random routing still produces much more meaningful sparse masks
than randomized or shuffled-token routing. The low routing-history/mask
correlation in this suite (`0.1237 +/- 0.1610`) is a useful nuance: the dense
causal effect is strong, but simple pairwise route agreement alone does not
explain mask overlap under this tokenizer.

### Long-Budget Multi-Domain Causal Controls

Source: [multidomain_long_causal_controls/multidomain_causal_results.md](multidomain_long_causal_controls/multidomain_causal_results.md)

This suite repeated the balanced multi-domain causal controls with a longer
10,000-step budget and 64 validation blocks. It completed all 12 train/prune
runs across seeds 7, 17, and 29.

Final-checkpoint results:

| Condition | Mean loss | Std | Delta vs normal |
|---|---:|---:|---:|
| normal | 1.5236 | 0.0415 | - |
| fixed_random | 1.4175 | 0.0517 | -6.96% |
| random_every_step | 1.4313 | 0.0174 | -6.06% |
| shuffled_usage | 1.4417 | 0.0086 | -5.37% |

Shuffled usage again exactly matched normal expert-count logs for all three
seeds. Direct pruning still showed non-random mask structure, but the final
10k dense comparison is not a clean causal result:

| Routing condition | 50% magnitude | 50% random | 80% magnitude | 80% random |
|---|---:|---:|---:|---:|
| normal | 1.6175 | 7.6520 | 6.8705 | 20.1388 |
| fixed_random | 1.4990 | 5.7106 | 4.5983 | 17.7759 |
| random_every_step | 1.4835 | 3.0755 | 2.6844 | 5.0695 |
| shuffled_usage | 1.4593 | 1.9987 | 1.7413 | 2.6651 |

Interpretation: the longer-budget run exposed a final-checkpoint selection
problem. Normal learned routing improved rapidly through the mid-training
region, then overfit or destabilized before step 10,000. For seed 7, normal
validation loss reached about `1.3002` at step 5,500 but ended at `1.5634`.
Fixed-random showed a similar but milder pattern, while randomized controls
continued improving later. This made final 10k loss the wrong comparison point.

### Long-Budget Best-Checkpoint Pruning

Source: [multidomain_long_best_checkpoint_pruning/best_checkpoint_pruning_results.md](multidomain_long_best_checkpoint_pruning/best_checkpoint_pruning_results.md)

This follow-up selected the best validation checkpoint for each seed and
condition, then reran pruning at the best saved checkpoint without overwriting
the original final-10k artifacts.

| Condition | Final 10k loss | Best observed loss | Best saved-checkpoint loss | Delta vs normal best saved |
|---|---:|---:|---:|---:|
| normal | 1.5236 | 1.3050 | 1.3072 | - |
| fixed_random | 1.4175 | 1.3037 | 1.3289 | +1.65% |
| random_every_step | 1.4313 | 1.4313 | 1.4313 | +9.49% |
| shuffled_usage | 1.4417 | 1.4417 | 1.4417 | +10.29% |

Direct pruning at the best saved checkpoints:

| Routing condition | Dense | 50% magnitude | 50% random | 80% magnitude | 80% random |
|---|---:|---:|---:|---:|---:|
| normal | 1.3072 | 1.3905 | 5.7336 | 7.7513 | 15.5672 |
| fixed_random | 1.3289 | 1.4003 | 5.3759 | 7.8883 | 19.9285 |
| random_every_step | 1.4313 | 1.4835 | 3.0754 | 2.6843 | 5.0693 |
| shuffled_usage | 1.4417 | 1.4593 | 1.9987 | 1.7413 | 2.6652 |

Interpretation: the long-budget caveat now cuts in favor of careful
checkpointing, not against the hypothesis. At each condition's best saved
checkpoint, normal learned routing is again the best dense model. Fixed-random
routing remains close, but worse. Random-every-step and shuffled-usage routing
remain much worse. The learned masks also remain strongly non-random in the
normal and fixed-random conditions, especially at 50% sparsity.

### Reduced Long-Budget Rewind

Source: [multidomain_long_best_checkpoint_rewinds/long_best_checkpoint_rewind_results.md](multidomain_long_best_checkpoint_rewinds/long_best_checkpoint_rewind_results.md)

This reduced suite tests 80% expert-local magnitude masks extracted from the
best saved normal-routing checkpoint in each long-budget seed. The learned-mask
lane covers initialization, 10%, and 25% rewinds. The random-mask,
random-reinit, and randomized-routing controls were then run at the winning
initialization rewind point.

Best-saved dense normal loss: `1.3072 +/- 0.0238`.

| Condition | Sparsity | Rewind fraction | Mean loss | Std | Delta vs best-saved dense |
|---|---:|---:|---:|---:|---:|
| learned_mask | 80% | 0% | 1.2984 | 0.0055 | -0.68% |
| learned_mask | 80% | 10% | 1.3552 | 0.0087 | +3.67% |
| learned_mask | 80% | 25% | 1.4125 | 0.0650 | +8.05% |
| random_mask | 80% | 0% | 1.3110 | 0.0070 | +0.29% |
| random_reinit | 80% | 0% | 1.3088 | 0.0074 | +0.12% |
| randomized_routing | 80% | 0% | 1.5011 | 0.0129 | +14.83% |

Interpretation: this is a control-complete initialization-rewind result at the
winning rewind point. The learned 80% mask slightly beats the best saved dense
checkpoint and narrowly beats the random-mask and random-reinit controls.
However, the margins against those two controls are small, so this should not
be overclaimed as a dramatic coordinate-level mask effect. The decisive effect
is randomized routing: keeping the learned mask but randomizing routes during
retraining is 14.83% worse than best-saved dense and 15.62% worse than the
learned-mask ticket. In this long-budget setting, stable learned routing is the
stronger signal than learned mask identity alone.

### Long-Budget Larger-Validation Extension

Source: [long_validation_extension/long_validation_extension_results.md](long_validation_extension/long_validation_extension_results.md)

This suite reran the closest long-budget 80% initialization-rewind comparison
on a larger held-out multi-domain validation file. The new validation file
combines TinyStories validation with WikiText validation plus WikiText test,
balanced and interleaved like the original multi-domain corpus.

```text
validation examples: 8,740
validation batches at batch size 128: about 69
previous multidomain validation batches: about 31
```

| Condition | Mean loss | Std | Delta vs dense best-saved |
|---|---:|---:|---:|
| dense_best_saved | 1.3213 | 0.0203 | - |
| learned_mask | 1.3096 | 0.0044 | -0.89% |
| random_mask | 1.3242 | 0.0080 | +0.22% |
| random_reinit | 1.3225 | 0.0099 | +0.09% |

Interpretation: this strengthens the most fragile long-budget result. On the
larger held-out validation file, the learned 80% initialization-rewound mask is
still best on average. The margins over random mask and random reinit remain
small, so the careful interpretation still stands, but the direction is now
more defensible because it survives a larger validation pass.

### Representative Iterative Magnitude Pruning

Source: [imp_representative/imp_representative_results.md](imp_representative/imp_representative_results.md)

This suite tested expert-local iterative magnitude pruning on the balanced
multi-domain representative cell:

```text
8 experts, top-1 routing, 4 layers
seeds: 7, 17, 29
target sparsity: 80%
IMP rounds: 4
rewind point: initialization
```

Dense representative loss: `1.4801 +/- 0.0347`.

| IMP round | Sparsity | Mean loss | Std | Delta vs dense |
|---:|---:|---:|---:|---:|
| 1 | 33.13% | 1.4938 | 0.0060 | +0.93% |
| 2 | 55.28% | 1.5013 | 0.0312 | +1.44% |
| 3 | 70.09% | 1.5168 | 0.0445 | +2.48% |
| 4 | 80.00% | 1.5687 | 0.0365 | +5.99% |

Interpretation: IMP closes an important methodological gap, but it does not
improve the main result in this representative setting. The earlier Phase 4
multi-domain 80% one-shot mask with 10% rewind reached `1.4792 +/- 0.0200`,
essentially dense-level performance. By contrast, this four-round IMP schedule
rewound to initialization finishes 5.99% worse than dense. That suggests the
paper should not claim classical IMP is necessary here; the stronger story is
that expert-local one-shot masks plus early rewinding already capture the
useful sparse structure, while repeated initialization-rewind IMP may be less
well matched to these small MoE training dynamics.

## What These Findings Mean

The project now supports four linked claims:

1. **MoE experts learn real sparse structure.** Learned masks beat random and
   transferred masks repeatedly, including under architectural variation.
2. **Routing is causally involved.** Replay, swap, randomized routing, and
   shuffled-usage controls show that routed-token histories change both
   performance and masks.
3. **The lottery-ticket claim is strongest after early rewinding.** The 50%
   case can often survive from initialization, but the robust high-sparsity
   case is a practical early-rewind ticket.
4. **Routing is not the whole story.** Cross-initialization replay shows that
   masks are not determined by route histories alone. They emerge from the
   interaction between initialization, data order, routing, and training.

This is a solid paper story because it avoids the easy overclaim. We can say
that routing histories induce sparse expert structure within a training
trajectory, while independent initializations do not share a simple coordinate
level mask identity.

## What Is Still Left

### Highest-Priority Experiments

1. **Optional long-budget control extension.** If compute allows, run
   random-mask, random-reinit, and randomized-routing controls at the 10% and
   25% long-budget rewind points too; this is lower priority because 0% is the
   winning learned-mask point.
2. **IMP variants, only if needed.** The representative initialization-rewind
   IMP run underperformed one-shot rewinding. A 10% rewind IMP variant could be
   run if reviewers demand it, but it is not necessary for the current story.
3. **Optional larger-validation routing control.** The larger-validation pass
   covered the close dense/learned/random/reinit comparison. Randomized routing
   could also be rerun on the larger validation file, but it is lower priority
   because its original margin was already large.

### Analysis and Paper Work

1. **Write the paper around four claims:** sparse structure, routing causality,
   lottery-ticket recovery, and initialization-routing interaction.
2. **Make final figures:** dense-condition bars, pruning curves, rewind curves,
   routing-vs-mask scatter, load-balance sweep, Phase 4 architecture grid, and
   representative rewind curves.
3. **Update the detailed report:** [execution_plan_1_results.md](execution_plan_1_results.md)
   should now include the representative Phase 4 rewind suite, balanced
   multi-domain causal controls, and broader swap interventions as complete.
4. **Create paper tables:** one table for main 3-seed results, one for causal
   controls, one for rewind tickets, one for Phase 4 robustness.
5. **Write limitations carefully:** small models, one lightweight subword
   tokenizer, limited validation size, one-shot pruning, non-exhaustive swap
   map, and cross-init symmetry concerns.

## Recommended Next Step

The next experiment should be:

> Either run the optional long-budget 10%/25% control extension, or shift effort
> into paper figures and tables.

Why this next: the representative Phase 4 rewind suite answered the largest
high-sparsity robustness question, and the balanced multi-domain causal-control
suite confirmed that usage counts are not enough on broader data. The broader
swap suite now confirms that routing-history permutations hurt performance
across global, layer-specific, and cyclic interventions. The subword suite now
reduces the byte-tokenization confound. The long-budget suite is complete, but
its final checkpoints are not the right comparison point because normal routing
overfits late. The checkpoint-aware follow-up resolved that comparison and
restored the expected causal ordering. The reduced long-budget rewind suite now
shows that 80% long-budget masks can retrain from initialization to slightly
better than best-saved dense loss, narrowly beat random-mask and random-reinit
controls, and strongly beat randomized-routing controls. The representative IMP
run now closes the main pruning-procedure gap: classical-style IMP was tested,
but it did not outperform the existing one-shot plus rewind result. The
larger-validation extension reduces the remaining uncertainty around the close
learned-mask versus random-control comparison.

## Artifact Index

Primary reports:

- [execution_plan_1_results.md](execution_plan_1_results.md)
- [tinystories_cpu_results.md](tinystories_cpu_results.md)
- [tinystories_gpu_multiseed/multiseed_results.md](tinystories_gpu_multiseed/multiseed_results.md)
- [wikitext103_gpu_multiseed/multiseed_results.md](wikitext103_gpu_multiseed/multiseed_results.md)
- [wikitext103_cross_init_replay/cross_init_replay_results.md](wikitext103_cross_init_replay/cross_init_replay_results.md)
- [wikitext103_functional_alignment/functional_alignment_results.md](wikitext103_functional_alignment/functional_alignment_results.md)
- [wikitext103_cross_init_rewind/cross_init_rewind_results.md](wikitext103_cross_init_rewind/cross_init_rewind_results.md)
- [wikitext103_fixed_random_rewind/fixed_random_rewind_results.md](wikitext103_fixed_random_rewind/fixed_random_rewind_results.md)
- [load_balance_sweep/load_balance_sweep_results.md](load_balance_sweep/load_balance_sweep_results.md)
- [phase4_robustness/phase4_results.md](phase4_robustness/phase4_results.md)
- [phase4_rewinds/phase4_rewind_results.md](phase4_rewinds/phase4_rewind_results.md)
- [multidomain_causal_controls/multidomain_causal_results.md](multidomain_causal_controls/multidomain_causal_results.md)
- [multidomain_swap_interventions/swap_interventions_results.md](multidomain_swap_interventions/swap_interventions_results.md)
- [multidomain_subword_causal_controls/subword_tokenized_results.md](multidomain_subword_causal_controls/subword_tokenized_results.md)
- [multidomain_long_causal_controls/multidomain_causal_results.md](multidomain_long_causal_controls/multidomain_causal_results.md)
- [multidomain_long_best_checkpoint_pruning/best_checkpoint_pruning_results.md](multidomain_long_best_checkpoint_pruning/best_checkpoint_pruning_results.md)
- [multidomain_long_best_checkpoint_rewinds/long_best_checkpoint_rewind_results.md](multidomain_long_best_checkpoint_rewinds/long_best_checkpoint_rewind_results.md)
- [imp_representative/imp_representative_results.md](imp_representative/imp_representative_results.md)
- [long_validation_extension/long_validation_extension_results.md](long_validation_extension/long_validation_extension_results.md)

Primary figures:

- [tinystories_gpu_multiseed/dense_conditions.png](tinystories_gpu_multiseed/dense_conditions.png)
- [tinystories_gpu_multiseed/rewind_0.8.png](tinystories_gpu_multiseed/rewind_0.8.png)
- [wikitext103_gpu_multiseed/dense_conditions.png](wikitext103_gpu_multiseed/dense_conditions.png)
- [wikitext103_gpu_multiseed/rewind_0.8.png](wikitext103_gpu_multiseed/rewind_0.8.png)
- [load_balance_sweep/tinystories_dense_loss.png](load_balance_sweep/tinystories_dense_loss.png)
- [load_balance_sweep/wikitext103_dense_loss.png](load_balance_sweep/wikitext103_dense_loss.png)
- [phase4_robustness/architecture_dense_loss.png](phase4_robustness/architecture_dense_loss.png)
- [phase4_rewinds/phase4_rewind_curves.png](phase4_rewinds/phase4_rewind_curves.png)
- [multidomain_causal_controls/dense_conditions.png](multidomain_causal_controls/dense_conditions.png)
- [multidomain_swap_interventions/swap_dense_loss.png](multidomain_swap_interventions/swap_dense_loss.png)
- [multidomain_subword_causal_controls/dense_conditions.png](multidomain_subword_causal_controls/dense_conditions.png)
- [multidomain_long_causal_controls/dense_conditions.png](multidomain_long_causal_controls/dense_conditions.png)

## Bottom Line

The results are currently supportive of the project hypothesis, with an
important refinement:

> Routing trajectories do causally shape expert-local sparse subnetworks, and
> those subnetworks can behave like lottery tickets after rewinding. However,
> the sparse mask is not determined by routing alone. It is a product of
> routing history interacting with initialization and training dynamics.

The subword-tokenized suite now shows that this pattern is not just a
byte-level-tokenization artifact. The long-budget suite adds a useful warning:
checkpoint selection matters, because final 10k normal-routing models can be
worse than their mid-training checkpoints. The best-checkpoint follow-up
restores the expected normal-routing advantage, so the overall evidence remains
supportive while becoming more precise. The reduced long-budget rewind suite is
especially encouraging but nuanced: its 80% initialization-rewound learned mask
slightly beats the best saved dense baseline and the matched random controls,
while randomized routing remains clearly harmful. The representative IMP result
does not beat the one-shot rewind result, which is useful negative evidence for
the methods section. The larger-validation extension confirms the close
learned-mask advantage on a broader held-out validation file. This is strong
enough to justify drafting the paper while running only optional robustness
controls in parallel.
