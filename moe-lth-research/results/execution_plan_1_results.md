# Execution Plan 1 Results

## Technical Report: Routing-Conditioned Lottery Tickets in Mixture-of-Experts Models

**Report date:** June 20, 2026  
**Execution plan:** [`../moe_lth_execution_plan.md`](../moe_lth_execution_plan.md)  
**Research question:** Do training-time routing trajectories causally induce stable lottery-ticket-like sparse subnetworks inside MoE experts?

---

## Executive Summary

The experiments completed so far provide consistent evidence across a CPU
pilot, three-seed TinyStories and WikiText-103 GPU replications, and a 36-run
Phase 4 architecture grid that learned MoE routing produces structured,
expert-specific sparse masks whose usefulness depends on both the selected
weights and the routing regime.

The strongest results are:

1. **Learned routing matters.** Random routing every step increased dense
   validation loss by 42.91% on TinyStories and 25.28% on the WikiText-103
   subset.
2. **The discovered masks are meaningful.** At 50% sparsity, magnitude masks
   substantially outperformed random masks, other-expert masks, and random
   reinitialization on both GPU datasets.
3. **50% sparse tickets satisfy strict initialization-rewind criteria.**
   Initialization-rewound learned masks finished within 1.55% of dense across
   TinyStories seeds and averaged 1.58% better than dense across three
   WikiText seeds.
4. **80% sparse tickets require early rewinding.** At a 10% rewind point,
   learned 80% masks finished within 2.03% of dense on TinyStories and averaged
   within 0.13% of dense across three WikiText seeds. Initialization rewind at
   80% did not meet the plan's 2-5% criterion.
5. **Routing history and mask structure move together.** Pairwise routing
   agreement and 80%-mask Jaccard similarity correlated at
   `0.7872 +/- 0.0144` across TinyStories seeds and `0.7785 +/- 0.0275`
   across WikiText seeds.
6. **Replay is exactly reproducible across seeds.** Replaying the normal
   routing history reproduced routing, masks, and final loss exactly in all
   three WikiText seeds.
7. **Specialization is depth-dependent.** The routed expert was the
   lowest-loss substitute for every expert distribution in layers 0 and 1,
   while this diagonal structure weakened in later layers.
8. **Usage counts are not enough.** Shuffling token identities while preserving
   every logged normal expert count increased loss by 26.56% on WikiText and
   50.07% on TinyStories, and nearly removed the learned-mask advantage.
9. **Routing and initialization interact.** Cross-initialization replay exactly
   reproduces source routes but does not reproduce source masks and is 6.54%
   worse than matched-data learned routing.
10. **Fixed-random routing also produces rewindable tickets.** With the router
    projection frozen throughout training and retraining, 50% and 80% learned
    masks beat the fixed-random dense baseline after 10% rewinding.
11. **Functional alignment does not rescue cross-init mask transfer.** Matching
    experts by output CKA and neurons by activation correlation changes
    cross-initialization mask Jaccard by less than 0.001.
12. **Cross-init replay masks are still rewindable.** Foreign-route masks do
    form usable tickets after rewinding, but matched-data learned routing
    remains better at both 50% and 80% sparsity.
13. **Load balancing helps, but does not explain away mask structure.**
    Sweeping `aux_loss_weight` from 0 to 0.3 improved dense loss and final
    usage entropy on both datasets; learned masks still beat random masks, but
    80% direct pruning became more brittle at stronger balancing.
14. **The main result survives the Phase 4 architecture and data grid.** Across
    36 WikiText architecture runs, top-2 routing and eight layers each improved
    dense loss in all six matched comparisons. Magnitude masks beat random
    masks for every architecture and dataset cell, including a balanced
    TinyStories/WikiText corpus with 32 validation batches.
15. **Representative Phase 4 rewind suites are complete.** At 80% sparsity,
    representative top-2/deeper, high-capacity, and balanced multi-domain cells
    all meet the full-loss criterion and beat random-mask, random-reinit, and
    randomized-routing controls.
16. **Balanced multi-domain causal controls are complete.** Shuffled usage
    exactly matched normal expert-count logs for all three multi-domain seeds
    but was 33.72% worse than normal, confirming on broader data that usage
    counts are not sufficient.
17. **Broader swap interventions are complete.** Six balanced multi-domain
    swap interventions across three seeds all worsened dense loss; the mildest
    was a layer-3 0/1 swap at +0.65%, while a global cyclic expert shift was
    strongest at +3.77%.
18. **Subword-tokenized causal controls are complete.** A 1024-ID byte-ngram
    subword tokenizer preserves the core causal-control pattern: fixed-random
    routing is +6.97% worse than normal, random-every-step is +32.83%, and
    shuffled usage is +33.04% despite exactly matched expert counts.

These results support the existence of routing-conditioned sparse expert
subnetworks and practical early-rewind lottery tickets. The strongest
routing-only claim is not supported: routing history shapes sparse masks, but
it is not sufficient to determine them independently of initialization. The
subword-tokenized setting now preserves the core causal pattern; the remaining
critical generalization tests are longer training, iterative pruning, and
broader validation.

---

## 1. Hypotheses and Current Verdicts

| Execution-plan question | Current verdict | Main evidence |
|---|---|---|
| Do experts specialize? | **Supported, depth-dependent** | High expert-ID probe accuracy and diagonal substitution matrices, strongest in layers 0-1 |
| Do experts contain stable sparse subnetworks? | **Supported at moderate sparsity** | 50% magnitude masks remain near dense and strongly beat controls |
| Are the subnetworks lottery tickets? | **Supported** | Strict 50% initialization tickets and practical 80% early-rewind tickets on both GPU datasets |
| Does stable routing matter? | **Supported** | Random-every-step routing substantially worsens dense and rewind performance |
| Do masks track routing history? | **Supported within the current controlled runs** | Exact replay and strong routing-history/mask-similarity correlations |
| Does routing history matter more than expert identity? | **Supported within an initialization** | Global, layer-specific, and cyclic swaps change masks and hurt loss; shuffled token identities fail despite exactly matched usage counts |
| Does routing history determine masks across initialization? | **Not supported** | Exact cross-init replay does not make masks more source-like and harms performance |
| Is usage imbalance sufficient to explain specialization? | **Rejected in the matched-usage and load-balance controls** | Shuffled usage exactly matches normal expert counts but is 26.56% worse on WikiText and 50.07% worse on TinyStories; stronger balancing improves dense loss without removing mask structure |
| Is the result robust? | **Supported across seeds, datasets, routing widths, depths, expert counts, swap controls, and subword tokenization** | Phase 4 covers 4/8/16 experts, top-1/top-2 routing, 4/8 layers, balanced multi-domain validation, representative rewinds, multi-domain causal controls, broader swap interventions, and a subword-tokenized causal-control suite; longer training remains open |

---

## 2. Experimental Progress Against the Plan

| Plan component | Status | Outcome |
|---|---|---|
| Small decoder-only MoE Transformer | Complete | 4 layers, 8 top-1 experts, 17,981,952 parameters |
| Router and expert-usage logging | Complete | Per-step usage, entropy, margins, and fixed-validation routes recorded |
| Fixed validation routing set | Complete | Fixed 12-block validation subset reused across checkpoints |
| Normal learned routing | Complete | Run on CPU pilot, TinyStories GPU, and WikiText GPU |
| Random routing every step | Complete | Run on all main suites |
| Replayed routing history | Complete | Exact reproduction on both GPU datasets |
| Swapped routing histories | Complete | Initial expert 0/1 swaps plus six balanced multi-domain global, layer-specific, and cyclic swap interventions |
| Expert-local magnitude pruning | Complete | 50%, 70%, 80%, 90%, and 95% sparsity |
| Random-mask control | Complete | Evaluated at every pruning sparsity and rewind point |
| Other-expert-mask control | Complete | Evaluated at every direct-pruning sparsity |
| Random-reinitialization control | Complete | Evaluated for direct pruning and rewind suites |
| Rewind to 0%, 1%, 5%, and 10% | Complete | Run at 50% and 80% sparsity on both GPU datasets |
| Randomized-routing rewind control | Complete | Run at every rewind point |
| Expert specialization analysis | Complete | Probe accuracy, silhouette, token divergence, and substitution matrices |
| Router/mask causal analysis | Complete | Pairwise route agreement, mask Jaccard, and correlation |
| Fixed random router | Complete | Three WikiText and TinyStories seeds; frozen random router projection, plus WikiText fixed-random rewind suites |
| Same usage, shuffled token identity | Complete | Three WikiText and TinyStories seeds; exact normal expert counts with token positions permuted |
| Strong load-balancing ablation | Complete | Three-seed sweep over aux weights 0, 0.01, 0.03, 0.1, and 0.3 on both GPU datasets |
| Top-2, larger depth, 4/16 experts | Complete | Full 3 x 2 x 2 factorial over 4/8/16 experts, top-1/top-2, and 4/8 layers, with three seeds per cell |
| Balanced multi-domain validation | Complete | Three seeds on interleaved equal-character TinyStories/WikiText data with 32 validation batches |
| Broader swap interventions | Complete | Six balanced multi-domain swap interventions across seeds 7, 17, and 29 |
| Three or more random seeds | Complete for WikiText and TinyStories | Seeds 7, 17, and 29 completed for routing, pruning, analysis, and both rewind sparsities |

---

## 3. Experimental Setup

### 3.1 GPU Suites

| Component | TinyStories GPU | WikiText-103 Subset GPU |
|---|---:|---:|
| Hardware | NVIDIA GeForce RTX 3080 Laptop GPU, 16 GB VRAM | Same |
| Training file | `data/processed/tinystories_train_50k.txt` | `data/wikitext103_subset/wikitext103_train.txt` |
| Training data size | 45,240,807 bytes | 2,943,403 bytes |
| Validation data size | 1,615,396 bytes | 247,035 bytes |
| Tokenization | Byte-level, vocabulary 256 | Byte-level, vocabulary 256 |
| Parameters | 17,981,952 | 17,981,952 |
| Layers / heads | 4 / 4 | 4 / 4 |
| Model width | 256 | 256 |
| Experts per layer | 8 | 8 |
| Expert hidden size | 1,024 | 1,024 |
| Routing | Top-1 learned routing | Top-1 learned routing |
| Capacity factor | 1.25 | 1.25 |
| Auxiliary loss weight | 0.1 | 0.1 |
| Sequence length / batch | 128 / 128 | 128 / 128 |
| Training steps | 2,500 | 2,500 |
| Training tokens per condition | 40.96 million | 40.96 million |
| Precision | FP16 | FP16 |
| Seed | 7, 17, and 29 | 7, 17, and 29 |
| Validation sample | Fixed 12 blocks | Fixed 12 blocks |

Configs:

- [`../configs/tinystories_gpu.yaml`](../configs/tinystories_gpu.yaml)
- [`../configs/tinystories_gpu_seed17.yaml`](../configs/tinystories_gpu_seed17.yaml)
- [`../configs/tinystories_gpu_seed29.yaml`](../configs/tinystories_gpu_seed29.yaml)
- [`../configs/wikitext103_gpu.yaml`](../configs/wikitext103_gpu.yaml)

### 3.2 CPU Pilot

The first TinyStories CPU pilot used a smaller 2-layer, 4-expert model for 500
steps. It revealed that the original `aux_loss_weight=0.01` setting allowed
expert collapse. Increasing it to `0.1` recovered balanced routing and removed
dead experts. This correction was carried into both GPU suites.

The pilot is documented in
[`tinystories_cpu_results.md`](tinystories_cpu_results.md).

### 3.3 Three-Seed WikiText Replication

The complete WikiText routing, pruning, analysis, specialization, 50% rewind,
and 80% rewind workflow was run independently with seeds 7, 17, and 29.

| Metric | Three-seed mean | Standard deviation |
|---|---:|---:|
| Normal dense loss | 1.6859 | 0.0160 |
| Random-every-step dense loss | 2.1121 | 0.0080 |
| Replay dense loss | 1.6859 | 0.0160 |
| Swapped dense loss | 1.7081 | 0.0151 |
| Routing-history/mask-similarity correlation | 0.7785 | 0.0275 |
| 50% learned mask, initialization rewind | 1.6593 | 0.0103 |
| 50% learned mask, 10% rewind | 1.5999 | 0.0103 |
| 80% learned mask, initialization rewind | 1.8399 | 0.0254 |
| 80% learned mask, 10% rewind | 1.6882 | 0.0297 |

Relative to the mean dense baseline, random-every-step routing is 25.28% worse,
swapping is 1.32% worse, the 50% initialization ticket is 1.58% better, and the
80% mask at 10% rewind is only 0.13% worse. Replay routing agreement, mask
Jaccard, and final loss are exactly equal to normal in every seed. In paired
seed comparisons, the 50% initialization ticket beats its own dense baseline
in all three seeds by 0.87% to 2.38%. The 80% ticket at 10% rewind remains
within 1.01% of its own dense baseline in every seed and beats dense in seed 29.

The complete aggregate is available in
[`wikitext103_gpu_multiseed/multiseed_results.md`](wikitext103_gpu_multiseed/multiseed_results.md).

![Multi-seed dense routing conditions](wikitext103_gpu_multiseed/dense_conditions.png)

![Multi-seed 80% rewind curves](wikitext103_gpu_multiseed/rewind_0.8.png)

### 3.4 Three-Seed TinyStories Replication

The complete TinyStories routing, pruning, analysis, specialization, 50%
rewind, and 80% rewind workflow was also run independently with seeds 7, 17,
and 29.

| Metric | Three-seed mean | Standard deviation |
|---|---:|---:|
| Normal dense loss | 1.0206 | 0.0096 |
| Random-every-step dense loss | 1.4586 | 0.0153 |
| Replay dense loss | 1.0206 | 0.0096 |
| Swapped dense loss | 1.0491 | 0.0111 |
| Routing-history/mask-similarity correlation | 0.7872 | 0.0144 |
| 50% learned mask, initialization rewind | 1.0365 | 0.0068 |
| 50% learned mask, 10% rewind | 0.9544 | 0.0070 |
| 80% learned mask, initialization rewind | 1.1492 | 0.0034 |
| 80% learned mask, 10% rewind | 1.0413 | 0.0138 |

Relative to the mean dense baseline, random-every-step routing is 42.91%
worse, swapping is 2.79% worse, the 50% initialization ticket remains within
1.55% of dense, and the 80% mask at 10% rewind remains within 2.03% of dense.
Replay again exactly reproduces the normal loss, routing, and masks in every
seed.

The complete aggregate is available in
[`tinystories_gpu_multiseed/multiseed_results.md`](tinystories_gpu_multiseed/multiseed_results.md).

![TinyStories multi-seed dense routing conditions](tinystories_gpu_multiseed/dense_conditions.png)

![TinyStories multi-seed 80% rewind curves](tinystories_gpu_multiseed/rewind_0.8.png)

### 3.5 Fixed-Random and Shuffled-Usage Causal Controls

Two additional routing controls were run across WikiText and TinyStories seeds
7, 17, and 29:

- **Fixed random:** the randomly initialized router projection is frozen while
  representations and experts continue training.
- **Shuffled usage:** each layer and training step receives the exact normal
  expert-count vector, but expert assignments are permuted across token
  positions.

For shuffled usage, all 3,232 logged expert-usage records per seed exactly
match the corresponding normal run on both datasets, including token counts
and usage fractions.

| Dataset | Routing condition | Mean loss | Std | Delta vs normal |
|---|---|---:|---:|---:|
| WikiText | Normal | **1.6859** | 0.0160 | - |
| WikiText | Fixed random | 1.7386 | 0.0252 | +3.13% |
| WikiText | Random every step | 2.1121 | 0.0080 | +25.28% |
| WikiText | Shuffled usage | 2.1337 | 0.0191 | +26.56% |
| TinyStories | Normal | **1.0206** | 0.0096 | - |
| TinyStories | Fixed random | 1.1105 | 0.0184 | +8.81% |
| TinyStories | Random every step | 1.4586 | 0.0153 | +42.91% |
| TinyStories | Shuffled usage | 1.5316 | 0.0151 | +50.07% |

Fixed random routing performs substantially better than either unstable
control on both datasets. This means learned router adaptation is useful but
not necessary for experts to learn useful computation; stable random router
geometry can induce a workable partition.

Shuffled usage performs slightly worse than random-every-step routing despite
exactly matching normal usage counts. The effect replicates on TinyStories and
is stronger there: shuffled usage is 50.07% worse than normal and 5.00% worse
than random-every-step routing. Therefore, update frequency and load balance
are not sufficient explanations for normal expert performance. Consistent
token identity and routed-token distribution are the critical factors
separated by this control.

| Dataset | Routing condition | 50% magnitude | 50% random mask | 80% magnitude | 80% random mask |
|---|---|---:|---:|---:|---:|
| WikiText | Normal | **1.7388** | 4.0564 | **4.5586** | 9.3660 |
| WikiText | Fixed random | **1.7803** | 3.1131 | **4.0933** | 6.1239 |
| WikiText | Random every step | **2.1257** | 2.4422 | **2.3651** | 2.8214 |
| WikiText | Shuffled usage | **2.1352** | 2.1645 | **2.1529** | 2.1835 |
| TinyStories | Normal | **1.0676** | 3.2700 | **3.5043** | 5.1919 |
| TinyStories | Fixed random | **1.1551** | 2.6055 | **4.1013** | 4.4096 |
| TinyStories | Random every step | **1.4844** | 2.1112 | **2.0078** | 2.7004 |
| TinyStories | Shuffled usage | **1.5349** | 1.5969 | **1.5743** | 1.6318 |

The learned-mask advantage is large under normal and fixed-random routing,
smaller under random-every-step routing, and nearly absent under shuffled
usage. At 50% sparsity, magnitude masks beat random masks by 2.3176 loss under
normal WikiText routing but only 0.0293 under shuffled usage. The TinyStories
replication has the same pattern: the 50% learned-mask advantage is 2.2024
under normal routing but only 0.0620 under shuffled usage. This indicates that
coherent token trajectories induce expert-specific sparse structure, whereas
matched usage with shuffled token identity produces largely interchangeable,
pruning-insensitive experts.

Normal-versus-shuffled routing agreement is near chance at `0.1282 +/- 0.0004`,
while mask Jaccard remains `0.5543 +/- 0.0115` on WikiText. TinyStories
replicates the near-chance routing agreement at `0.1282 +/- 0.0005`, with mask
Jaccard `0.5063 +/- 0.0082`. Usage counts therefore constrain some shared mask
structure, but they do not preserve routing identity, model quality, or
meaningful mask specificity.

![WikiText causal-control dense results](wikitext103_gpu_multiseed/dense_conditions.png)

![WikiText magnitude pruning by routing condition](wikitext103_gpu_multiseed/magnitude_pruning_by_routing.png)

![TinyStories causal-control dense results](tinystories_gpu_multiseed/dense_conditions.png)

![TinyStories magnitude pruning by routing condition](tinystories_gpu_multiseed/magnitude_pruning_by_routing.png)

### 3.6 Load-Balancing-Weight Sweep

A controlled sweep varied only `routing.aux_loss_weight` while holding model
size, data, optimizer, training steps, and pruning protocol fixed. The sweep
used seeds 7, 17, and 29 on both TinyStories and the WikiText-103 subset. The
existing `0.1` baseline runs were reused; all other weights were newly trained
and pruned.

| Dataset | Aux weight | Dense loss | Delta vs aux=0.1 | Final usage entropy | Dead experts/layer |
|---|---:|---:|---:|---:|---:|
| TinyStories | 0 | 1.2427 | +21.76% | 0.5112 | 1.08 |
| TinyStories | 0.01 | 1.0634 | +4.19% | 0.9610 | 0.00 |
| TinyStories | 0.03 | 1.0439 | +2.28% | 0.9856 | 0.00 |
| TinyStories | 0.1 | 1.0206 | +0.00% | 0.9957 | 0.00 |
| TinyStories | 0.3 | **1.0146** | -0.59% | **0.9971** | 0.00 |
| WikiText | 0 | 1.9252 | +14.19% | 0.2535 | 3.83 |
| WikiText | 0.01 | 1.7231 | +2.20% | 0.9582 | 0.00 |
| WikiText | 0.03 | 1.7040 | +1.07% | 0.9760 | 0.00 |
| WikiText | 0.1 | 1.6859 | +0.00% | 0.9932 | 0.00 |
| WikiText | 0.3 | **1.6502** | -2.12% | **0.9970** | 0.00 |

The no-balancing condition confirms the confound: without the auxiliary loss,
experts collapse, dead experts appear, and validation loss worsens sharply.
Increasing the weight from `0.1` to `0.3` slightly improves dense performance
and produces the most uniform usage on both datasets. This means the core
results are not an artifact of too-weak load balancing.

| Dataset | Aux weight | 50% magnitude | 50% random mask | 80% magnitude | 80% random mask |
|---|---:|---:|---:|---:|---:|
| TinyStories | 0 | 1.2747 | 1.9416 | 1.7854 | 2.4443 |
| TinyStories | 0.01 | 1.1073 | 2.4795 | 2.2598 | 3.6078 |
| TinyStories | 0.03 | 1.0882 | 2.7105 | 2.5058 | 4.0400 |
| TinyStories | 0.1 | 1.0676 | 3.2700 | 3.5043 | 5.1919 |
| TinyStories | 0.3 | 1.0647 | 4.2315 | 5.7921 | 8.8968 |
| WikiText | 0 | 1.9439 | 2.3248 | 2.2530 | 2.6468 |
| WikiText | 0.01 | 1.7622 | 2.7096 | 2.7351 | 3.4988 |
| WikiText | 0.03 | 1.7474 | 3.0594 | 3.8592 | 5.1037 |
| WikiText | 0.1 | 1.7388 | 4.0564 | 4.5586 | 9.3660 |
| WikiText | 0.3 | 1.7128 | 5.4444 | 7.1457 | 15.6605 |

Magnitude masks continue to beat random masks at every load-balancing weight.
However, stronger balancing makes direct 80% post-training pruning brittle:
the random-mask baseline gets much worse, and the learned mask also degrades
without rewinding. This does not weaken the lottery-ticket result; instead it
clarifies that highly balanced models still need the rewind/retrain protocol
at high sparsity.

![TinyStories load-balance dense loss](load_balance_sweep/tinystories_dense_loss.png)

![TinyStories load-balance usage](load_balance_sweep/tinystories_usage_balance.png)

![WikiText load-balance dense loss](load_balance_sweep/wikitext103_dense_loss.png)

![WikiText load-balance usage](load_balance_sweep/wikitext103_usage_balance.png)

Full report:
[`load_balance_sweep/load_balance_sweep_results.md`](load_balance_sweep/load_balance_sweep_results.md).

### 3.7 Fixed-Random Rewind

The fixed-random router was also evaluated with the same 50% and 80% rewind
suites used for normal routing. The router projection remained frozen during
both dense training and ticket retraining.

| Sparsity | Best learned-mask rewind | Mean loss | Delta vs fixed-random dense |
|---:|---:|---:|---:|
| 50% | 10% | **1.6315** | -6.16% |
| 80% | 10% | **1.7165** | -1.27% |

At 50% sparsity, fixed-random learned masks beat random masks and random
expert reinitialization at every rewind point. At 80% sparsity, the learned
mask is also best at every rewind point and recovers slightly better than the
fixed-random dense baseline by 10% rewind. Randomized routing remains far
worse, ending at `2.0863` for 50% sparsity and `2.1022` for 80% sparsity.

This result narrows the claim: learned router adaptation improves dense
performance, but a stable input partition is enough to produce lottery-ticket
behavior. The more important factor is not whether the router is learned or
random, but whether token trajectories are stable and coherent over training.

![Fixed-random rewind results](wikitext103_fixed_random_rewind/fixed_random_rewind.png)

Full report:
[`wikitext103_fixed_random_rewind/fixed_random_rewind_results.md`](wikitext103_fixed_random_rewind/fixed_random_rewind_results.md).

### 3.8 Cross-Initialization Replay

To separate routing history from initialization, seed 7 was used as the source
routing trajectory. Models initialized with seeds 17 and 29 were trained under
the seed-7 data order in two matched conditions:

- **Matched-data learned:** target initialization learns its own routes.
- **Cross-init replay:** target initialization is forced to use seed 7's exact
  routes.

| Target seed | Matched-data learned loss | Cross-init replay loss | Source vs learned routing | Source vs replay routing |
|---:|---:|---:|---:|---:|
| 17 | 1.7130 | 1.8093 | 0.1083 | **1.0000** |
| 29 | 1.6418 | 1.7648 | 0.1137 | **1.0000** |

Cross-init replay is 6.54% worse than matched-data learned routing on average,
despite exactly reproducing the source routes.

| Sparsity | Source vs matched-data learned mask | Source vs cross-init replay mask | Same-target-init learned vs replay |
|---:|---:|---:|---:|
| 50% | 0.3747 | 0.3733 | 0.5865 |
| 80% | 0.1931 | 0.1874 | 0.4684 |

Forcing source routes does not make independently initialized masks more
source-like. Yet changing routing under the same target initialization changes
the masks substantially. This supports an interaction:

```text
initialization x routing trajectory -> sparse mask
```

Routing history is causal, but it is not sufficient to determine coordinate-
level mask identity across initializations. Foreign routing trajectories also
transfer poorly, indicating that routes co-adapt with hidden representations
and expert weights.

Coordinate-level Jaccard across independently initialized networks should be
interpreted cautiously because hidden-unit permutation symmetries can hide
functional similarity, so a stronger alignment control was run next.

![Cross-initialization mask similarity](wikitext103_cross_init_replay/cross_init_mask_similarity.png)

### 3.9 Cross-Initialization Replay Rewind

The final masks discovered under matched-data learned routing and
cross-initialization replay were rewound and retrained at 50% and 80%
sparsity.

| Condition | Sparsity | Best rewind fraction | Mean loss | Delta vs own dense |
|---|---:|---:|---:|---:|
| Cross-init replay | 50% | 10% | **1.7018** | -4.78% |
| Cross-init replay | 80% | 10% | **1.7977** | +0.59% |
| Matched-data learned | 50% | 10% | **1.5944** | -4.95% |
| Matched-data learned | 80% | 10% | **1.6844** | +0.42% |

Foreign-route masks therefore do become usable early-rewind tickets. At 50%
sparsity, cross-init replay tickets beat their own dense replay baseline by
4.78%; at 80%, they recover to within 0.59% of dense. However, matched-data
learned routing remains better by 6.73% at 50% sparsity and 6.72% at 80%
sparsity. This strengthens the joint-mechanism view: a stable foreign route
history can induce tickets, but co-adapted routes induce better tickets.

![Cross-init rewind results](wikitext103_cross_init_rewind/cross_init_rewind.png)

Full report:
[`wikitext103_cross_init_rewind/cross_init_rewind_results.md`](wikitext103_cross_init_rewind/cross_init_rewind_results.md).

### 3.10 Functional Cross-Initialization Alignment

Experts were matched within each layer by linear CKA of their outputs on the
same 512 validation-token positions. Internal expert neurons were then matched
by activation correlation before recomputing mask Jaccard.

| Condition | Sparsity | Raw Jaccard | Functionally aligned Jaccard | Matched expert CKA |
|---|---:|---:|---:|---:|
| Matched-data learned | 50% | 0.3747 | 0.3750 | 0.5125 |
| Matched-data learned | 80% | 0.1931 | 0.1938 | 0.5125 |
| Cross-init replay | 50% | 0.3733 | 0.3735 | 0.4146 |
| Cross-init replay | 80% | 0.1874 | 0.1875 | 0.4146 |

Alignment barely changes mask overlap. This rules out a simple expert-ID or
hidden-neuron permutation explanation for the failed cross-initialization mask
transfer. It also strengthens the interaction claim: foreign replay routes
neither improve performance nor recover source-like sparse masks after this
functional alignment.

![Functional alignment mask similarity](wikitext103_functional_alignment/functional_alignment_mask_similarity.png)

Full report:
[`wikitext103_functional_alignment/functional_alignment_results.md`](wikitext103_functional_alignment/functional_alignment_results.md).

### 3.11 Phase 4 Architecture and Dataset Robustness Grid

The Phase 4 suite ran three seeds for every combination of 4, 8, or 16
experts; top-1 or top-2 routing; and 4 or 8 layers on the WikiText subset. It
also added a balanced TinyStories/WikiText corpus and increased that setting's
validation coverage from 12 to 32 batches. The architecture grid contains 36
runs, plus three TinyStories and three multi-domain dataset rows.

| Experts | Top-k | Layers | Parameters | Dense loss | 50% magnitude | 80% magnitude |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 4 | 9.57M | 1.7127 +/- 0.0304 | 1.7954 | 8.8331 |
| 4 | 1 | 8 | 19.04M | 1.5808 +/- 0.0110 | 1.6445 | 6.2745 |
| 4 | 2 | 4 | 9.57M | 1.6716 +/- 0.0183 | 1.8004 | 17.8248 |
| 4 | 2 | 8 | 19.04M | **1.5606 +/- 0.0102** | 1.6357 | 9.3844 |
| 8 | 1 | 4 | 17.98M | 1.6859 +/- 0.0160 | 1.7388 | 4.5586 |
| 8 | 1 | 8 | 35.87M | 1.5721 +/- 0.0070 | 1.6115 | 4.6839 |
| 8 | 2 | 4 | 17.98M | 1.6499 +/- 0.0154 | 1.7407 | 12.9197 |
| 8 | 2 | 8 | 35.87M | 1.5608 +/- 0.0061 | 1.6042 | 6.7442 |
| 16 | 1 | 4 | 34.81M | 1.6397 +/- 0.0436 | 1.6852 | 3.7368 |
| 16 | 1 | 8 | 69.52M | 1.5915 +/- 0.0058 | 1.6161 | 4.1889 |
| 16 | 2 | 4 | 34.81M | 1.5903 +/- 0.0045 | 1.6448 | 7.4676 |
| 16 | 2 | 8 | 69.52M | 1.5765 +/- 0.0172 | 1.6064 | 6.7282 |

Top-2 improved dense loss in all six matched comparisons, averaging a 1.75%
reduction relative to top-1. Eight layers also improved every matched
comparison, averaging a 5.05% reduction relative to four layers. Increasing
from 8 to 16 experts helped both four-layer models but slightly hurt both
eight-layer models, so expert count is not a monotonic quality lever at this
fixed 2,500-step budget. The best cell was the relatively economical
4-expert, top-2, 8-layer model; its `1.5606` mean was effectively tied with the
8-expert, top-2, 8-layer model at `1.5608`.

All architecture cells had zero dead experts and normalized usage entropy
between 0.9846 and 0.9952. At 50% direct pruning, loss increased by 1.55% to
7.71%. At 80%, one-shot pruning increased loss by 127.89% to 966.31%, with
the shallow top-2 models particularly brittle. Magnitude masks nevertheless
beat random masks in every architecture and dataset comparison. This supports
non-random mask structure, but the 80% direct-pruning values are not ticket
tests: full rewound retraining is required before extending the high-sparsity
lottery-ticket claim to these new architectures.

| Dataset | Validation batches | Dense loss | 50% magnitude / random | 80% magnitude / random |
|---|---:|---:|---:|---:|
| Balanced multi-domain | 32 | 1.4801 +/- 0.0425 | 1.5360 / 3.7565 | 5.5286 / 7.1896 |
| TinyStories | 12 | 1.0206 +/- 0.0096 | 1.0676 / 3.2700 | 3.5043 / 5.1919 |
| WikiText-103 subset | 12 | 1.6859 +/- 0.0160 | 1.7388 / 4.0564 | 4.5586 / 9.3660 |

![Phase 4 architecture dense loss](phase4_robustness/architecture_dense_loss.png)

![Phase 4 direct 80% pruning](phase4_robustness/architecture_pruning_80.png)

![Phase 4 dataset robustness](phase4_robustness/dataset_robustness.png)

Full report:
[`phase4_robustness/phase4_results.md`](phase4_robustness/phase4_results.md).

---

## 4. Dense Routing Results

### 4.1 Three-Seed Final Validation Loss

| Dataset | Normal | Random every step | Replay | Swapped experts 0/1 |
|---|---:|---:|---:|---:|
| TinyStories | **1.0206** | 1.4586 | **1.0206** | 1.0491 |
| WikiText-103 subset | **1.6859** | 2.1121 | **1.6859** | 1.7081 |

| Dataset | Random-routing penalty | Replay delta | Swap penalty |
|---|---:|---:|---:|
| TinyStories | +42.91% | 0.00% | +2.79% |
| WikiText-103 subset | +25.28% | 0.00% | +1.32% |

Random routing every step preserves access to all experts but destroys stable
token-to-expert histories. Its large loss penalty on both datasets indicates
that stable learned routing is functionally important. Swapping two route
identities changes internal computation but causes only a small dense-loss
penalty.

### 4.2 Representative Seed-7 Routing Stability

First-checkpoint to final-checkpoint same-token/same-expert agreement:

| Dataset | Layer 0 | Layer 1 | Layer 2 | Layer 3 | Overall |
|---|---:|---:|---:|---:|---:|
| TinyStories | 0.3112 | 0.1226 | 0.1383 | 0.1380 | 0.1776 |
| WikiText-103 subset | 0.3394 | 0.0890 | 0.1604 | 0.1420 | 0.1827 |

Chance agreement with eight experts is 0.125. Layer 0 retains substantially
more of its early routing identity than later layers.

![TinyStories routing stability](tinystories_gpu_suite/figures/routing_stability.png)

![WikiText routing stability](wikitext103_gpu_suite/figures/routing_stability.png)

---

## 5. Expert Usage and Collapse Control

### 5.1 CPU Pilot Finding

The standard CPU pilot collapsed under `aux_loss_weight=0.01`:

| Layer | Final normalized usage entropy | Dead experts |
|---|---:|---:|
| 0 | 0.506 | 1 |
| 1 | 0.194 | 1 |

With `aux_loss_weight=0.1`, normalized usage entropy rose to 0.968 and 0.979,
with zero dead experts. This established that load imbalance was a serious
confound and justified the stronger balancing used in the GPU suites.

### 5.2 Final GPU Usage

| Dataset | Layer 0 | Layer 1 | Layer 2 | Layer 3 | Dead experts |
|---|---:|---:|---:|---:|---:|
| TinyStories | 0.9877 | 0.9981 | 0.9978 | 0.9976 | 0 in every layer |
| WikiText-103 subset | 0.9870 | 0.9975 | 0.9994 | 0.9907 | 0 in every layer |

All final learned-routing GPU layers are highly balanced. The resulting
specialization and sparse-mask effects therefore cannot be explained only by
dead experts or severe final usage imbalance.

![TinyStories normal expert usage](tinystories_gpu_suite/figures/normal_expert_usage.png)

![WikiText normal expert usage](wikitext103_gpu_suite/figures/normal_expert_usage.png)

Random-routing and intervention usage plots:

- [TinyStories random-every-step usage](tinystories_gpu_suite/figures/random_every_step_expert_usage.png)
- [TinyStories replay usage](tinystories_gpu_suite/figures/replay_expert_usage.png)
- [TinyStories swapped usage](tinystories_gpu_suite/figures/swapped_expert_usage.png)
- [WikiText random-every-step usage](wikitext103_gpu_suite/figures/random_every_step_expert_usage.png)
- [WikiText replay usage](wikitext103_gpu_suite/figures/replay_expert_usage.png)
- [WikiText swapped usage](wikitext103_gpu_suite/figures/swapped_expert_usage.png)

---

## 6. Direct Post-Training Pruning

### 6.1 TinyStories

| Sparsity | Magnitude mask | Random mask | Other-expert mask | Learned mask + random reinit |
|---:|---:|---:|---:|---:|
| 0% dense | 1.0223 | - | - | - |
| 50% | **1.0680** | 3.2220 | 1.3751 | 5.4923 |
| 70% | **1.7037** | 4.6285 | 2.3015 | 5.4576 |
| 80% | **3.4488** | 5.1036 | 3.9227 | 5.4454 |
| 90% | **5.2314** | 5.3507 | 5.3056 | 5.4820 |
| 95% | 5.4478 | **5.3869** | 5.4472 | 5.4762 |

At 50% sparsity, the TinyStories magnitude mask is 4.47% worse than dense and
dramatically better than every control. At 80% sparsity, direct pruning is
237.36% worse than dense, so retraining is necessary.

### 6.2 WikiText-103 Subset

| Sparsity | Magnitude mask | Random mask | Other-expert mask | Learned mask + random reinit |
|---:|---:|---:|---:|---:|
| 0% dense | 1.6817 | - | - | - |
| 50% | **1.7331** | 4.3020 | 2.0216 | 8.5632 |
| 70% | **2.5098** | 6.6151 | 2.9254 | 8.4978 |
| 80% | **4.8942** | 7.6956 | 5.1422 | 8.5709 |
| 90% | 8.4087 | **8.2572** | 8.5496 | 8.5923 |
| 95% | 8.8405 | **8.3615** | 8.9272 | 8.6298 |

At 50% sparsity, the WikiText magnitude mask is only 3.06% worse than dense
and strongly outperforms every control. At 80% sparsity, direct pruning is
191.01% worse than dense.

### 6.3 Interpretation

Magnitude masks are clearly structured and expert-specific at moderate
sparsity. Their advantage over other-expert masks shows that masks are not
freely transferable. Their advantage over random reinitialization shows that
the trained weight values matter in addition to mask topology.

At 90-95% sparsity, all direct-pruning methods largely fail and distinctions
between masks narrow or reverse. The current evidence therefore supports useful
post-training sparse subnetworks at 50%, but not useful direct-pruned
subnetworks at the plan's target 80% sparsity.

![TinyStories pruning curves](tinystories_gpu_suite/normal/figures/pruning_curves.png)

![WikiText pruning curves](wikitext103_gpu_suite/normal/figures/pruning_curves.png)

---

## 7. Lottery-Ticket Rewind Results

The final magnitude masks were rewound to initialization, 1%, 5%, and 10% of
training and retrained with pruned weights held at zero. Controls used a random
mask, random expert reinitialization, or randomized routing.

### 7.1 Fifty Percent Sparsity

#### TinyStories

| Rewind point | Learned mask | Random mask | Random reinit | Randomized routing |
|---:|---:|---:|---:|---:|
| Initialization | **1.0365** | 1.0915 | 1.0785 | 1.5506 |
| 1% | **1.0212** | 1.0955 | 1.1126 | 1.5482 |
| 5% | **0.9734** | 1.0474 | 1.0460 | 1.4542 |
| 10% | **0.9544** | 1.0353 | 1.0356 | 1.4321 |

Across three seeds, the initialization-rewound learned mask is only 1.55%
worse than dense and beats all controls. It meets the execution plan's strict
ticket criterion. The 10%-rewound mask beats dense by 6.49%.

#### WikiText-103 Subset

| Rewind point | Learned mask | Random mask | Random reinit | Randomized routing |
|---:|---:|---:|---:|---:|
| Initialization | **1.6672** | 1.7408 | 1.7499 | 2.1595 |
| 1% | **1.6421** | 1.7149 | 1.7341 | 2.1603 |
| 5% | **1.6150** | 1.7025 | 1.7210 | 2.0898 |
| 10% | **1.6100** | 1.6701 | 1.6799 | 2.0743 |

The initialization-rewound learned mask beats dense by 0.87%, satisfying a
strong strict-ticket result. The 10%-rewound mask beats dense by 4.27%.

### 7.2 Eighty Percent Sparsity

#### TinyStories

| Rewind point | Learned mask | Random mask | Random reinit | Randomized routing |
|---:|---:|---:|---:|---:|
| Initialization | **1.1492** | 1.2488 | 1.2369 | 1.5998 |
| 1% | **1.1629** | 1.2510 | 1.2233 | 1.5821 |
| 5% | **1.0780** | 1.1748 | 1.1656 | 1.4899 |
| 10% | **1.0413** | 1.1606 | 1.1534 | 1.4579 |

At initialization, the learned mask is 12.60% worse than dense and does not
meet the strict criterion. At 10% rewind, it is only 2.03% worse than dense and
beats every control, satisfying the plan's practical early-rewind criterion.

#### WikiText-103 Subset

| Rewind point | Learned mask | Random mask | Random reinit | Randomized routing |
|---:|---:|---:|---:|---:|
| Initialization | **1.8329** | 1.9163 | 1.9219 | 2.1915 |
| 1% | **1.8680** | 1.9462 | 1.8943 | 2.1674 |
| 5% | **1.7377** | 1.8800 | 1.8597 | 2.1164 |
| 10% | **1.6953** | 1.8414 | 1.8338 | 2.1021 |

At initialization, the learned mask is 8.99% worse than dense. At 10% rewind,
it is only 0.81% worse than dense and strongly beats every control.

### 7.3 Ticket Classification

| Dataset | 50% initialization ticket | 50% best early ticket | 80% initialization ticket | 80% 10%-rewind ticket |
|---|---|---|---|---|
| TinyStories | **Yes**, +1.55% | **Yes**, -6.49% | No, +12.60% | **Yes**, +2.03% |
| WikiText-103 subset | **Yes**, -0.87% | **Yes**, -4.27% | No, +8.99% | **Yes**, +0.81% |

The cross-dataset pattern is consistent: moderate sparsity supports strict
initialization tickets, while 80% sparsity requires an early checkpoint.
Randomized routing is the weakest rewind condition at every tested point,
showing that routing consistency is necessary for ticket recovery.

Across the three WikiText seeds, this pattern remains stable. The 50%
initialization-rewound learned mask averages `1.6593 +/- 0.0103`, outperforming
the dense mean of `1.6859 +/- 0.0160`. At 80% sparsity, the 10%-rewound learned
mask averages `1.6882 +/- 0.0297`, compared with `1.8301 +/- 0.0540` for random
masks, `1.8245 +/- 0.0515` for random reinitialization, and `2.0905 +/- 0.0143`
for randomized routing.

---

## 8. Routing History and Mask Similarity

All mask comparisons below use expert-local 80% magnitude masks.

### 8.1 Pairwise Results

#### TinyStories

| Comparison | Routing agreement | Mask Jaccard |
|---|---:|---:|
| Normal vs random every step | 0.1254 | 0.4301 |
| Normal vs replay | **1.0000** | **1.0000** |
| Normal vs swapped | 0.7541 | 0.5365 |
| Random every step vs swapped | 0.1254 | 0.4208 |

Routing-history agreement and mask similarity correlation: **0.8037**.

#### WikiText-103 Subset

| Comparison | Routing agreement | Mask Jaccard |
|---|---:|---:|
| Normal vs random every step | 0.1253 | 0.4557 |
| Normal vs replay | **1.0000** | **1.0000** |
| Normal vs swapped | 0.7465 | 0.6024 |
| Random every step vs swapped | 0.1257 | 0.4561 |

Routing-history agreement and mask similarity correlation: **0.8656**.

### 8.2 Interpretation

Replay provides the cleanest result: forcing the same route sequence under the
same controlled initialization reproduces the same masks and final model
exactly. Swapping route identities moves both routing agreement and mask
similarity away from the normal run, while random-every-step routing reduces
agreement to chance.

Across the expanded WikiText six-condition suite, the routing-history/mask
similarity correlation is `0.7785 +/- 0.0275`. The high pairwise correlations
support the proposed relationship:

```text
routing history similarity -> sparse mask similarity
```

Replay within each seed uses the same initialization as that seed's normal
run, so exact replay is also a deterministic reproducibility check.
Cross-initialization replay supplies the stronger counterfactual: imposing the
same external route history on independent initializations exactly reproduces
the routes but does not make their masks more source-like. At the same time,
learned and replay masks under the same target initialization differ
substantially. Together, these results support a joint mechanism:

```text
initialization x routing trajectory -> sparse mask
```

![TinyStories mask similarity](tinystories_gpu_suite/figures/mask_similarity.png)

![TinyStories routing history versus mask similarity](tinystories_gpu_suite/figures/routing_history_vs_mask_similarity.png)

![WikiText mask similarity](wikitext103_gpu_suite/figures/mask_similarity.png)

![WikiText routing history versus mask similarity](wikitext103_gpu_suite/figures/routing_history_vs_mask_similarity.png)

---

## 9. Expert Specialization

### 9.1 Hidden-State Separability and Functional Specificity

| Dataset | Layer | Linear-probe accuracy | Silhouette score | Own expert is lowest loss |
|---|---:|---:|---:|---:|
| TinyStories | 0 | 0.9936 | 0.0514 | 8 / 8 |
| TinyStories | 1 | 0.9544 | 0.0451 | 8 / 8 |
| TinyStories | 2 | 0.8664 | -0.0149 | 7 / 8 |
| TinyStories | 3 | 0.7952 | -0.0198 | 3 / 8 |
| WikiText-103 subset | 0 | 0.9960 | 0.0563 | 8 / 8 |
| WikiText-103 subset | 1 | 0.9176 | 0.0103 | 8 / 8 |
| WikiText-103 subset | 2 | 0.8208 | -0.0516 | 4 / 8 |
| WikiText-103 subset | 3 | 0.7936 | -0.0235 | 2 / 8 |

Pre-router hidden states strongly predict expert identity, especially in the
first two layers. Low or negative silhouette scores show that these regions are
not cleanly separated compact clusters. Functional substitution is strongly
diagonal in early layers and progressively weaker in later layers.

This supports genuine but depth-dependent expert specialization. It also warns
against describing all experts and layers as equally specialized.

### 9.2 Expert-Specificity Heatmaps

TinyStories:

- [Layer 0](tinystories_gpu_suite/normal/figures/layer_0_expert_specificity.png)
- [Layer 1](tinystories_gpu_suite/normal/figures/layer_1_expert_specificity.png)
- [Layer 2](tinystories_gpu_suite/normal/figures/layer_2_expert_specificity.png)
- [Layer 3](tinystories_gpu_suite/normal/figures/layer_3_expert_specificity.png)

WikiText-103 subset:

- [Layer 0](wikitext103_gpu_suite/normal/figures/layer_0_expert_specificity.png)
- [Layer 1](wikitext103_gpu_suite/normal/figures/layer_1_expert_specificity.png)
- [Layer 2](wikitext103_gpu_suite/normal/figures/layer_2_expert_specificity.png)
- [Layer 3](wikitext103_gpu_suite/normal/figures/layer_3_expert_specificity.png)

---

## 10. Mask Stability During Training

For the WikiText normal run, the 80%-sparse mask's Jaccard similarity to its
final mask increased through training:

| Checkpoint step | 0 | 25 | 125 | 250 | 500 | 1,000 | 2,500 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal routing | 0.5795 | 0.5969 | 0.6023 | 0.6076 | 0.6199 | 0.6518 | 1.0000 |
| Random routing | 0.5147 | 0.5637 | 0.6840 | 0.6964 | 0.7267 | 0.7883 | 1.0000 |

Masks evolve continuously rather than appearing fully formed at
initialization. Random-routing masks stabilize earlier by this metric, but
their mask identity is less functionally meaningful: random routing has worse
dense loss, learned-mask rewind loss, and expert-specific differentiation.
Mask stability alone is therefore not evidence of a useful ticket.

---

## 11. Consolidated Interpretation

### 11.1 What the Results Support

The current evidence supports the following technical claims:

1. **Learned MoE routing creates functionally meaningful expert structure.**
   Learned routing outperforms random-every-step routing, and routed experts
   are usually the best experts for their assigned token distributions.
2. **Experts contain non-random sparse subnetworks.** Magnitude masks beat
   equal-sparsity random masks, transferred masks, and random reinitialization.
3. **Lottery-ticket behavior exists inside experts.** At 50% sparsity, masks
   can be rewound to initialization and retrained near or beyond dense
   performance. At 80%, early rewinding recovers near-dense performance.
4. **Ticket recovery depends on routing.** Randomized routing substantially
   degrades every rewind condition.
5. **Routing histories and sparse masks are linked.** Replay reproduces masks,
   swaps change masks, and route agreement strongly correlates with mask
   similarity.
6. **Specialization is not explained by usage counts alone.** Shuffled usage
   exactly preserves normal expert counts but destroys performance and nearly
   removes learned-mask specificity.
7. **Stable random routing can still produce tickets.** Fixed-random routing
   is worse than learned routing, but its learned masks become strong
   early-rewind tickets.
8. **Simple functional permutation alignment does not explain cross-init
   mismatch.** Expert-output CKA and hidden-neuron matching barely change
   cross-initialization mask overlap.
9. **Foreign-route tickets exist but are weaker than co-adapted tickets.**
   Cross-init replay masks become good early-rewind tickets, but matched-data
   learned routing remains better.
10. **Dense and mask-structure findings generalize across the Phase 4 grid.**
    Top-2 routing and eight layers improve all matched dense comparisons;
    magnitude masks beat random masks across 4/8/16 experts, 4/8 layers, both
    routing widths, and balanced multi-domain validation.

### 11.2 What the Results Do Not Yet Establish

The experiments do not yet establish:

1. Whether a stronger whole-model alignment method would reveal relationships
   not captured by expert-output CKA and hidden-neuron matching.
2. Whether fixed-random, shuffled-usage, and rewind controls replicate on the
   balanced multi-domain corpus.
3. That 80%-sparse tickets remain rewindable across the Phase 4 architecture
   grid; the grid currently provides direct-pruning controls only.
4. That the exact quantitative effects generalize beyond a byte-level
   tokenizer or to substantially larger training-token budgets.
5. That all layers specialize equally; later layers show notably weaker
   expert-specific substitution behavior.

### 11.3 Best Current Claim

> Small MoE language models trained with balanced learned routing contain
> expert-local sparse subnetworks whose mask identity, rewind performance, and
> functional specificity depend on the routing regime. Across three independent
> WikiText seeds, moderate-sparsity masks satisfy strict initialization-rewind
> lottery-ticket criteria, while 80%-sparse masks require early rewinding.
> Routing causally shapes masks within an initialization, but
> cross-initialization replay shows that routing alone does not determine a
> transferable coordinate-level sparse mask. The best-supported mechanism is
> an interaction between initialization and routing trajectory. Dense quality
> and non-random mask structure generalize across top-1/top-2 routing,
> 4/8/16 experts, 4/8 layers, and balanced multi-domain validation, while
> high-sparsity rewind robustness for those new settings remains open.

---

## 12. Limitations and Threats to Validity

- **Causal-control seed coverage:** Fixed-random and shuffled-usage controls
  now have three WikiText and three TinyStories seeds; cross-initialization
  controls remain WikiText-only.
- **Cross-initialization mask symmetry:** Expert-output CKA and hidden-neuron
  matching do not account for all possible whole-model symmetries across
  independently initialized networks.
- **Non-exhaustive swap intervention:** The new balanced multi-domain suite
  covers global pair swaps, layer-specific swaps, and a cyclic shift, but not
  every possible expert pair in every layer.
- **Limited scale:** Phase 4 spans 9.57M to 69.52M parameters, still far below
  production-scale language models and under one fixed 2,500-step budget.
- **Tokenizer coverage:** The subword control uses one deterministic
  byte-ngram tokenizer; a standard BPE/unigram tokenizer would still be useful
  before making broad tokenizer-invariance claims.
- **Validation coverage:** Main GPU suites use 12 fixed sequence blocks; the
  balanced multi-domain setting increases this to 32 but remains limited.
- **WikiText reuse:** The 2.94 MB WikiText training subset is reused for
  multiple passes.
- **Probe warnings:** WikiText checkpoint specialization emitted logistic
  regression convergence warnings, so probe accuracies are approximate.
- **One-shot pruning:** Iterative magnitude pruning has not been tested.
- **Load-balance rewind coverage:** The load-balancing sweep includes dense,
  usage, routing-stability, and direct-pruning metrics, but not full rewind
  suites for every auxiliary-loss weight.
- **Swap coverage:** Broader balanced multi-domain swaps are now complete, but
  the intervention map is still not exhaustive over every expert pair and
  every layer.

---

## 13. Required Next Experiments

Priority order for completing the execution plan:

1. Increase the training-token budget to separate architecture effects from
   undertraining.
2. Test iterative magnitude pruning to compare one-shot tickets against a more
   classical lottery-ticket pruning protocol.
3. Expand validation coverage beyond the current 12-block main suites and
   32-block balanced multi-domain setting.

---

## 14. Artifact Index

### 14.1 Primary Reports and Configs

- [Execution plan](../moe_lth_execution_plan.md)
- [TinyStories CPU pilot report](tinystories_cpu_results.md)
- [WikiText GPU report](wikitext103_gpu_results.md)
- [TinyStories GPU config](../configs/tinystories_gpu.yaml)
- [TinyStories seed-17 config](../configs/tinystories_gpu_seed17.yaml)
- [TinyStories seed-29 config](../configs/tinystories_gpu_seed29.yaml)
- [WikiText GPU config](../configs/wikitext103_gpu.yaml)
- [Load-balance sweep report](load_balance_sweep/load_balance_sweep_results.md)
- [Load-balance sweep JSON](load_balance_sweep/load_balance_sweep_summary.json)
- [Phase 4 robustness report](phase4_robustness/phase4_results.md)
- [Phase 4 aggregate JSON](phase4_robustness/phase4_summary.json)
- [Representative Phase 4 rewind report](phase4_rewinds/phase4_rewind_results.md)
- [Representative Phase 4 rewind JSON](phase4_rewinds/phase4_rewind_summary.json)
- [Balanced multi-domain causal-control report](multidomain_causal_controls/multidomain_causal_results.md)
- [Balanced multi-domain causal-control JSON](multidomain_causal_controls/multidomain_causal_summary.json)
- [Balanced multi-domain swap-intervention report](multidomain_swap_interventions/swap_interventions_results.md)
- [Balanced multi-domain swap-intervention JSON](multidomain_swap_interventions/swap_interventions_summary.json)
- [Subword-tokenized causal-control report](multidomain_subword_causal_controls/subword_tokenized_results.md)
- [Subword-tokenized causal-control JSON](multidomain_subword_causal_controls/multidomain_causal_summary.json)
- [Balanced multi-domain metadata](../data/multidomain_balanced/metadata.json)

### 14.2 TinyStories GPU Raw Results

- [Three-seed aggregate report](tinystories_gpu_multiseed/multiseed_results.md)
- [Three-seed aggregate JSON](tinystories_gpu_multiseed/multiseed_summary.json)
- [Suite summary](tinystories_gpu_suite/suite_summary.json)
- [Routing and mask analysis](tinystories_gpu_suite/tables/analysis_report.json)
- [Normal-run specialization analysis](tinystories_gpu_suite/normal/tables/checkpoint_analysis.json)
- [Normal-run pruning results](tinystories_gpu_suite/normal/tables/pruning_results.json)
- [50% rewind suite](tinystories_gpu_suite/normal/tables/rewind_suite_sparsity_0.5.json)
- [80% rewind suite](tinystories_gpu_suite/normal/tables/rewind_suite_sparsity_0.8.json)
- [Normal run summary](tinystories_gpu_suite/normal/summary.json)
- [Random-every-step summary](tinystories_gpu_suite/random_every_step/summary.json)
- [Replay summary](tinystories_gpu_suite/replay/summary.json)
- [Swapped summary](tinystories_gpu_suite/swapped/summary.json)
- [Seed-17 suite](tinystories_gpu_seed17_suite)
- [Seed-29 suite](tinystories_gpu_seed29_suite)

### 14.3 WikiText GPU Raw Results

- [Three-seed aggregate report](wikitext103_gpu_multiseed/multiseed_results.md)
- [Three-seed aggregate JSON](wikitext103_gpu_multiseed/multiseed_summary.json)
- [Cross-initialization replay report](wikitext103_cross_init_replay/cross_init_replay_results.md)
- [Cross-initialization replay JSON](wikitext103_cross_init_replay/cross_init_replay_summary.json)
- [Cross-init replay rewind report](wikitext103_cross_init_rewind/cross_init_rewind_results.md)
- [Cross-init replay rewind JSON](wikitext103_cross_init_rewind/cross_init_rewind_summary.json)
- [Functional alignment report](wikitext103_functional_alignment/functional_alignment_results.md)
- [Functional alignment JSON](wikitext103_functional_alignment/functional_alignment_summary.json)
- [Fixed-random rewind report](wikitext103_fixed_random_rewind/fixed_random_rewind_results.md)
- [Fixed-random rewind JSON](wikitext103_fixed_random_rewind/fixed_random_rewind_summary.json)
- [Suite summary](wikitext103_gpu_suite/suite_summary.json)
- [Routing and mask analysis](wikitext103_gpu_suite/tables/analysis_report.json)
- [Normal-run specialization analysis](wikitext103_gpu_suite/normal/tables/checkpoint_analysis.json)
- [Normal-run pruning results](wikitext103_gpu_suite/normal/tables/pruning_results.json)
- [50% rewind suite](wikitext103_gpu_suite/normal/tables/rewind_suite_sparsity_0.5.json)
- [80% rewind suite](wikitext103_gpu_suite/normal/tables/rewind_suite_sparsity_0.8.json)
- [Normal run summary](wikitext103_gpu_suite/normal/summary.json)
- [Random-every-step summary](wikitext103_gpu_suite/random_every_step/summary.json)
- [Replay summary](wikitext103_gpu_suite/replay/summary.json)
- [Swapped summary](wikitext103_gpu_suite/swapped/summary.json)
- [Seed-17 suite](wikitext103_gpu_seed17_suite)
- [Seed-29 suite](wikitext103_gpu_seed29_suite)

### 14.4 Figure Directories

- [TinyStories suite figures](tinystories_gpu_suite/figures)
- [TinyStories normal-run figures](tinystories_gpu_suite/normal/figures)
- [TinyStories aggregate figures](tinystories_gpu_multiseed)
- [WikiText suite figures](wikitext103_gpu_suite/figures)
- [WikiText normal-run figures](wikitext103_gpu_suite/normal/figures)
- [WikiText aggregate figures](wikitext103_gpu_multiseed)
- [Cross-initialization replay figures](wikitext103_cross_init_replay)
- [Cross-init replay rewind figures](wikitext103_cross_init_rewind)
- [Functional alignment figures](wikitext103_functional_alignment)
- [Fixed-random rewind figures](wikitext103_fixed_random_rewind)
- [Load-balance sweep figures](load_balance_sweep)
- [Phase 4 robustness figures](phase4_robustness)
- [Representative Phase 4 rewind figures](phase4_rewinds)
- [Balanced multi-domain causal-control figures](multidomain_causal_controls)
- [Balanced multi-domain swap-intervention figures](multidomain_swap_interventions)
- [Subword-tokenized causal-control figures](multidomain_subword_causal_controls)

---

## 15. Reproducibility and Verification

The final GPU environment test run completed successfully:

```text
16 passed in 11.94s
```

Each WikiText seed contains the full routing/pruning suite and two 16-condition
normal rewind suites plus the full fixed-random rewind suite. Each TinyStories
seed contains the full routing/pruning suite and two 16-condition normal rewind
suites. The three generated WikiText suites contain approximately 27.0 GB of
artifacts, the cross-initialization replay suite adds approximately 5.83 GB,
and the new TinyStories seed-17/29 suites add the second three-seed dataset.
The load-balancing sweep adds 24 new train/prune runs plus six reused
`aux_loss_weight=0.1` baselines. Phase 4 adds 36 architecture runs and six
additional dataset-validation runs, producing 12 architecture cells and three
dataset settings aggregated across seeds 7, 17, and 29. Representative Phase 4
rewinds add 288 ticket retraining evaluations. Balanced multi-domain causal
controls add 12 train/prune runs across normal, random-every-step,
fixed-random, and shuffled-usage routing.
Broader balanced multi-domain swap interventions add 18 train/prune runs across
global pair swaps, layer-specific swaps, and a global cyclic expert shift.
Subword-tokenized balanced multi-domain causal controls add another 12
train/prune runs across the same four causal-control conditions.

The report's tables were verified against the generated JSON suite summaries,
analysis reports, checkpoint analyses, pruning outputs, and rewind-suite
outputs listed above.

---

## Conclusion

Execution Plan 1 has moved beyond the initial MVP question of whether sparse
expert subnetworks exist. Across TinyStories and three independent
WikiText-103 subset seeds, the experiments show that:

- learned expert masks are materially better than random and transferred masks;
- moderate-sparsity masks satisfy strict lottery-ticket criteria;
- 80%-sparse masks recover near-dense performance after early rewinding;
- randomized routing prevents comparable ticket recovery;
- routing-history similarity strongly tracks mask similarity; and
- expert specialization is real but concentrated more strongly in early
  layers;
- preserving expert usage counts without preserving token identities is not
  sufficient; and
- preserving expert usage counts still fails on the balanced multi-domain
  corpus; and
- global, layer-specific, and cyclic swap interventions all worsen dense loss
  on the balanced multi-domain corpus; and
- stronger load balancing improves dense loss and balance without eliminating
  learned-mask structure; and
- exact foreign-route replay does not transfer source masks across
  initializations; and
- top-2 routing, added depth, expert-count changes, balanced multi-domain
  validation, and representative Phase 4 rewinds preserve non-random
  sparse-mask structure.

The current outcome is strong evidence for routing-conditioned lottery-ticket
behavior in small balanced MoE models. It supports a joint
initialization-routing mechanism rather than a routing-only causal law. Phase
4 now includes representative rewind suites, balanced multi-domain causal
controls, broader swap interventions, and subword-tokenized causal controls.
The next decisive experimental gaps are longer training, iterative pruning,
and broader validation.
