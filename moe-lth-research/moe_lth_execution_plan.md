# Execution Plan: Routing-Conditioned Lottery Tickets in Mixture-of-Experts Models

## Project Title

**Do Routing Trajectories Causally Induce Lottery-Ticket Subnetworks Inside MoE Experts?**

## Core Research Question

Mixture-of-Experts (MoE) models route each input token to a subset of experts. Experts often appear to specialize, but it is unclear whether this specialization reflects real learned structure inside the experts or whether it is mostly an artifact of router geometry, load imbalance, and uneven expert usage.

This project asks:

> Do training-time routing trajectories causally induce stable sparse subnetworks inside experts, or is observed expert specialization mostly explained by router geometry and uneven expert usage?

The strongest version of the claim is not merely that MoE experts can be pruned. The stronger claim is that experts may contain **routing-conditioned lottery tickets**:

> Sparse subnetworks inside experts that are trainable from early expert weights, preserve performance on the expert's routed-token distribution, and depend causally on the routing trajectory experienced during training.

---

# 1. Define the Hypothesis Clearly

## 1.1 Primary Hypothesis

Training-time routing trajectories shape the internal sparse structure of MoE experts.

For expert `e`, let:

```text
H_e = routing history experienced by expert e
M_e = sparse mask/subnetwork discovered inside expert e
```

The main causal claim is:

```text
H_e -> M_e
```

In words:

> Changing the routing history of an expert should change the sparse subnetwork that emerges inside it.

## 1.2 Stronger Lottery-Ticket Hypothesis

A sparse subnetwork inside expert `e` is a lottery ticket if:

```text
m_e ⊙ theta_e,t0
```

can be retrained from an early checkpoint `t0` and recover near-baseline performance.

Where:

- `m_e` is the binary mask inside expert `e`
- `theta_e,t0` is the expert's initialization or early checkpoint
- `⊙` means elementwise multiplication
- `D_e` is the distribution of tokens routed to expert `e`

A **routing-conditioned lottery ticket** exists if this sparse subnetwork succeeds under the original/replayed routing trajectory but weakens under randomized, shuffled, or swapped routing.

---

# 2. Decide the Initial Experimental Scope

## 2.1 Start Small

Do **not** begin with a giant MoE language model. The first version should be a controlled mechanistic study.

Recommended first setup:

| Component | Choice |
|---|---|
| Model type | Small decoder-only Transformer |
| Layers | 4 |
| Attention heads | 4 |
| Embedding dimension | 256 |
| MoE layers | Replace FFN blocks with MoE FFNs |
| Experts per MoE layer | 8 |
| Routing | Top-1 routing first; Top-2 later |
| Expert FFN hidden size | 1024 |
| Dataset | TinyStories, WikiText-103 subset, or OpenWebText subset |
| Objective | Next-token prediction |
| Training tokens | Start with 10M-50M for MVP |
| Hardware | Single GPU is enough for MVP |

## 2.2 Why This Setup Is Enough

The goal is not state-of-the-art language modeling. The goal is to test a causal mechanism:

1. Do experts develop stable sparse subnetworks?
2. Do those subnetworks depend on routing history?
3. Do they satisfy lottery-ticket-style rewind/retrain criteria?

A small model can answer those questions.

---

# 3. Build the Baseline MoE Model

## 3.1 Implement the MoE Layer

Each Transformer block should contain a standard attention block followed by an MoE feedforward block.

A normal dense FFN:

```text
FFN(x) = W2 * activation(W1 * x)
```

becomes:

```text
MoE(x) = sum over selected experts: gate_e(x) * Expert_e(x)
```

For the first version, use Top-1 routing:

```text
MoE(x) = Expert_r(x)(x)
r(x) = argmax_e Router(x)_e
```

## 3.2 Implement the Router

Use a simple learned linear router:

```text
p(e | x) = softmax(W_r x)
```

Track:

- selected expert ID
- router probabilities
- router entropy
- router margin
- token hidden state before routing

## 3.3 Add Load-Balancing Loss

Use a standard auxiliary load-balancing loss so that the model does not collapse onto one or two experts.

Recommended initial settings:

```text
aux_loss_weight = 0.01
top_k = 1
num_experts = 8
capacity_factor = 1.0 or 1.25
```

---

# 4. Train the Baseline Model

## 4.1 Baseline Condition

Train the first model normally:

```text
Model A: Normal learned-router MoE
- learned router
- normal load-balancing loss
- normal top-1 routing
- all experts train normally
```

## 4.2 Save Checkpoints

Save checkpoints at regular intervals:

```text
step 0
step 1k
step 5k
step 10k
step 25k
step 50k
final
```

For each checkpoint, save:

- full model weights
- router weights
- expert weights
- optimizer state if possible
- validation loss
- per-expert token counts
- routing logs on a fixed validation set

## 4.3 Fixed Validation Routing Set

Create a fixed validation set that is reused across checkpoints.

Example:

```text
validation_routing_set = 50k to 200k tokens
```

At every checkpoint, run this exact same validation set through the model and record routing decisions. This allows you to measure whether the same tokens keep going to the same experts over training.

---

# 5. Log Routing Trajectories

## 5.1 What to Log During Training

Do not store every training token's full routing path unless the dataset is tiny. That will become too large.

Instead, store aggregate usage and fixed validation routing.

### Per-Step Expert Usage

For layer `l`, expert `e`, and step `t`:

```text
U_e,l,t = tokens routed to expert e / total tokens in batch
```

Save:

```text
step
layer_id
expert_id
token_count
usage_fraction
router_entropy_mean
router_margin_mean
```

### Validation Routing Decisions

For a fixed validation set, save:

```text
token_position_or_id
layer_id
checkpoint_step
selected_expert
router_probability
router_margin
```

### Expert Token Distribution Samples

For each expert, periodically store a sample of routed tokens/contexts:

```text
layer_id
expert_id
step
token_ids
short decoded context
```

## 5.2 Routing Stability Metric

For token `i`, layer `l`, checkpoints `t1` and `t2`:

```text
S_i,l(t1,t2) = 1 if r_i,l,t1 == r_i,l,t2 else 0
```

Average across tokens:

```text
S_l(t1,t2) = mean_i S_i,l(t1,t2)
```

Interpretation:

- High stability means routing assignments have settled.
- Low stability means the router is unstable or noisy.

---

# 6. Measure Expert Specialization Before Pruning

Before claiming anything about lottery tickets, first establish whether experts specialize.

## 6.1 Expert Usage Entropy

For layer `l`:

```text
H_l = - sum_e U_e,l * log(U_e,l)
```

High entropy means balanced expert usage. Low entropy means uneven expert usage.

Also compute:

```text
coefficient of variation of expert usage
max expert usage / min expert usage
number of dead experts
```

## 6.2 Token Distribution Similarity

For each expert, collect token/context embeddings from the validation routing set.

Compare expert distributions using:

- Jensen-Shannon divergence
- cosine similarity of average context embeddings
- clustering overlap
- token frequency histograms
- syntactic category if available
- semantic cluster labels if using embedding clusters

## 6.3 Functional Specialization

For each expert `e`, collect its routed-token subset:

```text
D_e = {x_i : r(x_i) = e}
```

Evaluate expert-local behavior:

```text
Dense expert e on D_e
Dense expert e on D_j for other experts j
Other experts on D_e
```

You are checking whether expert `e` performs best on the distribution it normally receives.

---

# 7. Extract Sparse Subnetworks Inside Experts

## 7.1 Mask Definition

Each expert has weights:

```text
theta_e
```

A sparse expert is:

```text
m_e ⊙ theta_e
```

where:

- `m_e` is a binary mask
- 1 means the weight is kept
- 0 means the weight is pruned

## 7.2 Start with Magnitude Pruning

Use magnitude pruning first because it is simple, standard, and easy to defend.

For each expert:

1. Rank expert weights by absolute value.
2. Remove the lowest-magnitude weights.
3. Keep the desired sparsity level.
4. Evaluate the pruned expert.

Recommended sparsity levels:

```text
50%
70%
80%
90%
95%
```

## 7.3 Global vs Expert-Local Pruning

Start with expert-local pruning.

### Expert-Local Pruning

Each expert is pruned independently.

This tests:

> Does each expert contain its own sparse subnetwork?

### Global Expert-Layer Pruning

All experts in a layer compete under one pruning threshold.

This tests:

> Are some experts more compressible than others?

For the MVP, use expert-local pruning.

---

# 8. Test Whether Sparse Subnetworks Preserve Expert Function

## 8.1 Post-Training Sparse Subnetwork Test

Take the final trained model and prune expert `e`:

```text
m_e ⊙ theta_e,T
```

Evaluate on:

1. Full validation set
2. Expert-routed validation subset `D_e`
3. Other experts' validation subsets `D_j`

This tells you whether the trained expert contains a sparse functional subnetwork.

## 8.2 Required Controls

Compare:

```text
A. Dense expert
B. Magnitude-pruned expert
C. Random mask at same sparsity
D. Other expert's mask applied to this expert
E. Same mask but randomly reinitialized weights
```

## 8.3 Interpretation

If magnitude-pruned masks work better than random masks, then the sparse structure matters.

If the mask works best on its own routed-token distribution `D_e`, then the sparse subnetwork is expert-specific.

If masks transfer easily across experts, then specialization is weak.

---

# 9. Test Whether the Subnetworks Are Actually Lottery Tickets

Post-training pruning is not enough.

To claim lottery tickets, you need rewind/retrain experiments.

## 9.1 Rewinding Setup

After finding a mask `m_e`, rewind surviving weights to an earlier checkpoint:

```text
m_e ⊙ theta_e,t0
```

Test multiple rewind points:

```text
t0 = initialization
t0 = 1% of training
t0 = 5% of training
t0 = 10% of training
```

## 9.2 Retraining Protocol

For each sparse expert ticket:

1. Apply mask `m_e` to expert `e`.
2. Rewind surviving expert weights to `t0`.
3. Rewind or freeze router depending on the experimental condition.
4. Retrain using the original training schedule.
5. Keep pruned weights fixed at zero.
6. Compare final performance to the dense baseline.

## 9.3 Lottery-Ticket Criteria

A sparse expert subnetwork counts as a ticket if:

```text
sparsity >= 80%
full validation loss degradation <= 2-5%
expert-local loss degradation <= 2-5%
learned mask beats random mask
rewound weights beat random reinitialization
```

For strict LTH:

```text
initialization rewind must work
```

For practical modern LTH:

```text
early-checkpoint rewinding is acceptable
```

## 9.4 Key Controls

For each expert ticket, compare:

```text
A. Dense baseline
B. Final pruned expert, no rewind
C. Mask rewound to initialization
D. Mask rewound to early checkpoint
E. Random mask rewound to same checkpoint
F. Learned mask with random reinitialization
G. Learned mask trained under randomized routing
```

Only C and D support a lottery-ticket claim.

---

# 10. Run Counterfactual Routing Experiments

This is the central novelty of the project.

The question is not merely:

> Are there sparse expert subnetworks?

The real question is:

> Did routing history cause them?

## 10.1 Experiment A: Normal Learned Routing

```text
Model A:
- learned router
- standard load-balancing loss
- normal training
```

Purpose: baseline condition.

Expected result:

```text
Experts develop some specialization.
Some experts contain stable sparse subnetworks.
```

## 10.2 Experiment B: Fixed Random Router

```text
Model B:
- router initialized randomly
- router frozen
- token assignments are random but stable
```

Purpose: separates learned router geometry from stable token assignment.

Question:

> Does stable but random routing induce tickets?

## 10.3 Experiment C: Random Routing Every Step

```text
Model C:
- no stable routing trajectory
- tokens randomly assigned to experts each step
- load balanced by construction
```

Purpose: destroys consistent routing history while preserving expert update count.

Expected result if your hypothesis is right:

```text
Sparse expert subnetworks become less stable and less expert-specific.
```

## 10.4 Experiment D: Replayed Routing History

```text
Model D:
- train Model A normally
- record routing decisions
- train a second model using the recorded routing decisions
- router is bypassed or forced
```

Purpose: tests whether the same routing history reproduces similar masks.

Strong evidence:

```text
Same routing history produces similar sparse masks across runs.
```

## 10.5 Experiment E: Swapped Routing Histories

```text
Model E:
- Expert A receives Expert B's routed-token history
- Expert B receives Expert A's routed-token history
```

Purpose: separates expert identity from routing history.

Strong evidence for the hypothesis:

```text
Expert A receiving B's history develops a B-like mask.
```

## 10.6 Experiment F: Same Usage Counts, Shuffled Token Identities

```text
Model F:
- each expert receives the same number of tokens as baseline
- token identities are shuffled across experts
```

Purpose: separates update frequency from token-distribution specialization.

Strong evidence:

```text
Matching usage counts is not enough.
Preserving token trajectory matters.
```

## 10.7 Experiment G: Strong Load-Balancing Router

```text
Model G:
- learned router
- stronger load-balancing loss
- expert usage more uniform
```

Purpose: tests whether specialization survives when usage imbalance is reduced.

Strong evidence:

```text
Specialized masks still emerge under balanced usage.
```

---

# 11. Measure Mask Stability and Specificity

## 11.1 Mask Overlap

For expert `e`, compare masks from two checkpoints:

```text
Overlap(m_e,t1, m_e,t2) = |intersection| / |union|
```

This is Jaccard similarity.

Compute:

```text
same expert across checkpoints
same expert across random seeds
different experts in same layer
same expert under replayed routing
same expert under random routing
same expert under swapped routing
```

## 11.2 Expected Patterns

### Supports Routing-Conditioned Tickets

```text
Normal routing:
- high within-expert mask stability

Random routing:
- low mask stability

Replayed routing:
- similar masks to original

Swapped routing:
- masks follow routing history

Shuffled token identities:
- masks weaken or change
```

### Refutes Routing-Conditioned Tickets

```text
Random routing gives equally stable masks.
Masks are mostly determined by initialization.
Masks transfer freely across experts.
Replayed routing does not reproduce similar masks.
Swapped routing does not change mask identity.
```

---

# 12. Analyze Router Geometry

Because the proposal asks whether specialization is caused by router geometry, measure the router directly.

## 12.1 Router Vector Similarity

The router has vectors `w_e` for each expert.

Measure:

```text
cosine_similarity(w_e, w_j)
```

Question:

> Are experts with similar router vectors also similar in sparse-mask structure?

## 12.2 Router Margin

For input `x`, define router margin as:

```text
margin(x) = p(top expert | x) - p(second expert | x)
```

High margin means routing is confident.

Analyze:

```text
Do high-margin routed tokens produce stronger expert-specific tickets?
Do low-margin tokens correspond to unstable routing and weaker tickets?
```

## 12.3 Hidden-State Clustering

At each MoE layer:

1. Collect hidden states before routing.
2. Color points by selected expert.
3. Use PCA/UMAP/t-SNE for visualization.
4. Measure cluster separability quantitatively.

Useful metrics:

```text
silhouette score
linear probe accuracy for expert ID
intra-expert vs inter-expert distance
```

## 12.4 Router Geometry vs Routing History

Compare:

```text
router vector similarity
routing-history similarity
mask similarity
expert-local performance
```

The key question:

> Does mask similarity track static router geometry, or does it track actual token routing history?

---

# 13. Organize Experimental Phases

## Phase 1: MVP

Goal:

> Show whether stable sparse expert subnetworks exist at all.

Run:

```text
Model A: normal learned-router MoE
Model C: random routing every step
```

Measure:

```text
routing stability
expert usage
expert-local validation loss
magnitude-pruned masks
random-mask controls
mask overlap across checkpoints
```

Deliverable:

```text
Initial evidence that normal routing produces more stable and expert-specific sparse subnetworks than random routing.
```

## Phase 2: Lottery-Ticket Validation

Goal:

> Determine whether sparse expert subnetworks qualify as lottery tickets.

Run:

```text
rewind to initialization
rewind to 1%, 5%, 10% checkpoints
random mask controls
random reinitialization controls
```

Deliverable:

```text
Evidence for or against true lottery-ticket behavior inside experts.
```

## Phase 3: Causal Routing Tests

Goal:

> Test whether routing history causes the sparse masks.

Run:

```text
replayed routing
swapped routing
same usage counts with shuffled token identities
strong load-balancing router
```

Deliverable:

```text
Causal evidence that sparse expert tickets follow routing history rather than expert identity or usage count alone.
```

## Phase 4: Scaling and Robustness

Goal:

> Show the result is not an artifact of one toy setup.

Vary:

```text
number of experts: 4, 8, 16
top-k: 1 vs 2
dataset: TinyStories vs WikiText
model depth: 4 vs 8 layers
random seeds: at least 3
load-balancing strength
sparsity levels: 50%, 70%, 80%, 90%, 95%
```

Deliverable:

```text
Robustness tables and ablation results.
```

---

# 14. Minimum Viable Paper Experiment

If time is limited, run this:

## Models

```text
A. Normal learned-router MoE
B. Random routing every step
C. Replayed routing from A
D. Swapped routing histories
```

## Metrics

```text
routing stability
expert usage entropy
expert-local loss
mask Jaccard overlap
pruned performance
rewound-ticket performance
random-mask baseline
random-reinit baseline
```

## Main Claims You Could Test

```text
1. Normal MoE experts contain stable sparse subnetworks.
2. These subnetworks are more stable than those produced under random routing.
3. Rewinding shows whether they qualify as lottery tickets.
4. Replayed/swapped routing tests whether masks are caused by routing history.
```

This is enough for a serious research proposal and possibly an early workshop paper if the results are clean.

---

# 15. Expected Results and Interpretations

## 15.1 Result Pattern Supporting the Hypothesis

| Observation | Interpretation |
|---|---|
| Normal routing produces stable masks | Expert subnetworks are structured |
| Random routing weakens masks | Stable routing matters |
| Replayed routing reproduces masks | Routing history is causal |
| Swapped routing makes masks follow token history | Expert identity is not the main cause |
| Random masks fail | Mask structure matters |
| Random reinitialization fails | Early weights matter |
| Rewound masks retrain successfully | Lottery-ticket evidence |
| Masks specialize to own routed tokens | Expert-specific sparse computation |

## 15.2 Result Pattern Against the Hypothesis

| Observation | Interpretation |
|---|---|
| Random routing produces equally good masks | Routing history may not matter |
| Masks transfer across experts | Experts may be redundant |
| Random masks perform similarly | Sparse architecture may not be meaningful |
| Random reinitialization performs similarly | Not a classical lottery ticket |
| Replay does not reproduce masks | Routing history alone may not determine masks |
| Usage count predicts everything | Uneven update frequency may explain specialization |

Negative results are still valuable because they clarify whether MoE specialization is real or mostly superficial.

---

# 16. Figures to Produce

## Figure 1: Conceptual Diagram

Show:

```text
tokens -> router -> experts -> sparse masks inside experts
```

Highlight:

```text
routing trajectory H_e causes or fails to cause sparse mask M_e
```

## Figure 2: Routing Stability Over Training

Plot:

```text
x-axis: checkpoint
y-axis: same-token same-expert routing agreement
```

Compare:

```text
normal routing
random routing
fixed random routing
```

## Figure 3: Expert Usage Distribution

Plot expert usage per layer.

Show whether specialization is confounded with uneven expert usage.

## Figure 4: Mask Stability

Plot Jaccard overlap of masks over checkpoints.

Compare:

```text
normal routing
random routing
replayed routing
swapped routing
```

## Figure 5: Pruning Curves

Plot:

```text
x-axis: sparsity
y-axis: validation loss or perplexity
```

Compare:

```text
learned masks
random masks
other-expert masks
```

## Figure 6: Rewind Curves

Plot performance for masks rewound to:

```text
initialization
1%
5%
10%
final
```

## Figure 7: Routing History vs Mask Similarity

Scatterplot:

```text
x-axis: routing-history similarity
y-axis: mask similarity
```

If the hypothesis is correct, these should correlate.

## Figure 8: Expert-Local Specificity

Heatmap:

```text
rows: expert mask used
columns: routed-token subset evaluated
cell: loss/perplexity
```

Strong specialization appears as a strong diagonal.

---

# 17. Tables to Produce

## Table 1: Model Configurations

List all experimental conditions.

## Table 2: Main Performance Results

Include:

```text
dense baseline
pruned final
rewound ticket
random mask
random reinit
```

## Table 3: Routing Interventions

Include:

```text
normal
fixed random
random every step
replay
swap
shuffled usage
strong load balancing
```

For each:

```text
routing stability
usage entropy
mask stability
expert-local performance
```

## Table 4: Robustness

Break down by:

```text
seed
dataset
number of experts
top-k
sparsity level
```

---

# 18. Implementation Checklist

## 18.1 Code Components

You need modules for:

```text
MoE Transformer model
router logging
expert usage tracking
checkpointing
validation routing replay
mask generation
mask application
rewind/retrain protocol
counterfactual routing
metrics computation
plotting
```

## 18.2 Repository Structure

Recommended structure:

```text
moe-lth/
  configs/
    baseline.yaml
    random_routing.yaml
    fixed_router.yaml
    replay_router.yaml
    swapped_router.yaml

  data/
    raw/
    processed/

  src/
    models/
      transformer.py
      moe_layer.py
      router.py
      experts.py

    training/
      train.py
      evaluate.py
      checkpoint.py

    routing/
      log_routes.py
      replay_routes.py
      swap_routes.py
      random_routes.py

    pruning/
      magnitude_prune.py
      masks.py
      rewind.py

    analysis/
      routing_stability.py
      expert_usage.py
      mask_overlap.py
      expert_specificity.py
      router_geometry.py

    visualization/
      plot_routing.py
      plot_masks.py
      plot_pruning.py

  docs/
    experiments/
      phase1_mvp.md
      phase2_rewind.md
      phase3_counterfactuals.md
      phase4_scaling.md

  results/
    tables/
    figures/
    logs/

  README.md
```

---

# 19. Common Failure Modes

## 19.1 Expert Collapse

Problem:

```text
One or two experts get most tokens.
```

Fix:

```text
increase load-balancing loss
use capacity constraints
use router noise during training
monitor dead experts early
```

## 19.2 Routing Logs Become Too Large

Problem:

```text
Full routing histories are expensive to store.
```

Fix:

```text
store full routing only for fixed validation sets
store aggregate usage during training
sample training batches periodically
compress expert IDs as int8/int16 arrays
```

## 19.3 Pruning Is Too Expensive

Problem:

```text
IMP across all experts and seeds is costly.
```

Fix:

```text
start with one-shot magnitude pruning
only run IMP after MVP works
prune one MoE layer first
focus on 80% and 90% sparsity initially
```

## 19.4 MoE Training Is Noisy

Problem:

```text
Different seeds give different routing patterns.
```

Fix:

```text
run at least 3 seeds
report confidence intervals
use fixed validation routing set
separate robust results from anecdotal examples
```

## 19.5 Lottery-Ticket Claim Is Too Strong

Problem:

```text
Pruned subnetworks work, but rewound subnetworks fail.
```

Fix:

Use the more accurate claim:

```text
MoE experts contain stable sparse subnetworks, but not classical lottery tickets.
```

Or:

```text
Expert tickets are only visible under early checkpoint rewinding, not initialization rewinding.
```

That is still interesting.

---

# 20. Timeline

## Week 1: Literature and Setup

Deliverables:

```text
finalized research question
repository initialized
small Transformer baseline working
dataset pipeline working
```

## Week 2: MoE Baseline

Deliverables:

```text
MoE layer implemented
learned router working
load-balancing loss added
baseline training complete
routing logs saved
```

## Week 3: Routing Analysis

Deliverables:

```text
expert usage plots
routing stability plots
expert token distribution analysis
initial specialization metrics
```

## Week 4: Pruning Pipeline

Deliverables:

```text
expert-local magnitude pruning
random mask controls
pruning curves
expert-local evaluation
```

## Week 5: Rewind / Lottery-Ticket Tests

Deliverables:

```text
rewind to initialization
rewind to early checkpoint
random reinitialization controls
ticket criteria table
```

## Week 6: Counterfactual Routing MVP

Deliverables:

```text
random routing model
replayed routing model
comparison against baseline
mask stability results
```

## Week 7: Strong Causal Tests

Deliverables:

```text
swapped routing experiment
same-usage shuffled-token experiment
strong load-balancing experiment
```

## Week 8: Robustness and Writing

Deliverables:

```text
3-seed results
final figures
final tables
proposal/paper draft
limitations section
future work section
```

---

# 21. Final Paper/Proposal Structure

## Abstract

State the problem: MoE experts appear specialized, but the mechanism is unclear.

State the contribution: test whether routing trajectories induce sparse expert tickets.

State methods: routing logs, expert pruning, rewind tests, counterfactual routing interventions.

## Introduction

Cover:

```text
MoE models use sparse routing.
Expert specialization is often assumed.
Lottery-ticket theory studies sparse subnetworks.
No work directly tests whether routing trajectories cause sparse tickets inside experts.
```

## Related Work

Organize into:

```text
MoE routing and specialization
MoE pruning and compression
Lottery Ticket Hypothesis
Causal/mechanistic analysis of neural networks
```

## Method

Include:

```text
model architecture
routing trajectory logging
mask extraction
rewind/retrain tests
counterfactual routing interventions
metrics
```

## Experiments

Include:

```text
baseline MoE
random routing
fixed router
replay routing
swap routing
shuffled usage
load-balancing ablation
```

## Results

Organize around questions:

```text
Do experts specialize?
Do experts contain sparse subnetworks?
Are subnetworks lottery tickets?
Does routing history causally shape them?
```

## Discussion

Address:

```text
what counts as a routing-conditioned ticket
what this implies for MoE interpretability
what this implies for MoE pruning
what this implies for training dynamics
```

## Limitations

Be honest:

```text
small model scale
language-model setting only
routing logs approximate full history
LTH definitions vary
IMP is computationally expensive
larger MoEs may behave differently
```

## Conclusion

Return to the central claim:

> This work reframes MoE expert specialization as a possible sparse-subnetwork phenomenon shaped by routing trajectories.

---

# 22. The Most Important Experimental Logic

The project succeeds if it cleanly separates these explanations:

| Explanation | Test |
|---|---|
| Expert identity matters | swapped routing |
| Initialization matters | random reinitialization |
| Usage count matters | same-usage shuffled tokens |
| Router geometry matters | fixed/random/learned router comparisons |
| Routing trajectory matters | replay and swap experiments |
| Mask structure matters | learned mask vs random mask |
| Lottery-ticket behavior exists | rewind/retrain tests |

The central result you want is:

```text
Tickets follow routing history more than expert identity.
```

That is the clearest contribution.

---

# 23. Best Final Framing

The strongest version of the research proposal is:

> We investigate whether MoE expert specialization is encoded as routing-conditioned sparse subnetworks. Unlike prior work that studies expert usage, expert pruning, or router stability at the module level, we test whether training-time routing trajectories causally induce lottery-ticket-like subnetworks inside individual experts. By combining expert-local pruning, rewinding, and counterfactual routing interventions, we distinguish genuine sparse expert specialization from artifacts of router geometry and uneven expert usage.

This is specific, testable, and original enough to be worth pursuing.
