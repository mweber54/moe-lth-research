# ICLR 2027 Revision Plan
## Routing-Conditioned Optimization Shapes Sparse Supports in Mixture-of-Experts Models

This plan consolidates the two reviewer reports into one submission-focused revision strategy. The goal is not to add experiments indiscriminately. The goal is to remove the strongest intellectually defensible rejection arguments while making the paper's genuinely novel contribution unmistakable.

---

## Core Revision Principle

The strongest version of this paper is **not**:

> Routing changes MoE experts, or routing alone creates lottery tickets.

Both are either obvious or too strong.

The paper should instead establish:

> **Conditional on initialization and training configuration, interventions on the detailed routed optimization trajectory change which expert parameters form optimization-relevant sparse supports, even beyond coarse expert utilization statistics. The resulting supports are not determined by routing alone, but emerge through an interaction among routing, initialization, and optimization.**

The paper's novelty should be centered on the **controlled routing-intervention methodology + sparse-support/rewind probe**, not on magnitude pruning itself, not on the generic fact that routing changes gradients, and not on the equation \(A \rightarrow \nabla L \rightarrow \Theta \rightarrow m\).

---

# P0 — Must Fix Before Submission

These are the issues most likely to cause rejection. Complete these before spending substantial time on cosmetic changes or broad extra sweeps.

## P0.1 — Add a Compute-Matched Rewind Experiment

### Reviewer concern
The current early-rewind protocol restores parameters to \(t_0\) but then gives the sparse model a fresh **full \(T\)-step training budget**. Thus a 10% rewind receives information from the first 10% of dense training and then another complete training run. This is not directly compute-matched to the original dense trajectory.

This creates a serious alternative explanation for the 80%-sparsity result and makes the phrase "requires early rewinding" vulnerable.

### Experiment
For the reference TinyStories and WikiText settings, run:

- Sparsity: 50%, 80%
- Rewind point: 0%, 1%, 5%, 10%
- Conditions:
  - learned magnitude mask
  - matched random mask
  - learned mask + random reinitialization
  - dense reference

Add a second protocol:

**Compute-matched continuation**
- rewind to checkpoint \(t_0\)
- apply the final learned mask
- restore the appropriate optimizer/schedule state or clearly specify the restart rule
- train for only \(T-t_0\) additional steps
- continue from the corresponding point in the ordered data stream, or explicitly define and justify the data continuation

Keep the current full-budget-restart results as a separate protocol.

### Required output
Create one table/figure comparing:

| Rewind | Full-budget restart | Compute-matched continuation | Dense | Random mask | Random reinit |
|---|---:|---:|---:|---:|---:|

### Decision rule
- **If 80% remains near dense under compute matching:** retain the early-rewind interpretation with much stronger confidence.
- **If the effect weakens substantially:** stop calling the current protocol canonical early rewinding. Rename it to something like **early-state full-budget retraining** and limit the lottery-ticket claim to the 50% initialization-rewind result.
- **If 50% initialization rewind remains strong:** preserve that as the cleanest lottery-ticket result regardless of the 80% outcome.

---

## P0.2 — Deconfound the Shuffled-Usage Routing Intervention

### Reviewer concern
The current shuffled-usage intervention preserves primary expert-selection counts per layer and step, but does not preserve:

- raw gate values,
- capacity ranking,
- accepted-token identity,
- acceptance masks,
- gate-weighted gradient mass.

Therefore the current experiment establishes that **something about the realized routed update differs beyond counts**, but it does not cleanly isolate token identity/order as the cause.

### Required experiment
Implement at least one stronger intervention that separates token identity/history from gate/capacity effects.

Preferred hierarchy:

#### Option A — Fixed-gate replay shuffle
Archive enough information from a source run to replay:
- primary expert ID,
- gate value for the selected expert,
- capacity acceptance status.

Then permute token-to-expert assignments while preserving or explicitly replaying the gate/acceptance quantities.

This is the strongest option.

#### Option B — Accepted-token-only identity shuffle
Archive the exact accepted assignments and selected gate values, then permute token identities among accepted expert slots while preserving:
- accepted count,
- gate distribution,
- expert/step/layer allocation.

#### Option C — Matched gate-distribution shuffle
If exact replay is technically impractical, stratify or construct shuffled assignments so each expert receives a matched gate-weight distribution and accepted count.

### Required output
Report for each intervention:

- dense validation loss,
- route disagreement from learned routing,
- gate-distribution difference,
- accepted-count difference,
- magnitude-vs-random mask advantage,
- mask Jaccard vs. learned routing,
- rewind/retrain performance.

### Claim language after experiment
Only claim what the intervention isolates.

Examples:

- If identity-only perturbation changes support:  
  **"Token assignment identity/order contributes to support formation even when utilization and gate/acceptance statistics are controlled."**
- If the effect disappears after gate control:  
  **"The original effect is mediated substantially by gate/capacity-weighted optimization rather than token identity alone."**

Either outcome is scientifically useful.

---

## P0.3 — Add a Matched-Dense-Quality Routing Perturbation

### Reviewer concern
Random-every-step and shuffled routing substantially worsen dense-model performance. A reviewer can argue that the reduced magnitude-mask advantage is simply a consequence of training a worse or less-converged network, rather than evidence that coherent routing history shapes sparse structure.

Condition-specific normalization helps but does not fully eliminate this explanation.

### Experiment
Create a graded routing-corruption sweep that preserves per-expert counts while varying how much of the learned trajectory is perturbed.

For example:

- 0% shuffled
- 10%
- 25%
- 50%
- 75%
- 100%

At every level, preserve the strongest feasible utilization/gate/capacity controls from P0.2.

Alternatively, use expert-ID permutation interventions that already produce smaller dense-loss changes and expand those into the main mechanistic analysis.

### Required analysis
For every condition measure:

- dense validation loss,
- routing disagreement,
- mask Jaccard,
- magnitude-mask advantage,
- rewind performance.

Then identify pairs of conditions with similar dense loss but substantially different routed histories.

### Ideal result
The strongest causal result would be:

> Two conditions reach approximately the same dense validation loss, but the condition with more disrupted routing history forms substantially different/weaker expert-local sparse support.

This directly defeats the "you just trained a worse network" rejection argument.

---

## P0.4 — Properly Replicate Cross-Initialization Replay

### Reviewer concern
The current cross-initialization result uses one source seed and only two target seeds. It is a headline result but effectively has \(n=2\) targets and no meaningful uncertainty estimate.

### Experiment
Minimum:
- 1 source × at least 4–5 target initializations.

Preferred:
- 3–4 source routing histories × 3–4 target initializations.

For each pair:
- replay the exact source primary-ID trajectory,
- train matched learned-routing target,
- compare dense loss,
- compare 50% and 80% support overlap,
- compare rewindability,
- include symmetry-aligned mask overlap as currently done.

### Required reporting
Report:
- mean ± SD/CI across target initializations,
- source-level variation,
- within-initialization learned-vs-replay difference,
- source-vs-target support overlap.

### Reframing
Cross-init should become a **boundary-condition / interaction experiment**, not the main proof of causality.

Do not frame the primary scientific null as:

\[
m = g(A)
\]

independent of initialization, because that is an implausibly strong hypothesis.

Instead ask:

> **How much support variation is explained by route history conditional on initialization, and how much changes across initializations under the same route?**

---

## P0.5 — Run a Routing × Initialization Factorial / Variance-Decomposition Experiment

This combines and upgrades P0.4 into a genuinely mechanistic result.

### Design
Generate routing trajectories:

\[
A_1, A_2, A_3, \dots
\]

and initializations:

\[
\Theta_1, \Theta_2, \Theta_3, \dots
\]

Train/replay each route under multiple initializations.

For each \(A_i,\Theta_j\) pair, measure:

- dense loss,
- 50%/80% mask Jaccard,
- functional mask advantage,
- initialization-rewind performance,
- early-rewind performance if retained after P0.1.

### Analysis
Estimate:
- route-history effect,
- initialization effect,
- route × initialization interaction.

A mixed-effects model, ANOVA-style decomposition, or variance-components analysis is sufficient if assumptions are stated carefully.

### Why this matters
This turns the phrase **"routing, initialization, and optimization interact"** from a nearly tautological statement into a quantified empirical result.

If compute is limited, prioritize this experiment over adding many more architecture cells.

---

## P0.6 — Strengthen or Weaken the "Expert-Specific" Claim

### Reviewer concern
The main expert-transfer control only copies expert 0's mask to expert 1. That is not enough to justify a general claim that supports are "expert-specific."

### Preferred experiment
For each layer, construct a complete expert-mask transfer matrix:

\[
T_{ij} = \text{performance when mask from expert } i \text{ is applied to expert } j.
\]

Run this at 50% and 80% sparsity.

Report:
- diagonal vs. off-diagonal degradation,
- mean within-layer transfer penalty,
- whether expert matching/permutation alignment changes the result.

### Decision rule
- If diagonal/self masks consistently outperform off-diagonal transfers, keep **expert-specific sparse supports**.
- Otherwise change the paper everywhere to **non-random expert-local sparse supports**.

Do not overclaim uniqueness if the evidence only establishes locality.

---

## P0.7 — Test the Repeated-Data-Order / Memorization Alternative

### Reviewer concern
The current loader cycles a fixed shuffled batch order. Some datasets are traversed many times, and the long-budget run repeats the same small corpus heavily. Since this paper is explicitly about ordered optimization history, the fixed repeated order is scientifically relevant, not merely an implementation detail.

### Experiment
Run the reference causal experiment with:
- fresh reshuffling each epoch, or
- a substantially larger corpus that is not repeatedly traversed during the training budget.

Minimum conditions:
- learned routing,
- count-preserving shuffle,
- random/step or one matched-quality perturbation,
- direct-pruning mask comparison,
- 50% initialization rewind,
- 80% best relevant rewind protocol.

### Stronger version
Run one meaningfully larger-data experiment where total processed tokens are comparable but corpus reuse is drastically reduced.

### Goal
Show that routing-conditioned support is not an artifact of repeatedly seeing the same examples in the same order.

---

## P0.8 — Add One Meaningfully Larger-Scale Validation

### Reviewer concern
The existing robustness grid changes expert count, top-k, and depth, but remains in a small-model regime. Reviewers may interpret the paper as a carefully instrumented toy study.

### Experiment target
Do **one** larger-scale mechanistic replication rather than another broad grid.

Target roughly:
- 5–10× current parameter count if feasible,
- larger/non-repeated corpus,
- same core top-1 intervention protocol.

You do not need every current ablation.

Minimum:
- learned routing dense baseline,
- matched/count-preserving routing intervention,
- 50% magnitude vs random,
- 50% initialization rewind,
- 80% one key rewind condition,
- mask overlap.

### Interpretation
Treat this as an external-validity probe, not a scaling law.

If compute is constrained, P0.1–P0.7 are scientifically more important. However, a larger-scale result materially improves the ICLR significance case.

---

# P1 — Strongly Recommended

These changes are likely to increase reviewer scores materially once the causal core is fixed.

## P1.1 — Increase Seeds Only for the Mechanistic Core

Do not rerun the entire architecture grid with 10 seeds.

Use approximately 8–10 seeds for the reference conditions most central to the causal claim:

- learned routing,
- exact replay,
- deconfounded shuffled routing,
- matched-quality perturbation,
- optionally random-every-step.

Report paired uncertainty across seeds.

Three seeds remain acceptable for broad robustness screens with very large effect sizes.

---

## P1.2 — Standardize Statistical Reporting

### Fix immediately
Use one convention throughout:
- preferably **sample standard deviation** for seed-to-seed variability.

Do not mix sample and population SD across tables.

### Add
For the main paired comparisons:
- paired bootstrap confidence intervals across seeds, and/or
- paired effect sizes.

Do not pretend experts/tokens/layers are independent replicates.

For the route-agreement/mask-Jaccard correlation:
- report the correlation with and without the exact replay point \((1,1)\),
- explicitly note that pairwise condition comparisons are not independent,
- demote this analysis if it remains unstable across datasets.

For \(n=3\), avoid over-relying on p-values. Confidence intervals/effect sizes and replication across conditions are more informative.

---

## P1.3 — Demote or Replace the Route-Agreement/Jaccard Correlation

The correlation is approximately 0.78 in the original single-domain experiments but drops dramatically in mixture/tokenizer suites.

Therefore it should not be presented as a universal relationship.

### Preferred action
Replace Figure 3b with a stronger mechanistic result from:
- the graded routing perturbation sweep,
- matched-dense-quality intervention,
- or routing × initialization decomposition.

Move the correlation to the appendix.

If retained:
- label it explicitly as descriptive,
- show the no-replay-point estimate,
- state in the main text that it is not stable across all evaluated datasets/tokenizers.

---

## P1.4 — Promote IMP Enough to Establish Protocol Dependence

The current representative IMP result shows that pruning method can alter high-sparsity behavior.

Do not turn the paper into an IMP paper.

Instead run a focused comparison on the two principal datasets:

- one-shot magnitude mask,
- iterative magnitude pruning,
- 50% initialization rewind,
- 80% initialization rewind,
- 80% best early/compute-matched rewind.

### Goal
Answer:

> Is the claimed 50%-vs-80% boundary a property of routing-conditioned support itself, or partly a property of one-shot magnitude mask extraction?

If IMP changes the conclusion, rewrite the paper accordingly rather than hiding it.

---

## P1.5 — Clarify Top-1 vs. Top-2 Generalization

The full causal intervention suite is fundamentally a top-1 result because the archive lacks complete secondary top-k information.

State this clearly.

The architecture grid may still show structural robustness for top-2, but do not imply that the routing-history causal mechanism has been fully demonstrated for top-2.

If practical, add one full top-2 trace/replay experiment with complete top-k IDs, gates, and acceptance information. This is useful, but lower priority than the top-1 causal fixes.

---

## P1.6 — Update and Sharpen Related Work

The Related Work section should directly separate the submission from the closest prior art.

### Must explicitly distinguish

#### Frankle & Carbin / Frankle et al.
Established:
- winning tickets,
- initialization rewind,
- early/matching rewinds.

Your contribution:
- routing as an experimentally manipulated candidate mechanism shaping expert-local support.

#### Paul et al. — Unmasking the Lottery Ticket Hypothesis
Established:
- mask identity carries optimization-relevant information.

Your contribution:
- test whether MoE training-time routing helps determine which mask identity emerges.

#### StableMoE
Established:
- unstable routing changes expert training exposure.

Your contribution:
- test whether training-time routing leaves persistent, rewindable sparse structure inside expert parameters.

#### Routing the Lottery
Verify carefully before submission.
State the exact distinction between:
- designed/adaptive sparse subnetworks or predefined heterogeneous partitions,
- versus sparse supports emerging inside ordinarily trained MoE experts and then being probed with routing interventions.

#### MoE-Pruner / STUN / expert-pruning literature
Established:
- trained MoEs are compressible and router/weight/activation information can guide pruning.

Your contribution:
- final-checkpoint compression is not the endpoint; you ask whether the resulting support has rewindable optimization significance and whether training-time routing causally affects its formation.

#### Recent router–expert geometry / routed-gradient-history work
Independently verify and cite the closest 2026 work showing router/expert parameters accumulating geometry from routed-token histories.

Do **not** claim that "routing history changes expert gradients" is novel.

Your novelty must remain the sparse-support + intervention + rewind connection.

---

## P1.7 — Rewrite the Theoretical/Problem-Formulation Section

### Current issue
The current formalism risks presenting an intuitive computational dependency as a theoretical contribution.

### New structure
Keep only the notation needed for the experiments:

1. routed trajectory \(A\),
2. realized accepted/gated update trajectory \(G\) or equivalent,
3. support extraction \(m^{(\rho)}\),
4. rewind metric \(\Delta_{\text{ret}}\).

Then define actual empirical questions:

- **Q1 Structural:** Does trained routing produce support that outperforms matched sparse controls?
- **Q2 Optimization:** Does that support remain useful after initialization/early-state rewinding?
- **Q3 Routing intervention:** At fixed initialization, does altering routing history change support?
- **Q4 Utilization control:** Does the effect persist beyond matched selection/acceptance statistics?
- **Q5 Initialization interaction:** Does the same route produce identical support across initializations?

Avoid treating \(m=g(A,\Theta_0,O)\) as a theory. Call it a **dependency framework** or **experimental hypothesis decomposition**.

---

## P1.8 — Reframe the Paper's Contribution Hierarchy

The three current contributions should not appear equally novel.

Recommended hierarchy:

1. **Primary mechanistic contribution:** controlled routing interventions show that detailed training-time routing affects the formation of optimization-relevant expert-local support beyond coarse utilization.
2. **Necessary optimization evidence:** rewind/retrain establishes that the support is not merely a final-checkpoint compression artifact.
3. **Supporting structural evidence:** magnitude masks outperform random/transfer controls across architecture/data variants.
4. **Boundary condition:** cross-initialization shows routing history is influential but not a portable coordinate-level blueprint.

This makes the story coherent and prevents reviewers from dismissing the work as "magnitude pruning applied to MoEs."

---

# P2 — Writing, Figures, and Presentation

## P2.1 — Rewrite the Abstract Last

After experiments are complete, the abstract should contain only claims that survive the P0 tests.

Avoid:
- "token identity causally determines..."
- "routing alone..."
- "80% requires early rewind" unless compute-matched results justify it.

Preferred structure:
1. unresolved question,
2. probe,
3. strongest structural finding,
4. rewind finding,
5. cleanest routing intervention,
6. bounded conclusion.

Use "causal" only when the intervention target is explicitly stated.

---

## P2.2 — Rewrite the Introduction Around One Question

The introduction should converge quickly on:

> Does the detailed routed optimization trajectory leave persistent, optimization-relevant sparse support inside MoE experts?

Then establish:
- final-checkpoint pruning is insufficient,
- semantic specialization is not the question,
- rewindability tests optimization significance,
- routing interventions test mechanism.

Reduce defensive prose and repeated caveats.

---

## P2.3 — Compress Section 3 Formalism by ~30–40%

Keep enough notation to define interventions and metrics.

Cut equations or prose that merely restate:
- routing selects gradients,
- gradients change weights,
- weights determine magnitude masks.

Use the saved page budget for:
- the deconfounded intervention,
- compute-matched rewind,
- stronger cross-init/factorial result.

---

## P2.4 — Replace Figure 1 With a True Experimental Schematic

Current Figure 1 is mostly a textual table.

New Figure 1 should visually show four training lanes:

1. **Learned routing**
2. **Exact replay**
3. **Count/gate-controlled shuffled routing**
4. **Cross-initialization replay**

Show what is held fixed and what changes:
- initialization,
- data order,
- expert counts,
- token identity,
- gates,
- capacity acceptance.

End each lane with:
- train → extract mask → rewind/retrain.

The figure should let a reviewer understand the causal design without reading a paragraph.

---

## P2.5 — Make Figure 3 the Main Evidence Figure

Recommended panels:

### Panel A
Magnitude-mask advantage under routing interventions.

### Panel B
Graded route perturbation / matched-dense-quality result.

### Panel C
Routing × initialization variance decomposition or replicated cross-init result.

Move the original route/Jaccard correlation to the appendix unless it becomes substantially more informative.

---

## P2.6 — Improve Figure 2

Keep the direct-pruning and rewind visualization.

Add the 50% initialization-rewind result somewhere visible because it is the cleanest lottery-ticket result.

If compute-matched and full-budget protocols differ, show both clearly rather than hiding the distinction in text.

---

## P2.7 — Simplify Figure 4

Use Figure 4 only for robustness:
- architecture grid,
- dataset/tokenizer robustness,
- optional larger-scale result.

Move representative rewind controls elsewhere if the figure becomes too dashboard-like.

---

## P2.8 — Tighten Terminology

Use the following consistently:

- **Sparse support:** binary coordinate mask.
- **Expert-local support:** support extracted independently within an expert.
- **Expert-specific support:** only if full transfer evidence establishes specificity.
- **Winning ticket:** support + original initialization satisfying the initialization-rewind criterion.
- **Early-rewind / matching subnetwork:** only for a protocol aligned with the intended definition.
- **Full-budget early-state retraining:** if the current non-compute-matched protocol is retained.
- **Routing-conditioned optimization:** experimental/mechanistic framing, not a proven general theory.
- **Causal contribution:** only for the precisely manipulated routing variable.
- **Semantic specialization:** explicitly not established.

---

## P2.9 — Report Sparsity More Precisely

For every 50%/80% result, report:

- expert-weight sparsity,
- total-model/global parameter sparsity,
- optionally active parameters per token.

Avoid letting "80%-sparse model" imply 80% of the entire model is pruned when only expert weight matrices are masked.

---

## P2.10 — Clean Statistical Precision

- Standardize SD convention.
- Round means/SDs to sensible precision.
- Stop reporting four decimals when the seed uncertainty does not justify it.
- Keep exact values in appendix tables or machine-readable artifacts.
- Use the Results prose for effect direction and magnitude, not long lists of exact numbers.

---

## P2.11 — Move Important Caveats Into the Main Results

Do not leave these only in the appendix/limitations:

- shuffle does not perfectly isolate identity under the old protocol,
- cross-init replication count,
- top-2 archive limitation,
- route/Jaccard correlation is not universal,
- early rewind currently uses a full retraining budget,
- long-budget best-checkpoint selection is validation-biased.

Reviewers will find them. Put them next to the corresponding result and control the interpretation yourself.

---

# Recommended Experimental Execution Order

This ordering maximizes information gained per unit compute and avoids wasting runs before protocol problems are resolved.

## Phase 1 — Fix identification
1. Implement richer routing trace storage: IDs, gate values, acceptance status.
2. Run deconfounded shuffle on reference WikiText.
3. Run graded/matched-quality routing corruption.
4. Run compute-matched rewind on WikiText.
5. Repeat the successful designs on TinyStories.

## Phase 2 — Fix replication
6. Expand reference routing interventions to 8–10 seeds.
7. Expand cross-init to a multi-source/multi-target design.
8. Run routing × initialization factorial analysis.

## Phase 3 — Fix alternative explanations
9. Run reshuffled-each-epoch or non-repeated-data experiment.
10. Run complete expert-mask transfer matrix.
11. Run focused IMP comparison.

## Phase 4 — External validity
12. Run one larger-scale/non-repeated-corpus replication.
13. If feasible, run one full top-2 causal intervention with complete trace logging.

## Phase 5 — Rewrite paper
14. Rewrite claims based on actual outcomes.
15. Rewrite Related Work.
16. Rewrite Section 3.
17. Redesign Figures 1–4.
18. Standardize statistics and terminology.
19. Rewrite abstract last.
20. Final hostile-review pass.

---

# Minimum Viable Revision vs. Strong Revision

## Minimum viable revision for resubmission
If compute/time is limited, do these first:

1. Compute-matched rewind.
2. Deconfounded routing shuffle.
3. Matched-quality routing perturbation.
4. Properly replicated cross-init.
5. Full expert-transfer matrix or weaken "expert-specific."
6. Standardize statistics.
7. Rewrite causal/rewind claims.
8. Update closest related work.

Without these, the strongest current rejection arguments remain intact.

## Strong ICLR revision
Add:

9. Routing × initialization factorial.
10. 8–10 seeds on the core causal experiment.
11. Fresh-epoch-shuffle/non-repeated-data control.
12. Focused IMP comparison.
13. One meaningfully larger-scale replication.
14. Redesigned intervention schematic and main-evidence figure.

---

# Experiments We Should Not Prioritize

Do **not** spend major compute on:

- more random datasets just to increase dataset count,
- more architecture-grid cells at the same scale,
- many more sparsity levels,
- additional descriptive route/mask correlations,
- semantic expert-labeling studies,
- broad interpretability probes unrelated to the sparse-support mechanism,
- production-scale models before fixing the causal design.

These add breadth but do not resolve the actual rejection arguments.

---

# Claim Rewrite Matrix

| Current-style claim | Revised target claim |
|---|---|
| "Routing history causally determines sparse support." | "Interventions on the routed optimization trajectory causally alter expert-local support under fixed training conditions." |
| "Token identity shapes support." | Use only if identity is isolated from gate/capacity effects. |
| "Routing alone does not determine support." | "The effect of routing history is conditional on initialization; identical imposed routes need not reproduce coordinate-level support across initializations." |
| "Experts contain expert-specific tickets." | "Experts contain non-random expert-local supports"; upgrade to "expert-specific" only after full transfer tests. |
| "80% tickets require early rewind." | Use only if compute-matched rewind supports it; otherwise describe the exact protocol. |
| "Routing-conditioned optimization is a theory of specialization." | "Routing-conditioned optimization is an experimental framework for studying how routed gradients shape persistent expert-local parameter structure." |
| "Architecture robustness proves generality." | "The structural effect is robust across the evaluated small-model architecture variants." |

---

# Target Final Story

The final paper should tell one sequential story:

### Step 1 — There is structure to explain
Ordinarily trained MoE experts contain non-random expert-local sparse supports.

### Step 2 — The structure matters for optimization
At moderate sparsity, these supports survive initialization rewind; at higher sparsity, report exactly what the corrected rewind protocol establishes.

### Step 3 — Routing helps create the structure
Changing the detailed routing trajectory at fixed initialization changes the resulting support even when coarse utilization—and ideally gate/capacity statistics and dense optimization quality—are controlled.

### Step 4 — Routing is not the whole explanation
The same route across different initializations does not uniquely specify the same coordinate support; quantify route, initialization, and interaction effects rather than merely asserting this.

### Step 5 — Scope the conclusion
This is evidence for a routing-conditioned sparse-support formation mechanism in the tested MoE regimes, not evidence of semantic expert specialization and not yet a universal theory of production-scale MoEs.

---

# Final Pre-Submission Acceptance Checklist

## Central causal claim
- [ ] Deconfounded route intervention completed.
- [ ] Dense-quality-matched route comparison completed.
- [ ] Result replicated with sufficient seeds.
- [ ] Causal wording names exactly what was manipulated.

## Rewind / lottery-ticket claim
- [ ] Compute-matched rewind completed.
- [ ] Full-budget and compute-matched protocols clearly distinguished.
- [ ] 50% initialization-ticket claim still supported.
- [ ] 80% wording reflects corrected result.

## Initialization interaction
- [ ] Cross-init has adequate replication.
- [ ] Multiple source/target combinations if feasible.
- [ ] Route × initialization effects quantified.
- [ ] Cross-init framed as boundary condition, not straw-man null rejection.

## Expert-specificity
- [ ] Full mask-transfer matrix completed, or terminology weakened.

## Alternative explanations
- [ ] Fresh epoch reshuffling/non-repeated-data control completed.
- [ ] Gate/capacity confound addressed.
- [ ] Dense optimization-quality confound addressed.
- [ ] Pruning-method dependence checked with focused IMP comparison.

## External validity
- [ ] One meaningfully larger-scale/non-repeated-corpus replication if compute permits.
- [ ] Top-2 causal claims limited to evidence actually collected.

## Statistics
- [ ] Single SD convention.
- [ ] Paired uncertainty for main comparisons.
- [ ] Cross-init uncertainty reported.
- [ ] Correlation analysis demoted or corrected for replay-point sensitivity.
- [ ] Numeric precision reduced appropriately.

## Literature
- [ ] Routing the Lottery independently re-read and precisely distinguished.
- [ ] Paul et al. distinction expanded.
- [ ] StableMoE distinction sharpened.
- [ ] MoE-Pruner/STUN distinction sharpened.
- [ ] Recent router–expert gradient/geometry work independently verified and cited.
- [ ] No broad novelty claim remains that depends on literature nonexistence.

## Writing
- [ ] Abstract rewritten after experiments.
- [ ] Contribution hierarchy centered on routing intervention.
- [ ] Section 3 compressed.
- [ ] Figure 1 redesigned as intervention schematic.
- [ ] Figure 3 replaced/strengthened with causal evidence.
- [ ] Important caveats moved next to corresponding main-text claims.
- [ ] Semantic specialization remains explicitly outside scope.
- [ ] Global vs expert-weight sparsity clearly reported.

---

# Expected Effect on Reviewer Score

The current reviewer consensus is approximately **5/10: marginal reject / just below acceptance**.

The changes most likely to move the paper upward are not additional breadth. They are:

1. **cleaner causal isolation,**
2. **a correct/comparable rewind protocol,**
3. **proper cross-init replication,**
4. **evidence that the result is not merely poor optimization or repeated-data memorization,**
5. **a tighter novelty claim tied specifically to routing-conditioned rewindable support.**

If those results are favorable, the paper has a plausible path to the **6–7 range** because its strongest existing advantage is already present: the research question is coherent, the intervention mindset is good, the structural effect is robust, and the manuscript is unusually explicit about the limits of its evidence.
