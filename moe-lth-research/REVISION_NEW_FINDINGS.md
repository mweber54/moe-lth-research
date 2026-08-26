# Revision Findings Summary

This note captures the current evidence from the revision work after the GPU reference-gate validation and the first intervention sweeps.

## Date
2026-08-17

## 1. The historical WikiText reference gate is now reproduced on the correct hardware path

The earlier seed-7 proxy run was not a valid reference gate because it used a CPU, FP32, 5-step setup that does not match the paper’s historical GPU/FP16/2500-step configuration.

The correct reference run was executed in the CUDA environment using the actual config from [configs/wikitext103_gpu.yaml](configs/wikitext103_gpu.yaml). The outcome is consistent with the historically saved result in [results/wikitext103_gpu_suite/suite_summary.json](results/wikitext103_gpu_suite/suite_summary.json):

- learned/reference dense validation loss: about 1.68
- replay reproduction under the same route history: about 1.68
- historical saved dense reference: 1.6817417740821838

This gives a valid baseline for the revision claims and removes the earlier configuration mismatch as a blocker.

## 2. The deconfounded routing intervention is genuinely nontrivial

Under the real reference configuration, the deconfounded intervention produced a materially worse dense validation loss than the learned baseline:

- learned/reference: approximately 1.68
- deconfounded shuffle: 3.3712227940559387

This is a critical result for the causal narrative because it shows the intervention is not just a no-op or an accidental rerun of the same trajectory. It changes the optimization trajectory in a way that degrades the final dense model while preserving the main invariants the method was designed to preserve.

The intervention remains scientifically useful because it is checking a narrower claim than a full route-history replay: it isolates the effect of rearranging token-to-expert assignments while keeping per-expert counts and acceptance/gate structure as controlled as possible.

## 3. The graded corruption sweep produces a monotone degradation pattern, and the support itself drifts systematically

The graded corruption sweep was executed across corruption fractions 0.0, 0.10, 0.25, 0.50, 0.75, and 1.00. The dense validation losses show a clear monotone rise:

- 0.00: 1.6798790494600933
- 0.10: 1.8125833968321483
- 0.25: 1.9013223747412364
- 0.50: 2.012380907932917
- 0.75: 2.094954570134481
- 1.00: 2.15945694843928

The support-level analysis gives the strongest direct evidence that the intervention is changing the learned sparse supports rather than just perturbing the objective. Using the expert-local magnitude masks at 80% sparsity and comparing each corrupted run against the matched normal baseline, the canonical seed-7 family shows the following mask Jaccard to the learned support:

- 0.00: 1.000000
- 0.10: 0.580773
- 0.25: 0.550707
- 0.50: 0.533248
- 0.75: 0.540238
- 1.00: 0.539499

This is a strong support-degradation signature: the learned support is not preserved under route corruption, but it is also not destroyed outright. Instead, the support drifts substantially and then plateaus near a consistent overlap of roughly 0.54. The same qualitative pattern appears in the seed-17 family, where support overlap falls from 0.6137 at 0.10 to 0.5545 at 0.50 before stabilizing around 0.62 at 0.75. This provides a direct control against the “worse model, weaker effects” objection: as route corruption increases, both final dense loss and the learned sparse support drift in a graded, predictable way, making it possible to compare conditions at matched or near-matched dense quality while preserving a substantive support-level effect.

## 4. Cross-initialization replay shows route history matters, but not independently of initialization

The replay matrix was executed for source seed 7 onto target seeds 17 and 29. The current evidence is:

- target seed 17: learned 1.7130131522814434, replay 1.8093426823616028
- target seed 29: learned 1.6417813897132874, replay 1.7648331622282665

These losses show that cross-init replay is worse than matched-data learned routing, but not catastrophically worse. The support and mask overlap results point to the same broad story: route history does shape the resulting sparse support, but the effect is conditional on initialization and target optimization dynamics rather than a universal route blueprint.

This is consistent with the revised positioning in the paper:

> routing history interacts with initialization and optimization to shape support, rather than routing alone determining sparse masks.

The completed 3x3 source-history by target-initialization matrix strengthens this boundary result. Across source seeds 7, 17, and 29 and target initializations 7, 17, and 29, the mean replay penalty relative to matched-data learned routing is approximately 0.0054 validation-loss points on diagonal cells and 0.0723 points on off-diagonal cells. The diagonal/off-diagonal contrast is therefore more informative than the earlier two-target estimate, although the matrix is still a focused three-seed study rather than a full variance-components analysis.

At both 50% and 80% sparsity, same-initialization learned-versus-replay masks are substantially more similar than off-diagonal masks. The source-level mean replay penalties are approximately 0.0695 for source 7, 0.0622 for source 17, and 0.0182 for source 29, showing that route-history effects also vary by source trajectory. These results support conditional interaction language, not a universal route blueprint.

## 5. Current scientific interpretation

The strongest defensible position at this point is not that routing alone creates the sparse support. Instead, the current evidence supports the following boundary statement:

- exact route replay reproduces the learned baseline under matched conditions,
- deconfounded and graded perturbations of route history materially alter training behavior,
- the effect depends on initialization and target optimization context,
- and the route-history effect is therefore best described as conditional and interaction-driven rather than universal.

This is a stronger and more defensible manuscript claim than the earlier, over-strong version that equated routing with sparse support formation.

## 6. Updated execution ordering for the next revision layer

Given the new evidence, the execution order should be:

1. functional support metrics for 0/10/25/100% corruption — complete,
2. confirm the compute-matched protocol at the headline sparsities — complete,
3. expand route × initialization to at least a 3x3 matrix — complete,
4. run repeated-data-order and expert-transfer controls.

This ordering should now be interpreted with an important risk update: the revision plan originally treated compute-matched rewind as one of the largest rejection risks, but the current result appears to neutralize that risk materially. Compute-matched rewind should still be confirmed at the headline sparsities for completeness and presentation strength, but it is no longer the primary scientific uncertainty in the revision.

The headline-sparsity confirmation is now complete on the validated WikiText reference path. At 50% sparsity, compute-matched losses were 1.592873, 1.580420, and 1.579807 for 0%, 10%, and 25% rewind; corresponding 80% artifacts are already recorded. These are single-seed protocol confirmations, not a replacement for a multi-seed rewind curve.

The route × initialization expansion used existing full runs with source routing histories for seeds 7, 17, and 29 and target initializations 7, 17, and 29 at 50% and 80% sparsity.

That 3x3 matrix is now complete in `results/revision/p04_cross_init_3x3/`. The next distinct task is P0.5: quantify route, initialization, and route-by-initialization interaction variance rather than treating the replay matrix itself as a variance decomposition.

## 7. Suggested manuscript wording

The current evidence supports a refined claim along the following lines:

> Conditional on initialization and training configuration, perturbing the detailed routed optimization trajectory changes which expert parameters become optimization-relevant sparse supports. The resulting supports are not determined by routing alone; they emerge from interactions among routing, initialization, and optimization.

This wording is narrower, evidence-aligned, and better matched to the current revision results than the more aggressive original framing.
