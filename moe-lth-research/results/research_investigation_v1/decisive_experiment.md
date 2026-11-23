# Decisive experiment: separate gate values from gate Jacobians

**Proposal only. No training or intervention was launched.** The immediate goal is to falsify the forward-value interpretation of the strongest repository result before paying for architecture or seed sweeps.

## Smallest discriminating screen

Use the existing **8E, .85 one-shot learned-mask cohort**, ages R20 and R100, seeds 7/17/29, batch 128×128, original deterministic 2500-attempt recovery stream, original initialization/shared/expert states, original masks, fresh AdamW .0003/wd .01, clip 1, FP16 scaler, no scheduler. This is deliberately not the 32E IMP experiment. Keep the same complete R100 donor assignments, capacity priorities, accepted token identities and chronological batches for every arm and both dense/sparse recipients.

Factorial cells:

| Arm | Forward gate coefficient | Gate task-loss Jacobian |
|---|---|---|
| A, existing assignment replay | native recipient probability g | native dg/dθ |
| B, existing full detached replay | stored donor coefficient c | zero |
| C, new value-only substitution | c | native recipient dg/dθ, via explicit straight-through surgery |
| D, new derivative-only removal | native g | zero via detach |

C can be implemented locally as `(c-g).detach()+g`; D as `g.detach()`. This preserves the native **gate Jacobian**, not the entire native parameter gradient: altered forward coefficients change upstream derivatives and state trajectories. It is a specified path intervention, not a claim that the surrogate backward is the derivative of the unmodified forward objective. All arms retain the same native auxiliary-loss computation and coefficient .1, including its gradients into router and upstream states. Do not detach auxiliary probabilities. All capacity decisions use donor priorities, not intervention gate values.

Two new arms ×two ages ×two recipients ×three seeds = **24 new recoveries**. The existing 24 A/B recoveries are reused only after code/config/data/mask/probe/precision identities pass. If their execution fingerprints are incompatible, rerunning all four cells requires 48, and that expanded compute needs approval; do not silently replace baselines. No broad pruning or mask-discovery search is part of this screen.

Before training, test on identical states/batches that A/D and B/C have equal logits and accepted sets, B/D have zero gate task derivative, C has the native gate Jacobian, and all cells retain the expected auxiliary derivative. Test that C equals A when c equals g. Log exact forward/Jacobian tolerances separately; do not demand equal final models for different backward passes.

## Checkpoints and measurements

Save 0,25,100,250,500,1000,1500,2000,2500, with full non-mutating readout at each. Primary outcome is the seed-specific endpoint contrast

`C_arm = (CE_sparse,R20 − CE_dense,R20) − (CE_sparse,R100 − CE_dense,R100)`.

Every training arm uses the **same complete donor assignment-plus-gate readout** for the primary comparison. Report absolute sparse CE and dense CE separately as well. Secondary readouts: native and assignment-only, with their labels retained. Require observed common-readout states at all timing comparisons; do not interpolate missing baseline observations.

Primary effects: C−A (value intervention with native Jacobian), D−A (Jacobian removal with native values), and B−C−D+A (interaction). B−A is the combined intervention already observed. Keep n=3 exploratory and every seed value; no token/expert/time-level p-values. Use absolute nats/token effects; ratio attenuation is secondary and flagged when the baseline contrast is near zero. Specify .02 nats/token as a practical effect-size floor for the screen before execution, not as a ticket criterion.

Record actual applied updates, AMP skips, clipping, gradients before and after clipping, native/donor gate differences and auxiliary norms. At fixed diagnostic batches log E task gradient, auxiliary gradient and actual AdamW update. For tied E, obtain input-lookup and output-head contributions using separate leaves with identical forward values and verify their sum equals the tied gradient. These are **diagnostic derivatives**, not retroactively saved historical gradients. Expert-branch VJPs are separately defined observations; do not call overlapping layer VJPs an additive causal decomposition. Compare subsequent E displacement to measured optimizer updates, accounting for momentum and weight decay.

## Competing predictions and stopping rule

- **Forward values matter:** C moves the age contrast substantially toward B; D stays nearer A. A meaningful sparse-minus-dense interaction remains under common readout.
- **Backward path explains collapse:** D approaches B; C stays nearer A. Stop promoting a forward-amplitude mechanism and investigate a bounded gate-gradient claim only if it is not ordinary removal of the distinguishing pathway.
- **Both or auxiliary/state interaction:** neither isolated intervention reproduces B; a large factorial interaction remains. Do not conclude unique mediation.
- **Readout-specific fitting:** primary donor-readout effects are large but native/other valid readouts fail or reverse. Do not claim general recovery improvement; assess donor portability before expansion.

Stop the screen after the fixed 24 new recoveries. Proceed only if at least one isolated effect exceeds .02 nats/token with the same direction in all three seeds, the forward/Jacobian controls pass, and the result is not solely an evaluation-policy reversal. Otherwise NO-GO for substantial mechanistic investment; mixed outcomes are reported, not followed by an automatic sweep. This is a decision rule, not statistical proof.

## Confirmatory replication if the screen passes

Pre-register **five new seeds**, e.g. 101/137/173/211/257, before seeing their outcomes. Use all four arms, both ages and both dense/sparse conditions: 80 recoveries, plus five original source runs and required donor captures. Existing outcome-selected seeds remain exploratory. Do not claim fresh seeds exist now. Use a documented untouched held-out split, or explicitly label validation-only inference if none can be established. A donor-schedule portability check and a matched-count random-mask control are mandatory before claiming a learned-mask-specific mechanism or algorithmic benefit; they are a separately budgeted second stage, not silently included in the 80-run estimate.

Seeds are the independent units. Compute paired means and seed ranges for the three primary factorial effects, with df=4 t intervals explicitly assumption-dependent and Holm correction across these three tests. Report results in absolute nats and normalize only as sensitivity. For a meaningful mechanism, require an isolated effect beyond .02 nats in at least 4/5 fresh seeds plus a compatible corrected mean effect; all scripts/definitions freeze before the new outcomes. If intervals are inconclusive at five seeds, stop as inconclusive rather than automatically adding seeds until significant. A confidently tiny effect or failure of portability is a NO-GO for the broad claim.

## Compute and implementation

Existing 32E 2500-step segments took 559–2484 seconds, median 741 seconds, in `high_sparsity_coupling_v1/reruns/*/attempt_*/segment_0/record.json`. This is only an observed runtime proxy, not an 8E intervention benchmark. Twenty-four runs imply 3.7–16.6 serial GPU-hours at those rates; reserve **5–25 GPU-hours** for the screen including diagnostics/I/O, then measure a single authorized first run and revise before committing the remainder. The 80-run fresh-seed stage is approximately **20–70 GPU-hours plus source/donor and random-mask/portability overhead**, with substantial host/dependence uncertainty. Parallel hosts reduce elapsed time, not total GPU-hours. Model snapshots require tens of GB; schedule duplication should be avoided.

Implementation requires one isolated route-override/Jacobian adapter, exact same-state tests, E-gradient diagnostic instrumentation, and common-readout analysis. Do not change the production training code or launch any stage without explicit authorization. A full-scale model sweep, arbitrary embedding freeze, or additional descriptive geometry plots would provide less information per unit compute than resolving this existing confound.
