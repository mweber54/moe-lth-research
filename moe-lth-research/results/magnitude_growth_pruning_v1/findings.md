# Magnitude growth pruning intervention v1

Prospective intervention motivated by retrospective results; the overall project was not preregistered. All five original unpruned 32E source seeds were branched at step 1000. History 250/500/1000 alone determined matching. All masks were sealed before any loss measurement. All three branches used the same fresh AdamW/scaler/RNG reset because source optimizer/scaler/RNG states were unavailable. Historical uninterrupted endpoints are not controls.

| Seed | Immediate CE G−S | Final CE G−S (primary) | Final G−control | Final S−control |
|---|---:|---:|---:|---:|
| 7 | 0.00033494 | -0.00062366 | 0.00080952 | 0.00143318 |
| 17 | 0.00061353 | -0.00306180 | -0.00168532 | 0.00137648 |
| 29 | 0.00034231 | 0.00494551 | 0.00622115 | 0.00127564 |
| 43 | 0.00091069 | 0.00073534 | 0.00124486 | 0.00050952 |
| 61 | 0.00082401 | -0.00186822 | 0.00056614 | 0.00243436 |

Final paired effects: mean 0.00002543, median -0.00062366, range -0.00306180–0.00494551, sample SD 0.00309231, SE 0.00138292. Positive sign in 2/5 seeds. Source seed is the independent replication unit (n=5); coordinates, experts, layers, batches, and tokens are nested observations. No nonsignificance/equivalence inference is made.

## Required interpretation

1. Magnitude matching achieved 0.9998% of all eligible expert-weight coordinates per branch in every seed (2,621 per tensor, 670,976 per seed). Each growing/stable pair is within its seed/layer/expert/tensor and current-magnitude stratum. Worst absolute pair mismatch is 1.1781801e-05 tensor RMS; worst tensor standardized mean difference is 1.347338e-06. Squared removed-magnitude relative differences across tensors range -0.0001%–0.0001%. This is a local small-dose intervention, not full 80/85% pruning.
2. Immediate damage: growing minus stable is positive in 5/5 seeds; mean 0.00060510. These immediate effects are separate from the final endpoint.
3. After 1,500 attempted updates with the same subsequent original batches, final G−S is positive in 2/5 seeds; mean 0.00002543. Applied updates and AMP skips are reported for every branch; duration was never adapted to the result.
4. Consistency is assessed from the five individual effects above, rather than from coordinates or validation batches. Immediate and final effects need not have the same sign. Excess-loss AUC uses trapezoidal integration at registered steps 0,25,100,250,500,1000,1500 relative to the newly run control, without interpolation beyond linear segments. AUC and normalized mean excess CE are in paired_seed_effects.csv.
5. Magnitude matching does not establish equal connectivity. Row/column occupancy, concentration, and neuron-level residual differences are retained in structural_balance.csv. Residual magnitude, squared-energy, sign, and structural differences could contribute to any effect. Growing/stable is a compound selection rule; a positive result would support its predictive value under this intervention, not show that growth itself causes importance or establish a routing mechanism.
6. A full pruning schedule is not implemented or authorized here. Review whether the final effects are consistent, larger than residual balance differences plausibly explain, and aligned with immediate/AUC behavior before deciding whether growth protection merits a later test. Small-dose null or reversed effects do not establish equivalence at high sparsity.

## Protocol and limitations

Currently small means the bottom half within each fc1/fc2 tensor. Growing candidates strictly increase in absolute magnitude at both observed intervals and have high tensor-RMS-normalized OLS slopes; stable candidates have low total absolute variation, rather than near-zero net slope alone. Sustained growth refers only to the observed checkpoints. Signed trajectories and sign changes are diagnostics in matched_group_summary.csv and coordinate_examples.csv. No individual-magnitude denominator, future checkpoint, gradient, validation result, or IMP membership enters selection.

Everything else remains trainable, including native routers, shared layers, biases and surviving expert weights. Gradient masking and post-update zeroing maintain exact zeros. Canonical LR, decay, clipping, FP16/autoscaling, auxiliary loss and native capacity settings are preserved. Validation is task-only FP32 with TF32 disabled on identical complete batches and actual-token weighting. Unpruned shared-reset controls were newly run.

The seed-17 engineering pilot remained in the cohort and only numerical/resource checks controlled dispatch of the remaining seeds; effect direction did not enter the gate. Unique attempts, locks, heartbeats, immutable masks, serialized resume checks and checkpoint hashes are preserved. Every registered branch is reported, including any failures; no historical winners were selected.


## Reviewed scientific conclusion

The registered hypothesis separates into two results. Recent magnitude growth predicted **greater immediate pruning sensitivity** in this matched small-weight population, but it did **not show a consistent larger final recovery deficit** after the fixed continuation budget.

All five immediate G−S effects are positive: mean 0.00060510 CE, range 0.00033494–0.00091069. These small effects arise at effectively equal current magnitude, count and removed squared magnitude, supporting immediate sensitivity of this prespecified selection rule beyond current magnitude alone under the tested intervention. They do not show that growth itself caused importance: connectivity and other history-associated differences remain uncontrolled.

The final per-seed primary effects, in source-seed order 7, 17, 29, 43, 61, are −0.00062366, −0.00306180, +0.00494551, +0.00073534 and −0.00186822 CE. Only 2/5 are positive. Mean is +0.00002543, median −0.00062366, sample SD 0.00309231 and SE 0.00138292. Thus the immediate difference did not persist consistently through equal attempted-budget continuation. This is a mixed, imprecise result, not evidence of equivalence or proof that growth can never matter.

Integrated excess-loss differences also vary by seed: G−S AUC is positive in 3/5 seeds and negative in 2/5. The mean is influenced by seed 29; neither uniformly slower recovery nor a persistent deficit is established. Immediate damage, integrated burden and the final endpoint must remain distinct.

Matching succeeded at the full registered dose in all 1,280 tensors: 2,621 coordinates per tensor and 670,976 per branch per seed (0.999832% of all eligible expert-weight coordinates, about 2% of the currently-small half). Worst pair mismatch is 0.0000117818 tensor RMS, and the largest absolute tensor removed-energy relative mismatch is approximately 0.000101%. The magnitude balance is therefore strong at this modest local dose. It says little about high-sparsity or repeated pruning schedules.

Structural balance is a material limitation. Absolute row-occupancy differences average 5.98 percentage points across tensors and reach 34.47 points. Row/column occupancy and concentration were measured, not matched. Such residual structure could contribute to immediate sensitivity and later adaptation, even with nearly exact magnitude and squared-energy balance. This experiment neither isolates a routing mechanism nor equates growth with functional importance.

The completed experiment has 22,500 attempted production updates. Seed 43's growing-small branch had one correctly handled AMP skip (1,499 applied versus 1,500 for the other branches); no duration was extended. The differing applied count is retained as a protocol/numerical limitation, not used to drop that seed. All three branches shared fresh optimizer/scaler/RNG resets because original optimizer state was absent. Results concern this shared-reset continuation and cannot be claimed as exact uninterrupted-training effects.

**Decision: revise the candidate rule before considering a full growth-protection pruning schedule.** The observed history signal identifies a small, consistent immediate perturbation effect, but this experiment does not support the proposed larger residual recovery damage. A full schedule is not justified by these results alone. Do not claim equivalence, launch another dose, reverse the rule after inspecting outcomes, or extrapolate to other architectures. No further experiment or full pruning algorithm is implemented or launched here.
