# Complete recovery-trajectory routing intervention

**PAIRWISE ORDERING PRESERVED UNDER ROUTE REPLAY.** Six R100 dense donor captures and all 18 matched intervention pairs completed. Every donor reproduced its historical final model exactly; every dense R100 self-replay reproduced its donor exactly. All 36 recipient replays passed full assignment, accepted-token, input-identity, and chronological schedule-hash checks.

## Intervention and primary comparison

For each architecture/seed, all three router ages and both sparse/dense models replay the same complete R100 dense recovery trajectory. This fixes endogenous token selection to a shared pre-recorded sequence, including capacity decisions; the sequence itself can change with recovery step. Native differentiable gate amplitudes, trainable routers, shared parameters, the original initialization/masks, data order, fresh AdamW, fp16/GradScaler, clipping, and the 2500-step budget are retained.

The primary outcome compares native-trained historical checkpoints with clamped-trained checkpoints under the SAME replayed donor validation routes. The interaction is (clamped target-minus-control ticket gap) minus (native-trained target-minus-control ticket gap). Negative values shift the comparison toward the architecture-specific target. Native evaluation of both arms is reported separately. Changes caused solely by changing evaluation routes are not evidence of a training effect.

## Paired results

| experts | target_age | control_age | interaction_mean | interaction_seed_sd | positive_interactions | negative_interactions | ordering_flips | native_readout_ordering_flips | evaluation_only_flips |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 20 | 0 | 1.82244 | 0.65295 | 3 | 0 | 0 | 0 | 0 |
| 8 | 20 | 100 | 1.61835 | 0.22571 | 3 | 0 | 0 | 1 | 1 |
| 12 | 60 | 20 | -0.70263 | 0.67082 | 0 | 3 | 0 | 2 | 3 |
| 12 | 60 | 100 | 1.10089 | 0.37805 | 3 | 0 | 0 | 1 | 1 |

| experts | seed | target_age | control_age | baseline_fixed_contrast | clamped_fixed_contrast | primary_training_interaction | primary_pairwise_order_flipped | baseline_native_contrast | clamped_native_contrast | secondary_native_readout_interaction | secondary_pairwise_order_flipped | evaluation_only_pairwise_order_flipped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 7 | 20 | 0 | -1.15126 | -0.08275 | 1.06851 | False | -0.07338 | -0.09627 | -0.02289 | False | False |
| 8 | 7 | 20 | 100 | -1.52291 | -0.01751 | 1.50540 | False | -0.07882 | -0.07111 | 0.00771 | False | False |
| 8 | 17 | 20 | 0 | -2.23049 | -0.03762 | 2.19287 | False | -0.07944 | -0.07564 | 0.00379 | False | False |
| 8 | 17 | 20 | 100 | -1.89578 | -0.01754 | 1.87824 | False | 0.01347 | -0.08249 | -0.09597 | True | True |
| 8 | 29 | 20 | 0 | -2.27288 | -0.06695 | 2.20593 | False | -0.12323 | -0.07875 | 0.04448 | False | False |
| 8 | 29 | 20 | 100 | -1.49847 | -0.02706 | 1.47140 | False | -0.04345 | -0.07373 | -0.03028 | False | False |
| 12 | 7 | 60 | 20 | 1.51030 | 0.05149 | -1.45880 | False | -0.07312 | 0.10103 | 0.17415 | True | True |
| 12 | 7 | 60 | 100 | -0.84503 | -0.00730 | 0.83773 | False | -0.02544 | -0.05676 | -0.03133 | False | False |
| 12 | 17 | 60 | 20 | 0.47468 | 0.00469 | -0.46999 | False | -0.04435 | -0.02163 | 0.02271 | False | True |
| 12 | 17 | 60 | 100 | -1.54641 | -0.01231 | 1.53410 | False | 0.00986 | -0.02201 | -0.03187 | True | True |
| 12 | 29 | 60 | 20 | 0.19522 | 0.01612 | -0.17910 | False | -0.00830 | 0.01451 | 0.02281 | True | True |
| 12 | 29 | 60 | 100 | -0.96330 | -0.03246 | 0.93084 | False | -0.00218 | -0.12856 | -0.12638 | False | False |

## Seed rankings and architecture means

Rankings run from lowest (best) to highest ticket gap, always within the prespecified router grid.

| experts | seed | baseline_fixed | baseline_native | clamped_fixed | clamped_native |
| --- | --- | --- | --- | --- | --- |
| 8 | 7 | R20 > R0 > R100 | R20 > R0 > R100 | R20 > R100 > R0 | R20 > R100 > R0 |
| 8 | 17 | R20 > R100 > R0 | R100 > R20 > R0 | R20 > R100 > R0 | R20 > R0 > R100 |
| 8 | 29 | R20 > R100 > R0 | R20 > R100 > R0 | R20 > R100 > R0 | R20 > R100 > R0 |
| 12 | 7 | R20 > R60 > R100 | R60 > R100 > R20 | R20 > R60 > R100 | R20 > R60 > R100 |
| 12 | 17 | R20 > R60 > R100 | R100 > R60 > R20 | R20 > R60 > R100 | R60 > R20 > R100 |
| 12 | 29 | R20 > R60 > R100 | R60 > R100 > R20 | R20 > R60 > R100 | R20 > R60 > R100 |

| experts | age | policy | mean | seed_sd | seeds |
| --- | --- | --- | --- | --- | --- |
| 8 | 0 | baseline_fixed | 0.87812 | 0.52305 | 3 |
| 8 | 0 | baseline_native | 0.25974 | 0.03064 | 3 |
| 8 | 0 | clamped_fixed | 0.28358 | 0.04194 | 3 |
| 8 | 0 | clamped_native | 0.47070 | 0.16390 | 3 |
| 8 | 20 | baseline_fixed | -1.00676 | 0.29355 | 3 |
| 8 | 20 | baseline_native | 0.16773 | 0.01142 | 3 |
| 8 | 20 | clamped_fixed | 0.22114 | 0.01906 | 3 |
| 8 | 20 | clamped_native | 0.38715 | 0.15348 | 3 |
| 8 | 100 | baseline_fixed | 0.63229 | 0.08505 | 3 |
| 8 | 100 | baseline_native | 0.20399 | 0.03528 | 3 |
| 8 | 100 | clamped_fixed | 0.24184 | 0.02041 | 3 |
| 8 | 100 | clamped_native | 0.46292 | 0.14782 | 3 |
| 12 | 20 | baseline_fixed | -1.34823 | 0.63475 | 3 |
| 12 | 20 | baseline_native | 0.22658 | 0.05702 | 3 |
| 12 | 20 | clamped_fixed | 0.19644 | 0.02610 | 3 |
| 12 | 20 | clamped_native | 0.28609 | 0.08039 | 3 |
| 12 | 60 | baseline_fixed | -0.62150 | 0.39178 | 3 |
| 12 | 60 | baseline_native | 0.18465 | 0.02455 | 3 |
| 12 | 60 | clamped_fixed | 0.22055 | 0.03159 | 3 |
| 12 | 60 | clamped_native | 0.31739 | 0.07322 | 3 |
| 12 | 100 | baseline_fixed | 0.49675 | 0.02237 | 3 |
| 12 | 100 | baseline_native | 0.19057 | 0.03780 | 3 |
| 12 | 100 | clamped_fixed | 0.23790 | 0.01856 | 3 |
| 12 | 100 | clamped_native | 0.38650 | 0.03183 | 3 |

## Interpretation

The primary fixed-readout comparison contains 0 seed-level pairwise ordering flips across the 12 preregistered target/control comparisons. 0 of the four architecture/control contrasts flip in all three seeds. The secondary native-readout comparison contains 4 flips. Merely reevaluating unchanged native-trained models under donor routes produces 5 flips; these evaluation-only effects must be separated from the training intervention.

This is a controlled intervention on one donor-specific token-selection schedule. A training-arm difference at a common readout supports an effect of that schedule substitution in these matched runs. It does not show that all natural router-age effects are mediated by token selection, or that any alternative common schedule would have the same effect. Gates and upstream expert inputs remain endogenous; forcing them as well would answer a different question. With only three seeds per architecture, report paired effects and SD without claiming statistical significance or architecture independence.

## Audit and storage

| donors_exactly_reproduce_history | dense_R100_self_replays_exact | matched_pairs | complete_recipient_replays | snapshots | snapshot_mutation_audits | all_primary_values_finite | all_architecture_seed_schedule_hashes_identical_across_recipients |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 6 | 18 | 36 | 378 | 378 | True | True |

Primary data: outcomes.csv, paired_contrasts.csv, paired_effect_summary.csv, ranking_comparison.csv, and architecture_summary.csv. Full replay evidence: donor_reproduction.csv, self_replay_control.csv, replay_audit.csv, schedule manifests, and snapshot audits. Native routing-stat probes in recovery logs describe unforced proposals and are not the routes used for parameter updates. Historical recovery-fraction/time-to-threshold fields use the old helper denominator and are not used in this intervention analysis.

Bulk schedules, recovery artifacts and checkpoints are stored at `/tmp/mweber54_fixed_schedule_hxk19_zd` and linked from this result directory. The historical-results filesystem was nearly full; preserve that local directory together with this report. Nothing has been automatically deleted.

## Verdict

PAIRWISE ORDERING PRESERVED UNDER ROUTE REPLAY

**Most important implication:** The conclusion applies to the measured ordering changes under this specific complete R100 donor replay, with training and evaluation effects separated. The previously observed optimization-geometry correlations are not substituted for this intervention result. No additional experiment was launched.
