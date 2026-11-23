# Gate-amplitude replay intervention

8E target versus R0: target is better after full replay in 2/3 seeds; incremental contrast change 0.061632 ± 0.118098 seed SD; attenuation 94.7%; flips 2/3.
8E target versus R100: target is better after full replay in 1/3 seeds; incremental contrast change 0.317086 ± 0.198151 seed SD; attenuation 98.7%; flips 2/3.
12E target versus R20: target is better after full replay in 2/3 seeds; incremental contrast change -0.930294 ± 0.508118 seed SD; attenuation 99.8%; flips 2/3.
12E target versus R100: target is better after full replay in 1/3 seeds; incremental contrast change 0.422484 ± 0.512759 seed SD; attenuation 99.5%; flips 2/3.

## Paired contrasts at the common assignment-plus-gate readout

| experts | seed | target_age | control_age | native_contrast | assignment_only_contrast | assignment_gate_contrast | gate_increment_interaction | total_interaction | attenuation_fraction | assignment_to_gate_order_flip | level1_favors_target | secondary_native_gate_contrast | secondary_native_assignment_contrast | secondary_native_gate_interaction | secondary_native_order_flip | near_zero_level1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 7 | 20 | 0 | -0.591474 | 0.0589256 | -0.00701421 | -0.0659399 | 0.584459 | 0.880965 | True | False | -0.445919 | -0.0962688 | -0.34965 | False | False |
| 8 | 7 | 20 | 100 | -0.681872 | -0.2138 | -0.00622444 | 0.207575 | 0.675647 | 0.970887 | False | True | 0.105398 | -0.0711085 | 0.176506 | True | False |
| 8 | 17 | 20 | 0 | -2.53757 | -0.0829194 | 0.000772865 | 0.0836922 | 2.53834 | 0.990679 | True | True | -0.00394904 | -0.0756445 | 0.0716955 | False | False |
| 8 | 17 | 20 | 100 | -1.9777 | -0.196281 | 0.00158009 | 0.197861 | 1.97928 | 0.99195 | True | True | -0.134911 | -0.0824944 | -0.0524167 | False | False |
| 8 | 29 | 20 | 0 | -1.9892 | -0.172488 | -0.0053443 | 0.167144 | 1.98386 | 0.969016 | False | True | 0.0413697 | -0.0787512 | 0.120121 | True | False |
| 8 | 29 | 20 | 100 | -1.2956 | -0.544331 | 0.00149106 | 0.545822 | 1.29709 | 0.997261 | True | True | -0.0226314 | -0.0737321 | 0.0511008 | False | False |
| 12 | 7 | 60 | 20 | 0.821394 | 0.405732 | -0.0016963 | -0.407428 | -0.82309 | 0.995819 | True | False | -0.120278 | 0.10103 | -0.221309 | True | False |
| 12 | 7 | 60 | 100 | -0.648448 | -0.120973 | -0.000652749 | 0.12032 | 0.647795 | 0.994604 | False | True | 0.177146 | -0.056763 | 0.233909 | True | False |
| 12 | 17 | 60 | 20 | 0.614153 | 0.962411 | 0.00120922 | -0.961202 | -0.612943 | 0.998744 | False | False | -0.490054 | -0.0216315 | -0.468422 | False | False |
| 12 | 17 | 60 | 100 | -1.37002 | -0.132061 | 0.000544643 | 0.132606 | 1.37056 | 0.995876 | True | True | 0.143471 | -0.0220078 | 0.165479 | True | False |
| 12 | 29 | 60 | 20 | 0.543265 | 1.4219 | -0.000355419 | -1.42225 | -0.543621 | 0.99975 | True | False | -0.24081 | 0.0145103 | -0.25532 | True | False |
| 12 | 29 | 60 | 100 | -0.68468 | -1.01016 | 0.00436062 | 1.01452 | 0.689041 | 0.995683 | True | True | -0.171993 | -0.128558 | -0.043435 | False | False |

| experts | target_age | control_age | mean_gate_increment_interaction | seed_sd | seeds_toward_target | seeds_away_from_target | ordering_flips | mean_attenuation | attenuation_sd | seeds_attenuated | seeds_at_least_half_attenuated | full_three_seed_flip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 20 | 0 | 0.0616322 | 0.118098 | 1 | 2 | 2 | 0.946887 | 0.0581084 | 3 | 3 | False |
| 8 | 20 | 100 | 0.317086 | 0.198151 | 0 | 3 | 2 | 0.986699 | 0.0139491 | 3 | 3 | False |
| 12 | 60 | 20 | -0.930294 | 0.508118 | 3 | 0 | 2 | 0.998104 | 0.00204193 | 3 | 3 | False |
| 12 | 60 | 100 | 0.422484 | 0.512759 | 0 | 3 | 2 | 0.995388 | 0.000685387 | 3 | 3 | False |

## Architecture means (all three training levels at common readout)

| experts | age | historical_native_gap | assignment_replay_gap | assignment_gate_replay_gap | native_readout_after_gate_replay |
| --- | --- | --- | --- | --- | --- |
| 8 | 0 | 0.965692 | 0.0766236 | 0.232291 | -4.15581 |
| 8 | 20 | -0.74039 | 0.0111295 | 0.228429 | -4.29197 |
| 8 | 100 | 0.578 | 0.329267 | 0.22948 | -4.27459 |
| 12 | 20 | -1.09915 | -0.972354 | 0.20771 | -3.06073 |
| 12 | 60 | -0.439548 | -0.0423405 | 0.207429 | -3.34445 |
| 12 | 100 | 0.461501 | 0.378725 | 0.206011 | -3.39399 |

## Interpretation and limits

All amplitudes are detached exact stored coefficients. Router task gradients through gates are removed; historical auxiliary-loss gradients remain active into router and upstream states. Thus any remaining age dependence can travel through auxiliary-loss dynamics; surviving effects cannot be attributed exclusively to hidden/expert state independent of routing. Donor final-model identity was explicitly waived by the user because the backward pass changes. Exact same-state forward and gradient checks replace that invalid requirement.

The primary incremental contrast compares Level 2 and Level 1 under the SAME assignment-plus-gate evaluation. Native and assignment-only bridge readouts are separate columns in outcomes.csv. Absolute loss changes and evaluation-only differences are not substituted for paired sparse-minus-dense contrasts. Attenuation uses epsilon 1e-12; near-zero Level-1 contrasts are flagged. Three seeds are the independent units; there are no significance claims or token/expert pseudoreplicates.

Trajectory timing is resolved only at nine saved common-readout baseline snapshots. Level 2 has all 51 validation observations. Original 51-point baseline curves are retained with their original readout labels, not treated as common-readout evidence. Gate observations describe native probabilities on recipient intervention hidden states and donor-accepted edges. Layer/expert variation is descriptive and is not a factorial intervention on mean, variance, pattern, or curriculum. Actual training-gradient logs and matched FP32 diagnostic gradients are reported separately; nonfinite-step frequency is not an elementwise gradient fraction.

## Answers to the remaining scientific questions

5–8. Incremental and total changes, literal flips, attenuation, and three-seed sign consistency are tabulated above. There are 8/12 primary seed-level flips; 4/12 contrasts attenuate by at least half without a flip. Unfavorable and near-zero Level-1 contrasts must not be counted as removal of a favorable advantage.

| experts | control_age | seeds_toward_target | seeds_away_from_target | ordering_flips | seeds_attenuated | seeds_at_least_half_attenuated |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 0 | 1 | 2 | 2 | 3 | 3 |
| 8 | 100 | 0 | 3 | 2 | 3 | 3 |
| 12 | 20 | 3 | 0 | 2 | 3 | 3 |
| 12 | 100 | 0 | 3 | 2 | 3 | 3 |

9. First observed common-readout interaction exceeding 1e-5 loss units, and first observed halving of a favorable contrast, are below. This is a descriptive numerical threshold, not significance. Only nine baseline states exist; onset between them cannot be localized. Initialization contrasts under common fixed gates can be zero by construction, so a zero step-0 contrast is not evidence that training removed an initialization effect.

| experts | seed | control_age | first_observed_change_step | first_observed_half_attenuation_step |
| --- | --- | --- | --- | --- |
| 8 | 7 | 0 | 100 | nan |
| 8 | 7 | 100 | 100 | 100 |
| 8 | 17 | 0 | 100 | 1500 |
| 8 | 17 | 100 | 100 | 100 |
| 8 | 29 | 0 | 100 | 1500 |
| 8 | 29 | 100 | 100 | 100 |
| 12 | 7 | 20 | 100 | nan |
| 12 | 7 | 100 | 100 | 100 |
| 12 | 17 | 20 | 100 | nan |
| 12 | 17 | 100 | 100 | 100 |
| 12 | 29 | 20 | 100 | nan |
| 12 | 29 | 100 | 100 | 250 |

10. Actual training expert-gradient gate/assignment ratios across seed-age-kind runs range from 0.9242 to 1.8575, with mean 1.1809. See gradient_scale_summary.csv for clipping frequency, nonfinite-step fraction and scaler ranges, and common_readout_gradient_diagnostics.csv for elementwise nonfinite fractions in matched FP32 diagnostic gradients. These scales are a sanity check, not a causal endpoint.

11–12. Native-minus-donor amplitude differences vary across layer/seed means from -0.864687 to -0.005445, and across expert/layer/seed means from -0.969332 to 0.849224. Detailed means, SD, quantiles, entropy, margin and sparse/dense differences are retained. These show amplitude heterogeneity; the experiment does not identify separate layer-specific or expert-specific causal effects.

13. The following correlations use three seed-level target-minus-control amplitude differences per contrast. They describe proposals on gate-intervened hidden states, not an independently measured native-training mediator. They are observational, highly unstable at n=3, and cannot establish mediation.

| experts | control_age | seed_count | descriptive_pearson_r |
| --- | --- | --- | --- |
| 8 | 0 | 3 | 0.940974 |
| 8 | 100 | 3 | 0.382563 |
| 12 | 20 | 3 | 0.994872 |
| 12 | 100 | 3 | -0.845257 |

14. Native readout disagrees with the primary Level-2 target/control sign in 8/12 cases. All secondary native interactions are in paired_contrasts.csv; the assignment-only bridge and common-gate outcomes are in outcomes.csv. Native-readout flips are not the primary causal result.

15–16. A paired common-readout change supports a contribution of this specific detached gate substitution beyond assignment replay; consistent substantial attenuation of favorable contrasts supports a major contribution within the tested donor schedule. It cannot establish that amplitude is the unique or universal explanation: detachment changes gradient paths as well as forward coefficients, the donor is fixed, and only three seeds were tested. The descriptive verdict below applies only to these runs.

17. If favorable ordering survives, native auxiliary-loss gradients and endogenous hidden/expert-state evolution remain available pathways. This experiment cannot rule out router-dependent auxiliary dynamics or attribute the remaining effect solely to expert inputs. No further experiment was launched.

## Artifacts and validation

| donor_archives_verified | same_state_donor_controls_passed | recipients | matched_pairs | exact_full_replays | maximum_amplitude_error | snapshots | common_readout_snapshot_evaluations | figure_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 6 | 36 | 18 | 36 | 0 | 324 | 972 | 10 |

All new bulk artifacts are in stable storage at /users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/gate_amplitude_replay_intervention. No donor or Level-1 recovery was rerun.

GATE-AMPLITUDE MECHANISM SUPPORTED

**Most important implication for the paper:** The incremental paired effect measures the contribution of replacing native amplitudes with this specific donor schedule, beyond assignment replay. It does not establish formal or complete mediation; auxiliary-loss and representation dynamics remain endogenous.

**Most important next experiment:** Replay identical expert input representations in addition to assignments and gates, while preserving the current auxiliary-loss protocol, to separate input-state evolution from the already clamped routing output. This is a proposal only; it has not been launched.
