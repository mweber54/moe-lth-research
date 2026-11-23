# Router gate factorial mechanism experiment

**NO-GO** — The value-only intervention reproduces the attenuation under the imposed donor-gate readout, but its effect reverses under both native and assignment-only readouts in every seed. Jacobian removal alone and the primary interaction have mixed signs across seeds. The predeclared promotion criterion fails; this does not justify fresh-seed mechanistic confirmation.

Exactly 24 new C/D recoveries completed. E8 only; router ages 20/100; seeds 7/17/29; existing 85% learned masks; matched dense recoveries. The audited 24 A/B models are reused. No fresh-seed confirmation or expanded sweep was launched. All numerical/protocol inputs, source weights, masks, configurations and schedules remain unchanged. One concurrent trailing-newline-only change in the earlier narrative verdict is documented and was left untouched.

## What the factorial changes

A: native gate values/native gate Jacobian. B: donor values/detached gate Jacobian. C: donor values/native gate Jacobian. D: native values/detached gate Jacobian. A denotes the audited **assignment-replay** control, not the original freely routed native training arm. Donor IDs and capacity priorities are identical across all four cells. C supplies an explicit surrogate backward path, not a finite-difference derivative of its donor-constant forward function. Native means the recipient's current selected probability/Jacobian on its current state, not a saved A trajectory.

The native auxiliary-probability loss and coefficient .1 remain active in every cell. Gate detach also cuts indirect paths from later auxiliary losses through earlier gates, exactly as in the existing B intervention. This is therefore not a CE-exclusive graph surgery.

## Validation and preservation

[Pre-training validation](validation_report.md) passed before any recovery: all 12243 donor occurrences passed hashes; twelve real initial-state/data/mask identities passed; exact gate forward values, independent native/detached derivative references, FP32 and AMP legacy equivalence, active expert/upstream gradients, and tied-embedding lookup/head decomposition passed. The actual E8 router uses full softmax before top 1; it is not the structurally constant post-selection-normalized top 1 case. Exact selected probability 1 alone is not proof that every softmax Jacobian coordinate is numerically zero, so actual local Jacobians were measured separately.

All three C/R100/dense runs reproduce all 51 native donor validation values and every final model weight bit-for-bit, including workers paused for GPU scheduling. C/R100/dense is an identity cell by construction on the donor's own trajectory: its value-only C-A effect is zero there, while the native gate Jacobian remains active. It is a compatibility control, not an independent mechanistic replicate. All 24 final checkpoints match their step 2500 snapshots, preserve tied embedding values and sparse mask zeros, and contain finite weights. All 216 snapshot diagnostics preserve parameters, optimizer, scaler, gradient buffers and RNG. [Post-run checks](postrun_validation.json) reconstruct 192 subsequent embedding AdamW updates from clipped gradients, moments and weight decay; maximum error is 2.38e-07, within explicit FP32 rounding bounds. These are measured optimizer displacements, not inferred gradient directions.

Retained B endpoints were re-evaluated under all three readouts and checked against their audited values in [the endpoint recheck](combined_endpoint_readout_recheck.csv). Their new fixed-probe gradient measurements are explicitly diagnostic derivatives, not historical training gradients. A checkpoints and A/B intermediate snapshots are unavailable; their audited scalar readouts and execution provenance are reused without claiming new evaluations of missing states. [Reuse hashes](audited_reuse_hash_verification.json) and [scientific-input preservation audit](preservation_audit.json) passed. The strict byte check detected one removed final newline in the earlier narrative verdict; its original bytes are proven by SHA256 after adding the newline in memory. That report was left untouched. See [the formatting exception](preservation_exception.md). The frozen analyzer remains intact; a separately saved analyzer changes only this preservation audit and metadata, with an exact retained patch and unchanged estimands/thresholds.

## Full outcomes and router-age contrasts

The complete 48-row table with primary, native and assignment-only losses is [Markdown](full_factorial_results.md) / [CSV](full_factorial_results.csv). Absolute condition losses, averaged over the three seeds, are:

| arm | age | kind | primary_loss | native_loss | assignment_loss |
| --- | --- | --- | --- | --- | --- |
| A | 20 | dense | 1.805676 | 1.549840 | 1.461605 |
| A | 20 | sparse | 1.816805 | 1.936985 | 1.682744 |
| A | 100 | dense | 1.420451 | 1.420451 | 1.420451 |
| A | 100 | sparse | 1.749718 | 1.883375 | 1.662296 |
| B | 20 | dense | 1.415073 | 8.877186 | 9.033322 |
| B | 20 | sparse | 1.643503 | 4.585214 | 4.295943 |
| B | 100 | dense | 1.415256 | 8.880464 | 8.826128 |
| B | 100 | sparse | 1.644736 | 4.605874 | 4.348589 |
| C | 20 | dense | 1.428464 | 5.278764 | 5.074372 |
| C | 20 | sparse | 1.642848 | 3.165118 | 2.686633 |
| C | 100 | dense | 1.420451 | 1.420451 | 1.420451 |
| C | 100 | sparse | 1.636196 | 3.078391 | 2.579916 |
| D | 20 | dense | 28.488392 | 2.747641 | 2.260680 |
| D | 20 | sparse | 4.605340 | 2.330035 | 2.017969 |
| D | 100 | dense | 32.612153 | 2.656545 | 2.127985 |
| D | 100 | sparse | 4.922607 | 2.249892 | 2.017223 |


Define G=CE_sparse-CE_dense and T=G_R20-G_R100. Negative T means a smaller sparse-minus-dense penalty at R20. Primary evaluation forces the identical donor assignments **and coefficients**, with FP32 token-weighted CE on the identical complete validation batches. Primary T, by seed:

| arm | 7 | 17 | 29 | mean | sample_SD |
| --- | --- | --- | --- | --- | --- |
| A | -0.213800 | -0.196281 | -0.544331 | -0.318137 | 0.196085 |
| B | -0.006224 | 0.001580 | 0.001491 | -0.001051 | 0.004480 |
| C | -0.002285 | -0.006162 | 0.004363 | -0.001361 | 0.005323 |
| D | -1.400015 | 17.998295 | -5.178797 | 3.806494 | 12.434838 |


## Attenuation and factorial effects

Per-seed absolute attenuation is 1-|T_arm|/(|T_A|+1e-12); it is secondary to the signed nats contrasts. Near-zero baseline flags use the predeclared 1e-5 diagnostic. Primary attenuation:

| arm | 7 | 17 | 29 | mean | sample_SD |
| --- | --- | --- | --- | --- | --- |
| A | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| B | 0.970887 | 0.991950 | 0.997261 | 0.986699 | 0.013949 |
| C | 0.989314 | 0.968607 | 0.991985 | 0.983302 | 0.012796 |
| D | -5.548260 | -90.696416 | -8.514059 | -34.919578 | 48.326915 |


The isolated native-reference effects are C-A (value substitution) and D-A (Jacobian removal). Combined is B-A. Interaction is B-C-D+A. The balanced factorial main effects average both conditional effects: marginal value=((C-A)+(B-D))/2; marginal detach=((D-A)+(B-C))/2. These are effects on T, not effects on absolute CE, and the two notions of main effect should not be conflated when interaction is large.

| effect | 7 | 17 | 29 | mean | sample_SD |
| --- | --- | --- | --- | --- | --- |
| combined | 0.207575 | 0.197861 | 0.545822 | 0.317086 | 0.198151 |
| detach_native_values | -1.186215 | 18.194576 | -4.634466 | 4.124631 | 12.306304 |
| interaction | 1.182275 | -18.186834 | 4.631595 | -4.124321 | 12.300006 |
| marginal_detach | -0.595078 | 9.101159 | -2.318669 | 2.062471 | 6.156301 |
| marginal_value | 0.802653 | -8.903298 | 2.864491 | -1.745385 | 6.284074 |
| value_native_J | 0.211515 | 0.190119 | 0.548694 | 0.316776 | 0.201131 |


[Sparse/dense age components](age_effect_components.csv), [absolute-CE factorial effects](factorial_effects_on_condition_CE.csv), [ticket-gap factorial effects](factorial_effects_on_ticket_gap.csv), [all per-seed effects](factorial_effects_per_seed.csv), [aggregate effects](factorial_effects_aggregate.csv), [attenuation statistics](attenuation_aggregate.csv), and [attenuation-of-means versus mean-of-attenuations](aggregate_attenuation_estimands.csv) retain the distinct estimands. Seeds are the independent units; n=3 is exploratory. No token/expert/time-point pseudo-replication or confirmatory p-values are used.

## Readout sensitivity

Native-readout T:

| arm | 7 | 17 | 29 | mean | sample_SD |
| --- | --- | --- | --- | --- | --- |
| A | -0.071109 | -0.082494 | -0.073732 | -0.075778 | 0.005962 |
| B | 0.105398 | -0.134911 | -0.022631 | -0.017382 | 0.120240 |
| C | -3.392765 | -4.616315 | -3.305681 | -3.771587 | 0.732851 |
| D | -0.038835 | -0.092887 | 0.098865 | -0.010953 | 0.098870 |


Assignment-only-readout T:

| arm | 7 | 17 | 29 | mean | sample_SD |
| --- | --- | --- | --- | --- | --- |
| A | -0.017511 | -0.017542 | -0.027064 | -0.020706 | 0.005506 |
| B | -0.507887 | -0.089753 | -0.181881 | -0.259840 | 0.219698 |
| C | -3.422452 | -4.496644 | -2.722515 | -3.547204 | 0.893619 |
| D | 0.017016 | -0.147970 | -0.264892 | -0.131949 | 0.141635 |


C gives 98.330% mean per-seed absolute attenuation under the common readout, versus 98.670% for the existing combined B intervention. Its native-reference value effect is +0.316776 nats on average under that readout, but -3.695809 under native evaluation and -3.526498 under assignment-only evaluation; both reversals occur in all three seeds. Because assignment-only retains donor IDs and capacity, releasing assignments cannot explain the entire reversal: the evaluation gate coefficients themselves matter.

Absolute losses expose the dense-control contribution. C native CE averages 5.278764 at R20/dense versus 1.420451 at R100/dense, while sparse CE is 3.165118 versus 3.078391. The resulting very negative native age contrast is not evidence of sparse recovery improvement. Under the common readout, C reduces the dense age-CE difference from 0.385224 to 0.008013 and the sparse age difference from 0.067087 to 0.006652. Thus much of the primary contrast change comes from the dense control.

Averages can hide cancellation: D has mean native T=-0.010953, but its mean per-seed absolute attenuation is -0.433%, rather than a robust collapse. B likewise has -14.151% mean per-seed native attenuation despite a small signed mean T. The distinct aggregate estimands and all seed values are retained. The primary change largely removes R20's relative sparse advantage by improving its dense control. Native and assignment-only CE worsen in every non-identity C cell on the seed-averaged table, so the reversed gap is not a portable recovery gain.

Readouts measure the same models, not additional training arms. Native evaluation releases assignments and coefficients; assignment-only retains donor assignment/capacity but uses native coefficients. Primary uses donor assignment/capacity and donor coefficients. These are distinct evaluation interventions, so cross-readout reversals cannot be averaged away.

## Mechanistic assessment and stopping rule

Assessment: a readout-dependent forward-value effect is supported; natural causal mediation is not established. C and B have primary age contrasts within 0.007742 nats in each seed, without claiming statistical equivalence. C retains an active native gate Jacobian and nonzero router task gradients, so removing that backward path is not necessary for the audited primary attenuation. D alone does not reproduce the collapse consistently.

The primary factorial interaction is seed-inconsistent (+1.182275, -18.186834, +4.631595 nats), and the averaged main effects are consequently different from the native-reference simple effects. Native and assignment-only interactions are positive in every seed (means +3.689380 and +3.398607 nats), so the data do show interaction in those defined outcomes. Its readout dependence and the dense-control degradation prevent interpreting it as a unique mechanism for the original common-readout attenuation. The auxiliary-path consequence of gate detach further prevents a CE-exclusive backward mediation claim.

The predeclared screen stops after 24. Promotion requires at least one isolated effect exceeding .02 nats in the same direction in all three seeds, passed intervention controls, and an effect not solely explained by evaluation-policy reversal. The frozen numerical rule checks are in [stopping_rule_evaluation.json](stopping_rule_evaluation.json); native-readout seed effects are retained alongside primary effects. In seed order 7/17/29, C-A is +0.211515, +0.190119, +0.548694 nats: every value exceeds .02 with the same sign, but all native and assignment-only effects reverse. D-A is -1.186215, +18.194576, -4.634466 nats and fails same-direction consistency. Neither isolated intervention satisfies the full predeclared criterion. No new success threshold, equivalence bound, extra seed, or extended recovery budget was introduced.

The controlled gate interventions identify behavior under the imposed donor schedule and readout. Attenuation alone does not establish natural causal mediation of router age, learned-mask specificity, or general recovery improvement. Stop this direction before fresh-seed confirmation. Any separately authorized reopening should first test donor/gate portability with an evaluation policy specified independently of the training intervention; a CE-exclusive claim also needs a control that separates indirect auxiliary gate paths. Learned-mask-specific claims additionally require the previously specified matched random-mask control. None of these follow-ups was launched.

## Gradients, saturation, updates and budgets

Fixed-step 0-probe FP32 CE/auxiliary gradient norms across all new snapshot states:

| arm | router_task_min | router_task_max | router_aux_min | expert_task_min | shared_task_min | max_tied_decomposition_error |
| --- | --- | --- | --- | --- | --- | --- |
| C | 0.051252 | 8.484593 | 0.005509 | 0.057223 | 1.446358 | 0.000000 |
| D | 0.000000 | 0.000000 | 0.000968 | 0.010492 | 1.145864 | 0.000000 |


[Forward diagnostics](forward_gate_diagnostics.csv) show zero forward-source error for every logged layer/step. [Actual local FP16 Jacobians](fixed_probe_gate_jacobians.csv) match the requested policy in all 864 measurements; mean native zero-Jacobian fractions by arm are {'C': 0.0025182905749177166, 'D': 0.02474950526859039}. The native Jacobian can change along the optimization trajectory, so selected-gate 0/1 fractions are numerical diagnostics, not automatic mediation evidence. [Embedding gradient/update diagnostics](optimizer_update_diagnostics.csv) identify the fixed probe's source step, observed displacement and moment norms. Lookup/head contributions sum to tied CE gradients; auxiliary gradients remain separately logged. No layer VJP is called an additive causal decomposition.

All arm budgets, including reused controls:

| arm | models | attempts_min | attempts_max | applied_min | applied_max | total_AMP_skips |
| --- | --- | --- | --- | --- | --- | --- |
| A | 12 | 2500 | 2500 | 2498 | 2499 | 18 |
| B | 12 | 2500 | 2500 | 2498 | 2500 | 7 |
| C | 12 | 2500 | 2500 | 2498 | 2500 | 18 |
| D | 12 | 2500 | 2500 | 2492 | 2499 | 45 |


Every model used 2500 attempts, fresh AdamW, LR .0003, decay .01, clipping 1, FP16 AMP and the same native auxiliary objective. Applied-update counts can differ because AMP skips remain uncompensated, matching the audited protocol. Preclip norms and every attempted update are in legacy gradient logs; postclip norms and actual E updates are in measurements. Raw wall seconds for the two initially paused cells include waiting; [runtime accounting](runtime_accounting.csv) subtracts the logged pause intervals and does not call paused time GPU compute.

Snapshots are 0,25,100,250,500,1000,1500,2000,2500. There is no donor validation occurrence at 25: its native readout and fixed-step 0 derivative diagnostic are labeled, and it is excluded from primary timing comparisons. Primary evaluation remains every 50 updates; the full-range trajectory figure uses only jointly observed common steps, with no interpolation represented as evidence. The limited-axis figure is a zoom; [full-range primary trajectories](figures/primary_contrasts_full_range.png), [readout sensitivity](figures/readout_sensitivity.png), and [gradient policies](figures/gradient_policies.png) also have PDF exports.

Only previously used validation data were evaluated. Original A states are missing, all seeds are outcome-selected exploratory units, the Jacobian factor also affects indirect auxiliary paths, and donor schedule portability is not tested here. These limitations constrain the verdict and are not resolved by small scalar attenuation.
