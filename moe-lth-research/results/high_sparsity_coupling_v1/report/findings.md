# Provenance verdict and findings

**The duplication is legitimate:** all 75 shared historical/rerun checkpoints have identical model, expert, router and shared-component hashes. All 870 comparable raw output files are byte-identical. There are 315 logical measurements but 240 distinct model states; these tracks are not independent replications. No measurement cache required invalidation.

**The CE discrepancy is numerical:** all 15 historical endpoint CEs reproduce exactly with TF32 off; all 15 rerun endpoint CEs reproduce exactly with TF32 on. Historical continuation omits configure_device; reruns enable TF32. Example seed 7/.80: 1.3295407060960809 versus 1.3295155456539003. Neither checkpoint resolution nor model weights changed. Both settings retain the same four strict winners.

Execution, measurement verification and analysis are complete. Historical coverage remains 75 states (0/2500/3000/4000/5000); rerun coverage is 240 states; 165 historical slots remain missing. This is exploratory/post hoc, with five independent source seeds.

## Coupling is probe-dependent and differs from dense recovery

Means below average five paired source-seed effects separately at each sparsity; primary Pearson summaries use one common valid layer–expert population across the four cells.

| Input / selection | Sparsity | Sparse change | Dense change | Difference in changes |
|---|---:|---:|---:|---:|
| fixed_source / all_expert | 0.80 | +0.1396 | -0.3652 | +0.5048 |
| fixed_source / all_expert | 0.85 | +0.1712 | -0.3652 | +0.5363 |
| native / fixed_initial | 0.80 | +0.1658 | -0.3190 | +0.4848 |
| native / fixed_initial | 0.85 | +0.2022 | -0.3190 | +0.5212 |
| native / current_accepted | 0.80 | -0.1567 | -0.0366 | -0.1202 |
| native / current_accepted | 0.85 | -0.1558 | -0.0393 | -0.1165 |

All ten sparse runs increase association on fixed-source/all-expert and native/fixed-initial probes; matched dense changes are negative in all five seeds. Identical-fixed-pair sensitivity preserves the contrast: native/fixed-initial difference-in-changes is +0.4882/.5248 at .80/.85. Native/current-accepted association instead falls in all ten sparse runs; its paired effect is negative in 5/5 seeds at .80 and 4/5 at .85. Current accepted populations change; fixed token IDs do not freeze native hidden inputs. These probe contrasts are not an additive causal decomposition.

This does **not** distinguish tickets: both four winners and six nonwinners show the fixed-probe increases. The winning conditions remain seed 7/.80 and .85, 43/.80, and 61/.80; rerun gaps are approximately −0.035739, −0.021304, −0.001772, and −0.026802. No tolerance was used. Applied updates span 4997–5000 of 5000 attempts; AMP skips were reconciled directly with all 60 segment logs.

The former report contrasted separately valid sparse/dense populations. The corrected four-cell intersection changes some paired estimates materially (largest correction: +0.11552 for seed 61/.85 fixed-source/current-accepted, with 51 common layer–expert units). Pearson, Spearman, equal-expert, count-weighted and within-token statistics are separately labeled; units/tokens are not replications.

See [paired seed contrasts](paired_seed_coupling_changes.csv), [population corrections](paired_population_corrections.csv), [exclusions](coupling_exclusions.csv), [seed uncertainty](coupling_seed_uncertainty.csv), and [ticket tables](rerun_tickets.csv).

### Accepted-pair coverage sensitivity

The primary four-endpoint contrast uses roughly 125–126 common valid units on average and shows a negative sparse accepted-pair change in all ten conditions. However, the common population across **all 16 trajectory steps** contains only **11–14 of 128 units**. On this highly restricted subset, sparse changes increase in six conditions and decrease in four; differences-in-changes are positive in all ten. This is a different, traffic-selected expert population, not a contradiction in the underlying cache. Thus the accepted-pair result is **not robust to the valid-population definition**, and a universal decline is unsupported. The trajectory figure explicitly labels this restriction. Fixed-source/all-expert and native/fixed-initial signs survive Pearson/Spearman and equal-expert/count-weighted choices in every condition.

See [all-time population sensitivity](all_time_population_sensitivity.csv) and [correlation/weighting sensitivity](correlation_weighting_sensitivity.csv). Neither sensitivity treats ten related conditions as ten independent seeds.

### Pairing controls

On the same fixed-source/all-expert inputs and pair identities, final identity association exceeds unmatched-expert association by 0.2332/0.2845 at .80/.85; dense identity-minus-unmatched association is -0.3330. These signs hold in all five seeds. Shared input covariance alone therefore does not account for the signed identity contrast. Unmatched-router and unmatched-expert controls are separately exported; their global all-expert means are algebraically identical under the common complete selection and are not independent evidence. A negative dense signed activation association is not absence of geometry or poor routing.

See [matched pairing controls](pairing_specificity_controls.csv), using four-cell common valid intersections for each control.

## Geometry gives contradictory but compatible answers

- Relative to **same-step dense**, normalized full Gram distance rises from .58095→.81194 at .80 and .69915→.83889 at .85. Yet rank-8 overlap rises .15347→.29678 and .11088→.28181; rank-16 rises .27082→.32365 and .21050→.29476. Rank-32 falls .41166→.38332 and .34488→.34264. Ranks are not averaged.
- Relative to **fixed dense final**, full Gram distance also rises modestly (.78875→.81194; .80264→.83889), and all three average subspace overlaps decrease slightly. Moving- and fixed-reference initial geometries are different. SVD comparisons preserve reference-specific common-population rank/degeneracy exclusions; differing reference populations can produce slightly different final means.
- Relative to the **direct IMP mask-selection state**, .80 moves closer by full Gram shape (.48132→.39392) and all three overlaps increase; .85 moves away (.37244→.44846) and all three overlaps decrease. Actual masked initialization is a distinct reference: its initial overlap is one, then falls to roughly .53–.56.
- Expert fc1 trace increases from 198.34→418.39 at .80 and 162.63→393.38 at .85; this scale growth is separate from normalized shape. Centered-router q increases .00424→.00773 and .00436→.00839, but dense q increases more, .00391→.01094 (isotropic 1/256=.00390625). Dense therefore has larger unsigned directional energy despite declining signed score/activation association.

There is evidence of leading-subspace alignment, whole-spectrum separation, and sparsity-dependent source restoration at once. “Sparse models do not realign geometrically” is unsupported. See [rank/reference endpoint effects](geometry_endpoint_changes.csv), [geometry trajectories/exclusions](geometry_trajectories.csv), and [raw/centered q](q_trajectories.csv).

## E changes modestly; downstream differences are much larger

The tied byte embedding/LM-head E drifts 6.53%/6.72% from initialization at .80/.85, versus 6.47% dense; sparse–dense drift is 3.68%/3.51% with CKA .99934/.99938. Effective rank changes only slightly (154.75 initially to 154.89/154.88); top-eight energy stays near .1122. Mean row norms fall from 15.983 to 15.681/15.681 (dense 15.700). Token–token similarity and individual byte-row drifts are tabulated. E serves output prediction as well as input representation.

On identical probe positions, final token-embedding CKA is .99998, but after the first attention residual it is .7833/.7626, and normalized router-input CKA ranges roughly .56–.78 through depth. Differences already exist in deeper initial sparse states; the first attention stage starts at CKA 1 and diverges by the step-25 measurement. CKA changes are not monotonic through depth.

The reset changes no weights. From 2500→2525, E drift decreases slightly in both sparse and dense runs; deeper router-input CKA can rise before falling again by 5000. All before/after windows are reported; this is neither an instantaneous reset effect nor evidence that E mediates or causally precedes coupling. See [E trajectories/spectra](embedding_trajectories.csv), [depth summary](depth_endpoint_summary.csv), [all stages](representation_trajectories.csv), and [reset windows](optimizer_reset_representation_windows.csv).

## Functional agreement does not require expert equality

All conditions have substantial complete-expert discrepancies on the same fixed-source inputs: final relative errors span 1.72–2.36 and cosines .196–.261, while probe KL spans .319–.503 nats. Four models nevertheless beat their dense validation controls. For example, winner seed 43/.80 has KL .503 and gap −.001772; nonwinner seed 29/.80 has lower KL .337 and gap +.003709. Winner seed 7/.80 has native fixed-initial coupling gain +.1814; nonwinner 17/.85 has a larger +.2230. No simple cross-family separation of winners is established.

See [every condition across metric families](behavior_geometry_cases.csv), [expert output errors](expert_function_trajectories.csv), and [model CE/KL](model_function_trajectories.csv).

## Limits and review artifacts

Five source seeds, two dependent sparsities, deterministic historical repeats, validation-selected labels, missing historical intermediates, and no established unused test split limit inference. Seed bootstrap intervals are descriptive; no pooled token/expert/time p-values are used. CKA is not computational equivalence, mean activation is not expert usefulness, and none of these trends establishes causation. No follow-up intervention is executed or claimed necessary by this observational result.

Supported: the quantified probe-dependent paired effects, rank/reference-specific geometry, modest tied-E drift, large downstream representation/function differences and four strict validation winners. Unsupported: a universal coupling trend, coupling as a winner discriminator, blanket lack of geometric realignment, or causal representation mediation. Historical intermediates and independent-test behavior remain unresolved.

Audit: [provenance diagnosis](provenance_diagnosis.md), [all checkpoint/component identities](measurement_provenance.csv), [every coupling row/pair](coupling_row_provenance.csv), [coverage](coverage.csv), [full numerical detail](analysis_details.md). Figures: [coupling](coupling_sparse_dense.png), [geometry](weight_geometry.png), [E/depth](embedding_and_depth.png), [function versus loss](coupling_function_vs_loss.png).
