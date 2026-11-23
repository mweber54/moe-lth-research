# Cross-architecture optimization geometry: completed diagnostic rerun

All **18 matched pairs / 36 recoveries** completed the historical 2500-step protocol. All 36 final validation losses and all 18 ticket gaps reproduce history exactly at the stored numeric precision. All 36 final model states are also bit-identical to their historical checkpoints. All **324 complete model/optimizer snapshots** passed their mutation audits. The requested mechanism is not supported: favorable mean ticket gaps do not coincide with a common advantage in forced gradient, optimizer-update, and expert-function alignment.

## Protocol and measurement integrity

Only 85% sparsity was used: 8E R0/R20/R100 and 12E R20/R60/R100, each at seeds 7/17/29. Historical artifacts and the availability audit were preserved. E0, ET-derived masks, shared state, routers, and materialized training/validation sequences passed the preflight hash checks. Training retained fresh AdamW, its historical hyperparameters, no scheduler, fp16, trainable routers, mask enforcement, 2500 steps, and validation every 50 steps.

Snapshots at 0/100/250/500/750/1000/1500/2000/2500 retain the model, optimizer, GradScaler, RNGs, LR, step, mask, architecture/configuration, seed, historical router metadata, and next-batch identity. Diagnostics use a single persisted eight-sequence validation probe (1024 input-token occurrences), shared literally across architectures and seeds. They use fp32 backward with unit loss scale: gradients are already unscaled and the training scaler is untouched. Vectors are fp32 with fp64 reductions.

Forced routing shares the sparse model’s expert IDs **and capacity priority scores**, with exact accepted-token equality asserted. Dense gradients/updates are selected on identical surviving coordinates in sorted named-parameter order; unpruned expert biases survive in full. Native gate amplitudes and upstream representations may still differ. Functional comparisons separately hold expert inputs literally equal. Optional dense-route forcing, covariance similarity, and CKA were not run. Coherence uses two fixed diagnostic microbatches.

The run remained sequential. An attempted parallel scheduling change was rejected by automatic approval review and was not executed. This did not affect the scientific protocol or results.

## Primary results

Entries are means ± sample SD over three seed-level summaries. Gradient/function means weight the nine saved steps equally; update means weight the eight intervals equally. Experts are averaged within layer, then layers within seed. Seed is the independent replication unit; no token/expert pseudoreplication or significance claim is made. “Favorable” refers to the **matched ticket gap**, not necessarily the lowest absolute sparse loss.

| experts | age | forced_gradient | functional_similarity | update_alignment | ticket gap ± SD |
| --- | --- | --- | --- | --- | --- |
| 8 | 0 | 0.1405 ± 0.0164 | 0.3657 ± 0.0172 | 0.0649 ± 0.0047 | 0.2597 ± 0.0306 |
| 8 | 20 | 0.1554 ± 0.0084 | 0.3284 ± 0.0157 | 0.0423 ± 0.0043 | 0.1677 ± 0.0114 |
| 8 | 100 | 0.1791 ± 0.0221 | 0.3413 ± 0.0177 | 0.0385 ± 0.0085 | 0.2040 ± 0.0353 |
| 12 | 20 | 0.1515 ± 0.0097 | 0.3191 ± 0.0121 | 0.0377 ± 0.0089 | 0.2266 ± 0.0570 |
| 12 | 60 | 0.1699 ± 0.0109 | 0.3351 ± 0.0152 | 0.0394 ± 0.0055 | 0.1847 ± 0.0245 |
| 12 | 100 | 0.1919 ± 0.0095 | 0.3530 ± 0.0115 | 0.0391 ± 0.0032 | 0.1906 ± 0.0378 |

The age ordering for mean forced alignment is R100 > R20 > R0 in 8E and R100 > R60 > R20 in 12E. Both native and forced gradient rankings select R100 for every seed in both architectures. Thus forcing tokens does not reveal a favorable-checkpoint advantage that explains the intermediate-age ticket-gap optimum.

### Paired effects against each control

| experts | metric | control_age | mean_difference | seed_sd | seeds_favoring_target |
| --- | --- | --- | --- | --- | --- |
| 8 | forced_gradient | 0 | 0.0149 | 0.0248 | 2 |
| 8 | forced_gradient | 100 | -0.0237 | 0.0210 | 0 |
| 8 | functional_similarity | 0 | -0.0373 | 0.0327 | 1 |
| 8 | functional_similarity | 100 | -0.0130 | 0.0200 | 1 |
| 8 | native_gradient | 0 | 0.0086 | 0.0177 | 2 |
| 8 | native_gradient | 100 | -0.0162 | 0.0037 | 0 |
| 8 | trajectory_distance | 0 | 0.0449 | 0.0121 | 0 |
| 8 | trajectory_distance | 100 | 0.0180 | 0.0153 | 0 |
| 8 | update_alignment | 0 | -0.0226 | 0.0028 | 0 |
| 8 | update_alignment | 100 | 0.0038 | 0.0126 | 2 |
| 12 | forced_gradient | 20 | 0.0184 | 0.0204 | 3 |
| 12 | forced_gradient | 100 | -0.0220 | 0.0078 | 0 |
| 12 | functional_similarity | 20 | 0.0160 | 0.0273 | 2 |
| 12 | functional_similarity | 100 | -0.0179 | 0.0070 | 0 |
| 12 | native_gradient | 20 | 0.0091 | 0.0093 | 3 |
| 12 | native_gradient | 100 | -0.0108 | 0.0094 | 0 |
| 12 | trajectory_distance | 20 | -0.0075 | 0.0061 | 3 |
| 12 | trajectory_distance | 100 | 0.0090 | 0.0039 | 0 |
| 12 | update_alignment | 20 | 0.0018 | 0.0067 | 1 |
| 12 | update_alignment | 100 | 0.0004 | 0.0026 | 2 |

Differences are target minus control; positive favors the target except for trajectory distance, where negative is favorable. All three seeds favor R100 over the architecture-specific target on average forced gradient alignment. R20 loses to R0 on update alignment in all three 8E seeds. The small 12E mean update advantage is inconsistent across seeds.

## Answers to the scientific questions

1. **Does 8E:R20 improve forced alignment against both controls?** No. Its mean is 0.1554 versus 0.1405 (R0) and 0.1791 (R100). It beats R0 in only two seeds and loses to R100 in all three.

2. **Does 12E:R60 improve forced alignment against both controls?** No. Its mean is 0.1699 versus 0.1515 (R20) and 0.1919 (R100). It beats R20 in all three seeds but loses to R100 in all three.

3. **Does native alignment show the same pattern?** Yes for the failure of the proposed optimum: native averages also rank R100 first in every architecture/seed group. Architecture means are 0.1297/0.1383/0.1545 for 8E R0/R20/R100 and 0.1254/0.1345/0.1453 for 12E R20/R60/R100.

4. **Does controlling routing preserve or eliminate a favorable alignment advantage?** There is no consistent favorable-target alignment advantage to preserve. Forced and native rankings both prefer R100, so neither a common token-selection explanation nor a common expert-state explanation of the ticket-gap optimum is established.

5. **Do optimizer updates support the gradient mechanism?** No. 8E update means favor R0 (0.0649) over R20 (0.0423) and R100 (0.0385), whereas forced gradients favor R100. The 12E means (0.0377/0.0394/0.0391) have small, seed-dependent differences.

6. **Are sparse trajectories closer under favorable routers?** Not commonly. R20 has greater distance than both 8E controls in every seed (mean 0.4346 versus 0.3897/0.4166). R60 lies between the 12E controls (0.4232 versus 0.4307/0.4142) in every seed.

7. **Is dense expert function preserved more closely?** Not commonly. 8E R20’s mean function cosine is lower than both controls; 12E R60 is intermediate. Functional winners match the ticket-gap winner in only two of six architecture/seed groups.

8. **Is the mechanism present at step 0?** No shared effect. 8E R20 has greater forced alignment than R0 in all seeds, but not than R100 in seed 29. In 12E, R60 starts below R20 in forced alignment in all seeds. All parameter trajectory distances are exactly zero at initialization, as required.

9. **Does a common effect emerge later?** No. R100 becomes the consistently strongest forced-alignment router despite not being the mean ticket-gap optimum. Some target-versus-earlier-control improvements develop, but the required advantage against both controls does not emerge consistently.

10. **Does alignment separate before or alongside the loss advantage?** No common favorable-target sequence is supported. The preregistered first-all-seed concurrence finds no forced advantage over R100 at any saved step for either architecture. Against the earlier control, 8E’s R20 gradient difference can exist at step 0, while 12E R60 first wins across all seeds at step 250, after a ticket-gap difference is already observed. These first occurrences can be transient: initial gaps are negative and early curves cross. The supplemental seed-specific persistent-onset table explicitly distinguishes first-positive from positive-at-every-later-measurement timing, and separates absolute sparse loss from ticket gap. No causal timing claim is based on a transient step-0 crossing.

11. **Do metrics track seed-specific recovery winners?** Poorly. Forced/native gradient winners, update winners, and function winners each agree with the actual winner in only 2/6 groups; trajectory distance agrees in 1/6. Seed 17 favors R100 in both architectures in this exact grid. Seed 7 favors R20 in 8E and R60 in 12E; the historical note about an earlier 12E optimum must not override these measured outcomes within the restricted grid.

12. **Is the effect consistent across experts?** No common benefit across nearly all experts. Against R100, only about 34% of 8E and 24% of 12E experts favor the target in time-averaged forced alignment, averaged over seeds/layers. Full expert rows and distributions retain the exceptions.

13. **Is it consistent across MoE layers?** The failure against R100 holds in the seed-mean forced-gradient contrast of all four MoE layers in both architectures. Relative to the earlier control, all four layer means improve, but seed/expert effects vary. Thus the layer profiles also fail to identify the intermediate router as best against both controls. The table below retains each layer, and the full expert/layer table retains seed-specific signs; expert counts are not extra independent replicates.

14. **Is the shared property token selection, expert optimization geometry, both, or neither?** Neither is established as the common explanation of the favorable ticket-gap routers by this probe. This is not proof that routing or geometry is irrelevant to recovery.

15. **What is the strongest supported mechanistic statement?** Under exactly reproduced recoveries, higher matched dense/sparse gradient, update, or function alignment on this fixed probe does not reliably identify the router checkpoint with the smallest ticket gap. Common accepted-token forcing fails to recover the proposed intermediate-age advantage.

16. **Which tempting claims remain unsupported?** Universal geometry, architecture independence, causal mediation, an established token-selection mechanism, and a shared emergent co-adaptation explanation remain unsupported. Correlations with absolute sparse loss must not be substituted post hoc for the preregistered matched ticket-gap test.

## Initialization, seeds, and heterogeneity

| experts | age | forced_gradient | functional_similarity | native_gradient |
| --- | --- | --- | --- | --- |
| 8 | 0 | 0.45980 ± 0.00483 | 0.09368 ± 0.00467 | 0.45797 ± 0.00405 |
| 8 | 20 | 0.46981 ± 0.00266 | 0.09268 ± 0.00333 | 0.46830 ± 0.00178 |
| 8 | 100 | 0.46769 ± 0.00412 | 0.09020 ± 0.00253 | 0.46523 ± 0.00446 |
| 12 | 20 | 0.46364 ± 0.00742 | 0.09162 ± 0.00182 | 0.46084 ± 0.00764 |
| 12 | 60 | 0.45779 ± 0.00577 | 0.09179 ± 0.00446 | 0.45350 ± 0.00537 |
| 12 | 100 | 0.45634 ± 0.00627 | 0.09109 ± 0.00499 | 0.44992 ± 0.00565 |

Rankings below run from best to worst, with lower distance and ticket gap treated as better.

| experts | seed | ticket_gap_ranking | forced_gradient | functional_similarity | native_gradient | trajectory_distance | update_alignment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 7 | R20 > R0 > R100 | R100 > R0 > R20 | R0 > R100 > R20 | R100 > R0 > R20 | R0 > R100 > R20 | R0 > R20 > R100 |
| 8 | 17 | R100 > R20 > R0 | R100 > R20 > R0 | R0 > R100 > R20 | R100 > R20 > R0 | R0 > R100 > R20 | R0 > R100 > R20 |
| 8 | 29 | R20 > R100 > R0 | R100 > R20 > R0 | R20 > R0 > R100 | R100 > R20 > R0 | R0 > R100 > R20 | R0 > R20 > R100 |
| 12 | 7 | R60 > R100 > R20 | R100 > R60 > R20 | R100 > R60 > R20 | R100 > R60 > R20 | R100 > R60 > R20 | R60 > R100 > R20 |
| 12 | 17 | R100 > R60 > R20 | R100 > R60 > R20 | R100 > R20 > R60 | R100 > R60 > R20 | R100 > R60 > R20 | R100 > R20 > R60 |
| 12 | 29 | R60 > R100 > R20 | R100 > R60 > R20 | R100 > R60 > R20 | R100 > R60 > R20 | R100 > R60 > R20 | R20 > R60 > R100 |

Fraction of experts favoring the target, averaged over seeds and layers (time averaged within each expert):

| experts | metric | control_age | fraction_experts_favoring_target_mean_over_seeds_layers |
| --- | --- | --- | --- |
| 8 | forced_gradient | 0 | 0.6354 |
| 8 | forced_gradient | 100 | 0.3438 |
| 8 | functional_similarity | 0 | 0.3333 |
| 8 | functional_similarity | 100 | 0.4896 |
| 8 | native_gradient | 0 | 0.4375 |
| 8 | native_gradient | 100 | 0.3542 |
| 8 | trajectory_distance | 0 | 0.4167 |
| 8 | trajectory_distance | 100 | 0.3333 |
| 8 | update_alignment | 0 | 0.1979 |
| 8 | update_alignment | 100 | 0.5833 |
| 12 | forced_gradient | 20 | 0.6250 |
| 12 | forced_gradient | 100 | 0.2361 |
| 12 | functional_similarity | 20 | 0.5000 |
| 12 | functional_similarity | 100 | 0.4514 |
| 12 | native_gradient | 20 | 0.5625 |
| 12 | native_gradient | 100 | 0.3958 |
| 12 | trajectory_distance | 20 | 0.5625 |
| 12 | trajectory_distance | 100 | 0.3889 |
| 12 | update_alignment | 20 | 0.5000 |
| 12 | update_alignment | 100 | 0.4931 |

Layer-specific forced-gradient contrasts (target minus control; SD over seeds):

| experts | control_age | layer | mean | std |
| --- | --- | --- | --- | --- |
| 8 | 0 | 0 | 0.0399 | 0.0386 |
| 8 | 0 | 1 | 0.0116 | 0.0331 |
| 8 | 0 | 2 | 0.0078 | 0.0265 |
| 8 | 0 | 3 | 0.0021 | 0.0093 |
| 8 | 100 | 0 | -0.0487 | 0.0201 |
| 8 | 100 | 1 | -0.0118 | 0.0256 |
| 8 | 100 | 2 | -0.0101 | 0.0209 |
| 8 | 100 | 3 | -0.0255 | 0.0169 |
| 12 | 20 | 0 | 0.0239 | 0.0285 |
| 12 | 20 | 1 | 0.0154 | 0.0166 |
| 12 | 20 | 2 | 0.0168 | 0.0241 |
| 12 | 20 | 3 | 0.0185 | 0.0198 |
| 12 | 100 | 0 | -0.0313 | 0.0054 |
| 12 | 100 | 1 | -0.0177 | 0.0013 |
| 12 | 100 | 2 | -0.0186 | 0.0089 |
| 12 | 100 | 3 | -0.0220 | 0.0175 |

## Coverage and limitations

| metric | total | valid | minimum_condition_step_coverage |
| --- | --- | --- | --- |
| forced_gradient | 6480 | 6389 | 0.7812 |
| functional_similarity | 6480 | 6389 | 0.7812 |
| native_gradient | 6480 | 6298 | 0.4062 |
| trajectory_distance | 6480 | 6480 | 1.0000 |
| update_alignment | 5760 | 5751 | 0.9583 |

Undefined cosines come from empty or zero-norm comparisons and are excluded with counts retained. The lowest single native condition/step coverage is 40.6%; forced/function coverage is at least 78.1%. Conditional means can depend on which experts are active, so routing-dependent coverage limits interpretation. This is one small fixed probe, three seeds per architecture, fp32 diagnostics alongside historical fp16 training, and two-microbatch coherence. The negative conclusion is limited to the tested common alignment explanation; it does not exclude mechanisms outside this probe or these metrics.

## Artifacts and verification

- `reproduction_check.csv`: all 18 sparse/dense comparisons, every delta zero.
- `snapshot_manifest.csv`, `mutation_audit.csv`, and `artifact_integrity.json`: snapshot identities, non-mutation evidence, and completed artifact checks.
- `run_manifest.json`, `protocol_audit.md`, `provenance/`, and probe/data manifests: fixed protocol, primary metrics, implementation and data identities.
- `paired_seed_effect_summary.csv`, `seed_rankings_review.csv`, `expert_layer_paired_effects.csv`: paired effects and heterogeneity.
- `temporal_mechanism_summary.csv`: preregistered first-concurrence timing. `temporal_seed_specific_review.csv`: explicitly supplemental persistent/seed-specific timing.
- `figures/`: all required figure families in PNG and PDF, plus paired temporal contrasts. Thin lines in original time courses identify seeds 7 (solid), 17 (dashed), and 29 (dotted); thick lines are architecture means.

## Primary verdict

NO COMMON OPTIMIZATION GEOMETRY FOUND

**Most important implication for the paper:**
The historical intermediate-router ticket-gap optima reproduce exactly, but they should not be explained by a shared tendency for sparse experts to follow dense gradients, updates, or functions more closely. R100 has the strongest average forced alignment in every seed while often losing on ticket gap, and the other primary metrics do not resolve this mismatch. Present this as a bounded negative mechanism result, not proof that optimization geometry is universally irrelevant.

**Most important next experiment:**
Run one matched recovery intervention that fixes the complete routed-token schedule across router initializations, to test whether evolving token selection causally changes the ticket-gap ordering. Do not launch it automatically.
