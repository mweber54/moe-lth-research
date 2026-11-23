from common import *
t=pd.read_csv(OUT/'decision_thresholds.csv').iloc[0];c=pd.read_csv(OUT/'phase2_cost_estimate.csv');storage=pd.read_csv(OUT/'phase2_storage_estimate.csv').iloc[0];X=float(t.near_zero_X);Y=float(t.reopens_Y);tc=math.sqrt(2*.95**2/(1-.95**2))
pins=['decision_thresholds.csv','noise_floor_per_step.csv','phase2_cost_estimate.csv','phase2_staging_cost.csv','phase2_storage_estimate.csv','recovery_compatibility.csv','ce_targets.csv']
hashes='\n'.join(f'- `{n}`: `{sha(OUT/n)}`' for n in pins)
text=f'''# E32/IMP80 recovery preregistration — proposed, awaiting approval

No Phase 2 run has started. This file is to be committed before any Phase 2 training. User approval of this preregistration and its cost estimate is required; committing it does not authorize training.

## Cohort and fixed protocol

Only E32, seeds 7, 17, 29, 43, 61. Final expert-weight mask nominal sparsity 0.8, actual sparsity 0.7999992370605469. The final cut follows round 07, with parent nominal sparsity 0.7902848 and actual sparsity 0.7902851104736328. This is iterative IMP with original-initialization rewinds, not a one-shot dense-to-80% prune. Phase-1 dense round 00 and sparse round 07 analyses remain separate from these recovery interventions.

Reuse the 30 verified original recovery runs (IMP and exact legacy random masks 1/2, five seeds, two orders); rerun every full-shuffle realization under this three-draw design. The 40 old full-shuffle jobs are calibration/provenance data, not Phase-2 intervention readouts. Pin original sources/masks/variants and A/B index arrays from `results/imp_initial_value_placement_v1/variant_manifest.csv` and `recovery_order_manifest.json`. Compatibility evidence: `recovery_compatibility.csv`, `reconstructed_data_orders.csv`, `sealed_source_audit.csv`.

Three independent perturbation draws per seed, mask and arm. A draw's perturbed model is held fixed across orders A/B; the orders share init/mask and are never independent replication units. The random masks for Phase 2 are the two exact legacy full-matrix random masks. They are not the newly sampled round-07-alive controls used in the primary Phase-1 analysis.

Native 5000 attempted updates in two 2500-attempt segments; batch size 128, sequence length 128; all surviving experts, expert biases, routers, attention, embeddings and normalization parameters train. AdamW, LR 0.0003 constant, weight decay 0.01, default native betas/epsilon, no scheduler, gradient clip 1, auxiliary loss coefficient 0.1, FP16/GradScaler. Preserve native gates, top-k=1 and capacity factor 1.25. At 2500 retain the complete model and reset AdamW, GradScaler and source-seed RNG exactly as before. Do not compensate for AMP-skipped updates. No hyperparameter, data, dose, draw-count or endpoint changes based on outcomes.

Fixed readouts: 0, 250, 500, 1000, 1500, 2000, 2500, 2525, 3000, 4000, 5000. Fixed validation stream `09a073e9f03bbb854d3903081440b9e0869feb8e04b50b3b1be76a1d42658b22`, 492416 target tokens in native complete batches. FP32, TF32 off, actual-token weighting and the existing CE accumulation procedure; evaluation preserves model/RNG state. Initial CE is numerically valid but comes from an extremely overconfident input-copying initialization; it is not treated as trained language-model performance or used to infer the mid-training outcome.

## Arms and immutable transformations

Original (reused); full shuffle (new draws/reruns); sign permutation; magnitude permutation within sign groups; distance-matched partial sign permutation. Perturb only survivors of each expert's fc1/fc2 matrix independently. Masked coordinates stay zero; biases, router and shared tensors are exactly unchanged before training.

Sign permutation keeps position-wise absolute values and uniformly permutes nonzero signs within a matrix. Magnitude permutation keeps position-wise signs and permutes absolute values separately in positive/negative survivor groups. Exact-zero survivors remain fixed in these sign-defined operations. Full shuffle uniformly permutes every retained value, including exact zeros, within a matrix. The matched sign arm uses the step-0 `B_matched` construction: independently randomize positive/negative index lists, pair them, and flip the closest prefix of opposite-sign pairs to the magnitude arm's squared displacement in that matrix. This preserves sign counts, absolute values and energy but is a different conditional permutation distribution from uniform sign permutation.

Generator: isolated CUDA `torch.Generator`; seed = little-endian first eight SHA256 bytes of `sign-recovery-phase2-v1|E32|arm|source_seed|mask_id|draw|tensor_name`, modulo 2**63. Use arm IDs `full_shuffle`, `sign`, `magnitude`, `matched_sign`; mask IDs `imp`, `random1`, `random2`; draws 1, 2, 3. Matched-sign target magnitudes use the same magnitude draw/key as the magnitude arm. Record every tensor RNG seed, index/permutation hashes, actual displacement and sign-flip fraction before training. Do not consume training RNG or regenerate an unfavorable draw.

Distance is the Frobenius displacement over all expert fc1/fc2 weights divided by their original masked norm. Match independently within each matrix; record per-matrix residuals. Require absolute global relative-distance mismatch <=0.0001 and per-matrix displacement-norm mismatch <=1% when the target is positive; zero target requires zero displacement. Stop if discreteness prevents this; do not search alternative seeds. The targets come from weights only, never CE.

Assert bitwise mask equality and identical binary row/column survivor counts (not numerical nonzero counts). Assert masked zeros and unchanged biases/router/shared state before training. For magnitude: position signs and per-sign-group magnitude multisets. For sign and matched-sign: position magnitudes and matrix sign counts. For full shuffle: retained-value multisets and sign counts. A/B/full/matched preserve squared matrix magnitude with FP64 rtol=atol=1e-12. Identity engineering checks and original baseline reproduction must pass before any production training. A baseline CE discrepancy >0.001, invariant failure, nonfinite readout or incompatible source/configuration stops work and is reported; do not tune around it.

## Metrics and replication

Record absolute CE for every seed/order/draw/mask/step and the pinned unpruned comparator (`legacy_unpruned_comparator.csv`). Delta is perturbed minus the original with the same seed/order/mask. For each draw j, interaction is Delta_IMP,j minus the mean of Delta_random1,j and Delta_random2,j. IMP advantage is mean random CE minus IMP CE, for originals and every arm. Average A/B equally within each seed/draw before inferential summaries. Keep all per-seed, per-order and per-draw rows.

A seed's interval is mean across its three order-averaged draw interactions +/- `{tc:.15g}` times sample SD/sqrt(3), a two-sided 95% t interval with df=2. Paired arm comparisons use draw-index differences after averaging orders/masks as above; different arm RNG streams remain independent. These intervals are conditional on the two fixed random masks. For pooled displays, use equal seed weights and a 10000-replicate hierarchical bootstrap: resample five source seeds, then three draws within each sampled seed; average orders first and keep the two fixed random masks paired. Fixed bootstrap seed 320803. Report the small number of source seeds/draws and no universal-mechanism claim.

Time-to-threshold targets are fixed per seed/order from original IMP CE at steps 1000, 2000, 5000 (`ce_targets.csv`). Use the earliest linear crossing on the fixed readout grid, no extrapolation. If absent by 5000 report 'not reached by 5000', not a permanent failure. Report step-5000 effects separately from steps 1500–2500; do not average them together or select a peak to classify outcomes.

## Numeric decision rules fixed from Phase 0.4

From `noise_floor_per_step.csv`, X is the maximum over all post-initialization logged steps of: empirical q95 absolute random penalties, q95 absolute centered seed-mean random penalties, and q95 absolute within-seed paired-interaction deviations. The legacy data have only two IMP draws and one shuffle per random mask; their random-mask and shuffle variability cannot be separated. These are empirical planning thresholds, not a proven population noise bound. Across-seed interaction spread is reported separately because it can include true heterogeneity.

- Near-zero margin X = **{X:.17g} nats**.
- Reopening threshold Y = 2X = **{Y:.17g} nats**.
- Replication requirement K = **4 of 5 seeds**; primary decision step = **2000**.
- 'Reopens': seed mean I at step 2000 > Y and its draw-level interval has lower bound >0; at least K seeds meet both.
- 'Near zero': seed |mean I| < X and its interval lies wholly within [-X,+X]; at least K seeds meet both. Failure to reopen is not equivalence.
- 'About as much as full shuffle': the interval for I_sign - I_full lies within [-X,+X] in at least K seeds.
- 'Matched sign exceeds magnitude': mean(I_matched - I_magnitude) > X and its difference interval has lower bound >0, in at least K seeds.
- 'Matched sign does not materially exceed magnitude': the difference interval lies within [-X,+X] in at least K seeds; lack of significance alone is insufficient.

First require the rerun full-shuffle arm to reopen; otherwise report that the reference effect failed this registered criterion and no listed explanatory outcome is selected. Then apply these mutually constrained rules:

1. Sign reopens about as much as full shuffle; magnitude is near zero; matched sign reopens and exceeds magnitude by the above rule. Report intervention evidence for sign-specific recovery sensitivity beyond the tested displacement control.
2. Sign reopens; magnitude is near zero; matched sign does not materially exceed magnitude. Report that the tested partial-sign contrast does not establish sign specificity at matched distance; size remains a possible explanation.
3. Both sign and magnitude reopen. Report that neither isolated treatment alone accounts for all sensitivity. Report I_full - I_sign - I_magnitude descriptively, but do not call it a clean factorial interaction: full shuffle is not exactly the composed A-by-B manipulation. A true factorial extension would need a separately approved design and budget.
4. Sign, magnitude and matched sign are all near zero while full shuffle reopens. Report that step-0 sign-only dependence does not reproduce the mid-training gap in these interventions, without claiming statistical independence of mechanisms.

If none fits, report 'none of the preregistered outcomes fits'. No outcome is chosen from plots. Report step-5000 effects and threshold delays separately even if their signs differ from the step-2000 decision.

## Staging, cost and storage

N=3 means **360 new full-length perturbation runs**, 90 per non-original arm, 1800000 attempted updates. This includes all **90 full-shuffle reruns**, with no reuse of the old 40 shuffled jobs. The 30 original runs are reused. Prices use observed native segment elapsed time on H100 NVL hosts, including historical slow runs; they are elapsed GPU-slot planning proxies, not GPU-active-time measurements. Sources and variability are in `observed_runtime.csv`, `runtime_by_seed.csv`, `runtime_by_segment.csv`; no new training benchmark was run. Add 15% planning overhead.

1. Sign only, seeds 7/17/29, both orders, all three mask variants, three draws: **54 runs x 2500 attempts**, then pause at the native reset boundary. Estimated **25.8492 slot-hours**, including overhead; storage **188.629 GB** including allowance. Use step 2000 for the gate. Advance only if at least **2 of these 3 seeds** have I_sign > Y and a draw interval excluding zero. If not, stop for user review without selecting a final outcome for unrun arms. Stage 1 is the smallest approval unit proposed here.
2. After a passing Stage-1 review, finish those 54 sign runs with their second 2500 attempts, then run full shuffle, magnitude and matched sign for those same three seeds: **54 continuations x 2500 + 162 new runs x 5000**. Estimated **220.6246 slot-hours** including overhead. Preserve complete first-segment model states; the continuation uses the original fresh-optimizer/scaler/RNG reset, not a new initialization.
3. Seeds 43/61, all four perturbation arms: **144 new runs x 5000**, estimated **188.8476 slot-hours** including overhead. Reassess only approved resource availability, never scientific thresholds or hyperparameters.

Total mean-based planning estimate **{c.planning_hours_mean_with_15pct_overhead.sum():.4f} slot-hours**; median-based reference **{(c.estimated_segment_hours_median.sum()*1.15):.4f} hours**; summed per-seed/segment q90 reference **{(c.estimated_segment_hours_q90.sum()*1.15):.4f} hours**. These references are not statistical confidence bounds. In particular, legacy seed 29 includes a very slow run; no runtime observation was excluded.

Measured available filesystem space at estimation: **{storage.available_bytes/1e9:.3f} GB**. Full-grid native retention plus allowance: **{storage.full_grid_with_15pct_allowance_bytes/1e12:.4f} TB**, which does not fit. First-stage files plus allowance fit the measured free space, but available quota/space must be rechecked before starting. Preserve existing results; do not reduce retention or change destinations silently. Additional stages require a concrete approved storage/archival plan. No guaranteed parallel wall time is quoted because host availability and runtime contention vary.

## Pinned audit inputs

SHA256 hashes at preregistration generation:
{hashes}

No Phase-2 outcome has been observed or classified. User approval and a recorded commit of this exact file are required before any Phase-2 run.
'''
(OUT/'preregistration.md').write_text(text)
print('preregistration written',sha(OUT/'preregistration.md'))
