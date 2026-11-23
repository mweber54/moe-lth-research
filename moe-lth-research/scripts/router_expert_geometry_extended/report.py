from common import *

def values_table(frame,metric,keys):
    lines=['| '+' | '.join(keys+['Seed 7','Seed 17','Seed 29','Seed 43','Seed 61','Mean','95% seed bootstrap'])+' |',
           '| '+' | '.join(['---']*len(keys)+['---:']*6+['---'])+' |']
    for key,g in frame.groupby(keys,dropna=False):
        key=key if isinstance(key,tuple) else (key,);vals={int(r.seed):r[metric] for _,r in g.iterrows()}
        m,lo,hi=bootstrap(list(vals.values()))
        f=lambda x:'NA' if x is None or not np.isfinite(x) else f'{x:.5f}'
        lines.append('| '+' | '.join([str(x) for x in key]+[f(vals.get(s)) for s in SEEDS]+[f(m),f'[{f(lo)}, {f(hi)}]'])+' |')
    return lines

def write_report(cases,inventory):
    status=read(OUT/'analysis_status.json');correction=read(OUT/'svd_correction.json')
    svd=pd.read_csv(OUT/'svd_endpoint_changes.csv');gram=pd.read_csv(OUT/'gram_endpoint_changes.csv');act=pd.read_csv(OUT/'activation_endpoint_changes.csv')
    trajectory=pd.read_csv(OUT/'svd_seed_trajectories.csv');coverage=pd.read_csv(OUT/'activation_coverage.csv')
    endpoint=svd[svd.layer.eq(-1)&svd.membership.eq('endpoint_reference_valid')]
    counts=[]
    for (rank,kind),g in endpoint.groupby(['rank','mask_type']):counts.append(dict(rank=int(rank),mask_type=kind,n_runs=len(g),source_overlap_decreases=int(g.source_subspace_overlap_change.lt(0).sum()),source_overlap_increases=int(g.source_subspace_overlap_change.gt(0).sum())))
    frame(OUT/'svd_direction_counts.csv',counts)
    lines=['# Geometry and score–activation extension v1','',
      '**Post hoc exploratory analysis.** These additional metrics and aggregation corrections were selected after inspecting original rank-16 results. Existing analysis outputs, checkpoints, and Dataset B were preserved. No training or optimizer steps were run. Rank 16 remains the original primary SVD metric; ranks 8/32, Gram metrics, and GELU associations are complementary exploratory measurements.','',
      '## Input and reference audit','',
      f"Measured {status['recovery_states']}/525 recovery states (75 runs × seven steps) and {status['measured_existing_states']} existing source/pruning/initialization/historical states. State identities, probe hashes, checkpoint origins, mask-zero verification, and sample coverage are exported. Mid-trajectory diagnostic snapshots did not carry an independent prior file hash; this pass establishes SHA-256 identity, checks their step/probe metadata, and verifies finite tensors and mask zeros. Step zero is checked against the exact constructed initialization; step 2500 is tensor-compared against the original hash-attested native endpoint.",
      '', 'Source-final dense, current matched dense recovery at the same router age/step, and sparse initialization are separate Gram references. Historical dense comparators remain historical. Source/pruning-only states have no invented recovery comparator. Summary CSVs are used for recorded SVD metrics only; new Gram and token statistics are computed from actual weights and probes. See `metric_definitions.md`, `state_coverage_inventory.csv`, `coverage/`, and per-state `completed.json`.',
      '', '## Corrected SVD findings','',
      f"The original rank-16 learned-mask claim changes from {correction['original_learned_decreases']}/30 to **{correction['endpoint_matched_learned_decreases']}/30** when both endpoints use identical valid (layer, expert) membership. Seed 17/R0/50% and seed 43/R0/50% instead increase slightly. The original step-specific mean changed which experts were included; therefore its universal sign claim is invalid.",
      '', '| Rank | Mask type | Decreasing source overlap | Increasing source overlap |', '| --- | --- | ---: | ---: |']
    for r in counts:lines.append(f"| {r['rank']} | {r['mask_type']} | {r['source_overlap_decreases']}/{r['n_runs']} | {r['source_overlap_increases']}/{r['n_runs']} |")
    lines += ['', 'Endpoint matching, fixed trajectory intersections, and learned/random joint intersections are reported separately. The main figures, direction counts and cross-metric cases require valid current and source-reference bases; the direct original-rule correction is preserved separately. Both reproduce 28/30. Cutoff gaps and retained energy are preserved in the expert raw tables. An endpoint decrease is not a monotonic trajectory claim.', '', '## Full-spectrum Gram findings','']
    for kind in ('learned','random','dense'):
        d=cases[cases.mask_type.eq(kind)]
        lines.append(f"- {kind}: {int(d.source_final_shape_distance_change.gt(0).sum())}/{len(d)} runs move farther from source-final Gram shape; {int(d.source_final_shape_distance_change.lt(0).sum())}/{len(d)} move closer. This is a separate measurement from top-rank subspace overlap.")
    lines+=['','All seed values for source-final Gram shape distance changes (negative = approach):','']
    g=gram[gram.layer.eq(-1)&gram.reference.eq('source_final')]
    lines+=values_table(g,'shape_distance_change',['mask_type','sparsity','router_age'])
    lines+=['','Matched dense reference and scale are separately reported in `gram_endpoint_seed_summary.csv`. The normalized Frobenius cosine and shape distance satisfy d²=2−2cosine and are not independent evidence. Trace and trace ratio measure projection energy; relative unnormalized distance combines shape and scale. Coordinatewise comparisons do not remove upstream representation drift.', '', '## GELU score–activation association','',
      'The statistic uses the raw linear router score and mean signed post-GELU fc1 activation before route-weight multiplication. Fixed source hidden inputs and native current representations are separate modes. Proposed assignments, capacity-accepted assignments, and the fixed 128-token all-expert counterfactual subset are distinct. Each layer/expert/selection group is z-scored separately before pooling. Pearson is primary; Spearman is sensitivity. Per-expert associations, equal-expert summaries, and token-weighted standardized pooled Pearson are kept distinct.','']
    for mode in ('fixed','native'):
        for selection in ('proposed','accepted','all_expert'):
            metric=f'{mode}_{selection}_pearson_expert_mean_change';above=f'{mode}_{selection}_above_shuffle_final'
            lines.append(f'### {mode}, {selection}')
            lines.append('')
            for kind in ('learned','random','dense'):
                d=cases[cases.mask_type.eq(kind)];v=d[metric].dropna();a=d[above].dropna()
                lines.append(f"- {kind}: association increases at the endpoint in {int(v.gt(0).sum())}/{len(v)} evaluable runs; final association exceeds within-layer shuffled pairings in {int(a.gt(0).sum())}/{len(a)}. Run counts are descriptive; source seed is the replication unit.")
            lines+=['','Equal-expert endpoint Pearson changes with all five seed values:','']
            a=act[act['mode'].eq(mode)&act.selection.eq(selection)&act.layer.eq(-1)&act.membership.eq('endpoint_matched')]
            lines+=values_table(a,'pearson_expert_mean_change',['mask_type','sparsity','router_age']);lines.append('')
    lines+=['## Joint observations and discordant cases','']
    for kind in ('learned','random','dense'):
        d=cases[cases.mask_type.eq(kind)];sub=d.source_subspace_overlap_change.lt(0);away=d.source_final_shape_distance_change.gt(0)
        coupling=d.native_proposed_pearson_expert_mean_change.gt(0)
        positive=d.native_proposed_pearson_expert_mean_final.gt(0)&d.native_proposed_above_shuffle_final.gt(0)
        lines.append(f"- {kind}: leading-rank overlap decreases in {int(sub.sum())}/{len(d)}; both overlap decreases and full Gram distance increases in {int((sub&away).sum())}/{len(d)}. Native proposed-pair coefficients increase despite decreasing overlap in {int((sub&coupling).sum())}/{len(d)}, and despite increasing full Gram distance in {int((away&coupling).sum())}/{len(d)}. Positive final association above shuffled controls coexists with full Gram departure in {int((away&positive).sum())}/{len(d)}. A higher coefficient can remain negative; endpoint increases alone are not evidence of positive coupling.")
    lines+=['','Every concordant and discordant case is retained in `combined_seed_cases.csv`; loss and sparse-minus-matched-dense gap accompany each metric. `joint_expert_endpoint_effects.parquet` and `joint_matched_expert_seed_effects.csv` additionally enforce identical expert eligibility across SVD, Gram and activation measurements. Learned/random and router-age joint comparisons match experts across conditions before seed bootstrap uncertainty. See `existing_reference_findings.md` for separate source and pruning-only findings. The following illustrative rows are selected descriptively, not for a hypothesis test:','',
       '| Condition | Δ rank-16 overlap | Δ source Gram distance | Δ native proposed Pearson | Final dense gap |','| --- | ---: | ---: | ---: | ---: |']
    selections=[]
    for query in [cases.source_subspace_overlap_change.lt(0)&cases.source_final_shape_distance_change.lt(0),
                  cases.source_final_shape_distance_change.gt(0)&cases.native_proposed_pearson_expert_mean_change.gt(0),
                  cases.source_subspace_overlap_change.gt(0),cases.native_proposed_pearson_expert_mean_change.lt(0)]:
        sample=cases[query].sort_values('matched_dense_gap').head(3)
        selections.extend(sample.condition_id.tolist())
    for cid in dict.fromkeys(selections):
        r=cases[cases.condition_id.eq(cid)].iloc[0]
        lines.append(f'| {cid} | {r.source_subspace_overlap_change:.5f} | {r.source_final_shape_distance_change:.5f} | {r.native_proposed_pearson_expert_mean_change:.5f} | {r.matched_dense_gap:.5f} |')
    invalid=coverage[~coverage.status.eq('valid')]
    consistency=read(OUT/'metric_consistency_validation.json')
    cohort=[]
    for filename in ('activation_endpoint_seed_summary.csv','joint_matched_expert_seed_summary.csv'):
        d=pd.read_csv(OUT/filename);limited=d[d.n_seeds.lt(5)].copy();limited['source_table']=filename;cohort.append(limited)
    frame(OUT/'summary_coverage_exceptions.csv',pd.concat(cohort,ignore_index=True))
    lines+=['','## Coverage and uncertainty','',f"There are {int(invalid.n_experts.sum())} excluded expert/group/time/mode/selection observations, with reasons and counts in `activation_coverage.csv`. Groups need at least eight samples and nonzero variance. Repeated tokens, experts, layers, and checkpoints are not independent training replicates. The original probe was not enlarged; sample-poor routed experts remain explicitly missing. This probe is descriptive and may underpower routed correlations.",
       '',f"All 89,600 expert/state Gram calculations used actual weights. Of these, {consistency['independent_energy_checks']} also match the independently recorded original SVD energy within relative tolerance 2e-6. Older SVD tables omit that field for {consistency['original_svd_energy_unavailable_rows']} rows (23 states); their actual Gram trace is available, but that extra archival energy check cannot be made. Exact seed/condition rows are marked in `svd_gram_weight_identity_consistency.csv`. This is a missing legacy field, not a missing weight checkpoint.",
       '', 'All SVD and Gram endpoint summary strata retain five seeds. Some activation layer/mode/selection strata retain zero to four seeds after fixed expert matching, and joint three-metric strata can retain only two seeds. Empty strata have undefined means/intervals, not zero effects. These coverage-limited results are not five-seed evidence; all exceptions and available seed values are in `summary_coverage_exceptions.csv`. Two-seed intervals are particularly unstable and support no strong inference.',
       '', 'Paired learned/random comparisons match valid experts across both mask types and required steps before computing effects. Uncertainty uses 10,000 paired seed bootstraps at fixed seed 8103, n≤5, and all five individual values are provided. These are pointwise exploratory intervals, not simultaneous confidence bands. No formal p-values or multiplicity-adjusted significance claims are made; multiplicity is not corrected, so results must not be interpreted as confirmatory. Within-condition loss associations use five seed-level values and remain unstable descriptive sensitivities.',
       '', 'Equal Gram matrices do not imply identical nonlinear experts: left-orthogonal input-projection transformations preserve G but need not preserve GELU. Gram excludes bias, fc2, nonlinear computation, and routing. Higher score–activation association is not proof of source restoration or a causal mechanism. Representation drift is measured separately, and conclusions vary with reference, probe mode, rank, and layer.', '', '## Conclusion','']
    learned=cases[cases.mask_type.eq('learned')]
    departure=int(learned.source_final_shape_distance_change.gt(0).sum());lead=int(learned.source_subspace_overlap_change.lt(0).sum())
    assoc=int((learned.source_final_shape_distance_change.gt(0)&learned.native_proposed_pearson_expert_mean_change.gt(0)).sum())
    positive=int((learned.source_final_shape_distance_change.gt(0)&learned.native_proposed_pearson_expert_mean_final.gt(0)&learned.native_proposed_above_shuffle_final.gt(0)).sum())
    lines += [f"**(c) Metric-dependent behavior** is the appropriate overall conclusion: rank-16 overlap decreases in {lead}/30 learned runs, while full-spectrum source Gram distance increases in {departure}/30. Source approach occurs in some states/metrics; it is not universal restoration. Native proposed coefficients increase while full Gram shape departs in {assoc}/30 learned runs, but only {positive}/30 simultaneously show positive final proposed association above shuffled controls and full Gram departure. Selected, accepted and all-expert results must be distinguished. Where favorable losses coexist with positive above-control association and departure, they support limited descriptive evidence for (b), not a causal mechanism. Read the per-layer, per-seed, mask-control, probe-mode and reference-choice results before narrowing this conclusion.",
       '', '## Reproduction','', 'Run from repository root:', '', '```bash',
       'OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/router_expert_geometry_extended/validate.py',
       'OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/router_expert_geometry_extended/svd.py',
       'OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/router_expert_geometry_extended/run.py --seed 7 --benchmark',
       'OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/router_expert_geometry_extended/run.py',
       'OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/router_expert_geometry_extended/analyze.py',
       'OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/router_expert_geometry_extended/existing_summary.py',
       'OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/router_expert_geometry_extended/audit.py','```','',
       'Inference-only per-seed workers may use `--seed 7`, `17`, `29`, `43`, or `61`; completion caches are keyed by checkpoint/reference/probe/config and measurement-code hashes. Original artifacts are read-only. All outputs live in `analysis_extended/v1/`. Figures are PNG and PDF. Raw token pairs retain sample identity in `cache/**/token_pairs.npz`.']
    (OUT/'combined_findings.md').write_text('\n'.join(lines)+'\n')
    save(OUT/'findings_counts.json',dict(svd=counts,learned_source_gram_departure=departure,learned_native_coupling_strengthens_with_gram_departure=assoc))
