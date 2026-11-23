from common import *
import platform,socket,datetime
# Mark checks that are inapplicable explicitly, rather than as failed/implicitly passed.
inv=pd.read_csv(OUT/'sign_invariants.csv');app=inv.analysis.eq('discovery');inv['random_subset_parent_check_applicable']=app;inv['random_subset_parent_passed']=np.where(app,True,None);inv=inv.drop(columns='random_subset_parent',errors='ignore');inv.to_csv(OUT/'sign_invariants.csv',index=False)
sc=pd.read_csv(OUT/'initialization_scaling.csv');sc['survivor_value_check_applicable']=sc.tensor.str.contains('.moe.experts.',regex=False);sc.loc[~sc.survivor_value_check_applicable,'survivor_values_unchanged']=None;sc.to_csv(OUT/'initialization_scaling.csv',index=False)
# Keep rerunnable report definitions consistent with the clearer applicability fields.
p0=OUT/'sign_agreement.py';s=p0.read_text().replace("random_subset_parent=analysis=='discovery'","random_subset_parent_check_applicable=analysis=='discovery',random_subset_parent_passed=True if analysis=='discovery' else None");p0.write_text(s)
p0=OUT/'high_ce.py';s=p0.read_text().replace("if mask is not None else True,variance_compensation_factor", "if mask is not None else None,survivor_value_check_applicable=mask is not None,variance_compensation_factor");p0.write_text(s)
checks=[]
def check(name,condition,details):
 assert condition,(name,details);checks.append(dict(check=name,passed=True,details=details))
h=pd.read_csv(OUT/'high_ce_audit.csv');check('20_original_CE_evaluations',len(h)==20,'five sources times dense/IMP/two fixed random masks')
check('baseline_reproduction_tolerance',h.baseline_difference.abs().max()<=.001,f'max difference {h.baseline_difference.abs().max():.17g}')
check('independent_CE_arithmetic',h.independent_CE_difference.abs().max()<=.001,f'max difference {h.independent_CE_difference.abs().max():.17g}')
co=pd.read_csv(OUT/'recovery_compatibility.csv');check('70_legacy_run_protocol_checks',len(co)==70 and co.protocol_compatible.all(),'30 originals reusable; shuffled jobs rerun by user decision')
check('sealed_source_matches',pd.read_csv(OUT/'sealed_source_audit.csv').matches.all(),'actual files hashed against sealed registration')
check('reconstructed_A_B_data_orders',len(pd.read_csv(OUT/'reconstructed_data_orders.csv'))==20,'five sources x two orders x two segments; actual content hashes compared to run metadata')
for analysis in ('discovery','dense'):
 d=pd.read_csv(OUT/f'{analysis}_matrix_sign_metrics.csv');check(f'{analysis}_matrix_rows',len(d)==15360,'five sources x 256 matrices x (two fixed groups plus ten random groups)')
 check(f'{analysis}_finite_statistics',np.isfinite(d[['sign_agreement','corr_abs_trained_abs_init','corr_delta_init']]).all().all(),'all groups have nonzero counts and defined correlations')
 check(f'{analysis}_valid_correlations',d[['corr_abs_trained_abs_init','corr_delta_init']].abs().max().max()<=1+1e-10,'Pearson ranges')
 key='retained' if analysis=='discovery' else 'eventual_imp_survivors'
 base=d[d.group==key].set_index(['seed','tensor']).n
 rnd=d[d.group=='random_survivors'].set_index(['seed','tensor'])
 check(f'{analysis}_random_matrix_budgets',all(float(row.n)==float(base.loc[idx]) for idx,row in rnd.iterrows()),'each draw matches the corresponding final IMP survivor count per matrix')
 for seed in SEEDS:
  total=d[(d.seed==seed)&(d.draw==0)].n.sum();expected=14073728 if analysis=='discovery' else 67108864
  check(f'{analysis}_seed{seed}_eligible_count',total==expected,f'eligible count={int(total)}')
check('primary_subset_and_zero_invariants',inv[app].final_subset_parent.all() and inv[app].trained_previously_pruned_zero_checked.all(),'the executed sign-analysis assertions also checked zeros outside the parent mask')
check('legacy_threshold_rows',len(pd.read_csv(OUT/'legacy_time_to_threshold.csv'))==210,'five sources x two orders x seven variants x three fixed targets')
cost=pd.read_csv(OUT/'phase2_cost_estimate.csv');check('planned_run_count',cost.new_distinct_runs.sum()==360 and cost.total_attempted_steps.sum()==1800000,'proposed budget only, no training')
roles=pd.read_csv(OUT/'checkpoint_role_audit.csv');check('checkpoint_role_and_mask_identity',roles.iloc[:,4:].all().all(),'actual selecting path, parent/final mask hashes, initial identity and seed checked')
check('all_checkpoint_hashes_verified',pd.read_csv(OUT/'checkpoint_provenance.csv').hash_matches.all(),'round07 and round00 trained checkpoints verified against stage hashes')
write('validation_results.csv',checks)
write('cost_assumptions.csv',[dict(new_training_benchmark_run=False,runtime_unit='observed native-segment elapsed slot time, not GPU-active time',runtime_observations_excluded=0,planning_allowance_fraction=.15,phase2_training_started=False)])
# Per-seed first: no pooled sign-agreement number replaces these rows.
primary=pd.read_csv(OUT/'discovery_per_seed.csv');dense=pd.read_csv(OUT/'dense_per_seed.csv');legacy=pd.read_csv(OUT/'dense_legacy_random_per_seed.csv')
lines=[]
for seed in SEEDS:
 p1=primary[primary.seed==seed];d1=dense[dense.seed==seed];l1=legacy[legacy.seed==seed]
 get=lambda df,g:100*df[df.group==g].sign_agreement.mean()
 lines.append(f'|{seed}|{get(p1,"retained"):.2f}|{get(p1,"newly_pruned"):.2f}|{get(p1,"random_survivors"):.2f}|{get(d1,"eventual_imp_survivors"):.2f}|{get(d1,"eventual_imp_pruned"):.2f}|{get(l1,"legacy_random_survivors"):.2f}|')
text='''**E32/IMP80 only.** The selecting checkpoint is round 07, trained for 2500 attempts from an original-init rewind with actual input sparsity **79.028511%**; the final mask is **79.999924%** sparse. The separate dense checkpoint is round 00, also at step 2500. This is not a one-shot dense-to-80% prune. ([checkpoint_provenance.csv](checkpoint_provenance.csv): `num_experts`, `checkpoint_round`, `checkpoint_step`, `trained_input_actual_sparsity`, `final_mask_actual_sparsity`, `rewind_policy`.)

Sign agreement, percent, by seed. Discovery random controls use the same round-07 alive set and matrix survivor budgets (five draws); dense random controls below are the two exact legacy masks. Already-pruned discovery coordinates are excluded. ([discovery_per_seed.csv](discovery_per_seed.csv), [dense_per_seed.csv](dense_per_seed.csv), [dense_legacy_random_per_seed.csv](dense_legacy_random_per_seed.csv): `seed`, `group`, `draw`, `sign_agreement`, `n`.)

|Seed|Discovery retained|Newly pruned|Alive-set random|Dense eventual IMP|Dense eventual pruned|Dense legacy random|
|---|---|---|---|---|---|---|
'''+ '\n'.join(lines)+'''

Retained discovery weights agree more than newly pruned weights and alive-set random controls in every seed. Earlier rounds already selected that alive set, so this contrast isolates the last cut only. Agreement is already strong in dense round 00 and is actually higher there on eventual IMP survivors than in round 07; the simple “rounds strengthen sign agreement” narrative is not supported. Random controls also show strong agreement, contradicting the proposed absence of random-mask correlation. These are survival/training associations, not a demonstrated causal recovery mechanism. The requested magnitude and growth correlations are included separately by matrix/layer/projection. ([sign_contrasts_per_seed.csv](sign_contrasts_per_seed.csv): `metric`, `comparison_group`, `difference`; [discovery_per_seed.csv](discovery_per_seed.csv), [dense_per_seed.csv](dense_per_seed.csv): `sign_agreement`, `corr_abs_trained_abs_init`, `corr_delta_init`; [phase_status.csv](phase_status.csv): `phase`, `status`.)

The high step-0 CE is **real arithmetic in a pathological initialization regime**, not a broken CE evaluation: all 20 baselines reproduce within 2.9e-14 nats and independent log-sum-exp CE agrees within 5.7e-6. Dense init also has CE 171–178. Every baseline copies the input token at every position, with mean maximum probability numerically 1 and next-token accuracy 2.04%; LayerNorm variance is about 0.999995. Native unscaled embeddings tied to the output head produce very large logits; masking is not the source of the high-CE regime. Step-0 damage therefore measures initial function/confidence, not trained prediction quality. ([high_ce_audit.csv](high_ce_audit.csv): `CE`, `baseline_difference`, `independent_CE_difference`, `input_copy_fraction`, `mean_max_probability`, `accuracy`, `embedding_std`, `head_tied`, `logit_abs_max`; [layernorm_audit.csv](layernorm_audit.csv): `mean_output_variance`; [initialization_code.csv](initialization_code.csv): `topic`, `file`, `start_line`, `quote`.)

All 70 legacy runs pass the reconstructed protocol checks; 30 originals can be reused, while all full-shuffle draws will be rerun. Legacy random variability mixes mask and shuffle variation because there were two IMP draws but only one shuffle per random mask. The post-init empirical noise envelope gives **X=0.02486047**, **Y=0.04972095** nats, with **4/5 seeds** required for reopening at step 2000. This limited legacy sample does not establish a population noise bound. Step 5000 remains separate from the intermediate trajectory. ([recovery_compatibility.csv](recovery_compatibility.csv): `protocol_compatible`, `approved_for_reuse`, `disposition`; [noise_floor_per_step.csv](noise_floor_per_step.csv): `random_min`, `random_max`, `random_sd`, `interaction_across_seed_sd`, `interaction_within_seed_sd_max`, `legacy_shuffle_draws_per_random_mask`; [decision_thresholds.csv](decision_thresholds.csv): `near_zero_X`, `reopens_Y`, `required_seeds_K`.)

The proposed first stage is **54 sign runs × 2500 attempts**, about **25.85 elapsed GPU-slot hours** including 15% allowance, followed by a step-2000 gate. All four perturbation arms would require **360 full runs**, including 90 new full-shuffle runs: about **435.32 slot-hours** using historical means, with substantial runtime variation retained in the estimate. First-stage storage is about **188.63 GB** versus **234.20 GB** measured free; the full **2.29 TB** retention budget does not fit. Additional stages need an approved storage plan. No new training benchmark was run. ([phase2_cost_estimate.csv](phase2_cost_estimate.csv): `new_distinct_runs`, `total_attempted_steps`, `planning_hours_mean_with_15pct_overhead`; [observed_runtime.csv](observed_runtime.csv): `segment_wall_seconds_per_5000_attempts`; [phase2_storage_estimate.csv](phase2_storage_estimate.csv): `available_bytes`, `first_stage_with_15pct_allowance_bytes`, `full_grid_with_15pct_allowance_bytes`, `full_grid_fits_current_free_space`; [cost_assumptions.csv](cost_assumptions.csv): `new_training_benchmark_run`, `runtime_unit`, `runtime_observations_excluded`.)

**No Phase-2 outcome rule has fired: Phase 2 has not run.** These audits cannot establish whether sign placement explains the mid-training gap or the step-5000 endpoint. Cost and preregistration approval are still required. ([phase_status.csv](phase_status.csv): `phase`, `status`, `note`.)
'''
(OUT/'findings.md').write_text(text)
write('analysis_environment.csv',[dict(cohort='E32/IMP80',python=sys.version,torch=torch.__version__,cuda_build=torch.version.cuda,gpu=torch.cuda.get_device_name(0),host=socket.gethostname(),recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase2_started=False)])
write('analysis_code_hashes.csv',[dict(file=f.name,sha256=sha(f)) for f in sorted(OUT.glob('*.py'))])
print('validated',len(checks),'checks; findings written')
