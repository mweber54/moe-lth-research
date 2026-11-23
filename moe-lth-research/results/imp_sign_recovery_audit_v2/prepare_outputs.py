from common import *
import shutil
# Preserve prior read-only evidence in this new results directory.
for name in ('full_shuffle_cohort_audit.csv','initial_checkpoint_audit.csv','mask_discovery_provenance.csv','source_evidence.csv','draw_design_audit.csv'):
 shutil.copyfile(ROOT/'results/imp_sign_recovery_audit_v1'/name,OUT/f'phase0_1_{name}')
source=[]
for file,lo,hi,topic in [('src/moe_lth/models/transformer.py',78,88,'embedding_and_tied_head_initialization'),('src/moe_lth/models/transformer.py',103,103,'unscaled_input_embedding'),('src/moe_lth/models/transformer.py',126,126,'unscaled_output_head'),('src/moe_lth/models/experts.py',7,16,'expert_initialization_and_no_mask_compensation'),('src/moe_lth/data.py',151,165,'next_token_targets'),('scripts/imp_80_85/run.py',89,95,'each_discovery_round_rewinds_all_weights_to_original_init')]:
 lines=(ROOT/file).read_text().splitlines();source.append(dict(topic=topic,file=file,start_line=lo,end_line=hi,quote='\n'.join(f'{i}: {lines[i-1]}' for i in range(lo,hi+1))))
write('initialization_code.csv',source)
# Ensure future re-execution reproduces the corrected source-code citations.
path=OUT/'high_ce.py';s=path.read_text().replace("106,106,'unscaled_input_embedding'","103,103,'unscaled_input_embedding'").replace("132,132,'unscaled_output_head'","126,126,'unscaled_output_head'");path.write_text(s)
path=OUT/'sign_agreement.py';s=path.read_text().replace("assert read(OUT/'high_ce_status.csv') if False else True", "status=pd.read_csv(OUT/'high_ce_status.csv').iloc[0]; assert status.status=='complete' and not status.evaluation_broken");path.write_text(s)
prov=pd.read_csv(OUT/'checkpoint_provenance.csv');prov['num_experts']=32
prov['trained_input_actual_sparsity']=[read(ROOT/f'results/imp_80_85/trajectories/e32/seed_{r.seed}/round_{r.checkpoint_round:02d}/stage.json')['actual_sparsity'] for r in prov.itertuples()]
prov['selection_parent_actual_sparsity']=[read(ROOT/f'results/imp_80_85/trajectories/e32/seed_{r.seed}/mask_07.json')['actual_sparsity'] for r in prov.itertuples()]
prov['training_order']='native discovery order, not recovery orders A/B';prov.to_csv(OUT/'checkpoint_provenance.csv',index=False)
write('interpretation_plan.csv',[dict(analysis='primary_discovery',question='Do retained initial signs agree with trained round07 values more than newly pruned coordinates and alive-set random controls?',contrast='retained vs newly_pruned; random controls sampled from the same alive set with matched per-matrix budgets',caveat='Earlier IMP rounds already selected this alive set; contrast isolates only the last cut.',source='user instructions supplied before sign measurements; plan serialized with outputs'),dict(analysis='secondary_dense',question='Does init-sign agreement precede pruning selection?',contrast='eventual IMP survivors vs eventual pruned; exact legacy random masks plus separate uniform matrix-matched controls',caveat='Dense and sparse checkpoints answer different conditional questions; comparisons are not a causal intervention on IMP history.',source='user instructions supplied before sign measurements; plan serialized with outputs'),dict(analysis='joint',question='Is agreement already present in dense training, and does discovery show stronger agreement?',contrast='Report measured differences without forcing one of the proposed narratives.',caveat='Phase1 associations cannot establish which Phase2 recovery-outcome rule applies.',source='user interpretation guide supplied before analysis')])
metric=['sign_agreement','corr_abs_trained_abs_init','corr_delta_init','outward_growth_fraction']
contrasts=[];nulls=[]
for analysis in ('discovery','dense'):
 raw=pd.read_csv(OUT/f'{analysis}_matrix_sign_metrics.csv');fixed=raw[raw.draw==0]
 kg='retained' if analysis=='discovery' else 'eventual_imp_survivors';pg='newly_pruned' if analysis=='discovery' else 'eventual_imp_pruned'
 k=fixed[fixed.group==kg].set_index(['seed','tensor']);p0=fixed[fixed.group==pg].set_index(['seed','tensor']);joined=k[['n','sign_matches']].join(p0[['n','sign_matches']],rsuffix='_removed')
 joined['expected_random_matches']=joined.n*(joined.sign_matches+joined.sign_matches_removed)/(joined.n+joined.n_removed)
 expected=joined.groupby('seed')[['n','expected_random_matches']].sum();expected['expected_random_agreement']=expected.expected_random_matches/expected.n
 d=pd.read_csv(OUT/f'{analysis}_per_seed.csv')
 for seed in SEEDS:
  z=d[d.seed==seed];ret=z[z.group==kg].iloc[0];pr=z[z.group==pg].iloc[0];rnd=z[z.group=='random_survivors']
  for m in metric:
   contrasts.append(dict(analysis=analysis,seed=seed,metric=m,retained=ret[m],comparison_group=pg,comparison=pr[m],difference=ret[m]-pr[m],comparison_draws=0))
   contrasts.append(dict(analysis=analysis,seed=seed,metric=m,retained=ret[m],comparison_group='new_random_survivors_mean',comparison=rnd[m].mean(),difference=ret[m]-rnd[m].mean(),comparison_draws=5))
  nulls.append(dict(analysis=analysis,seed=seed,expected_random_sign_agreement=expected.loc[seed,'expected_random_agreement'],observed_random_mean=rnd.sign_agreement.mean(),observed_random_sd=rnd.sign_agreement.std(),difference_from_expectation=rnd.sign_agreement.mean()-expected.loc[seed,'expected_random_agreement'],expectation='weighted by final survivor count within each matrix; not the unweighted full-population mean'))
write('sign_contrasts_per_seed.csv',contrasts);write('random_control_expectation.csv',nulls)
# Fixed targets and linear first-crossing times for legacy curves, with no extrapolation.
legacy=pd.read_csv(OLD/'trajectory_losses.csv');legacy.to_csv(OUT/'legacy_absolute_ce.csv',index=False)
ce=[]
for ref in rows(OLD/'dense_reference_checks.csv'):
 for line in Path(ref['common_readout_path']).read_text().splitlines():
  row=json.loads(line);ce.append(dict(seed=int(ref['seed']),order=ref['order'],step=row['step'],unpruned_CE=row['cross_entropy'],source_path=ref['common_readout_path']))
pd.DataFrame(ce).to_csv(OUT/'legacy_unpruned_comparator.csv',index=False)
targets=[];times=[]
for seed in SEEDS:
 for order in ('A','B'):
  base=legacy[(legacy.seed==seed)&(legacy.order==order)&(legacy.variant=='imp_original')]
  for anchor in (1000,2000,5000):
   target=float(base[base.step==anchor].cross_entropy.iloc[0]);targets.append(dict(seed=seed,order=order,target_anchor_step=anchor,target_CE=target,source_variant='imp_original'))
   for variant,g in legacy[(legacy.seed==seed)&(legacy.order==order)].groupby('variant'):
    g=g.sort_values('step');xs=g.step.to_numpy();ys=g.cross_entropy.to_numpy();hit=np.flatnonzero(ys<=target)
    if len(hit)==0:time=np.nan;status='not reached by 5000'
    else:
     j=int(hit[0]);status='reached'
     time=float(xs[0]) if j==0 else float(xs[j-1]+(target-ys[j-1])*(xs[j]-xs[j-1])/(ys[j]-ys[j-1]))
    times.append(dict(seed=seed,order=order,variant=variant,target_anchor_step=anchor,target_CE=target,time_to_threshold=time,status=status,interpolation='linear first crossing on fixed logged steps; no extrapolation'))
write('ce_targets.csv',targets);write('legacy_time_to_threshold.csv',times)
# Descriptive nested bootstrap: average orders before resampling seeds and paired legacy realizations.
z=pd.read_csv(OUT/'legacy_order_averaged_realizations.csv');rng=np.random.default_rng(3208025);intervals=[]
for step in STEPS:
 mat=z[z.step==step].pivot(index='seed',columns='realization',values='paired_interaction').loc[list(SEEDS)].to_numpy()
 boot=[]
 for _ in range(5000):
  sampled=rng.integers(0,5,5);boot.append(np.mean([mat[s,rng.integers(0,2,2)].mean() for s in sampled]))
 intervals.append(dict(step=step,mean_interaction=mat.mean(),bootstrap_low=np.quantile(boot,.025),bootstrap_high=np.quantile(boot,.975),replicates=5000,rng_seed=3208025,unit='source seed; legacy realizations nested within seed; A/B averaged first',limitation='only two legacy paired realizations; random-mask and shuffle variability are confounded'))
write('legacy_interaction_intervals.csv',intervals)
# Explicit status: no Phase2 results are fabricated for plots or decision rules.
write('phase_status.csv',[dict(phase='0.1',status='complete',note='E32/80% full-shuffle configs and initialization code/provenance audited'),dict(phase='0.2',status='complete',note='20 baseline evaluations and independent CE checks; evaluation is not broken; initialization predictions pathological'),dict(phase='0.3',status='complete',note='70 protocol checks; 30 originals reusable; 40 full-shuffle runs rejected for reuse and scheduled for new 3-draw runs'),dict(phase='0.4',status='complete',note='all logged-step spreads and threshold components recorded; legacy random draws confound mask and shuffle variation'),dict(phase='1_primary',status='complete',note='round07 retained vs newly-pruned and alive-set random controls; no already-pruned comparison'),dict(phase='1_secondary',status='complete',note='all five round00 dense checkpoints exist; analyzed separately'),dict(phase='2',status='not_launched',note='cost and preregistration await user approval; no recovery outcome rule evaluated')])
print('prepared tables')
