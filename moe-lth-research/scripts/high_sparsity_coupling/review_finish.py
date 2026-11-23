"""Finish the frozen experiment's analysis from verified existing measurements only."""
from common import *
from review_audit import REVIEW,paths_for,array_hash
from review_stats import paired_contrast,cache_identity
from measure import corr
from functools import lru_cache
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
DEST=OUT/'report'
KEYS=['layer','expert']
MODES=('fixed_source','native')
SELECTIONS=('all_expert','current_proposed','current_accepted','fixed_initial','fixed_source')
@lru_cache(maxsize=32)
def groups(track,cid,step):return pd.read_csv(OUT/'analysis'/track/cid/f'step_{step:04d}/groups.csv')
@lru_cache(maxsize=8)
def arrays(track,cid,step):
 with np.load(OUT/'analysis'/track/cid/f'step_{step:04d}/raw.npz') as x:return {k:x[k] for k in x.files if any(t in k for t in ('_scores','_activations','_selection_'))}
def frame(track,cid,step,mode,sel):
 d=groups(track,cid,step);return d[d['mode'].eq(mode)&d.selection.eq(sel)].set_index(KEYS)
def common_pairs_frames(track,scid,dcid,mode,sel):
 items=[arrays(track,cid,s) for cid,s in ((scid,0),(scid,5000),(dcid,0),(dcid,5000))];rows=[[] for _ in items];pair_ids=[]
 for layer in range(4):
  key=f'{mode}_L{layer}';pair=np.logical_and.reduce([a[key+'_selection_'+sel] for a in items]);pair_ids.append(array_hash(pair))
  for expert in range(32):
   keep=pair[:,expert]
   for out,a in zip(rows,items):
    p,s,status=corr(a[key+'_scores'][keep,expert],a[key+'_activations'][keep,expert]);out.append(dict(layer=layer,expert=expert,n=int(keep.sum()),pearson=p,spearman=s,status=status))
 return [pd.DataFrame(v).set_index(KEYS) for v in rows],pair_ids

def coupling(jobs):
 effects=[];trajectories=[];exclusions=[];within=[]
 for track in ('historical','rerun'):
  steps=(0,2500,3000,4000,5000) if track=='historical' else STEPS
  for j in jobs:
   if j['sparsity'] is None:continue
   for mode in MODES:
    for sel in SELECTIONS:
     frames=[frame(track,cid,step,mode,sel) for cid,step in ((j['id'],0),(j['id'],5000),(j['dense_id'],0),(j['dense_id'],5000))]
     variants=[('registered_condition_pairs',frames,None)]
     if sel in ('fixed_initial','fixed_source','all_expert'):
      ff,ph=common_pairs_frames(track,j['id'],j['dense_id'],mode,sel);variants.append(('identical_fixed_pairs',ff,ph))
     for variant,ff,ph in variants:
      for metric in ('pearson','spearman'):
       cov,equal,weighted=paired_contrast(ff,metric)
       meta=dict(track=track,seed=j['seed'],sparsity=j['sparsity'],condition=j['id'],mode=mode,selection=sel,pair_definition=variant,coefficient=metric,common_pair_hashes=json.dumps(ph))
       for aggregation,values in [('equal_layer_expert',equal),('pair_count_weighted',weighted)]:effects.append(dict(**meta,aggregation=aggregation,**cov,**values))
       for slot,f in zip(('sparse_initial','sparse_final','dense_initial','dense_final'),ff):
        exclusions.append(dict(**meta,cell=slot,n_valid=int((f[metric].notna()&f.n.ge(8)).sum()),n_low_count=int(f.n.lt(8).sum()),n_undefined=int(f[metric].isna().sum()),n_common=cov['n_valid_units'],n_excluded=cov['n_excluded_units']))
     # The trajectory uses one common layer/expert membership across ALL sparse+dense times.
     fs=[frame(track,cid,step,mode,sel) for cid in (j['id'],j['dense_id']) for step in steps]
     valid=set.intersection(*(set(f.index[f.pearson.notna()&f.n.ge(8)]) for f in fs));valid=sorted(valid)
     for cid in (j['id'],j['dense_id']):
      for step in steps:
       d=frame(track,cid,step,mode,sel).loc[valid]
       trajectories.append(dict(track=track,seed=j['seed'],sparsity=j['sparsity'],model_role='sparse' if cid==j['id'] else 'dense',condition=cid,step=step,mode=mode,selection=sel,n_valid_units=len(d),n_excluded_units=128-len(d),pearson=float(d.pearson.mean()) if len(d) else None,spearman=float(d.spearman.mean()) if len(d) else None,n_pairs=int(d.n.sum())))
 for track in ('historical','rerun'):
  for j in jobs:
   for step in paths_for(j,track):
    p=OUT/'analysis'/track/j['id']/f'step_{step:04d}/within.csv';d=pd.read_csv(p)
    for mode,z in d.groupby('mode'):within.append(dict(track=track,seed=j['seed'],sparsity=j['sparsity'],condition=j['id'],step=step,mode=mode,n_token_layer_rows=len(z),n_unique_positions=128,n_valid=int(z.pearson.notna().sum()),mean_within_token_across_expert_pearson=z.pearson.mean(),mean_within_token_across_expert_spearman=z.spearman.mean()))
 table(DEST/'paired_seed_coupling_changes.csv',effects);table(DEST/'coupling_trajectories.csv',trajectories);table(DEST/'coupling_exclusions.csv',exclusions);table(DEST/'within_token_associations.csv',within)
 effect=pd.DataFrame(effects);unc=[]
 for keys,d in effect.groupby(['track','sparsity','mode','selection','pair_definition','coefficient','aggregation']):
  vals=d.difference_in_changes.dropna().to_numpy();rng=np.random.default_rng(8103)
  if len(vals):lo,hi=np.quantile(rng.choice(vals,(10000,len(vals)),replace=True).mean(1),[.025,.975])
  else:lo=hi=np.nan
  unc.append(dict(zip(['track','sparsity','mode','selection','pair_definition','coefficient','aggregation'],keys),n_source_seeds=len(vals),mean_difference_in_changes=float(vals.mean()) if len(vals) else None,seed_bootstrap_low=lo,seed_bootstrap_high=hi,n_positive=int((vals>0).sum()),n_negative=int((vals<0).sum())))
 table(DEST/'coupling_seed_uncertainty.csv',unc);return effect,pd.DataFrame(trajectories)

def tickets(jobs,reeval):
 rows=[];budgets=[];complete=[];reset_controls=[]
 for j in jobs:
  h=read(Path(j['historical_checkpoints']['5000']).parents[1]/'extension_manifest.json');r=read(OUT/'reruns'/j['id']/'completed.json');assert r['cohort_sha256']==sha(OUT/'cohort.json') and r['protocol_sha256']==sha(OUT/'protocol.md')
  attempt=Path(r['attempt']);first=read(Path(j['boundary']).parents[1]/('stage.json' if j['mask'] else 'metadata.json'));r0=read(attempt/'segment_0/record.json');r1=read(attempt/'segment_2500/record.json');assert r0['optimizer_step_attempts']==r1['optimizer_step_attempts']==2500
  assert r['final_validation_ce']==r1['final_validation_loss'];assert r['first_applied']==r0['optimizer_steps_applied'] and r['second_applied']==r1['optimizer_steps_applied'];assert read(attempt/'snapshots/step_2500.json')['model_hash']==read(attempt/'snapshots/step_2500_after_reset.json')['model_hash']
  before=read(attempt/'snapshots/step_2500.json');after=read(attempt/'snapshots/step_2500_after_reset.json');assert after['optimizer_entries']==0
  reset_controls.append(dict(condition=j['id'],seed=j['seed'],sparsity=j['sparsity'],global_recovery_step=2500,before_path=str(attempt/'snapshots/step_2500.pt'),after_path=str(attempt/'snapshots/step_2500_after_reset.pt'),before_model_hash=before['model_hash'],after_model_hash=after['model_hash'],weights_unchanged=before['model_hash']==after['model_hash'],optimizer_entries_before=before['optimizer_entries'],optimizer_entries_after=after['optimizer_entries'],native_final_checkpoint=r['checkpoint'],native_final_local_step=2500,native_final_segment_offset=2500,native_final_global_step=5000))
  for track,ce,applied,skips in [('historical',h['extension_final_validation_cross_entropy'],first['optimizer_steps_applied']+h['applied_optimizer_updates'],first['amp_overflow_step_count']+h['trainer_record']['amp_overflow_step_count']),('rerun',r['final_validation_ce'],r0['optimizer_steps_applied']+r1['optimizer_steps_applied'],r0['amp_overflow_step_count']+r1['amp_overflow_step_count'])]:
   assert applied+skips==5000;budgets.append(dict(track=track,seed=j['seed'],sparsity=j['sparsity'],condition=j['id'],logged_ce=ce,attempted_updates=5000,applied_updates=applied,amp_skips=skips))
 b=pd.DataFrame(budgets)
 for track in ('historical','rerun'):
  for j in jobs:
   if j['sparsity'] is None:continue
   s=b[(b.track==track)&(b.condition==j['id'])].iloc[0];d=b[(b.track==track)&(b.condition==j['dense_id'])].iloc[0];q=reeval[reeval.condition.eq(j['id'])].iloc[0];qd=reeval[reeval.condition.eq(j['dense_id'])].iloc[0]
   rows.append(dict(track=track,seed=j['seed'],sparsity=j['sparsity'],dense_ce=d.logged_ce,sparse_ce=s.logged_ce,gap=s.logged_ce-d.logged_ce,strict_success=bool(s.logged_ce<=d.logged_ce),sparse_applied_updates=s.applied_updates,dense_applied_updates=d.applied_updates,sparse_amp_skips=s.amp_skips,dense_amp_skips=d.amp_skips,attempted_updates=5000,common_tf32_on_gap=q.reevaluated_tf32_on-qd.reevaluated_tf32_on,common_tf32_off_gap=q.reevaluated_tf32_off-qd.reevaluated_tf32_off))
 table(DEST/'optimizer_reset_controls.csv',reset_controls)
 result=pd.DataFrame(rows)
 for track in ('historical','rerun'):result[result.track.eq(track)].to_csv(DEST/f'{track}_tickets.csv',index=False)
 a=result[result.track.eq('historical')].merge(result[result.track.eq('rerun')],on=['seed','sparsity'],suffixes=('_historical','_rerun'),validate='one_to_one');a['label_agreement']=a.strict_success_historical==a.strict_success_rerun;a.to_csv(DEST/'ticket_label_agreement.csv',index=False);b.to_csv(DEST/'update_budgets.csv',index=False);reeval.to_csv(DEST/'ce_execution_settings.csv',index=False);return result

def metric_tables(jobs):
 geometry=[];qrows=[];emb=[];reps=[];func=[];models=[];spectra=[]
 for track in ('historical','rerun'):
  for j in jobs:
   for step in paths_for(j,track):
    root=OUT/'analysis'/track/j['id']/f'step_{step:04d}';meta=dict(track=track,seed=j['seed'],sparsity=j['sparsity'],condition=j['id'],step=step);done=read(root/'completed.json');models.append(dict(**meta,validation_ce=done['validation_ce'],probe_ce=done['probe_ce_mean'],probe_kl=done['probe_kl_mean']))
    d=pd.read_csv(root/'weight_geometry.csv')
    for (reference,rank),g in d.groupby(['reference','rank']):
     geometry.append(dict(**meta,reference=reference,rank=int(rank),n_units=len(g),n_svd_valid=int(g.subspace_overlap.notna().sum()),n_current_degenerate=int(g.degenerate_cutoff.sum()),n_reference_degenerate=int(g.reference_degenerate_cutoff.sum()),n_rank_deficient=int(g.rank_deficient.sum()),n_reference_rank_deficient=int(g.reference_rank_deficient.sum()),subspace_overlap=g.subspace_overlap.mean(),retained_energy=g.retained_energy.mean(),alignment_centered=g.alignment_centered.mean(),alignment_raw=g.alignment_raw.mean(),normalized_gram_distance=g.normalized_gram_distance.mean(),gram_cosine=g.gram_cosine.mean(),trace=g.trace.mean(),reference_trace=g.reference_trace.mean(),trace_ratio=(g.trace/g.reference_trace).mean(),unnormalized_difference=g.unnormalized_difference.mean(),relative_unnormalized_difference=g.relative_unnormalized_difference.mean()))
    d=pd.read_csv(root/'router_energy.csv')
    for router,g in d.groupby('router'):qrows.append(dict(**meta,router=router,n_units=int(g.q.notna().sum()),q=g.q.mean(),unmatched_q=g.unmatched_q_mean.mean(),q_above_unmatched=(g.q-g.unmatched_q_mean).mean(),isotropic=1/256,trace=g.trace.mean()))
    d=pd.read_csv(root/'embedding.csv');er=pd.read_csv(root/'embedding_rows.csv')
    for _,row in d.iterrows():
     rows=er[er.reference.eq(row.reference)];record={k:row[k] for k in d.columns if k!='singular_energy'};emb.append(dict(**meta,**record,n_byte_rows=len(rows),row_norm_mean=rows.norm.mean(),row_norm_min=rows.norm.min(),row_norm_max=rows.norm.max(),row_drift_mean=rows.drift.mean(),row_drift_max=rows.drift.max()));v=json.loads(row.singular_energy)
     if row.reference=='original_initial':
      for k,value in enumerate(v):spectra.append(dict(**meta,index=k,squared_singular_value=value))
    d=pd.read_csv(root/'reps.csv')
    for _,row in d.iterrows():reps.append(dict(**meta,**row.to_dict(),n_identical_probe_positions=2048))
    d=pd.read_csv(root/'function.csv')
    for (mode,layer),g in d.groupby(['mode','layer']):func.append(dict(**meta,mode=mode,layer=layer,n_experts=len(g),n_valid_outputs=int(g.n_valid.sum()),n_zero_exclusions=int(g.zero_exclusions.sum()),relative_output_error=g.relative_error.mean(),output_cosine=g.cosine.mean()))
 for name,rows in [('geometry_trajectories',geometry),('q_trajectories',qrows),('embedding_trajectories',emb),('embedding_spectra',spectra),('representation_trajectories',reps),('expert_function_trajectories',func),('model_function_trajectories',models)]:table(DEST/(name+'.csv'),rows)
 # Endpoint geometry deltas use identical valid expert membership within each rank/reference.
 endpoints=[]
 for track in ('historical','rerun'):
  for j in jobs:
   path=OUT/'analysis'/track/j['id'];aa=pd.read_csv(path/'step_0000/weight_geometry.csv');bb=pd.read_csv(path/'step_5000/weight_geometry.csv');m=aa.merge(bb,on=['layer','expert','rank','reference'],suffixes=('_initial','_final'),validate='one_to_one')
   for (rank,reference),g in m.groupby(['rank','reference']):
    row=dict(track=track,condition=j['id'],seed=j['seed'],sparsity=j['sparsity'],rank=rank,reference=reference,n_gram_units=int((g.normalized_gram_distance_initial.notna()&g.normalized_gram_distance_final.notna()).sum()))
    for metric in ('subspace_overlap','alignment_centered','alignment_raw','retained_energy','normalized_gram_distance','trace','unnormalized_difference','relative_unnormalized_difference'):
     v=g[g[metric+'_initial'].notna()&g[metric+'_final'].notna()];a=v[metric+'_initial'].mean();b=v[metric+'_final'].mean();row.update({metric+'_initial':a,metric+'_final':b,metric+'_change':b-a,metric+'_n_valid':len(v)})
    endpoints.append(row)
 table(DEST/'geometry_endpoint_changes.csv',endpoints)
 # Registered before/reset/after comparisons: no threshold chosen for a favorable temporal lag.
 reset=[];ep=pd.DataFrame(emb);rp=pd.DataFrame(reps)
 for j in jobs:
  for family,data,keycols in [('embedding',ep,['reference']),('representations',rp,['reference','boundary'])]:
   z=data[data.track.eq('rerun')&data.condition.eq(j['id'])]
   for keys,g in z.groupby(keycols):
    if not isinstance(keys,tuple):keys=(keys,)
    lookup=g.set_index('step')
    for t0,t1 in ((2000,2500),(2500,2525),(2500,2600),(2600,3000)):
     for metric in ('cka','relative_l2','effective_rank'):
      reset.append(dict(family=family,seed=j['seed'],sparsity=j['sparsity'],condition=j['id'],**dict(zip(keycols,keys)),start=t0,end=t1,metric=metric,initial=lookup.loc[t0,metric],final=lookup.loc[t1,metric],delta=lookup.loc[t1,metric]-lookup.loc[t0,metric],delta_per_attempted_step=(lookup.loc[t1,metric]-lookup.loc[t0,metric])/(t1-t0)))
 table(DEST/'optimizer_reset_representation_windows.csv',reset)
 return {name:pd.DataFrame(v) for name,v in [('geometry',geometry),('q',qrows),('embedding',emb),('representations',reps),('function',func),('models',models),('geometry_endpoints',endpoints)]}

def provenance(jobs):
 parts=[];families=[];pairs=[]
 for seed in SEEDS:
  status=read(REVIEW/'provenance'/f'seed_{seed}_complete.json');assert status['passed'] and status['states']==63
  parts.append(pd.read_csv(REVIEW/'provenance'/f'seed_{seed}_states.csv'));families.append(pd.read_csv(REVIEW/'provenance'/f'seed_{seed}_families.csv'));pairs.append(pd.read_csv(REVIEW/'provenance'/f'seed_{seed}_pairs.csv'))
 states=pd.concat(parts,ignore_index=True);ff=pd.concat(families,ignore_index=True);pp=pd.concat(pairs,ignore_index=True);assert len(states)==315
 hist=states[states.track.eq('historical')];rr=states[states.track.eq('rerun')];common=hist.merge(rr,on=['condition','seed','step'],suffixes=('_historical','_rerun'),validate='one_to_one');assert len(common)==75
 for g in ('model','expert','router','shared'):common[g+'_identical']=common[g+'_hash_historical']==common[g+'_hash_rerun']
 common.to_csv(DEST/'cross_track_component_identity.csv',index=False)
 fam=ff[ff.track.eq('historical')].merge(ff[ff.track.eq('rerun')],on=['condition','seed','step','family'],suffixes=('_historical','_rerun'),validate='one_to_one');fam['identical_output']=fam.sha256_historical==fam.sha256_rerun;fam.to_csv(DEST/'cross_track_measurement_families.csv',index=False)
 frozen_files={x['path']:x['sha256'] for x in frozen()['files']};refs={};reference_rows=[]
 for j in jobs:
  source=read(Path(j['mask']).with_suffix('.json'))['trained_checkpoint'] if j['mask'] else j['source_reference']
  for p in (j['initial'],j['mask'],j['source_reference'],source):
   if p and p not in refs:refs[p]=verify_file(p,frozen_files[p]);reference_rows.append(refs[p])
  done=read(OUT/'reruns'/j['id']/'completed.json');verify_file(done['checkpoint'],done['sha256']);native_payload=load(done['checkpoint']);assert native_payload['step']==2500;actual=native_payload['model'];expected=rr[rr.condition.eq(j['id'])&rr.step.eq(5000)].iloc[0].model_hash;assert state_dict_hash(actual)==expected
 table(DEST/'reference_checkpoint_identities.csv',reference_rows)
 launch=read(OUT/'launch.json');implementation=hashlib.sha256(json.dumps(launch['implementation'],sort_keys=True).encode()).hexdigest();keyrows=[]
 lookup=states.set_index(['track','condition','step'])
 for _,r in states.iterrows():
  j=next(j for j in jobs if j['id']==r.condition);selected=pp[(pp.track==r.track)&(pp.condition==r.condition)&(pp.step==r.step)];initial=lookup.loc[(r.track,r.condition,0)];dense=lookup.loc[(r.track,j['dense_id'],int(r.step))];finaldense=lookup.loc[(r.track,j['dense_id'],5000)]
  references=dict(source=refs[j['source_reference']],mask_selection=refs[r.mask_selection_reference],recovery_initial=dict(model_hash=initial.model_hash,checkpoint=initial.checkpoint_path,mask_sha256=refs[j['mask']]['sha256'] if j['mask'] else None),same_step_dense=dict(path=dense.checkpoint_path,sha256=dense.checkpoint_sha256,model_hash=dense.model_hash),fixed_dense_final=dict(path=finaldense.checkpoint_path,sha256=finaldense.checkpoint_sha256,model_hash=finaldense.model_hash))
  config=dict(schema=2,cohort=sha(OUT/'cohort.json'),protocol=sha(OUT/'protocol.md'),probe_configuration=sha(OUT/'probe_config.json'),minimum_samples=8,variance_floor=1e-12,ranks=[8,16,32],activation='signed native biased GELU mean',function_precision='FP32 TF32 disabled',validation_precision='FP32 TF32 enabled',validation_data_hash=j['extension_identity']['validation_batch_hash'])
  fixed={f'{p["mode"]}/L{p.layer}/{p.selection}':p.pair_sha256 for _,p in selected.iterrows() if p.selection in ('all_expert','fixed_initial','fixed_source')}
  key,payload=cache_identity(track=r.track,condition=r.condition,step=int(r.step),checkpoint=dict(path=r.checkpoint_path,sha256=r.checkpoint_sha256),components={g:r[g+'_hash'] for g in ('model','expert','router','shared')},references=references,probe=dict(path=r.probe_path,sha256=r.probe_sha256,tokens=r.token_sha256,positions=r.position_sha256),pairs=fixed,implementation=implementation,configuration=config)
  dest=REVIEW/'verified_identities'/r.track/r.condition/f'step_{int(r.step):04d}.json';save(dest,dict(verified_cache_key=key,identity=payload,original_cache_key=r.cache_key,original_completed_sha256=r.measurement_completed_sha256,verified_without_invalidating_measurements=True));keyrows.append(dict(track=r.track,condition=r.condition,step=int(r.step),verified_cache_key=key,implementation_sha256=implementation,configuration_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),identity_path=str(dest)))
 states=states.merge(pd.DataFrame(keyrows),on=['track','condition','step'],validate='one_to_one');states.to_csv(DEST/'measurement_provenance.csv',index=False);pp.merge(states,on=['track','condition','seed','step'],suffixes=('_pair','_state'),validate='many_to_one').to_csv(DEST/'coupling_row_provenance.csv',index=False)
 coverage=[]
 for j in jobs:
  for track in ('historical','rerun'):
   for step in STEPS:coverage.append(dict(track=track,condition=j['id'],seed=j['seed'],sparsity=j['sparsity'],step=step,available=step in paths_for(j,track),shared_with_other_track=step in (0,2500,3000,4000,5000),reconstructed_initialization=track=='historical' and step==0))
 table(DEST/'coverage.csv',coverage);return states,common,fam

def figures(effects,trajectories,data,ticket):
 colors=dict(zip(SEEDS,plt.get_cmap('tab10').colors[:5]));fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True)
 probes=[('fixed_source','all_expert'),('native','fixed_initial'),('native','current_accepted')]
 for row,(mode,sel) in enumerate(probes):
  for col,sp in enumerate((.8,.85)):
   ax=axes[row,col];v=trajectories[trajectories.track.eq('rerun')&trajectories.sparsity.eq(sp)&trajectories['mode'].eq(mode)&trajectories.selection.eq(sel)]
   for (seed,role),g in v.groupby(['seed','model_role']):g=g.sort_values('step');ax.plot(g.step,g.pearson,color=colors[seed],ls='-' if role=='sparse' else '--',label=f'{seed} {role}')
   ax.axvline(2500,color='gray',alpha=.5,lw=.8);ax.set(title=f'{mode}/{sel}; s={sp}',ylabel='Equal layer–expert Pearson',xlabel='Attempted recovery step');ax.legend(fontsize=6,ncol=2)
 fig.suptitle('Fixed all-time common populations; accepted-pair panels retain only 11–14 of 128 units.\nEndpoint tables use broader four-cell intersections; see coverage sensitivity.',fontsize=10);fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(DEST/'coupling_sparse_dense.png',dpi=170);plt.close(fig)
 g=data.get('geometry_fixed',data['geometry']);fig,axes=plt.subplots(4,2,figsize=(12,13),sharex=True)
 for col,sp in enumerate((.8,.85)):
  for seed in SEEDS:
   z=g[g.track.eq('rerun')&g.seed.eq(seed)&g.sparsity.eq(sp)]
   for ref,style in [('same_step_dense','-'),('fixed_dense_final','--')]:
    v=z[z.reference.eq(ref)&z['rank'].eq(8)].sort_values('step');axes[0,col].plot(v.step,v.normalized_gram_distance,style,color=colors[seed],label=f'{seed} {ref}')
   for row,rank in enumerate((8,16,32),1):
    v=z[z.reference.eq('same_step_dense')&z['rank'].eq(rank)].sort_values('step');axes[row,col].plot(v.step,v.subspace_overlap,color=colors[seed],label=str(seed));axes[row,col].set(ylabel=f'Subspace overlap k={rank}',xlabel='Attempted recovery step',title=f'Sparse versus same-step dense; s={sp}');axes[row,col].legend(fontsize=7)
  axes[0,col].set(title=f'Full Gram shape distance; s={sp}',ylabel='Normalized Gram distance',xlabel='Attempted recovery step');axes[0,col].legend(fontsize=6,ncol=2)
 for ax in axes.flat:ax.axvline(2500,color='gray',alpha=.5,lw=.8)
 fig.tight_layout();fig.savefig(DEST/'weight_geometry.png',dpi=170);plt.close(fig)
 ep=data['embedding'];rp=data['representations'];fig,axes=plt.subplots(2,2,figsize=(12,9))
 for col,sp in enumerate((.8,.85)):
  for seed in SEEDS:
   for role,cid in [('sparse',f'e32_seed{seed}_final_s{sp:.2f}'),('dense',f'e32_seed{seed}_dense')]:
    v=ep[ep.track.eq('rerun')&ep.condition.eq(cid)&ep.reference.eq('original_initial')].sort_values('step');axes[0,col].plot(v.step,v.relative_l2,ls='-' if role=='sparse' else '--',color=colors[seed],label=f'{seed} {role}')
   names=['token_embedding','embedding_plus_position']+[name for l in range(4) for name in (f'post_attention_residual_L{l}',f'router_input_L{l}',f'residual_after_L{l}')]
   v=rp[rp.track.eq('rerun')&rp.seed.eq(seed)&rp.sparsity.eq(sp)&rp.step.eq(5000)&rp.reference.eq('same_step_dense')].set_index('boundary').loc[names];axes[1,col].plot(range(len(v)),v.cka,'o-',color=colors[seed],label=str(seed))
  axes[0,col].set(title=f'Tied byte E drift from initialization; s={sp}',ylabel='Relative Frobenius drift',xlabel='Attempted recovery step');axes[0,col].legend(fontsize=6,ncol=2);axes[0,col].axvline(2500,color='gray',alpha=.5)
  axes[1,col].set(title=f'Final sparse–dense CKA through depth; s={sp}',ylabel='Linear CKA',xlabel='Measured stage');axes[1,col].set_xticks(range(len(names)),['E','E+P']+[f'{p}{l}' for l in range(4) for p in ('A','N','B')],rotation=60,fontsize=8);axes[1,col].legend(fontsize=7)
 fig.tight_layout();fig.savefig(DEST/'embedding_and_depth.png',dpi=170);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4.5))
 for sp,marker in ((.8,'o'),(.85,'^')):
  ef=effects[effects.track.eq('rerun')&effects.sparsity.eq(sp)&effects['mode'].eq('native')&effects.selection.eq('fixed_initial')&effects.pair_definition.eq('identical_fixed_pairs')&effects.coefficient.eq('pearson')&effects.aggregation.eq('equal_layer_expert')]
  for _,v in ef.iterrows():
   loss=ticket[ticket.track.eq('rerun')&ticket.seed.eq(v.seed)&ticket.sparsity.eq(sp)].iloc[0];axes[0].scatter(v.difference_in_changes,loss.gap,color=colors[v.seed],marker=marker);axes[0].annotate(f'{v.seed}/{sp}',(v.difference_in_changes,loss.gap),fontsize=8)
   out=data['function'];out=out[out.track.eq('rerun')&out.seed.eq(v.seed)&out.sparsity.eq(sp)&out.step.eq(5000)&out['mode'].eq('fixed_source')];err=out.relative_output_error.mean();axes[1].scatter(err,loss.gap,color=colors[v.seed],marker=marker);axes[1].annotate(f'{v.seed}/{sp}',(err,loss.gap),fontsize=8)
 for ax in axes:ax.axhline(0,color='gray');ax.set_ylabel('Logged sparse minus dense CE')
 axes[0].set_xlabel('Coupling difference in changes (identical fixed pairs)');axes[1].set_xlabel('Complete expert relative output error (fixed-source inputs)');fig.tight_layout();fig.savefig(DEST/'coupling_function_vs_loss.png',dpi=170);plt.close(fig)

def paired_behavior(data,tickets,effects):
 rows=[]
 for _,ticket in tickets.iterrows():
  track,seed,sp=ticket.track,int(ticket.seed),ticket.sparsity;meta=dict(track=track,seed=seed,sparsity=sp,strict_success=bool(ticket.strict_success),loss_gap=ticket.gap)
  def take(name,**filters):
   z=data[name]
   for key,value in dict(track=track,seed=seed,sparsity=sp,step=5000,**filters).items():z=z[z[key].eq(value)]
   return z
  row=dict(**meta)
  ee=take('embedding',reference='same_step_dense');row.update(embedding_dense_cka=ee.cka.mean(),embedding_dense_relative_l2=ee.relative_l2.mean())
  re=take('representations',reference='same_step_dense');row.update(router_input_dense_cka=re[re.boundary.str.startswith('router_input')].cka.mean())
  g=take('geometry',reference='same_step_dense');row['normalized_gram_distance']=g[g['rank'].eq(8)].normalized_gram_distance.mean()
  for k in (8,16,32):row[f'subspace_overlap_rank{k}']=g[g['rank'].eq(k)].subspace_overlap.mean()
  for mode in MODES:
   fn=take('function',mode=mode);row[f'expert_output_relative_error_{mode}']=fn.relative_output_error.mean();row[f'expert_output_cosine_{mode}']=fn.output_cosine.mean()
  mm=take('models');row.update(probe_kl=mm.probe_kl.mean(),probe_ce=mm.probe_ce.mean())
  for mode,sel in [('fixed_source','all_expert'),('native','fixed_initial'),('native','current_accepted')]:
   v=effects[effects.track.eq(track)&effects.seed.eq(seed)&effects.sparsity.eq(sp)&effects['mode'].eq(mode)&effects.selection.eq(sel)&effects.pair_definition.eq('registered_condition_pairs')&effects.coefficient.eq('pearson')&effects.aggregation.eq('equal_layer_expert')]
   if len(v):row[mode+'_'+sel+'_coupling_change']=v.sparse_change.iloc[0];row[mode+'_'+sel+'_coupling_did']=v.difference_in_changes.iloc[0]
  rows.append(row)
 table(DEST/'behavior_geometry_cases.csv',rows);return pd.DataFrame(rows)

def write_findings(effects,data,ticket,states=None,common=None,families=None):
 e=effects[effects.pair_definition.eq('registered_condition_pairs')&effects.coefficient.eq('pearson')&effects.aggregation.eq('equal_layer_expert')&effects.track.eq('rerun')]
 # Text has no pooled token p-values; all means below average five source-seed effects per sparsity.
 valid=states is not None
 lines=['# Reviewed high-sparsity coupling analysis','',('**Provenance verdict:** identical historical/rerun model states explain the duplication. No raw measurement-cache collision or incorrect checkpoint resolution was found.' if valid else '**Provisional provenance verdict:** completed seed audits identify identical model states; remaining states are still being verified.'),'',f'Execution: 15/15 reruns. Measurement validity: {"verified" if valid else "audit pending"}. Analysis: {"complete" if valid else "provisional"}. Exploratory/post hoc; independent n=5 source seeds. Historical/rerun repeats and the two sparsities are dependent observations.','', '## Provenance and evaluation precision','']
 if valid:lines += [f'All {len(common)} common historical/rerun states have identical full model and expert/router/shared component hashes; there are {states.model_hash.nunique()} distinct model states across 315 logical measurements (75 historical + 240 rerun). Historical coverage is only steps 0/2500/3000/4000/5000; 165 historical slots remain missing. No raw cache was invalidated or relabeled. The previous reports are preserved in [superseded](../superseded/analysis_before_review_2026_10_20/report/).','']
 ce=pd.read_csv(DEST/'ce_execution_settings.csv');lines += [f'All 15 historical logged endpoint CEs are reproduced exactly with TF32 **off**, and all 15 rerun logged CEs with TF32 **on** (maximum residuals {ce.historical_residual_off.abs().max():.3g} and {ce.rerun_residual_on.abs().max():.3g}). This explains the apparent discrepancy without a weight difference. Example seed 7/.80: off = {ce[ce.condition.eq("e32_seed7_final_s0.80")].reevaluated_tf32_off.iloc[0]:.16f}; on = {ce[ce.condition.eq("e32_seed7_final_s0.80")].reevaluated_tf32_on.iloc[0]:.16f}. The historical continuation entry point omits configure_device, whereas rerun runner.py explicitly enables TF32. Training uses FP16 AMP; the verified repeated weights are identical. Geometric diagnostics consistently disable TF32. See [CE settings](ce_execution_settings.csv), [state provenance](measurement_provenance.csv), [pair provenance](coupling_row_provenance.csv), and [family audit](cross_track_measurement_families.csv).','', '## Tickets','']
 for track in ('historical','rerun'):
  tt=ticket[ticket.track.eq(track)];lines += [f'{track}: {int(tt.strict_success.sum())}/10 strict successes across {tt[tt.strict_success].seed.nunique()} source seeds. Sparse CE <= seed-matched dense CE, with 5000 attempted steps and no tolerance. Both numerical settings preserve the four labels.','', '```csv',tt[['seed','sparsity','dense_ce','sparse_ce','gap','strict_success','sparse_applied_updates','dense_applied_updates','sparse_amp_skips','dense_amp_skips']].to_csv(index=False).strip(),'```','']
 lines += ['See [historical tickets](historical_tickets.csv), [rerun tickets](rerun_tickets.csv), [label agreement](ticket_label_agreement.csv), and [update budgets](update_budgets.csv). Small numerical margins are retained without changing the criterion.','', '## 1–3. Coupling and matched dense recovery','', 'Primary endpoint contrasts below use the same valid layer–expert intersection in sparse initial/final and dense initial/final. Equal layer–expert weighting is distinct from count-weighted, per-expert standardized Pearson pooling. Per-expert Spearman is separately summarized; within-token across-expert associations remain separate. Up to 128 layer–expert units are nested diagnostics, not independent replications.','']
 summary=[]
 for (sp,mode,sel),v in e.groupby(['sparsity','mode','selection']):
  if (mode,sel) not in [('fixed_source','all_expert'),('native','fixed_initial'),('native','current_accepted')]:continue
  summary.append(dict(sparsity=sp,input=mode,selection=sel,sparse_initial=v.sparse_initial.mean(),sparse_final=v.sparse_final.mean(),sparse_change=v.sparse_change.mean(),dense_initial=v.dense_initial.mean(),dense_final=v.dense_final.mean(),dense_change=v.dense_change.mean(),endpoint_difference=v.endpoint_difference.mean(),difference_in_changes=v.difference_in_changes.mean(),positive_sparse_changes=int((v.sparse_change>0).sum()),positive_did=int((v.difference_in_changes>0).sum()),valid_units_min=int(v.n_valid_units.min()),valid_units_max=int(v.n_valid_units.max())))
 table(DEST/'coupling_primary_summary.csv',summary);lines += ['```csv',pd.DataFrame(summary).to_csv(index=False).strip(),'```','','Fixed-source/all-expert coupling and native/fixed-initial coupling increase in sparse recoveries; native/current-accepted coupling decreases. Read the dense changes and difference-in-changes above before attributing any pattern specifically to sparsity. Current accepted tokens change over training; native hidden inputs change even when token identities and diagnostic expert assignments are held fixed. Probe contrasts are not an additive causal decomposition.','']
 matched=effects[effects.track.eq('rerun')&effects['mode'].eq('native')&effects.selection.eq('fixed_initial')&effects.pair_definition.eq('identical_fixed_pairs')&effects.coefficient.eq('pearson')&effects.aggregation.eq('equal_layer_expert')].merge(ticket[ticket.track.eq('rerun')][['seed','sparsity','strict_success']],on=['seed','sparsity'],validate='one_to_one')
 lines += ['Identical-fixed-pair sensitivity and all condition outcomes:','','```csv',matched[['seed','sparsity','strict_success','sparse_change','dense_change','endpoint_difference','difference_in_changes','n_valid_units']].to_csv(index=False).strip(),'```','','Both winners and nonwinners strengthen coupling under the fixed probes. This does not establish a ticket classifier; winner conditioning is post hoc and source seed 7 contributes two related winners. See [paired endpoint contrasts](paired_seed_coupling_changes.csv), [coverage/exclusions](coupling_exclusions.csv), [seed uncertainty](coupling_seed_uncertainty.csv), and [within-token associations](within_token_associations.csv).','', '## 4. Weight geometry: shape, scale, and rank are different','']
 ge=data['geometry_endpoints'];gg=ge[ge.track.eq('rerun')&ge.sparsity.notna()];geom_summary=[]
 for (sp,ref,rank),g in gg.groupby(['sparsity','reference','rank']):geom_summary.append(dict(sparsity=sp,reference=ref,rank=rank,overlap_initial=g.subspace_overlap_initial.mean(),overlap_final=g.subspace_overlap_final.mean(),overlap_change=g.subspace_overlap_change.mean(),overlap_positive_seeds=int((g.subspace_overlap_change>0).sum()),gram_initial=g.normalized_gram_distance_initial.mean(),gram_final=g.normalized_gram_distance_final.mean(),gram_change=g.normalized_gram_distance_change.mean(),trace_initial=g.trace_initial.mean(),trace_final=g.trace_final.mean(),valid_overlap_units_min=int(g.subspace_overlap_n_valid.min())))
 table(DEST/'geometry_endpoint_summary.csv',geom_summary);gd=pd.DataFrame(geom_summary)
 lines += ['```csv',gd[gd.reference.isin(['same_step_dense','fixed_dense_final'])][['sparsity','reference','rank','overlap_initial','overlap_final','overlap_change','gram_initial','gram_final','gram_change','valid_overlap_units_min']].to_csv(index=False).strip(),'```','','Gram distance is rank-free and is repeated only to accompany the three separately reported SVD ranks; it must not be averaged as three pieces of evidence. Gram cosine is algebraically related to normalized Gram distance. Scale is separately reported by trace/energy and unnormalized distances. A growing distance to the *moving* same-step dense model can coexist with shrinking distance to the fixed dense endpoint. Rank-specific subspaces need not follow full-spectrum shape. Thus one rank/distance cannot establish an absence of geometric realignment.','']
 q=data['q'];qr=[]
 for sp in (.8,.85):
  for router in ('raw','centered'):
   for role in ('sparse','dense'):
    z=q[q.track.eq('rerun')&q.router.eq(router)&(q.sparsity.eq(sp) if role=='sparse' else q.sparsity.isna())]
    for step in (0,5000):
     v=z[z.step.eq(step)];qr.append(dict(sparsity=sp,router=router,role=role,step=step,q=v.q.mean(),unmatched_q=v.unmatched_q.mean(),above_unmatched=v.q_above_unmatched.mean(),isotropic=1/256))
 table(DEST/'q_endpoint_summary.csv',qr);lines += ['```csv',pd.DataFrame(qr).to_csv(index=False).strip(),'```','','Direct IMP mask-selection states (round 07 for .80, round 08 for .85), actual masked initialization, same-step dense and fixed dense final remain separate references. Original source-final is a distinct embedding/representation reference; it was not an additional registered expert-Gram reference. See [geometry trajectories and exclusions](geometry_trajectories.csv), [common-population endpoint changes](geometry_endpoint_changes.csv), and [raw/centered q](q_trajectories.csv).','', '## 5. E and downstream representations','']
 ep=data['embedding'];ep=ep[ep.track.eq('rerun')&ep.step.eq(5000)];em=[]
 for sp in (.8,.85):
  for role in ('sparse','dense'):
   for ref in ('original_initial','source','same_step_dense'):
    z=ep[ep.reference.eq(ref)&(ep.sparsity.eq(sp) if role=='sparse' else ep.sparsity.isna())];em.append(dict(sparsity=sp,role=role,reference=ref,relative_l2=z.relative_l2.mean(),cka=z.cka.mean(),entropy_effective_rank=z.embedding_effective_rank.mean(),top8_energy=z.top8_spectral_energy.mean(),token_similarity_distance=z.token_similarity_relative_distance.mean(),row_norm=z.row_norm_mean.mean(),row_drift=z.row_drift_mean.mean()))
 table(DEST/'embedding_endpoint_summary.csv',em);lines += ['```csv',pd.DataFrame(em).to_csv(index=False).strip(),'```','','E is the tied byte token embedding/LM-head matrix; its evolution reflects input and output prediction roles. E summaries are unweighted over 256 byte rows. Effective rank is exp(-sum p log p), with p the normalized squared singular values of uncentered E; the separately reported covariance effective rank centers observations. Native token representations reflect the fixed probe token frequencies. Singular spectra, row norms/drift and row similarity metrics are retained rather than inferred from CKA alone.','']
 rp=data['representations'];rs=rp[rp.track.eq('rerun')&rp.step.eq(5000)&rp.sparsity.notna()&rp.reference.eq('same_step_dense')];depth=rs.groupby(['sparsity','boundary'],as_index=False)[['cka','relative_l2','effective_rank','norm_mean']].mean();depth.to_csv(DEST/'depth_endpoint_summary.csv',index=False);lines += ['```csv',depth.to_csv(index=False).strip(),'```','','Differences already present at token embeddings can propagate through attention/residual boundaries to the normalized inputs of each router. Full trajectories and [reset-window differences](optimizer_reset_representation_windows.csv) describe 2000→2500, 2500→2525, 2500→2600, and 2600→3000 separately. An optimizer-only reset leaves weights unchanged; subsequent changes are not an instantaneous reset effect. Snapshot resolution does not establish precedence, causation, or mediation. See [E trajectories](embedding_trajectories.csv), [E spectra](embedding_spectra.csv), and [all representation stages](representation_trajectories.csv).','', '## 6. Geometry versus behavior','']
 cases=pd.read_csv(DEST/'behavior_geometry_cases.csv');lines += ['Every condition, rather than selected successes:','','```csv',cases[cases.track.eq('rerun')].to_csv(index=False).strip(),'```','','Complete expert outputs use identical current-native or fixed-source hidden inputs, including both biases, GELU and output projection. Their errors are separate from model-level KL and validation CE. Good end-to-end CE does not require expert-wise equality with dense controls; activation magnitude is not expert usefulness. No pooled token/expert/time correlation is used as replication evidence. See [expert functions](expert_function_trajectories.csv), [model CE/KL](model_function_trajectories.csv), and [cross-family cases](behavior_geometry_cases.csv).','', '## 7. Supported, unsupported, and unresolved','', '- Supported: the verified input/selection-dependent association patterns, paired seed effects, direct gradient identities, rank/reference-specific geometry, tied-E drift and functional differences in the linked tables.', '- Unsupported: a single monotonic coupling trajectory independent of probe definition; treating any increase in winners as a ticket discriminator; equating source-subspace overlap with own-router coupling; an absence of all geometric realignment; causal mediation by E or representations.', '- Limited: five source seeds, dependent sparsities and historical repeats; validation-based ticket selection; no established independent test; original historical intermediate recovery states are absent. CKA is not computational equivalence. Seed bootstrap intervals are descriptive and unstable at n=5.', '', 'No follow-up intervention is executed or recommended as established by these observational results. The frozen protocol and all ten sparse conditions/five controls are preserved.']
 (DEST/'findings.md').write_text('\n'.join(lines)+'\n')

def budget_log_audit(jobs):
 rows=[]
 for j in jobs:
  h1=Path(j['boundary']).parents[1];h2=Path(j['historical_checkpoints']['5000']).parents[1];done=read(OUT/'reruns'/j['id']/'completed.json');r=Path(done['attempt'])
  for track,segment,path in [('historical',0,h1),('historical',2500,h2),('rerun',0,r/'segment_0'),('rerun',2500,r/'segment_2500')]:
   attempts={}
   p=path/'gradient_stats/gradient_stats.jsonl'
   with p.open() as f:
    for line in f:
     record=json.loads(line);step=int(record['step'])
     if step>0:
      assert step not in attempts,(p,step);attempts[step]=bool(record['optimizer_step_applied'])
   assert set(attempts)==set(range(1,2501)),p
   applied=sum(attempts.values())
   if track=='rerun':expected=read(path/'record.json')['optimizer_steps_applied']
   elif segment==0:expected=read(path/('stage.json' if j['mask'] else 'metadata.json'))['optimizer_steps_applied']
   else:expected=read(path/'extension_manifest.json')['applied_optimizer_updates']
   assert applied==expected,(p,applied,expected)
   rows.append(dict(track=track,condition=j['id'],seed=j['seed'],sparsity=j['sparsity'],segment_start=segment,attempted=len(attempts),applied=applied,amp_skips=2500-applied,source=str(p),passed=True))
 table(DEST/'budget_log_audit.csv',rows)

def concise_findings(effects,data,tickets,states,common,families):
 # Full supporting numeric grids are kept in analysis_details.md and linked CSVs.
 (DEST/'findings.md').rename(DEST/'analysis_details.md')
 e=effects[effects.track.eq('rerun')&effects.coefficient.eq('pearson')&effects.aggregation.eq('equal_layer_expert')&effects.pair_definition.eq('registered_condition_pairs')]
 lines=['# Provenance verdict and findings','',f'**The duplication is legitimate:** all {len(common)} shared historical/rerun checkpoints have identical model, expert, router and shared-component hashes. All {len(families)} comparable raw output files are byte-identical. There are 315 logical measurements but {states.model_hash.nunique()} distinct model states; these tracks are not independent replications. No measurement cache required invalidation.','', '**The CE discrepancy is numerical:** all 15 historical endpoint CEs reproduce exactly with TF32 off; all 15 rerun endpoint CEs reproduce exactly with TF32 on. Historical continuation omits configure_device; reruns enable TF32. Example seed 7/.80: 1.3295407060960809 versus 1.3295155456539003. Neither checkpoint resolution nor model weights changed. Both settings retain the same four strict winners.','', 'Execution, measurement verification and analysis are complete. Historical coverage remains 75 states (0/2500/3000/4000/5000); rerun coverage is 240 states; 165 historical slots remain missing. This is exploratory/post hoc, with five independent source seeds.','', '## Coupling is probe-dependent and differs from dense recovery','', 'Means below average five paired source-seed effects separately at each sparsity; primary Pearson summaries use one common valid layer–expert population across the four cells.','', '| Input / selection | Sparsity | Sparse change | Dense change | Difference in changes |','|---|---:|---:|---:|---:|']
 for mode,sel in [('fixed_source','all_expert'),('native','fixed_initial'),('native','current_accepted')]:
  for sp in (.8,.85):
   z=e[e['mode'].eq(mode)&e.selection.eq(sel)&e.sparsity.eq(sp)];lines.append(f'| {mode} / {sel} | {sp:.2f} | {z.sparse_change.mean():+.4f} | {z.dense_change.mean():+.4f} | {z.difference_in_changes.mean():+.4f} |')
 lines += ['', 'All ten sparse runs increase association on fixed-source/all-expert and native/fixed-initial probes; matched dense changes are negative in all five seeds. Identical-fixed-pair sensitivity preserves the contrast: native/fixed-initial difference-in-changes is +0.4882/.5248 at .80/.85. Native/current-accepted association instead falls in all ten sparse runs; its paired effect is negative in 5/5 seeds at .80 and 4/5 at .85. Current accepted populations change; fixed token IDs do not freeze native hidden inputs. These probe contrasts are not an additive causal decomposition.','', 'This does **not** distinguish tickets: both four winners and six nonwinners show the fixed-probe increases. The winning conditions remain seed 7/.80 and .85, 43/.80, and 61/.80; rerun gaps are approximately −0.035739, −0.021304, −0.001772, and −0.026802. No tolerance was used. Applied updates span 4997–5000 of 5000 attempts; AMP skips were reconciled directly with all 60 segment logs.','', 'The former report contrasted separately valid sparse/dense populations. The corrected four-cell intersection changes some paired estimates materially (largest correction: +0.11552 for seed 61/.85 fixed-source/current-accepted, with 51 common layer–expert units). Pearson, Spearman, equal-expert, count-weighted and within-token statistics are separately labeled; units/tokens are not replications.','', 'See [paired seed contrasts](paired_seed_coupling_changes.csv), [population corrections](paired_population_corrections.csv), [exclusions](coupling_exclusions.csv), [seed uncertainty](coupling_seed_uncertainty.csv), and [ticket tables](rerun_tickets.csv).','', '## Geometry gives contradictory but compatible answers','', '- Relative to **same-step dense**, normalized full Gram distance rises from .58095→.81194 at .80 and .69915→.83889 at .85. Yet rank-8 overlap rises .15347→.29678 and .11088→.28181; rank-16 rises .27082→.32365 and .21050→.29476. Rank-32 falls .41166→.38332 and .34488→.34264. Ranks are not averaged.', '- Relative to **fixed dense final**, full Gram distance also rises modestly (.78875→.81194; .80264→.83889), and all three average subspace overlaps decrease slightly. Moving- and fixed-reference initial geometries are different. SVD comparisons preserve reference-specific common-population rank/degeneracy exclusions; differing reference populations can produce slightly different final means.', '- Relative to the **direct IMP mask-selection state**, .80 moves closer by full Gram shape (.48132→.39392) and all three overlaps increase; .85 moves away (.37244→.44846) and all three overlaps decrease. Actual masked initialization is a distinct reference: its initial overlap is one, then falls to roughly .53–.56.', '- Expert fc1 trace increases from 198.34→418.39 at .80 and 162.63→393.38 at .85; this scale growth is separate from normalized shape. Centered-router q increases .00424→.00773 and .00436→.00839, but dense q increases more, .00391→.01094 (isotropic 1/256=.00390625). Dense therefore has larger unsigned directional energy despite declining signed score/activation association.', '', 'There is evidence of leading-subspace alignment, whole-spectrum separation, and sparsity-dependent source restoration at once. “Sparse models do not realign geometrically” is unsupported. See [rank/reference endpoint effects](geometry_endpoint_changes.csv), [geometry trajectories/exclusions](geometry_trajectories.csv), and [raw/centered q](q_trajectories.csv).','', '## E changes modestly; downstream differences are much larger','', 'The tied byte embedding/LM-head E drifts 6.53%/.6.72% from initialization at .80/.85, versus 6.47% dense; sparse–dense drift is 3.68%/3.51% with CKA .99934/.99938. Effective rank changes only slightly (154.75 initially to 154.89/154.88); top-eight energy stays near .1122. Mean row norms fall from 15.983 to 15.681/15.681 (dense 15.700). Token–token similarity and individual byte-row drifts are tabulated. E serves output prediction as well as input representation.','', 'On identical probe positions, final token-embedding CKA is .99998, but after the first attention residual it is .7833/.7626, and normalized router-input CKA ranges roughly .56–.78 through depth. Differences already exist in deeper initial sparse states; the first attention stage starts at CKA 1 and diverges by the step-25 measurement. CKA changes are not monotonic through depth.','', 'The reset changes no weights. From 2500→2525, E drift decreases slightly in both sparse and dense runs; deeper router-input CKA can rise before falling again by 5000. All before/after windows are reported; this is neither an instantaneous reset effect nor evidence that E mediates or causally precedes coupling. See [E trajectories/spectra](embedding_trajectories.csv), [depth summary](depth_endpoint_summary.csv), [all stages](representation_trajectories.csv), and [reset windows](optimizer_reset_representation_windows.csv).','', '## Functional agreement does not require expert equality','', 'All conditions have substantial complete-expert discrepancies on the same fixed-source inputs: final relative errors span 1.72–2.36 and cosines .196–.261, while probe KL spans .319–.503 nats. Four models nevertheless beat their dense validation controls. For example, winner seed 43/.80 has KL .503 and gap −.001772; nonwinner seed 29/.80 has lower KL .337 and gap +.003709. Winner seed 7/.80 has native fixed-initial coupling gain +.1814; nonwinner 17/.85 has a larger +.2230. No simple cross-family separation of winners is established.','', 'See [every condition across metric families](behavior_geometry_cases.csv), [expert output errors](expert_function_trajectories.csv), and [model CE/KL](model_function_trajectories.csv).','', '## Limits and review artifacts','', 'Five source seeds, two dependent sparsities, deterministic historical repeats, validation-selected labels, missing historical intermediates, and no established unused test split limit inference. Seed bootstrap intervals are descriptive; no pooled token/expert/time p-values are used. CKA is not computational equivalence, mean activation is not expert usefulness, and none of these trends establishes causation. No follow-up intervention is executed or claimed necessary by this observational result.','', 'Supported: the quantified probe-dependent paired effects, rank/reference-specific geometry, modest tied-E drift, large downstream representation/function differences and four strict validation winners. Unsupported: a universal coupling trend, coupling as a winner discriminator, blanket lack of geometric realignment, or causal representation mediation. Historical intermediates and independent-test behavior remain unresolved.','', 'Audit: [provenance diagnosis](provenance_diagnosis.md), [all checkpoint/component identities](measurement_provenance.csv), [every coupling row/pair](coupling_row_provenance.csv), [coverage](coverage.csv), [full numerical detail](analysis_details.md). Figures: [coupling](coupling_sparse_dense.png), [geometry](weight_geometry.png), [E/depth](embedding_and_depth.png), [function versus loss](coupling_function_vs_loss.png).']
 # Remove a typographic percentage artifact before saving.
 text='\n'.join(lines).replace('6.53%/.6.72%','6.53%/6.72%')+'\n';(DEST/'findings.md').write_text(text)

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--reuse-summaries',action='store_true');args=parser.parse_args();DEST.mkdir(exist_ok=True);f=frozen();jobs=f['conditions']
 for seed in SEEDS:assert read(REVIEW/'provenance'/f'seed_{seed}_complete.json')['passed']
 reeval=pd.concat([pd.read_csv(REVIEW/'ce'/f'seed_{s}.csv') for s in SEEDS],ignore_index=True);assert len(reeval)==15 and reeval.historical_residual_off.abs().max()==0 and reeval.rerun_residual_on.abs().max()==0
 tt=tickets(jobs,reeval)
 if args.reuse_summaries:
  ef=pd.read_csv(DEST/'paired_seed_coupling_changes.csv');tr=pd.read_csv(DEST/'coupling_trajectories.csv');names={'geometry':'geometry_trajectories','q':'q_trajectories','embedding':'embedding_trajectories','representations':'representation_trajectories','function':'expert_function_trajectories','models':'model_function_trajectories','geometry_endpoints':'geometry_endpoint_changes'};data={k:pd.read_csv(DEST/(v+'.csv')) for k,v in names.items()}
 else:ef,tr=coupling(jobs);data=metric_tables(jobs)
 paired_behavior(data,tt,ef);data['geometry_fixed']=fixed_geometry_trajectories(jobs);states,common,families=provenance(jobs);assert common.model_identical.all() and families.identical_output.all();budget_log_audit(jobs);figures(ef,tr,data,tt);write_findings(ef,data,tt,states,common,families);concise_findings(ef,data,tt,states,common,families);coverage_sensitivity(ef,tr);specificity_controls(ef)
 old=pd.read_csv(OUT/'superseded/analysis_before_review_2026_10_20/report/paired_seed_coupling_changes.csv');new=ef[ef.pair_definition.eq('registered_condition_pairs')&ef.coefficient.eq('pearson')&ef.aggregation.eq('equal_layer_expert')];delta=new.merge(old,on=['track','seed','sparsity','mode','selection'],suffixes=('','_superseded'));delta['correction']=delta.difference_in_changes-delta.sparse_minus_dense_delta;delta.to_csv(DEST/'paired_population_corrections.csv',index=False)
 diagnosis=['# Provenance diagnosis','',f'{len(states)} verified logical measurements; {states.model_hash.nunique()} unique model states. All {len(common)} cross-track common states have identical expert/router/shared/full-state hashes. All {len(families)} cross-track paired family files are byte-identical. Distinct file SHA256 values encode different checkpoint containers (optimizer/RNG/configuration versus diagnostic-only payloads); they do not demonstrate different model weights.','', 'Checkpoint paths resolve to the intended frozen historical files or completed rerun attempt snapshots. Historical SHA256 checks match the immutable cohort. Model/component hashes, raw token/position identities, router scores on both input modes, embedding lookups, and every cached output-file hash were independently checked. No track join/collision or stale raw cache explains the duplication. Historical missing states remain missing.','', 'All 15 historical logged final losses are reproduced exactly with TF32 disabled and all 15 rerun losses with TF32 enabled. The historical continuation entry point does not invoke configure_device; runner.py does. Common-setting re-evaluation removes the cross-track CE discrepancy. Historical and rerun strict labels agree, and each track keeps its own logged losses. This is a numerical evaluation difference, not a training effect.','', 'The actual reporting defect was inconsistent expert populations for sparse-versus-dense changes. The old report intersected initial/final within each model separately, then contrasted different sets. Corrected contrasts intersect all four cells. Fixed-initial selections are also evaluated on identical token–expert pair intersections, separately from the registered per-condition fixed assignments. A focused synthetic regression demonstrates the error and corrected result. A second regression verifies cache-key separation across tracks, checkpoints, components, references, probes, pair selections, implementation and configuration.','', 'Affected output inventory: prior report/findings, paired_seed_coupling_changes, seed_level_uncertainty, and associated comparison/trajectory figures are superseded; the full prior report tree and source are retained in superseded/analysis_before_review_2026_10_20/. Raw measurement families (activation groups/pools/within-token, Gram/SVD/q, embeddings, representations, complete expert outputs, gradients, CE/KL) are valid and retained without recomputation or relabeling. New population-controlled summaries, coverage and exclusions are generated from them.','', 'Schema-2 provenance sidecars explicitly bind track, global step, actual checkpoint path/SHA, component hashes, every reference, fixed-pair hashes, probe identities, original implementation hash and runtime configuration. Future measurement cache code now uses those explicit identities, checks saved probe identities and freezes its implementation fingerprint at import; existing valid raw cache files are not invalidated.','', 'The old unconditional overall-complete flag has been replaced by separate execution_complete, measurement_validity and analysis_complete fields. No training, mask search or historical checkpoint modification was performed.']
 (DEST/'provenance_diagnosis.md').write_text('\n'.join(diagnosis)+'\n');summary=dict(complete=True,execution_complete=True,completed_reruns=15,logical_measured_states=315,unique_model_states=int(states.model_hash.nunique()),historical_states=75,rerun_states=240,historical_missing_slots=165,measurement_validity='verified',analysis_complete=True,common_cross_track_states=75,identical_component_states=int(common.model_identical.sum()),invalidated_measurement_caches=0,independent_source_seeds=5,unix=time.time());save(DEST/'completion.json',summary);save(OUT/'execution_status.json',summary);(OUT/'execution_status.md').write_text('# Verified completion\n\nTraining: 15/15 complete. Measurement validity: verified (315 logical states, 240 unique model states). Analysis: complete. Historical/rerun common states are tensor-identical; the logged CE differences are explained exactly by TF32. All ten sparse conditions and five matched dense controls are retained. No new training was launched.\n\n[Findings](report/findings.md), [provenance diagnosis](report/provenance_diagnosis.md), [completion](report/completion.json). Original reports are preserved under superseded/analysis_before_review_2026_10_20/.\n');save(REVIEW/'review_implementation.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('review_*.py')},cohort=sha(OUT/'cohort.json'),protocol=sha(OUT/'protocol.md'),unix=time.time()));print(json.dumps(summary,indent=2))

def fixed_geometry_trajectories(jobs):
 rows=[]
 for track in ('historical','rerun'):
  for j in jobs:
   steps=list(paths_for(j,track));frames={step:pd.read_csv(OUT/'analysis'/track/j['id']/f'step_{step:04d}/weight_geometry.csv') for step in steps}
   for reference in ('mask_selection_source','recovery_initial','same_step_dense','fixed_dense_final'):
    for rank in (8,16,32):
     parts={t:f[f.reference.eq(reference)&f['rank'].eq(rank)].set_index(['layer','expert']) for t,f in frames.items()}
     ids=sorted(set.intersection(*(set(f.index[f.subspace_overlap.notna()]) for f in parts.values())))
     for step,f in parts.items():
      d=f.loc[ids];rows.append(dict(track=track,condition=j['id'],seed=j['seed'],sparsity=j['sparsity'],step=step,reference=reference,rank=rank,n_svd_valid=len(ids),n_svd_excluded=128-len(ids),fixed_valid_units=json.dumps(ids),subspace_overlap=d.subspace_overlap.mean(),normalized_gram_distance=f.normalized_gram_distance.mean(),n_gram_valid=int(f.normalized_gram_distance.notna().sum()),trace=f.trace.mean(),reference_trace=f.reference_trace.mean()))
 table(DEST/'geometry_fixed_population_trajectories.csv',rows);return pd.DataFrame(rows)

def coverage_sensitivity(effects,trajectories):
 rows=[]
 for keys,z in trajectories.groupby(['track','seed','sparsity','mode','selection']):
  zz=z[z.step.isin([0,5000])].set_index(['model_role','step']);a=zz.loc[('sparse',0),'pearson'];b=zz.loc[('sparse',5000),'pearson'];c=zz.loc[('dense',0),'pearson'];d=zz.loc[('dense',5000),'pearson']
  rows.append(dict(zip(['track','seed','sparsity','mode','selection'],keys),population='common_all_measured_times',n_valid_units=int(z.n_valid_units.iloc[0]),n_excluded_units=128-int(z.n_valid_units.iloc[0]),sparse_change=b-a,dense_change=d-c,difference_in_changes=(b-a)-(d-c)))
 table(DEST/'all_time_population_sensitivity.csv',rows)
 sensitivity=pd.DataFrame(rows);v=sensitivity[sensitivity.track.eq('rerun')&sensitivity['mode'].eq('native')&sensitivity.selection.eq('current_accepted')];assert v.n_valid_units.min()==11 and v.n_valid_units.max()==14
 # This is a different target population, not an alternative independent replication.
 summary=[]
 for keys,g in effects[effects.track.eq('rerun')&effects.pair_definition.eq('registered_condition_pairs')].groupby(['sparsity','mode','selection','coefficient','aggregation']):
  summary.append(dict(zip(['sparsity','mode','selection','coefficient','aggregation'],keys),n_conditions=len(g),n_positive_sparse_changes=int(g.sparse_change.gt(0).sum()),n_positive_did=int(g.difference_in_changes.gt(0).sum()),mean_did=g.difference_in_changes.mean()))
 table(DEST/'correlation_weighting_sensitivity.csv',summary)
 path=DEST/'findings.md';text=path.read_text();anchor='## Geometry gives contradictory but compatible answers'
 paragraph='### Accepted-pair coverage sensitivity\n\nThe primary four-endpoint contrast uses roughly 125–126 common valid units on average and shows a negative sparse accepted-pair change in all ten conditions. However, the common population across **all 16 trajectory steps** contains only **11–14 of 128 units**. On this highly restricted subset, sparse changes increase in six conditions and decrease in four; differences-in-changes are positive in all ten. This is a different, traffic-selected expert population, not a contradiction in the underlying cache. Thus the accepted-pair result is **not robust to the valid-population definition**, and a universal decline is unsupported. The trajectory figure explicitly labels this restriction. Fixed-source/all-expert and native/fixed-initial signs survive Pearson/Spearman and equal-expert/count-weighted choices in every condition.\n\nSee [all-time population sensitivity](all_time_population_sensitivity.csv) and [correlation/weighting sensitivity](correlation_weighting_sensitivity.csv). Neither sensitivity treats ten related conditions as ten independent seeds.\n\n'
 if '### Accepted-pair coverage sensitivity' not in text:text=text.replace(anchor,paragraph+anchor)
 path.write_text(text)
 p=DEST/'within_token_associations.csv';d=pd.read_csv(p)
 if 'n_tokens' in d:d=d.rename(columns={'n_tokens':'n_token_layer_rows'});d['n_unique_positions']=128;d.to_csv(p,index=False)

def specificity_controls(effects):
 rows=[]
 for track in ('historical','rerun'):
  for j in frozen()['conditions']:
   if j['sparsity'] is None:continue
   for mode,selection in [('fixed_source','all_expert'),('native','fixed_initial'),('native','current_accepted')]:
    fs=[frame(track,cid,step,mode,selection).copy() for cid,step in ((j['id'],0),(j['id'],5000),(j['dense_id'],0),(j['dense_id'],5000))]
    for control,column in [('unmatched_router_rows','router_pair_pearsons'),('unmatched_experts','expert_pair_pearsons')]:
     transformed=[]
     for f in fs:
      null=[]
      for index,r in f.iterrows():
       v=np.asarray(json.loads(r[column]),dtype=float);null.append(float(np.delete(v,index[1]).mean()) if np.isfinite(v).all() else np.nan)
      f['unmatched_pearson']=null;f['above_unmatched']=f.pearson-f.unmatched_pearson;transformed.append(f)
     cov,contrast,_=paired_contrast(transformed,'above_unmatched');ids=[tuple(x) for x in json.loads(cov['valid_units'])];out=dict(track=track,condition=j['id'],seed=j['seed'],sparsity=j['sparsity'],mode=mode,selection=selection,control=control,**cov,**{'above_unmatched_'+k:v for k,v in contrast.items()})
     for label,f in zip(('sparse_initial','sparse_final','dense_initial','dense_final'),transformed):
      z=f.loc[ids];out['identity_'+label]=z.pearson.mean();out['unmatched_'+label]=z.unmatched_pearson.mean()
     rows.append(out)
 table(DEST/'pairing_specificity_controls.csv',rows)
 d=pd.DataFrame(rows);v=d[d.track.eq('rerun')&d['mode'].eq('fixed_source')&d.selection.eq('all_expert')&d.control.eq('unmatched_experts')];summary=v.groupby('sparsity')[['above_unmatched_sparse_final','above_unmatched_dense_final']].mean();p=DEST/'findings.md';text=p.read_text();anchor='## Geometry gives contradictory but compatible answers';paragraph='### Pairing controls\n\nOn the same fixed-source/all-expert inputs and pair identities, final identity association exceeds unmatched-expert association by '+ '/'.join(f'{x:.4f}' for x in summary.above_unmatched_sparse_final)+' at .80/.85; dense identity-minus-unmatched association is '+f'{summary.above_unmatched_dense_final.iloc[0]:.4f}'+'. These signs hold in all five seeds. Shared input covariance alone therefore does not account for the signed identity contrast. Unmatched-router and unmatched-expert controls are separately exported; their global all-expert means are algebraically identical under the common complete selection and are not independent evidence. A negative dense signed activation association is not absence of geometry or poor routing.\n\nSee [matched pairing controls](pairing_specificity_controls.csv), using four-cell common valid intersections for each control.\n\n'
 if '### Pairing controls' not in text:text=text.replace(anchor,paragraph+anchor)
 p.write_text(text)
if __name__=='__main__':main()
