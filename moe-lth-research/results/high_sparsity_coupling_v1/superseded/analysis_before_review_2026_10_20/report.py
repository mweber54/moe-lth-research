"""Explicit completeness and descriptive, seed-level paired results; no p-values."""
from common import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 manifest=frozen();endpoints=[];poolframes=[];reprframes=[];energy=[];grad=[];emb=[];weights=[];rerun=[];status=[]
 for j in manifest['conditions']:
  base=OUT/'reruns'/j['id'];done=base/'completed.json'
  status.append(dict(condition=j['id'],**(read(base/'status.json') if (base/'status.json').exists() else dict(status='NOT_STARTED'))))
  if done.exists():
   d=read(done);verify_file(d['checkpoint'],d['sha256']);rerun.append(dict(condition=j['id'],seed=j['seed'],sparsity=j['sparsity'],ce=d['final_validation_ce']))
 for track in ('historical','rerun'):
  for j in manifest['conditions']:
   for step in STEPS:
    dest=OUT/'analysis'/track/j['id']/f'step_{step:04d}';done=dest/'completed.json'
    if not done.exists():continue
    d=read(done)
    if not all(sha(dest/n)==h for n,h in d['outputs'].items()):raise RuntimeError('Corrupt completed analysis '+str(dest))
    meta={k:d[k] for k in ('condition','seed','sparsity','track','step')};endpoints.append({**meta,'validation_ce':d['validation_ce'],'probe_kl':d['probe_kl_mean']})
    for fname,collection in [('pools',poolframes),('reps',reprframes),('router_energy',energy),('gradient_identity',grad),('embedding',emb),('weight_geometry',weights)]:
     p=dest/(fname+'.csv')
     if p.exists():
      df=pd.read_csv(p)
      for k,v in meta.items():df[k]=v
      collection.append(df)
 out=OUT/'report';out.mkdir(exist_ok=True);table(out/'condition_status.csv',status);table(out/'model_metrics.csv',endpoints)
 for name,parts in [('coupling_expert_pools',poolframes),('representations',reprframes),('router_energy',energy),('gradient_checks',grad),('embedding_evolution',emb),('weight_geometry',weights)]:
  if parts:pd.concat(parts,ignore_index=True).to_csv(out/(name+'.csv'),index=False)
 df=pd.DataFrame(endpoints);comparisons=[]
 if len(df):
  for _,s in df[df.sparsity.notna()].iterrows():
   dense=df[df.track.eq(s.track)&df.seed.eq(s.seed)&df.step.eq(s.step)&df.sparsity.isna()]
   if len(dense)==1:comparisons.append(dict(track=s.track,seed=s.seed,sparsity=s.sparsity,step=s.step,sparse_ce=s.validation_ce,dense_ce=float(dense.iloc[0].validation_ce),gap=s.validation_ce-float(dense.iloc[0].validation_ce),strict_success=s.validation_ce<=float(dense.iloc[0].validation_ce)))
 table(out/'matched_loss_gaps.csv',comparisons)
 if rerun:
  rd=pd.DataFrame(rerun);rr=[]
  for _,s in rd[rd.sparsity.notna()].iterrows():
   dense=rd[rd.seed.eq(s.seed)&rd.sparsity.isna()]
   if len(dense):rr.append(dict(seed=s.seed,sparsity=s.sparsity,sparse_ce=s.ce,dense_ce=float(dense.iloc[0].ce),gap=s.ce-float(dense.iloc[0].ce),strict_success=s.ce<=float(dense.iloc[0].ce),label_origin='instrumented_rerun_validation'))
  table(out/'rerun_tickets.csv',rr)
 # Fixed expert populations across initial/final (all four layers, no favorable-layer selection).
 changes=[]
 for track in ('historical','rerun'):
  for j in manifest['conditions']:
   root=OUT/'analysis'/track/j['id'];a=root/'step_0000/groups.csv';b=root/'step_5000/groups.csv'
   if not a.exists() or not b.exists():continue
   da=pd.read_csv(a);db=pd.read_csv(b);keys=['mode','layer','selection','expert'];merged=da.merge(db,on=keys,suffixes=('_initial','_final'),validate='one_to_one');merged=merged[merged.status_initial.eq('valid')&merged.status_final.eq('valid')]
   for (mode,selection),d in merged.groupby(['mode','selection']):
    changes.append(dict(track=track,condition=j['id'],seed=j['seed'],sparsity=j['sparsity'],mode=mode,selection=selection,n_matched_experts=len(d),n_excluded=128-len(d),initial=float(d.pearson_initial.mean()),final=float(d.pearson_final.mean()),delta=float((d.pearson_final-d.pearson_initial).mean())))
 table(out/'fixed_expert_coupling_changes.csv',changes)
 if poolframes:
  p=pd.concat(poolframes,ignore_index=True);agg=p.groupby(['track','condition','seed','sparsity','step','mode','selection'],dropna=False).matched_identity.mean().reset_index();agg.to_csv(out/'seed_coupling_trajectories.csv',index=False)
  global_pools=[]
  for keys,d in p.groupby(['track','condition','seed','sparsity','step','mode','selection'],dropna=False):
   weights=d.n_pairs.to_numpy();good=d.paper_pooled_standardized_pearson.notna().to_numpy() & (weights>0)
   value=float(np.average(d.paper_pooled_standardized_pearson.to_numpy()[good],weights=weights[good])) if good.any() else None
   global_pools.append(dict(zip(['track','condition','seed','sparsity','step','mode','selection'],keys),paper_pooled_standardized_pearson=value,n_pairs=int(weights[good].sum()),n_valid_groups=int(d.n_valid.sum()),n_excluded_groups=int(d.n_excluded.sum())))
  table(out/'paper_global_standardized_pools.csv',global_pools)
  for track in agg.track.unique():
   fig,axs=plt.subplots(1,2,figsize=(11,4))
   for ax,sp in zip(axs,(.8,.85)):
    for seed in SEEDS:
     d=agg[agg.track.eq(track)&agg.seed.eq(seed)&agg.sparsity.eq(sp)&agg['mode'].eq('native')&agg.selection.eq('current_accepted')].sort_values('step');ax.plot(d.step,d.matched_identity,'o-',label=str(seed))
    ax.set(title=f'{track}: sparsity {sp}',xlabel='Attempted recovery steps',ylabel='Layer mean standardized association');ax.legend(title='Source seed')
   fig.tight_layout();fig.savefig(out/f'{track}_coupling.png',dpi=160);plt.close(fig)
 if reprframes:
  rep=pd.concat(reprframes,ignore_index=True)
  for track in rep.track.unique():
   fig,ax=plt.subplots(figsize=(10,5))
   d=rep[rep.track.eq(track)&rep.reference.eq('initial')&rep.boundary.str.startswith('router_input')&rep.sparsity.notna()]
   for (seed,sp),v in d.groupby(['seed','sparsity']):
    v=v.groupby('step').cka.mean();ax.plot(v.index,v.values,'o-',label=f'{seed}/{sp}')
   ax.set(xlabel='Attempted recovery steps',ylabel='Layer mean linear CKA with initialization',title=track);ax.legend(ncol=2);fig.tight_layout();fig.savefig(out/f'{track}_representations.png',dpi=160);plt.close(fig)
 if emb:
  ee=pd.concat(emb,ignore_index=True);fig,ax=plt.subplots(figsize=(10,5))
  for (track,seed,sp),v in ee[ee.reference.eq('original_initial')&ee.sparsity.notna()].groupby(['track','seed','sparsity']):
   v=v.sort_values('step');ax.plot(v.step,v.cka,'o-',label=f'{track}/{seed}/{sp}')
  ax.set(xlabel='Attempted recovery steps',ylabel='Embedding CKA with original E',title='Tied byte embedding evolution');ax.legend(ncol=2,fontsize=7);fig.tight_layout();fig.savefig(out/'embedding_evolution.png',dpi=160);plt.close(fig)
 if changes:
  cc=pd.DataFrame(changes);dd=cc[cc.sparsity.isna()][['track','seed','mode','selection','delta']].rename(columns={'delta':'dense_delta'})
  paired=cc[cc.sparsity.notna()].merge(dd,on=['track','seed','mode','selection'],validate='many_to_one');paired['sparse_minus_dense_delta']=paired.delta-paired.dense_delta;paired.to_csv(out/'paired_seed_coupling_changes.csv',index=False)
  uncertainty=[]
  for (track,sp,mode,selection),v in paired.groupby(['track','sparsity','mode','selection']):
   vals=v.sparse_minus_dense_delta.to_numpy();generator=np.random.default_rng(8103);lo,hi=np.quantile(generator.choice(vals,(10000,len(vals)),replace=True).mean(1),[.025,.975]);uncertainty.append(dict(track=track,sparsity=sp,mode=mode,selection=selection,n_independent_source_seeds=len(vals),cohort_complete=len(vals)==5,paired_mean=float(vals.mean()),seed_bootstrap_low=float(lo),seed_bootstrap_high=float(hi),descriptive=True))
  table(out/'seed_level_uncertainty.csv',uncertainty)
 if changes and comparisons:
  cc=pd.DataFrame(changes);gg=pd.DataFrame(comparisons);fig,ax=plt.subplots(figsize=(7,5))
  d=cc[cc['mode'].eq('native')&cc.selection.eq('fixed_initial')&cc.sparsity.notna()].merge(gg[gg.step.eq(5000)],on=['track','seed','sparsity'])
  for _,v in d.iterrows():ax.scatter(v.delta,v.gap);ax.annotate(f'{v.seed}/{v.sparsity}',(v.delta,v.gap),fontsize=8)
  ax.axhline(0,color='gray');ax.set(xlabel='Fixed-pair equal-expert coupling change',ylabel='Sparse minus dense validation CE');fig.tight_layout();fig.savefig(out/'loss_gap_vs_coupling_change.png',dpi=160);plt.close(fig)
 historical=pd.read_csv(OUT/'historical_tickets.csv');complete_runs=sum(s['status']=='COMPLETE' for s in status);measure_count=len(endpoints);is_complete=complete_runs==15 and len([x for x in endpoints if x['track']=='rerun'])==15*len(STEPS)
 lines=['# High sparsity coupling: execution report','',f'Exploratory/post hoc. Completed reruns: {complete_runs}/15. Verified measured states: {measure_count}. Overall complete: {is_complete}.', '',f'Historical strict matches: {int(historical.strict_success.sum())}/10, representing {historical[historical.strict_success].seed.nunique()} independent source seeds. Five distinct dense controls. Historical labels remain separate from reruns.','',historical[['seed','sparsity','gap','strict_success']].to_csv(index=False),'','## Findings and limits','']
 if changes:
  cc=pd.DataFrame(changes);selected=cc[cc.sparsity.notna()&cc['mode'].eq('native')&cc.selection.eq('fixed_initial')];lines+=['Measured fixed-initial-pair coupling changes (matched valid expert populations):','',selected[['track','seed','sparsity','initial','final','delta','n_matched_experts']].to_csv(index=False),'']
 if grad:
  g=pd.concat(grad);lines += [f"Masked single-token gradient identity maximum absolute residual: {g.masked_identity_max_error.max():.4g}. Router identity residual: {g.router_identity_max_error.max():.4g}. Unselected task-gradient logits observed: up to {int(g.unselected_nonzero_task_logit_gradients.max())} per layer. This repository's full-softmax gate differs from selected-logit normalization.",'']
 lines += ['## Answers supported by completed measurements','']
 if changes:
  cc=pd.DataFrame(changes);sparse=cc[cc.sparsity.notna()]
  for (track,sp,mode,selection),d in sparse.groupby(['track','sparsity','mode','selection']):
   if selection not in ('fixed_initial','current_accepted','all_expert'):continue
   lines += [f"1–2. {track}, sparsity {sp}, {mode}, {selection}: {len(d)} source seeds; coupling increased in {int((d.delta>0).sum())}, decreased in {int((d.delta<0).sum())}; mean initial {d.initial.mean():.4f}, final {d.final.mean():.4f}, change {d.delta.mean():+.4f}. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control."]
  lines += ['Input and selection contrasts are separately measured on the same token positions. Their differences are descriptive, not an additive causal decomposition. Missing experts are excluded using common initial/final membership within each contrast.','']
 else:lines += ['1–2. Initial/final trajectories are not yet complete; persistence and input-versus-selection effects remain unanswered.','']
 if grad:
  g=pd.concat(grad);lines += [f"3. Across completed checks, maximum masked expert identity residual is {g.masked_identity_max_error.max():.4g}; median retained input norm is {g.retained_input_norm.median():.4f}. Signed and absolute directions are separate raw columns; zero-gradient cases are undefined. Up to {int(g.unselected_nonzero_task_logit_gradients.max())} unselected router logits receive task gradients under full-softmax normalization. Batch-aggregated AdamW changes are separately recorded and are not the single-token identity.",'']
 else:lines += ['3. Numerical tests passed; checkpoint-specific gradient checks remain incomplete.','']
 if changes:
  hist=historical[['seed','sparsity','strict_success']];selected=sparse[sparse['mode'].eq('native')&sparse.selection.eq('fixed_initial')].merge(hist,on=['seed','sparsity']);lines += ['4. Native, fixed-initial-pair effects by historical status (validation labels; the two sparsities within seed are dependent):','','```csv',selected[['track','seed','sparsity','strict_success','delta']].to_csv(index=False).strip(),'```','Consistency must be judged across source seeds, with rerun status separately evaluated; these nested rows are not independent replications.','']
 else:lines += ['4. Success-versus-failure consistency remains unanswered pending paired trajectories.','']
 if emb and reprframes:
  ee=pd.concat(emb);rr=pd.concat(reprframes);lines += [f"5. Embedding and router-input trajectories are cached on identical positions (earliest measured post-initial step: {int(ee[ee.step.gt(0)].step.min()) if ee.step.gt(0).any() else 'none'}). Tied byte embeddings also serve output prediction. The observation grid bounds temporal resolution; these measurements alone do not establish causal precedence.",'']
 else:lines += ['5. Embedding/hidden temporal relationships remain incomplete.','']
 lines += ['6. Association, full-spectrum q, Gram/SVD, complete expert-function errors and model CE/KL are exported separately. No cross-metric mechanistic conclusion is claimed automatically from correlation, distance, or a shared time trend. Inspect paired seed effects and exclusions; incomplete families remain incomplete.','', '7. Candidate subsequent intervention: randomize whether shared representation parameters are frozen during a short continuation with identical initial weights, masks and data. Compare fixed-pair coupling and CE. This would isolate a contribution of representation evolution; it is not performed here, and its priority remains provisional until the complete observational results are reviewed.','','No independent test split has been established. No causal mechanism or publication claim is made. Historical missing states are listed in coverage. Exact checkpoint key inventory is in checkpoint_coverage.csv. Failed attempts and live heartbeats remain under reruns/.']
 (out/'findings.md').write_text('\n'.join(lines)+'\n');save(out/'completion.json',dict(complete=is_complete,completed_reruns=complete_runs,measured_states=measure_count,unix=time.time()))
 print('\n'.join(lines[:8]))
if __name__=='__main__':main()
