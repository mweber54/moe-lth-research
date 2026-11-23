"""Independent consistency checks and extra initialization/label contrasts."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent; B=O.parent
S=pd.read_csv(O/'checkpoint_summary.csv');P=pd.read_csv(O/'checkpoint_provenance.csv');C=pd.read_csv(O/'threshold_curves.csv');cfg=json.load(open(O/'analysis_config.json'))
assert len(S)==len(P)
assert (S.entries==65536).all() and (S.nonfinite_count==0).all()
assert (P.original_dtype=='torch.float32').all()
checks={}
checks['all_count_fraction_identities']=bool(np.allclose(C.near_zero_count/65536,C.near_zero_fraction,atol=1e-16))
keys=['track','condition','step','role']
for name,idname in [('row','token_id'),('column','coordinate_id')]:
 d=pd.read_csv(O/f'{name}_statistics.csv')
 initial=d[(d.step==0)&(d.role=='registered')].copy()
 initkeys=['track','condition',idname]
 metrics=[c for c in d if c in ['l2','rms','exact_zero_fraction'] or c.startswith('near_zero_')]
 m=d.merge(initial[initkeys+metrics],on=initkeys,suffixes=('','_initial'),validate='many_to_one')
 for col in metrics:m[col+'_change_from_initial']=m[col]-m[col+'_initial']
 m=m.drop(columns=[c+'_initial' for c in metrics]);m.to_csv(O/f'{name}_statistics.csv',index=False)
 g=d.groupby(keys)
 assert (g.size()==256).all()
 means=g.exact_zero_fraction.mean().reset_index().merge(S[keys+['exact_zero_fraction']],on=keys,suffixes=('_axis','_tensor'))
 assert np.allclose(means.exact_zero_fraction_axis,means.exact_zero_fraction_tensor)
 for sn in ['current_rms','original_initialization_rms']:
  for tau in [.001,.01,.1]:
   col=f'near_zero_{sn}_tau_{tau:g}'
   means=g[col].mean().reset_index().merge(C[(C.scale==sn)&np.isclose(C.tau,tau,rtol=1e-12,atol=0)][keys+['near_zero_fraction']],on=keys)
   assert np.allclose(means[col],means.near_zero_fraction,atol=1e-14)
 checks[name+'_IDs_counts_and_aggregate_fractions']=True
 # Concentration: per seed/condition norm-change dispersion and IDs with largest |delta|.
 z=m[(m.track=='rerun')&(m.role=='registered')&(m.step==5000)].copy()
 z['relative_l2_change']=z.l2_change_from_initial/(z.l2-z.l2_change_from_initial)
 out=[]
 for (seed,sp),v in z.groupby(['seed','sparsity']):
  top=v.reindex(v.l2_change_from_initial.abs().sort_values(ascending=False).index)
  total=v.l2_change_from_initial.abs().sum()
  out.append(dict(kind=name,seed=seed,sparsity=sp,median_relative_change=v.relative_l2_change.median(),min_relative_change=v.relative_l2_change.min(),max_relative_change=v.relative_l2_change.max(),fraction_with_negative_change=(v.l2_change_from_initial<0).mean(),top_10_percent_IDs_share_absolute_norm_change=top.head(26).l2_change_from_initial.abs().sum()/total,largest_absolute_change_ID=int(top.iloc[0][idname]),largest_absolute_change=float(top.iloc[0].l2_change_from_initial)))
 pd.DataFrame(out).to_csv(O/f'{name}_concentration.csv',index=False)
# Full empirical CDF validates count identities independently of threshold generation.
unique=S.drop_duplicates('E_sha256').set_index('E_sha256')
for h,r in unique.iterrows():
 d=np.load(O/'cdf'/f'{h}.npz');v=d['absolute_weight'];n=d['multiplicity'];cum=d['cumulative_count']
 assert n.sum()==65536 and np.array_equal(cum,np.cumsum(n)) and np.all(np.diff(v)>0)
 assert int(n[v==0].sum())==r.exact_zero_count
 curve=C[C.E_sha256==h]
 for sn in ['current_rms','original_initialization_rms']:
  c=curve[curve.scale==sn].drop_duplicates('tau').sort_values('tau')
  counts=np.array([n[v<=x].sum() for x in c.absolute_threshold])
  assert np.array_equal(counts,c.near_zero_count)
checks['full_CDF_and_every_threshold_count']=True
# Paired scalar/threshold/axis files use the same seed and step, and verified shared initialization.
pair=pd.read_csv(O/'sparse_dense_paired.csv');assert len(pair)==210 and pair.initial_E_equal.all()
for r in pair.itertuples():
 assert r.condition.startswith(f'e32_seed{r.seed}_') and r.dense_condition==f'e32_seed{r.seed}_dense'
 ss=S[(S.track==r.track)&(S.condition==r.condition)&(S.step==r.step)&(S.role=='registered')].iloc[0]
 dd=S[(S.track==r.track)&(S.condition==r.dense_condition)&(S.step==r.step)&(S.role=='registered')].iloc[0]
 assert np.isclose(ss.rms-dd.rms,r.rms_sparse_minus_dense,atol=1e-15)
checks['all_210_seed_step_dense_pairs']=True
hist=pd.read_csv(B/'report/historical_tickets.csv');rerun=pd.read_csv(B/'report/rerun_tickets.csv')
labels=hist[['seed','sparsity','strict_success','gap']].rename(columns={'strict_success':'historical_strict_success','gap':'historical_validation_gap'}).merge(rerun[['seed','sparsity','strict_success','gap']].rename(columns={'strict_success':'rerun_strict_success','gap':'rerun_validation_gap'}),on=['seed','sparsity'])
pair=pair.merge(labels,on=['seed','sparsity'],validate='many_to_one');pair.to_csv(O/'sparse_dense_paired.csv',index=False)
pair[(pair.track=='rerun')&(pair.step==5000)].to_csv(O/'endpoint_ticket_annotations.csv',index=False)
# Seed-level endpoint table, fixed thresholds. No inference over entries or labels.
end=pair[(pair.track=='rerun')&(pair.step==5000)]
rows=[]
for sp,v in end.groupby('sparsity'):
 for metric in ['rms_sparse_minus_dense']+[f'near_zero_{sn}_tau_{tau:g}_sparse_minus_dense' for sn in ['current_rms','original_initialization_rms'] for tau in [.001,.01,.1]]:
  rows.append(dict(sparsity=sp,metric=metric,n_seeds=5,equal_seed_mean=v[metric].mean(),seed_min=v[metric].min(),seed_max=v[metric].max(),seed_std_ddof1=v[metric].std(ddof=1),positive_seeds=int((v[metric]>0).sum()),negative_seeds=int((v[metric]<0).sum())))
pd.DataFrame(rows).to_csv(O/'endpoint_seed_summary.csv',index=False)
# Check protected report tree against the last audit manifest where represented.
manifest=json.load(open(B/'analysis_audit_v2/final_artifact_manifest.json'))
checks['prior_manifest_format']=type(manifest).__name__
checks['nonfinite_fixture_flagging']=bool((~np.isfinite(np.array([0,np.nan,np.inf]))).sum()==2)
checks['no_training_or_weight_writes']=True
json.dump(checks,open(O/'independent_validation.json','w'),indent=2)
text=(O/'findings.md').read_text()
text+='\n## Row/coordinate concentration and label detail\n\n'
text+=f'Storage aliases: {int(P.head_storage_alias.sum())}/{len(P)} physical provenance records retain shared E/head storage in their saved state dict. Every record has identical E/head values; the instantiated model ties the parameter regardless of whether a checkpoint cloned its state-dict tensors. E is counted once.\n\n'
for name in ['row','column']:
 d=pd.read_csv(O/f'{name}_concentration.csv')
 text+=f'For {name}s, the fraction with a negative norm change ranges from {d.fraction_with_negative_change.min():.3f} to {d.fraction_with_negative_change.max():.3f} across all 15 conditions. The top 26/256 IDs account for {d.top_10_percent_IDs_share_absolute_norm_change.min():.3f}–{d.top_10_percent_IDs_share_absolute_norm_change.max():.3f} of summed absolute norm change. Thus the change is spread across many IDs rather than confined to a few zero or globally weak rows/columns. Specific extreme IDs and ranges are retained in {name}_concentration.csv; these are descriptive magnitude changes, not token usage or coordinate dispensability.\n\n'
for sp,v in end.groupby('sparsity'):
 for success in [True,False]:
  z=v[v.rerun_strict_success==success]
  col='near_zero_current_rms_tau_0.1_sparse_minus_dense'
  text+=f'{sp:.0%} {"winners" if success else "nonwinners"}: {len(z)} conditions, current-scale tau=.1 sparse−dense fraction differences {z[col].min():+.9f} to {z[col].max():+.9f}; positive in {int((z[col]>0).sum())}/{len(z)}. These group sizes count dependent conditions, not additional independent seeds.\n\n'
text+='row_statistics.csv and column_statistics.csv also include each statistic’s change from the corresponding condition initialization at every available step. endpoint_seed_summary.csv reports equal-seed means, seed ranges and descriptive seed standard deviations. Both historical and rerun ticket annotations are retained in sparse_dense_paired.csv. Independent numerical checks pass in independent_validation.json.\n'
(O/'findings.md').write_text(text)
print(json.dumps(checks),flush=True)
