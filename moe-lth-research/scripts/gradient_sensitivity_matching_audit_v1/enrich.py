"""Complete concentration reporting from existing arrays; no new matching decisions."""
from common import *

def row_stats(h,l,width):
 rows_count=1024 if width==256 else 256;k=len(h)
 rh=np.bincount(h//width,minlength=rows_count);rl=np.bincount(l//width,minlength=rows_count)
 return dict(row_HHI_H=float(np.sum((rh/k)**2)) if k else None,row_HHI_L=float(np.sum((rl/k)**2)) if k else None,row_max_share_H=float(rh.max()/k) if k else None,row_max_share_L=float(rl.max()/k) if k else None,row_TV=float(np.abs(rh-rl).sum()/(2*k)) if k else None)

def main():
 frozen();overlap=rows(D/'feature_overlap.csv');bykey={(int(r['seed']),r['parameter']):r for r in overlap};prov=read(OLD/'checkpoint_and_calibration_provenance.json')
 for seed in SEEDS:
  c=prov['seeds'][str(seed)]['checkpoints']['1000'];assert sha(ROOT/c['path'])==c['sha256'];state=load(ROOT/c['path'])['model']
  for name in g.names(state):
   w=state[name].numpy().reshape(-1).astype(np.float64);small=np.argsort(np.abs(w),kind='stable')[:len(w)//2]
   A=torch.load(OLD/'scores'/f'seed{seed}'/(name+'.pt'),map_location='cpu',weights_only=True,mmap=True)['A'].numpy().reshape(-1)
   h=[];l=[]
   for ids in np.array_split(small,50):
    ordered=ids[np.lexsort((ids,A[ids]))];k=int(math.ceil(len(ids)*.25));l.extend(ordered[:k]);h.extend(ordered[-k:])
   h=np.asarray(h);l=np.asarray(l);r=bykey[(seed,name)];width=state[name].shape[1];new=row_stats(h,l,width)
   assert abs(new['row_TV']-float(r['row_TV']))<1e-12
   assert math.isclose(float(np.median(A[h])),float(r['A_H_q0.5']),rel_tol=1e-12,abs_tol=1e-30)
   r.update(new)
  print('Concentration report enriched',seed,flush=True)
 table(D/'feature_overlap.csv',overlap)
 balances=rows(D/'candidate_balance.csv')
 for r in balances:
  seed=int(r['seed']);name=r['parameter'];width=256 if r['projection']=='fc1' else 1024
  if r['method']=='binary_alternative_edges':
   p=D/'solver_inputs'/f'seed{seed}'/(name+'.npz');a=np.load(p);idx=np.load(D/'solver_outputs'/f'seed{seed}'/(name+'.'+r['design']+'.npz'))['selected_edges'];h=a['h'][idx];l=a['l'][idx]
  else:
   a=np.load(D/'coordinates'/f'seed{seed}'/(name+'.npz'));h=a[r['design']+'_h'];l=a[r['design']+'_l']
  assert len(h)==int(r['pairs']);r.update(row_stats(h,l,width))
 table(D/'candidate_balance.csv',balances)
 summaries=[]
 for seed in SEEDS:
  for layer in range(4):
   for projection in ('fc1','fc2'):
    rs=[r for r in overlap if int(r['seed'])==seed and int(r['layer'])==layer and r['projection']==projection]
    metrics={k:float(np.median([float(r[k]) for r in rs])) for k in ('magnitude_SMD','magnitude_KS','slope_SMD','slope_KS','variation_SMD','variation_KS','row_TV','column_TV','row_HHI_H','row_HHI_L','column_HHI_H','column_HHI_L') if all(r[k] not in ('',None) for r in rs)}
    summaries.append(dict(seed=seed,layer=layer,projection=projection,tensors=len(rs),**{'median_'+k:v for k,v in metrics.items()},interpretation='descriptive medians over32 experts, not training replications'))
 table(D/'feature_overlap_summaries.csv',summaries)
 save(D/'concentration_reporting_checks.json',{'passed':True,'existing_population_membership_reconstructed':True,'row_TV_and_A_quantiles_match_previous_records':True,'row_and_column_concentration_reported':True,'matching_decisions_changed':False,'model_operations':0,'scores_recomputed':0})
if __name__=='__main__':main()
