from common import *
from moe_lth.pruning.masks import load_masks
RAW=['n','sign_matches','init_positive','init_negative','init_zero','trained_positive','trained_negative','trained_zero','outward_growth_count','sum_abs_init','sum_abs_trained','sum_abs_init2','sum_abs_trained2','sum_abs_product','sum_init','sum_delta','sum_init2','sum_delta2','sum_init_delta']
@torch.inference_mode()
def stats(w,t,indices):
 x=w[indices].double();y=t[indices].double();delta=y-x;ax=x.abs();ay=y.abs()
 vals=[x.new_tensor(x.numel()),(x.sign()==y.sign()).sum(),(x>0).sum(),(x<0).sum(),(x==0).sum(),(y>0).sum(),(y<0).sum(),(y==0).sum(),(delta*x.sign()>0).sum(),ax.sum(),ay.sum(),ax.square().sum(),ay.square().sum(),(ax*ay).sum(),x.sum(),delta.sum(),x.square().sum(),delta.square().sum(),(x*delta).sum()]
 return dict(zip(RAW,torch.stack(vals).cpu().tolist()))
def derive(df):
 n=df.n
 def corr(sx,sy,sxx,syy,sxy):
  vx=sxx-sx*sx/n;vy=syy-sy*sy/n;den=(vx*vy).clip(lower=0)**.5
  return (sxy-sx*sy/n)/den.replace(0,np.nan)
 df['sign_agreement']=df.sign_matches/n
 df['sign_agreement_independent_marginals']=(df.init_positive*df.trained_positive+df.init_negative*df.trained_negative+df.init_zero*df.trained_zero)/n**2
 df['outward_growth_fraction']=df.outward_growth_count/n
 df['corr_abs_trained_abs_init']=corr(df.sum_abs_init,df.sum_abs_trained,df.sum_abs_init2,df.sum_abs_trained2,df.sum_abs_product)
 df['corr_delta_init']=corr(df.sum_init,df.sum_delta,df.sum_init2,df.sum_delta2,df.sum_init_delta)
 return df
@torch.inference_mode()
def main():
 status=pd.read_csv(OUT/'high_ce_status.csv').iloc[0]; assert status.status=='complete' and not status.evaluation_broken
 manifest=rows(OLD/'variant_manifest.csv');provenance=[];rnglog=[];invariants=[]
 # Complete all primary discovery analyses before opening dense trained weights.
 for analysis,round_idx in [('discovery',7),('dense',0)]:
  data=[]
  for seed in SEEDS:
   root=ROOT/f'results/imp_80_85/trajectories/e32/seed_{seed}'
   mmeta=read(root/'mask_08.json');stage=read(root/f'round_{round_idx:02d}/stage.json');checkpoint=root/f'round_{round_idx:02d}/checkpoints/final_recovered.pt'
   if not checkpoint.exists():
    if analysis=='dense':
     provenance.append(dict(analysis=analysis,seed=seed,checkpoint=str(checkpoint),exists=False,checkpoint_round=round_idx,checkpoint_step='',discovery_input_sparsity=stage['sparsity'],final_mask_nominal_sparsity=.8,final_mask_actual_sparsity=mmeta['actual_sparsity'],sha256='',hash_matches=False,rewind_policy='original initialization at each discovery round'));continue
    raise FileNotFoundError(checkpoint)
   h=sha(checkpoint);assert h==stage['checkpoint_sha256']
   payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
   r=next(r for r in manifest if int(r['seed'])==seed and r['variant']=='imp_original' and r['order']=='A')
   assert sha(r['initial_checkpoint'])==r['initial_checkpoint_sha256'];init=torch.load(r['initial_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
   assert payload['config']['model']['num_experts']==init['config']['model']['num_experts']==32
   masks=load_masks(root/'mask_08.pt');parent=load_masks(root/'mask_07.pt')
   provenance.append(dict(analysis=analysis,seed=seed,checkpoint=str(checkpoint),exists=True,checkpoint_round=round_idx,checkpoint_step=payload['step'],discovery_input_sparsity=stage['sparsity'],final_mask_nominal_sparsity=.8,final_mask_actual_sparsity=mmeta['actual_sparsity'],sha256=h,hash_matches=True,rewind_policy='original initialization at each discovery round'))
   for name,m in masks.items():
    alive=parent[name].cuda().flatten();keep=m.cuda().flatten();assert bool((~keep|alive).all())
    t=payload['model'][name].cuda().flatten();w=init['model'][name].cuda().flatten();pool=alive.nonzero().flatten() if analysis=='discovery' else torch.arange(len(w),device='cuda')
    if analysis=='discovery':assert torch.count_nonzero(t[~alive])==0
    else:assert stage['sparsity']==0
    fixed=[('retained',keep.nonzero().flatten()),('newly_pruned',(alive&~keep).nonzero().flatten())] if analysis=='discovery' else [('eventual_imp_survivors',keep.nonzero().flatten()),('eventual_imp_pruned',(~keep).nonzero().flatten())]
    count=int(keep.sum());layer=int(name.split('.')[1]);projection=name.split('.')[-2]
    for group,idx in fixed:
     data.append(dict(analysis=analysis,seed=seed,checkpoint_round=round_idx,layer=layer,expert=int(name.split('.')[4]),projection=projection,tensor=name,group=group,draw=0,**stats(w,t,idx)))
    for draw in range(1,6):
     g,s=seed_rng(analysis,seed,draw,name);perm=torch.randperm(len(pool),device='cuda',generator=g);selected=pool[perm[:count]];removed=pool[perm[count:]]
     assert len(selected)==count and (analysis!='discovery' or bool(alive[selected].all()))
     rnglog.append(dict(analysis=analysis,seed=seed,checkpoint_round=round_idx,tensor=name,draw=draw,rng_seed=s,random_pool='round07_alive' if analysis=='discovery' else 'all_coordinates',pool_count=len(pool),selected_count=count,selected_index_sha256=hashlib.sha256(selected.cpu().numpy().tobytes()).hexdigest()))
     for group,idx in [('random_survivors',selected),('random_removed',removed)]:
      data.append(dict(analysis=analysis,seed=seed,checkpoint_round=round_idx,layer=layer,expert=int(name.split('.')[4]),projection=projection,tensor=name,group=group,draw=draw,**stats(w,t,idx)))
    invariants.append(dict(analysis=analysis,seed=seed,tensor=name,final_subset_parent=True,final_count=count,parent_alive_count=int(alive.sum()),newly_pruned_count=int((alive&~keep).sum()),random_count_matched=True,random_subset_parent_check_applicable=analysis=='discovery',random_subset_parent_passed=True if analysis=='discovery' else None,trained_previously_pruned_zero_checked=analysis=='discovery'))
   print(analysis,seed,'complete',flush=True)
   write('checkpoint_provenance.csv',provenance);write('random_sign_controls_rng.csv',rnglog);write('sign_invariants.csv',invariants)
  df=derive(pd.DataFrame(data));df.to_csv(OUT/f'{analysis}_matrix_sign_metrics.csv',index=False)
  dims=['analysis','seed','checkpoint_round','group','draw']
  for suffix,extra in [('by_layer_projection',['layer','projection']),('by_projection',['projection']),('per_seed',[])]:
   agg=df.groupby(dims+extra)[RAW].sum().reset_index();derive(agg).to_csv(OUT/f'{analysis}_{suffix}.csv',index=False)
  print(analysis,'ALL SEEDS COMPLETE',flush=True)
 write('sign_analysis_status.csv',[dict(status='complete',primary_checkpoint_round=7,secondary_checkpoint_round=0,random_draws_per_matrix=5,phase2_started=False)])
if __name__=='__main__':main()
