"""Exact CPU coordinate analysis, bounded to the frozen cohort/grid. No model forwards."""
from audit import *
import numpy as np,concurrent.futures,math,re
from scipy.stats import spearmanr
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.models import TinyMoELanguageModel

def ranks(x):
 order=np.argsort(np.abs(x),kind='stable');r=np.empty(len(x),dtype=np.int32);r[order]=np.arange(1,len(x)+1,dtype=np.int32);return r

def exposure(seed,d,steps,pairs):
 logs=[json.loads(l) for l in (d/'logs/training_exposure.jsonl').open()];keys=[f'layer_{l}_expert_{e}' for l in range(4) for e in range(32)]
 a=np.array([[r['accepted'][k] for k in keys] for r in logs],dtype=np.int64);p=np.array([[r['proposed'][k] for k in keys] for r in logs],dtype=np.int64);ap=np.array([r['optimizer_step_applied'] for r in logs]);ca=np.vstack([np.zeros((1,128),dtype=np.int64),np.cumsum(a,axis=0)]);cp=np.vstack([np.zeros((1,128),dtype=np.int64),np.cumsum(p,axis=0)]);cap=np.vstack([np.zeros((1,128),dtype=np.int64),np.cumsum(a*ap[:,None],axis=0)])
 routes={};hashes={}
 for st in steps[1:]:
  path=d/f'logs/validation_routes_step_{st}.npz';hashes[str(st)]=sha(path)
  with np.load(path) as z:routes[st]=[np.concatenate([z[f'checkpoint_{st}_batch_{b}_layer_{l}'].reshape(-1) for b in range(31)]) for l in range(4)]
  assert all(len(x)==492416 and x.min()>=0 and x.max()<32 for x in routes[st])
 out={}
 for t,u in pairs:
  count=ca[u]-ca[t];proposed=cp[u]-cp[t];apcount=cap[u]-cap[t];apsteps=int(ap[t:u].sum());probe=[None]*128
  if t:
   for l in range(4):
    tab=np.bincount(routes[t][l].astype(np.int64)*32+routes[u][l],minlength=1024).reshape(32,32);inter=tab.diagonal();union=tab.sum(0)+tab.sum(1)-inter
    for e in range(32):probe[l*32+e]=float(1-inter[e]/union[e]) if union[e] else 0.
  for i,k in enumerate(keys):out[(t,u,i)]=dict(interval_attempts=u-t,interval_applied_updates=apsteps,interval_amp_skips=u-t-apsteps,accepted_tokens=int(count[i]),proposed_tokens=int(proposed[i]),dropped_tokens=int(proposed[i]-count[i]),accepted_tokens_applied=int(apcount[i]),accepted_tokens_skipped=int(count[i]-apcount[i]),accepted_rate_applied=float(apcount[i]/max(apsteps,1)),accepted_rate_attempt=float(count[i]/(u-t)),drop_fraction=float(1-count[i]/proposed[i]) if proposed[i] else 0.,cumulative_accepted_by_start=int(ca[t,i]),cumulative_applied_accepted_by_start=int(cap[t,i]),fixed_probe_proposed_membership_turnover=probe[i])
 return out,hashes

def work(args):
 seed,layer,expert,states,steps,pairs,ex=args;prefix=f'blocks.{layer}.moe.experts.{expert}';names=[prefix+'.fc1.weight',prefix+'.fc2.weight'];weights=[]
 for st in steps:
  assert all(n in states[st] for n in names)
  vals=[states[st][n].numpy().reshape(-1).copy() for n in names];assert all(v.shape==(262144,) and np.isfinite(v).all() for v in vals);weights.append(vals)
 result=[];trans=[];late=[];surv=[];persistence=[];examples=[];tiecount=0
 for unit in ('fc1','fc2','joint'):
  xs=[v[0] if unit=='fc1' else v[1] if unit=='fc2' else np.concatenate(v) for v in weights];rs=[ranks(x) for x in xs];n=len(xs[0]);meta=dict(seed=seed,trajectory=f'original_source_seed{seed}',layer=layer,expert=expert,ranking='expert_joint' if unit=='joint' else 'within_tensor',tensor=unit,coordinates=n)
  for si,st in enumerate(steps):
   if unit=='joint':
    for f in (.2,.15):
     keep=max(1,round(n*f));mask=rs[si]>n-keep
     for j,name in enumerate(names):surv.append(dict(**meta,step=st,retained_fraction=f,parameter_name=name,survivor_fraction=float(mask[j*262144:(j+1)*262144].mean()),survivor_count=int(mask[j*262144:(j+1)*262144].sum()),tie_at_boundary=int(np.count_nonzero(np.abs(xs[si])==np.partition(np.abs(xs[si]),n-keep)[n-keep]))))
  for t,u in pairs:
   i=steps.index(t);j=steps.index(u);a,b=rs[i],rs[j];x,y=xs[i],xs[j];ax=np.abs(x);ay=np.abs(y);delta=ay-ax;disp=y-x;dr=(b.astype(np.float64)-a)/n
   correlation=1-6*np.sum((a.astype(np.float64)-b)**2)/(n*(float(n)**2-1))
   common=dict(**meta,start_step=t,end_step=u,initialization_reference=t==0,optimizer_reset_crossed=False)
   base=dict(**common,rank_correlation=float(correlation),mean_absolute_magnitude_start=float(ax.mean(dtype=np.float64)),mean_absolute_magnitude_end=float(ay.mean(dtype=np.float64)),initial_mean_absolute_magnitude=float(np.abs(xs[0]).mean(dtype=np.float64)),current_low_half_mean_magnitude=float(ax[a<=n//2].mean(dtype=np.float64)),mean_absolute_magnitude_change=float(delta.mean(dtype=np.float64)),mean_abs_magnitude_change=float(np.abs(delta).mean(dtype=np.float64)),mean_signed_displacement=float(disp.mean(dtype=np.float64)),rms_signed_displacement=float(np.sqrt(np.mean(disp.astype(np.float64)**2))),mean_abs_rank_change=float(np.abs(dr).mean()),**ex[(t,u,layer*32+expert)])
   binsa=np.minimum(((a.astype(np.int64)-1)*5)//n,4);binsb=np.minimum(((b.astype(np.int64)-1)*5)//n,4);tab=np.bincount(binsa*5+binsb,minlength=25).reshape(5,5)
   assert tab.sum()==n
   for f in (.2,.15):
    keep=max(1,round(n*f));topa=a>n-keep;topb=b>n-keep;both=int(np.count_nonzero(topa&topb));low=a<=n//2;rise=low&topb
    count=int(rise.sum());removed=~topa
    base[f'top{round(f*100)}_overlap_fraction']=both/keep;base[f'top{round(f*100)}_turnover']=1-both/keep
    late.append(dict(**common,retained_fraction=f,early_low_count=int(low.sum()),late_rising_count=count,late_rising_fraction_of_early_low=count/int(low.sum()),late_rising_fraction_of_late_top=count/keep,removed_later_top_count=int((removed&topb).sum()),fraction_later_top_removed=int((removed&topb).sum())/keep,fraction_early_removed_later_top=int((removed&topb).sum())/int(removed.sum()),late_riser_mean_abs_change=float(delta[rise].mean(dtype=np.float64)) if count else None,late_riser_mean_start_abs=float(ax[rise].mean(dtype=np.float64)) if count else None,late_riser_mean_end_abs=float(ay[rise].mean(dtype=np.float64)) if count else None,late_riser_absolute_growth_fraction=float(np.mean(delta[rise]>0)) if count else None))
    if f==.2:base['late_rising_fraction']=count/int(low.sum())
   for aa in range(5):
    for bb in range(5):trans.append(dict(**common,early_bin=aa,late_bin=bb,count=int(tab[aa,bb]),fraction=int(tab[aa,bb])/n,row_fraction=int(tab[aa,bb])/int(tab[aa].sum())))
   result.append(base)
   if seed==7 and layer==0 and expert==0 and t>0 and unit=='joint':
    for label,mask in [('rise_with_growth',(dr>.05)&(delta>0)),('rise_without_growth',(dr>.00001)&(delta<=0))]:
     inds=np.flatnonzero(mask)
     if len(inds):
      k=int(inds[0]);pn=names[k//262144];index=k%262144;shape=states[t][pn].shape
      assert float(states[t][pn].reshape(-1)[index])==float(x[k]) and float(states[u][pn].reshape(-1)[index])==float(y[k])
      examples.append(dict(**common,example=label,parameter_name=pn,row=index//shape[1],column=index%shape[1],row_major_index=index,signed_start=float(x[k]),signed_end=float(y[k]),abs_start=float(ax[k]),abs_end=float(ay[k]),rank_start=int(a[k]),rank_end=int(b[k]),rank_quantile_change=float(dr[k])))
  nonzero=rs[1:];always_low=np.logical_and.reduce([r<=n//2 for r in nonzero]);always_high=np.logical_and.reduce([r>n-round(n*.2) for r in nonzero]);earlylow=nonzero[0]<=n//2
  persistence.append(dict(**meta,first_observed_nonzero=steps[1],last_observed=steps[-1],always_low_fraction=float(always_low.mean()),early_low_persistent_fraction=float(always_low.sum()/earlylow.sum()),always_top20_fraction=float(always_high.mean()),early_high_persistent_fraction=float(always_high.sum()/round(n*.2))))
 return result,trans,late,surv,persistence,examples

def main():
 cfg=json.loads((O/'analysis_config.json').read_text());specsha=sha(O/'analysis_config.json');steps=cfg['checkpoint_steps'];pairs=[tuple(p) for p in cfg['pairs']];allrows=[[] for _ in range(6)];routehash={};start=time.monotonic();torch.set_num_threads(1)
 for seed in cfg['source_seeds']:
  d=Path(cfg['source_paths'][str(seed)]);states={}
  for st in steps:
   cp=torch.load(d/f'checkpoints/step_{st}.pt',map_location='cpu',weights_only=True,mmap=True)
   assert cp['step']==st and cp['config']['model']==cfg['architecture']
   states[st]=cp['model']
  expected={f'blocks.{l}.moe.experts.{e}.{fc}.weight' for l in range(4) for e in range(32) for fc in ('fc1','fc2')}
  assert all({n for n in s if '.moe.experts.' in n and n.endswith('.weight')}==expected for s in states.values())
  ex,routehash[str(seed)]=exposure(seed,d,steps,pairs)
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   tasks=((seed,l,e,states,steps,pairs,ex) for l in range(4) for e in range(32))
   for i,outs in enumerate(pool.map(work,tasks)):
    for dest,rs in zip(allrows,outs):dest.extend(rs)
    if i%32==31:print('seed',seed,'experts',i+1,'elapsed',round(time.monotonic()-start),flush=True)
  del states
  for name,rs in zip(['rank_stability','rank_transition_summary','late_rising_weight_summary','realized_survivor_fractions','rank_persistence','coordinate_examples'],allrows):table(O/(name+'.csv'),rs)
  print('completed seed',seed,flush=True)
 save(O/'rank_computation_checks.json',dict(passed=True,analysis_config_sha256=specsha,exact_coordinates_per_checkpoint=67108864,coordinate_sampling=False,exact_name_and_shape_matching=True,rank_bijections=True,transition_counts_exact=True,fixed_probe_hashes=routehash,wall_seconds=time.monotonic()-start,manual_examples='coordinate_examples.csv; reread values by exact name and row-major index'))
if __name__=='__main__':main()
