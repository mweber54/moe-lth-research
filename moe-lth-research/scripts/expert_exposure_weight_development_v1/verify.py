"""Numerical/provenance checks, including actual repository pruning calls."""
from audit import *
import numpy as np,pandas as pd
from analyze import ranks
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.router_age import state_dict_hash

def main():
 cfg=json.loads((O/'analysis_config.json').read_text());checks={};examples=[];torch.set_num_threads(1)
 # Repeated ties retain row-major coordinate order deterministically.
 a=np.array([.1,-.1,.1,0.,2.,-2.],dtype=np.float32);assert np.array_equal(ranks(a),[2,3,4,1,5,6]);assert np.array_equal(ranks(a),ranks(a));checks['deterministic_tie_fixture']=True
 class Adapter:
  def __init__(self,state):self.state=state
  def named_parameters(self):return self.state.items()
 for st in cfg['checkpoint_steps']:
  c=torch.load(source(7)/f'checkpoints/step_{st}.pt',map_location='cpu',weights_only=True,mmap=True);assert c['step']==st
  for l in range(4):
   names=[f'blocks.{l}.moe.experts.0.{f}.weight' for f in ('fc1','fc2')];state={n:c['model'][n] for n in names};x=torch.cat([state[n].flatten() for n in names]).numpy();r=ranks(x)
   for sparsity in (.8,.85):
    actual=expert_local_magnitude_masks(Adapter(state),sparsity);mask=torch.cat([actual[n].flatten() for n in names]).numpy();k=max(1,round(len(x)*(1-sparsity)));det=r>len(x)-k;assert mask.sum()==k==det.sum();boundary=np.partition(np.abs(x),len(x)-k)[len(x)-k];diff=mask!=det;assert np.all(np.abs(x)[diff]==boundary)
    examples.append(dict(seed=7,layer=l,expert=0,step=st,sparsity=sparsity,actual_keep=int(mask.sum()),deterministic_keep=int(det.sum()),boundary_tie_count=int((np.abs(x)==boundary).sum()),tie_only_membership_disagreements=int(diff.sum()),actual_fc1_survivors=int(mask[:262144].sum()),actual_fc2_survivors=int(mask[262144:].sum())))
 checks['actual_repository_pruning_count_checks']=examples
 # Explicit duplicate model detection, despite container/serialization differences.
 manifest=json.loads((ROOT/'results/imp_adaptation_factorial_v1/manifest.json').read_text());duplicates=[]
 for seed in cfg['source_seeds']:
  p=source(seed)/'checkpoints/step_0.pt';cp=torch.load(p,map_location='cpu',weights_only=True,mmap=True);h=state_dict_hash(cp['model']);v=next(v for v in manifest['variants'] if v['seed']==seed and not v['sparsity']);assert h==v['initial_model_hash']
  aliases=[Path(v['initial']),Path(v['full_attempt'])/'segment_0'/'checkpoints'/'initial_state.pt']
  # Rerun reset snapshots retain the starting model even when no initial_state file exists.
  aliases += [Path(v['full_attempt'])/'snapshots/step_0000.pt']
  for alias in aliases:
   if alias.exists():
    h2=state_dict_hash(torch.load(alias,map_location='cpu',weights_only=True,mmap=True)['model']);assert h2==h;duplicates.append(dict(seed=seed,canonical=str(p),alias=str(alias),model_hash=h,used_as_independent_trajectory=False))
 checks['model_state_duplicate_aliases']=duplicates
 for seed in cfg['source_seeds']:
  d=source(seed);rows=[json.loads(l) for l in (d/'logs/training_exposure.jsonl').open()];assert len(rows)==2500 and [r['step'] for r in rows]==list(range(1,2501))
  for r in rows:
   totals=[sum(r['proposed'][f'layer_{l}_expert_{e}'] for e in range(32)) for l in range(4)];assert len(set(totals))==1
  # Check interval alignment by brute summation on one expert for every frozen interval.
  m=pd.read_csv(O/'rank_stability.csv');z=m[(m.seed==seed)&(m.layer==0)&(m.expert==0)&(m.tensor=='joint')]
  for r in z.to_dict('records'):
   lo,hi=int(r['start_step']),int(r['end_step']);assert r['accepted_tokens']==sum(x['accepted']['layer_0_expert_0'] for x in rows[lo:hi]);assert r['accepted_tokens_applied']==sum(x['accepted']['layer_0_expert_0'] for x in rows[lo:hi] if x['optimizer_step_applied'])
 checks['exposure_intervals_left_open_right_closed_exact']=True
 t=pd.read_csv(O/'rank_transition_summary.csv')
 # Every transition table sums to its explicit coordinate count.
 totals=t.groupby(['seed','layer','expert','ranking','tensor','start_step','end_step']).agg(total=('count','sum'),coordinates=('coordinates','first'));assert (totals.total==totals.coordinates).all()
 checks['exact_transition_totals']=True
 # Find an actual relative rise without own growth, rather than manufacture an example.
 q=pd.read_csv(O/'coordinate_examples.csv')
 if not (q.example=='rise_without_growth').any():
  start,end=1750,2500
  ss={st:torch.load(source(7)/f'checkpoints/step_{st}.pt',map_location='cpu',weights_only=True,mmap=True)['model'] for st in (start,end)}
  found=None
  for layer in range(4):
   for expert in range(32):
    names=[f'blocks.{layer}.moe.experts.{expert}.{f}.weight' for f in ('fc1','fc2')]
    x=np.concatenate([ss[start][n].numpy().reshape(-1) for n in names]);y=np.concatenate([ss[end][n].numpy().reshape(-1) for n in names]);a,b=ranks(x),ranks(y)
    eligible=np.flatnonzero((b>a)&(np.abs(y)<=np.abs(x)))
    if len(eligible):
     k=int(eligible[0]);name=names[k//262144];index=k%262144;shape=ss[start][name].shape
     found=dict(seed=7,trajectory='original_source_seed7',layer=layer,expert=expert,ranking='expert_joint',tensor='joint',coordinates=len(x),start_step=start,end_step=end,initialization_reference=False,optimizer_reset_crossed=False,example='rise_without_growth',parameter_name=name,row=index//shape[1],column=index%shape[1],row_major_index=index,signed_start=float(x[k]),signed_end=float(y[k]),abs_start=float(abs(x[k])),abs_end=float(abs(y[k])),rank_start=int(a[k]),rank_end=int(b[k]),rank_quantile_change=float((int(b[k])-int(a[k]))/len(x)))
     # Rank gain with unchanged/shrinking own magnitude requires other coordinates to cross downward.
     crossed=np.flatnonzero((a>a[k])&(b<b[k]));assert len(crossed)>0
     j=int(crossed[0]);found['crossing_other_parameter']=names[j//262144];found['crossing_other_row_major_index']=j%262144;found['crossing_other_abs_start']=float(abs(x[j]));found['crossing_other_abs_end']=float(abs(y[j]));assert abs(y[j])<abs(x[j])
     break
   if found is not None:break
  if found is not None:
   pd.concat([q,pd.DataFrame([found])],ignore_index=True).to_csv(O/'coordinate_examples.csv',index=False)
  checks['relative_rise_without_growth_search']='all seed7 experts at1750–2500, first qualifying coordinate; illustrative only'
 q=pd.read_csv(O/'coordinate_examples.csv');checks['illustrative_examples']={k:int((q.example==k).sum()) for k in ('rise_with_growth','rise_without_growth')};assert (q[q.example=='rise_with_growth'].abs_end>q[q.example=='rise_with_growth'].abs_start).all();assert (q[q.example=='rise_without_growth'].abs_end<=q[q.example=='rise_without_growth'].abs_start).all();assert (q.rank_end>q.rank_start).all()
 checks['examples_checked_by_exact_coordinate']=True
 checks['rank_checks']=json.loads((O/'rank_computation_checks.json').read_text());checks['prediction_checks']=json.loads((O/'prediction_checks.json').read_text());checks['analysis_config_unchanged']=checks['rank_checks']['analysis_config_sha256']==sha(O/'analysis_config.json');assert checks['analysis_config_unchanged'];checks['passed']=True
 save(O/'numerical_checks.json',checks)
if __name__=='__main__':main()

