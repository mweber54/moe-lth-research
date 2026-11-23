"""Analyze completed split replays; controls are compared within exposure-matched experts."""
import argparse,csv,json
from pathlib import Path
import numpy as np,torch
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=17);ap.add_argument('--order',default='B');ap.add_argument('--step',type=int,default=250);ap.add_argument('--permutations',type=int,default=1);ap.add_argument('--all-retained',action='store_true');ap.add_argument('--output-dir',type=Path,default=P);args=ap.parse_args()
base=f'seed{args.seed}_{args.order}_{args.step}'
def load(h,label):return torch.load(P/f'split_half_gradients_{base}_h{h}_{label}{'_all_retained' if args.all_retained else ''}.pt',map_location='cpu',weights_only=False)
def arrays(r):
 signs=[];known=[]
 for d in range(2):
  ds=[];dk=[]
  for j,(s,n) in enumerate(zip(r['sign_sum'][d],r['resolved_batches'][d])):
   s=s.numpy().reshape(-1); n=n.numpy().reshape(-1)
   ds.append(np.divide(s,n,out=np.zeros_like(s,dtype=np.float64),where=n>0))
   dk.append((n>=8)&(r['accepted_occurrences'][d,j]>=32)&(r['contributing_batches'][d,j]>=8))
  signs.append(ds);known.append(dk)
 return signs,known
def calc(pair,threshold,reverse,experts):
 arr=[arrays(x) for x in pair]; disc,valid=(1,0) if reverse else (0,1)
 all_s=[[],[]];all_k=[[],[]]
 for h in (0,1):
  for d in (0,1):
   all_s[h].append(np.concatenate([arr[h][0][d][j] for j in experts]))
   all_k[h].append(np.concatenate([arr[h][1][d][j] for j in experts]))
 sd,sv=all_s[disc],all_s[valid];kd,kv=all_k[disc],all_k[valid]
 known=kd[0]&kd[1]; eligible=known&kv[0]&kv[1]
 selected=known&(sd[0]*sd[1]<0)&(np.abs(sd[0])>=threshold)&(np.abs(sd[1])>=threshold)
 tested=selected&eligible;n=int(tested.sum())
 retained=[np.sign(sd[d])==np.sign(sv[d]) for d in (0,1)]
 strong=(sv[0]*sv[1]<0)&(np.abs(sv[0])>=threshold)&(np.abs(sv[1])>=threshold)
 frac=lambda x:float(x[tested].mean()) if n else None
 joint=tested&retained[0]&retained[1]&strong
 result={'eligible_both_halves':int(eligible.sum()),'qualify_both_same_directions_count':int(joint.sum()),'qualify_both_same_directions_fraction':float(joint.sum()/eligible.sum()) if eligible.sum() else None,'total_weights':len(known),'discovery_known':int(known.sum()),'selected':int(selected.sum()),'selected_validation_known':n,'selected_validation_unknown':int((selected&~eligible).sum()),'selection_coverage':float(selected.mean()),'unknown_rate':float((~eligible).mean()),'selected_unknown_rate':float((selected&~eligible).sum()/selected.sum()) if selected.sum() else None,'retain_tinystories':frac(retained[0]),'retain_wikitext103':frac(retained[1]),'retain_both':frac(retained[0]&retained[1]),'remain_strongly_opposed':frac(strong)}
 for d,name in enumerate(('tinystories','wikitext103')):
  both=all_k[0][d]&all_k[1][d]; x=all_s[0][d][both]; y=all_s[1][d][both]
  result['correlation_'+name]=float(np.corrcoef(x,y)[0,1]) if len(x)>1 and np.std(x)>0 and np.std(y)>0 else None
 # Independent-domain baseline among jointly eligible nonzero signs, with observed marginals.
 for h in (0,1):
  x,y=all_s[h]; valid_sign=all_k[h][0]&all_k[h][1]&(x!=0)&(y!=0)
  if valid_sign.any():
   px=float((x[valid_sign]>0).mean());py=float((y[valid_sign]>0).mean())
   result[f'half{h+1}_marginal_expected_opposition']=px*(1-py)+(1-px)*py
   result[f'half{h+1}_marginal_expected_agreement']=px*py+(1-px)*(1-py)
  else:result[f'half{h+1}_marginal_expected_opposition']=None;result[f'half{h+1}_marginal_expected_agreement']=None
 return result
def write(name,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 args.output_dir.mkdir(parents=True,exist_ok=True)
 with (args.output_dir/name).open('w',newline='') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
original=[load(1,'original'),load(2,'original')]
perms=[[load(h,f'perm{i}') for h in (1,2)] for i in range(args.permutations)]
rows=[]; null=[];coverage=[]
for h in (0,1):
 for d in range(2):
  for j in range(32):
   for i,pair in [(-1,original)]+list(enumerate(perms)):
    r=pair[h];orig=original[h];occ=int(r['accepted_occurrences'][d,j]);batches=int(r['contributing_batches'][d,j]);orig_occ=int(orig['accepted_occurrences'][d,j]);orig_batches=int(orig['contributing_batches'][d,j])
    coverage.append({'cell':base,'analysis':'all_retained' if args.all_retained else 'first_180','permutation':i,'half':h+1,'domain':d+1,'expert':j,'accepted_occurrences':occ,'contributing_batches':batches,'original_accepted_occurrences':orig_occ,'original_contributing_batches':orig_batches,'occurrence_shift':occ-orig_occ,'contributing_batch_shift':batches-orig_batches,'occurrence_ratio':occ/orig_occ if orig_occ else None,'contributing_batch_ratio':batches/orig_batches if orig_batches else None})
for t in (0.6,0.8,0.9):
 for rev in (False,True):
  common={'cell':base,'analysis':'all_retained' if args.all_retained else 'first_180','threshold':t,'direction':'reverse' if rev else 'primary'}
  rows.append(common|calc(original,t,rev,list(range(32))))
  for i,pair in enumerate(perms):
   unfiltered=calc(pair,t,rev,list(range(32)))
   # Require both domain/half exposure measures to lie within 20% of the
   # original for the same expert; no gradient-dependent filtering.
   matched=[]
   for j in range(32):
    ok=True
    for h in (0,1):
     for d in (0,1):
      for key in ('accepted_occurrences','contributing_batches'):
       a=int(original[h][key][d,j]);b=int(pair[h][key][d,j]);ok &= (a==b==0 or (a>0 and 0.8<=b/a<=1.2))
    if ok:matched.append(j)
   if matched:
    obs=calc(original,t,rev,matched);ctl=calc(pair,t,rev,matched)
    null.append(common|{'permutation':i,'matched_experts':len(matched),'matched_expert_ids':' '.join(map(str,matched))}|{'all_experts_'+k:v for k,v in unfiltered.items()}|{'observed_'+k:v for k,v in obs.items()}|{'null_'+k:v for k,v in ctl.items()})
   else:null.append(common|{'permutation':i,'matched_experts':0,'matched_expert_ids':''}|{'all_experts_'+k:v for k,v in unfiltered.items()})
write('split_half_reproducibility.csv',rows);write('split_half_permutation_null.csv',null);write('split_half_coverage.csv',coverage)
# Paired upper-tail comparison on each permutation's own matched expert set.
comparisons=[]
for t in (0.6,0.8,0.9):
 for rev in ('primary','reverse'):
  for metric in ('retain_both','remain_strongly_opposed','qualify_both_same_directions_fraction'):
   usable=[r for r in null if r['threshold']==t and r['direction']==rev and r.get('null_'+metric) is not None and r.get('observed_'+metric) is not None]
   count=sum(r['null_'+metric]>=r['observed_'+metric] for r in usable)
   comparisons.append({'cell':base,'analysis':'all_retained' if args.all_retained else 'first_180','threshold':t,'direction':rev,'metric':metric,'usable_permutations':len(usable),'null_at_least_observed':count,'empirical_upper_tail_p':(count+1)/(len(usable)+1) if usable else None})
write('split_half_permutation_comparison.csv',comparisons)
