from gcommon import *
import numpy as np

def mean(path,key):
 vals=[float(r[key]) for r in rows(path) if r['rank']=='16' and r[key] not in ('','None') and r['zero_matrix']=='False' and r['rank_deficient']=='False' and r['degenerate_cutoff']=='False']
 return float(np.mean(vals)) if vals else None

def summarize():
 # Revisit the fixed inventory: completed analyses skip; newly added optional
 # dense endpoint coverage is completed before the initial report/launch gate.
 from analyze_existing import analyze
 for seed in SEEDS:analyze(seed)
 state_summaries=[]
 for seed in SEEDS:
  root=OUT/'existing'/f'source_e32_seed{seed}_T2500'
  for state_dir in sorted(root.iterdir()):
   if not (state_dir/'weights.csv').exists():continue
   state_summaries.append(dict(seed=seed,state=state_dir.name,alignment_centered=mean(state_dir/'weights.csv','alignment_centered'),alignment_above_shuffled=mean(state_dir/'weights.csv','alignment_above_shuffled'),source_subspace_overlap=mean(state_dir/'weights.csv','source_subspace_overlap'),primary_rank=16))
 table(OUT/'analysis/existing_state_seed_summaries.csv',state_summaries)
 effects=[]
 for seed in SEEDS:
  root=OUT/'existing'/f'source_e32_seed{seed}_T2500'
  for sp in SP:
   l=root/f'masked_trained_learned_s{sp:.2f}'/'weights.csv';r=root/f'masked_trained_random_s{sp:.2f}'/'weights.csv'
   if not l.exists() or not r.exists():continue
   for metric in ('alignment_centered','alignment_above_shuffled','source_subspace_overlap'):
    lv=mean(l,metric);rv=mean(r,metric);effects.append(dict(seed=seed,sparsity=sp,metric=metric,learned=lv,random=rv,paired_difference=None if lv is None or rv is None else lv-rv))
 table(OUT/'analysis/pruning_seed_effects.csv',effects);summary=[];g=np.random.default_rng(8103)
 for sp in SP:
  for metric in ('alignment_centered','alignment_above_shuffled','source_subspace_overlap'):
   rs=[r for r in effects if r['sparsity']==sp and r['metric']==metric];v=np.array([r['paired_difference'] for r in rs if r['paired_difference'] is not None]);ci=np.quantile(g.choice(v,(10000,len(v)),replace=True).mean(1),[.025,.975]) if len(v) else [None,None]
   summary.append(dict(sparsity=sp,metric=metric,n_seeds=len(v),mean_paired_difference=float(v.mean()) if len(v) else None,bootstrap_lo=None if ci[0] is None else float(ci[0]),bootstrap_hi=None if ci[1] is None else float(ci[1])))
 table(OUT/'analysis/pruning_summary.csv',summary)
 lines=['# Initial checkpoint analysis','', 'Exploratory learned-minus-random pruning-only effects; primary rank 16; seed is the replication unit. All seed effects are in pruning_seed_effects.csv. Historical endpoints are separate directories; no trajectories inferred.','']
 for r in summary:lines.append(f"- sparsity {r['sparsity']}, {r['metric']}: n={r['n_seeds']}, mean={r['mean_paired_difference']}, seed bootstrap 95% interval [{r['bootstrap_lo']}, {r['bootstrap_hi']}].")
 lines+=['','## All seed effects for primary centered alignment','','| Sparsity | Seed 7 | Seed 17 | Seed 29 | Seed 43 | Seed 61 |','|---|---:|---:|---:|---:|---:|']
 for sp in SP:
  values=[next(r['paired_difference'] for r in effects if r['sparsity']==sp and r['seed']==seed and r['metric']=='alignment_centered') for seed in SEEDS];lines.append(f'| {sp:.0%} | '+' | '.join(f'{v:.5f}' for v in values)+' |')
 lines+=['','Learned pruning slightly raises alignment above the original source level; higher alignment is not literal identity with source geometry. Compare alignment and basis preservation separately.']
 for sp in SP:
  vals={}
  for kind in ('learned','random'):
   for state_kind,label in [('pruning',f'masked_trained_{kind}_s{sp:.2f}'),('rewind',f'masked_init_R100_{kind}_s{sp:.2f}')]:
    ss=[r for r in state_summaries if r['state']==label];vals[state_kind,kind]=float(np.mean([r['source_subspace_overlap'] for r in ss]));vals['alignment',kind]=float(np.mean([r['alignment_centered'] for r in ss])) if state_kind=='rewind' else vals.get(('alignment',kind))
  lines.append(f"Sparsity {sp:.0%}: source-subspace overlap learned/random is {vals['pruning','learned']:.3f}/{vals['pruning','random']:.3f} before rewind and {vals['rewind','learned']:.3f}/{vals['rewind','random']:.3f} after rewind. Actual R100 rewind alignment is {vals['alignment','learned']:.3f}/{vals['alignment','random']:.3f}. These are distinct reference states, not recovery findings.")
 lines+=['',f'{len(state_summaries)} existing states summarized separately in existing_state_seed_summaries.csv. Source-age geometry, unmasked rewind, masked rewind, pruning-only trained states and historical endpoints must not be conflated. All 15 canonical historical dense checkpoints are hash-verified on both hosts and included. Separate table-backed reference directories do not themselves contain model checkpoints.','No causal claim. Rewound initialization and source-age states are separate raw tables. New recovery findings remain pending.']
 (OUT/'analysis/initial_report.md').write_text('\n'.join(lines)+'\n')
 try:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  fig,ax=plt.subplots()
  for sp in SP:
   rs=[r for r in effects if r['sparsity']==sp and r['metric']=='alignment_centered'];ax.plot([r['seed'] for r in rs],[r['paired_difference'] for r in rs],'o-',label=str(sp))
  ax.axhline(0,color='gray');ax.set(xlabel='Source seed',ylabel='Learned − random centered alignment (rank 16)');ax.legend();fig.savefig(OUT/'analysis/pruning_effects.png');plt.close(fig)
 except ImportError:pass
 from domain_analysis import analyze as domains
 domains()
 save(OUT/'analysis/initial_analysis_status.json',dict(complete=all((OUT/'existing'/f'source_e32_seed{s}_T2500'/'complete.json').exists() for s in SEEDS),unix=time.time()))
if __name__=='__main__':summarize()
