"""Every-seed sparse/dense differences and factorial contrasts, no selection."""
from af_common import *
import statistics,math
from scipy.stats import t

def summary(values):
 n=len(values);mean=statistics.mean(values);sd=statistics.stdev(values) if n>1 else None;half=float(t.ppf(.975,n-1))*sd/math.sqrt(n) if n>1 else None
 return dict(n_seeds=n,mean=mean,sd=sd,min=min(values),max=max(values),positive=sum(x>0 for x in values),negative=sum(x<0 for x in values),descriptive_95_t_lower=mean-half if half is not None else None,descriptive_95_t_upper=mean+half if half is not None else None)
def analyze(manifest):
 results=[];statuses=[];curves=[];lookup={};curve_lookup={}
 for j in manifest['jobs']:
  st=status(j);statuses.append(dict(st,job=j['id'],reuse=j['reuse']))
  if st['status'] not in ('COMPLETE','REUSED'):continue
  r=read(st['result_path']);results.append(r);lookup[(j['seed'],j['sparsity'],j['condition'])]=r
  for point in read(Path(st['attempt'])/'curves.json'):
   x=dict(seed=j['seed'],sparsity=j['sparsity'],condition=j['condition'],step=point['step'],ce=point['loss'],job=j['id']);curves.append(x);curve_lookup[(j['seed'],j['sparsity'],j['condition'],point['step'])]=point['loss']
 table(OUT/'per_condition_endpoints.csv',results,None if results else ['job','seed','sparsity','condition','final_ce'])
 table(OUT/'job_status.csv',statuses);table(OUT/'ce_curves.csv',curves,['seed','sparsity','condition','step','ce','job'])
 effects=[];contrasts=[]
 for seed in SEEDS:
  for sparsity in (.8,.85):
   for step in range(0,5001,50):
    def ce(sp,c):return curve_lookup.get((seed,sp,c,step))
    base_dense,base_sparse=ce(0.,'full'),ce(sparsity,'full')
    if base_dense is None or base_sparse is None:continue
    g={}
    for c in CONDITIONS:
     d,s=ce(0.,c),ce(sparsity,c)
     if d is None or s is None:continue
     gap=s-d;g[c]=gap
     effects.append(dict(seed=seed,sparsity=sparsity,condition=c,step=step,sparse_ce=s,dense_ce=d,G=gap,G_full=base_sparse-base_dense,D=gap-(base_sparse-base_dense),sparse_loss_change=s-base_sparse,dense_loss_change=d-base_dense,same_condition_strict_match=gap<=0,sparse_at_or_below_original_full_dense=s<=base_dense,dense_restricted_vs_original_full=d-base_dense,annotation='match is descriptive; negative gap can reflect dense deterioration'))
    if len(g)==4:
     contrasts.append(dict(seed=seed,sparsity=sparsity,step=step,router_when_shared_frozen=g['expert_only']-g['expert_router'],router_when_shared_trainable=g['expert_shared']-g['full'],shared_when_router_frozen=g['expert_only']-g['expert_shared'],shared_when_router_trainable=g['expert_router']-g['full'],interaction=g['full']-g['expert_router']-g['expert_shared']+g['expert_only']))
 table(OUT/'seed_gap_effects.csv',effects,['seed','sparsity','condition','step','sparse_ce','dense_ce','G','G_full','D','sparse_loss_change','dense_loss_change','same_condition_strict_match','sparse_at_or_below_original_full_dense','dense_restricted_vs_original_full','annotation'])
 table(OUT/'seed_factorial_contrasts.csv',contrasts,['seed','sparsity','step','router_when_shared_frozen','router_when_shared_trainable','shared_when_router_frozen','shared_when_router_trainable','interaction'])
 aggregates=[]
 for sp in (.8,.85):
  for c in CONDITIONS:
   rr=[r for r in effects if r['sparsity']==sp and r['condition']==c and r['step']==5000]
   if rr:
    for metric in ('G','D','sparse_loss_change','dense_loss_change'):aggregates.append(dict(sparsity=sp,condition=c,metric=metric,**summary([r[metric] for r in rr])))
  rr=[r for r in contrasts if r['sparsity']==sp and r['step']==5000]
  if rr:
   for metric in ('router_when_shared_frozen','router_when_shared_trainable','shared_when_router_frozen','shared_when_router_trainable','interaction'):aggregates.append(dict(sparsity=sp,condition='factorial',metric=metric,**summary([r[metric] for r in rr])))
 table(OUT/'descriptive_seed_summaries.csv',aggregates)
 counts={s:sum(x['status']==s for x in statuses) for s in sorted({x['status'] for x in statuses})};complete=sum(x['status'] in ('COMPLETE','REUSED') for x in statuses);newdone=sum(x['status']=='COMPLETE' for x in statuses);launched=sum(x['status'] in ('RUNNING','COMPLETE','FAILED') and not x['reuse'] for x in statuses)
 save(OUT/'completion.json',dict(scientifically_complete=complete==60,status='COMPLETE' if complete==60 else 'STOPPED' if (OUT/'STOP.json').exists() else 'RUNNING_OR_PREPARED',counts=counts,total=60,verified_conditions=complete,new_training_launched=launched,new_training_complete=newdone,reused_full_complete=counts.get('REUSED',0),reuse_expected=manifest['reused_full'],new_expected=manifest['new_recoveries'],updated=time.time()))
 lines=['# Adaptation factorial findings','',f'Verified conditions: {complete}/60. New training jobs launched: {launched}; completed: {newdone}; completed reused full controls: {counts.get("REUSED",0)}. Counts: {counts}.','', '## Observations and differential effects','', 'Original native full-validation CE is primary; diagnostic FP32 losses do not replace it. All E32 source seeds and both IMP sparsities remain included. The five dense controls are shared across the two sparsity comparisons.']
 for r in [r for r in effects if r['step']==5000]:
  lines.append(f"- Seed {r['seed']}, sparsity {r['sparsity']:.0%}, {r['condition']}: sparse CE {r['sparse_ce']:.6f}, dense CE {r['dense_ce']:.6f}, G {r['G']:+.6f}, D {r['D']:+.6f}; sparse change {r['sparse_loss_change']:+.6f}, dense change {r['dense_loss_change']:+.6f}; original full dense level recovered={r['sparse_at_or_below_original_full_dense']}.")
 lines+=['','## Equal-seed endpoint summaries','']
 for sp in (.8,.85):
  for c in CONDITIONS:
   rr=[r for r in effects if r['sparsity']==sp and r['condition']==c and r['step']==5000]
   if not rr:continue
   d=summary([r['D'] for r in rr]);ss=statistics.mean(r['sparse_loss_change'] for r in rr);dd=statistics.mean(r['dense_loss_change'] for r in rr)
   direction='increases the sparse-minus-dense gap' if d['mean']>0 else 'decreases the sparse-minus-dense gap' if d['mean']<0 else 'leaves the mean gap unchanged'
   decomposition='both sparse and dense deteriorate' if ss>0 and dd>0 else 'sparse deteriorates while dense does not' if ss>0 else 'dense deteriorates while sparse does not' if dd>0 else 'neither shows mean deterioration'
   lines.append(f"- {sp:.0%} {c}, n={len(rr)} seeds ({'complete' if len(rr)==5 else 'partial'}): {direction}; mean D={d['mean']:+.6f}, descriptive 95% t interval [{d['descriptive_95_t_lower']}, {d['descriptive_95_t_upper']}], signs +{d['positive']}/-{d['negative']}. Mean sparse change {ss:+.6f}, dense change {dd:+.6f}: {decomposition}.")
 lines+=['','## Interpretation','', 'Positive D indicates greater relative sparse impairment. Interpret its sparse and dense components separately: a lower gap driven by poor dense performance is not successful recovery of the original performance level. Similar sparse/dense deterioration provides little explanation of sparse outperformance. Freezing initially untrained shared attention and tied embedding/output limits this protocol; fixed-budget failure can be slower learning. Frozen router parameters do not freeze decisions or exposure. Clipping/optimizer changes are part of the restriction, not a purely routing effect.','', 'Seed is the independent unit (n=5), with equal weights and separate .80/.85 summaries. The sparsities share seeds and dense controls. Descriptive t intervals and sign consistency are not equivalence tests or proof of a unique mechanism, compensation, generality or novelty. No performance-based stopping, extension or additional experiment is scheduled.','', 'Diagnostics use the same complete native-input probe, proposed/accepted assignment counts and explicit gradient denominators. Update norms are actual net parameter displacement across fixed intervals, not summed path length. Historical clipping is reconstructed and labelled; new clipping uses the actual native return. Primary curves and every seed effect precede aggregated summaries in the CSVs. Missing/pending endpoints are not observations.','', 'See protocol.md, manifest.json, reuse_decisions.json, input_hashes.json, parameter_groups.csv, resource_estimate.json, validation/, status/, workers/, and per-attempt diagnostic/training/state files.']
 if (OUT/'STOP.json').exists():lines+=['','Technical stop: '+str(read(OUT/'STOP.json'))]
 (OUT/'findings.md').write_text('\n'.join(lines)+'\n')
 if results:figures(results,effects,curves)
 return effects,contrasts

def figures(results,effects,curves):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 p=OUT/'figures';p.mkdir(exist_ok=True)
 fig,axes=plt.subplots(1,5,figsize=(19,4),sharey=True)
 for ax,seed in zip(axes,SEEDS):
  for sp,label in [(0.,'dense'),(.8,'IMP80'),(.85,'IMP85')]:
   rr=[r for r in results if r['seed']==seed and r['sparsity']==sp];ax.plot([CONDITIONS.index(r['condition']) for r in rr],[r['final_ce'] for r in rr],marker='o',label=label)
  ax.set_xticks(range(4),CONDITIONS,rotation=60);ax.set_title(f'Seed {seed}')
 axes[0].set_ylabel('Final native full-validation CE');axes[-1].legend();fig.tight_layout();fig.savefig(p/'individual_seed_endpoints.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
 for ax,sp in zip(axes,(.8,.85)):
  for seed in SEEDS:
   rr=[r for r in effects if r['seed']==seed and r['sparsity']==sp and r['step']==5000];ax.plot([CONDITIONS.index(r['condition']) for r in rr],[r['D'] for r in rr],marker='o',label=str(seed))
  ax.axhline(0,c='black');ax.set_xticks(range(4),CONDITIONS,rotation=40);ax.set_title(f'{sp:.0%}');ax.set_ylabel('D = restricted gap minus full gap')
 axes[-1].legend(title='Source seed');fig.tight_layout();fig.savefig(p/'individual_seed_gap_changes.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,5,figsize=(18,4),sharey=True)
 for ax,seed in zip(axes,SEEDS):
  for sp in (0.,.8,.85):
   for condition in CONDITIONS:
    rr=[r for r in curves if r['seed']==seed and r['sparsity']==sp and r['condition']==condition]
    if rr:ax.plot([r['step'] for r in rr],[r['ce'] for r in rr],label=f'{sp} {condition}')
  ax.set_yscale('log');ax.set_title(f'Seed {seed}');ax.set_xlabel('Attempted updates')
 axes[0].set_ylabel('Native validation CE (log)');fig.tight_layout();fig.savefig(p/'all_condition_ce_trajectories.png',dpi=160);plt.close(fig)
if __name__=='__main__':
 with lock(OUT/'report.lock'):analyze(read(OUT/'manifest.json'))
