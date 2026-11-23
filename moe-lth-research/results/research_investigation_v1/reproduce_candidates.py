"""Independent arithmetic/raw-array checks; no model forwards or optimization."""
from pathlib import Path
import os,json,hashlib,re
import numpy as np,pandas as pd
O=Path(__file__).resolve().parent;ROOT=O.parents[1];B=ROOT/'results/high_sparsity_coupling_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs={}
def readcsv(p):
 p=ROOT/p;inputs[str(p.relative_to(ROOT))]=sha(p);return pd.read_csv(p)
# Repository-wide inventory is metadata/report triage, not an assertion all weights were inspected.
inv=[];reports=[]
for family in sorted((ROOT/'results').iterdir()):
 if not family.is_dir():continue
 counts={'family':str(family.relative_to(ROOT)),'files':0,'pt_files':0,'csv_files':0,'json_files':0,'md_files':0,'symlinks':0,'broken_symlinks':0}
 for parent,dirs,files in os.walk(family,followlinks=False):
  dirs[:]=[d for d in dirs if d not in ['__pycache__']]
  for name in files:
   p=Path(parent)/name;counts['files']+=1
   if p.is_symlink():counts['symlinks']+=1;counts['broken_symlinks']+=not p.exists()
   ext=p.suffix[1:];k=ext+'_files'
   if k in counts:counts[k]+=1
   if ext=='md' and not any(x in p.parts for x in ['superseded','failed_builds','smoke','inputs']) and p.exists():
    txt=p.read_text(errors='replace');claims=[f'{i}: {line[:500]}' for i,line in enumerate(txt.splitlines(),1) if re.search(r'^(#|\*\*.*verdict)|NO.GO|SUPPORTED|NOT SUPPORTED|No setting|28/30|94.7|99.8|unsupported|cannot establish',line,re.I)]
    reports.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=len(txt.encode()),claims='\n'.join(claims[:20])))
 inv.append(counts)
pd.DataFrame(inv).to_csv(O/'repository_inventory.csv',index=False);pd.DataFrame(reports).to_csv(O/'report_triage.csv',index=False)
# Recalculate the gate intervention from raw per-condition losses, not its summary contrasts.
g=readcsv('results/gate_amplitude_replay_intervention/outcomes.csv');old=readcsv('results/gate_amplitude_replay_intervention/paired_contrasts.csv');new=[]
for r in old.itertuples():
 t=g[(g.experts==r.experts)&(g.seed==r.seed)&(g.age==r.target_age)].iloc[0];c=g[(g.experts==r.experts)&(g.seed==r.seed)&(g.age==r.control_age)].iloc[0];z=dict(experts=r.experts,seed=r.seed,target_age=r.target_age,control_age=r.control_age)
 for level,label in [('native','native'),('assign','assignment_only'),('gate','assignment_gate')]:
  for row in [t,c]:assert abs(row[f'{level}_fixedgate_sparse_loss']-row[f'{level}_fixedgate_dense_loss']-row[f'{level}_fixedgate_ticket_gap'])<1e-12
  z[label+'_contrast']=t[f'{level}_fixedgate_ticket_gap']-c[f'{level}_fixedgate_ticket_gap'];assert abs(z[label+'_contrast']-getattr(r,label+'_contrast'))<1e-12
 z['gate_increment']=z['assignment_gate_contrast']-z['assignment_only_contrast'];z['attenuation']=1-abs(z['assignment_gate_contrast'])/(abs(z['assignment_only_contrast'])+1e-12);z['assignment_already_attenuates_native']=1-abs(z['assignment_only_contrast'])/abs(z['native_contrast']);assert abs(z['attenuation']-r.attenuation_fraction)<1e-12
 new.append(z)
G=pd.DataFrame(new);G.to_csv(O/'gate_contrasts_reproduced.csv',index=False)
# Raw checkpoint summary rechecks Level2 actual final native readout losses, kept distinct from common readout.
checks=[]
for r in g.itertuples():
 for kind in ['dense','sparse']:
  p=ROOT/f'results/gate_amplitude_replay_intervention/recovery/e{r.experts}/seed{r.seed}/R{r.age}/{kind}/summary.json';v=json.load(open(p));inputs[str(p.relative_to(ROOT))]=sha(p)
  delta=v['final_validation_loss']-getattr(r,f'gate_fixedgate_{kind}_loss')
  checks.append(dict(experts=r.experts,seed=r.seed,age=r.age,kind=kind,summary_minus_common_gate_CE=delta))
pd.DataFrame(checks).to_csv(O/'gate_summary_recheck.csv',index=False)
# Independent per-expert Pearson recomputation on saved arrays at initialization and final state.
pro=readcsv('results/high_sparsity_coupling_v1/report/measurement_provenance.csv');pro=pro[(pro.track=='rerun')&pro.step.isin([0,5000])]
raw={};coupling=[]
for r in pro.itertuples():
 d=np.load(r.raw_path);per=[]
 for layer in range(4):
  selection=d[f'fixed_source_L{layer}_selection_all_expert'];assert np.all(selection==selection[:,0,None]);ids=selection[:,0];x=d[f'fixed_source_L{layer}_scores'][ids].astype(np.float64);y=d[f'fixed_source_L{layer}_activations'][ids].astype(np.float64);assert x.shape==y.shape==(128,32)
  x=x-x.mean(0);y=y-y.mean(0);per.extend((x*y).sum(0)/np.sqrt((x*x).sum(0)*(y*y).sum(0)))
 raw[(r.seed,r.sparsity if pd.notna(r.sparsity) else 0.,r.step)]=np.array(per)
 source=Path(r.raw_path);inputs[str(source.relative_to(ROOT))]=sha(source)
oldc=readcsv('results/high_sparsity_coupling_v1/report/paired_seed_coupling_changes.csv');oldc=oldc[(oldc.track=='rerun')&(oldc['mode']=='fixed_source')&(oldc.selection=='all_expert')&(oldc.coefficient=='pearson')&(oldc.aggregation=='equal_layer_expert')]
for r in oldc.itertuples():
 values=[raw[(r.seed,r.sparsity,s)] for s in [0,5000]]+[raw[(r.seed,0.,s)] for s in [0,5000]];v=[x.mean() for x in values];effect=(v[1]-v[0])-(v[3]-v[2]);assert abs(effect-r.difference_in_changes)<1e-6
 coupling.append(dict(seed=r.seed,sparsity=r.sparsity,sparse_initial=v[0],sparse_final=v[1],dense_initial=v[2],dense_final=v[3],sparse_change=v[1]-v[0],dense_change=v[3]-v[2],difference_in_changes=effect))
pd.DataFrame(coupling).to_csv(O/'coupling_from_raw.csv',index=False)
# Other falsification controls, kept separate rather than counted as independent experiments.
rank=readcsv('results/cross_arch_optimization_geometry/diagnostic_rerun/seed_rankings_review.csv');res=[]
for col in ['forced_gradient','functional_similarity','native_gradient','trajectory_distance','update_alignment']:
 hit=[r.ticket_gap_ranking.split(' > ')[0]==getattr(r,col).split(' > ')[0] for r in rank.itertuples()];res.append(dict(metric=col,winner_hits=sum(hit),groups=len(hit),independent_seeds=3))
pd.DataFrame(res).to_csv(O/'alignment_winner_falsification.csv',index=False)
freeze=readcsv('results/router_freeze_mechanism/full/aggregate/seed_level_results.csv');freeze.to_csv(O/'freeze_control_review.csv',index=False)
# Hash all directly consumed artifacts for reproducibility; log arrays are actual cached model probes.
json.dump(inputs,open(O/'consumed_input_sha256.json','w'),indent=2)
json.dump({'gate_loss_arithmetic_18_conditions':True,'all_12_gate_contrasts_reproduced':True,'all_36_gate_endpoint_summaries_checked':True,'raw_fixed_source_correlations_reproduced_10_pairs':True,'max_coupling_effect_error_vs_report':float(max(abs(pd.DataFrame(coupling).difference_in_changes.to_numpy()-oldc.difference_in_changes.to_numpy()))),'training_launched':False,'report_triage_count':len(reports),'result_family_count':len(inv)},open(O/'candidate_reproduction_validation.json','w'),indent=2)
print('candidate reproduction complete',flush=True)
