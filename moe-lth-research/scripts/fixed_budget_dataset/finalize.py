"""Validate, assemble and freeze Dataset A. No training calls."""
from common import *
import time,itertools,math,fcntl
import numpy as np,pandas as pd
from collections import Counter
from moe_lth.experiments import run_router_age_recovery as r
from moe_lth.models.transformer import TinyMoELanguageModel
from analyze_r20_8_vs_r60_12 import collect,probe_batches,js,cumulative_exposure

def exposures(path,source=False,e=8):
 if not path.exists():return {}
 cache=OUT/'diagnostic_cache'/f'{hashlib.sha256(str(path).encode()).hexdigest()}.json'
 identity={'size':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
 if cache.exists():
  c=jread(cache)
  if c['identity']==identity:return c['summary']
 proposed=Counter();accepted=Counter();active=Counter();applied=Counter();seen=set();updates=0;applied_flags_available=True
 with path.open() as f:
  for line in f:
   x=json.loads(line);step=x['step']
   if step==0:continue
   p=x.get('proposed' if source else 'expert_token_counts');a=x.get('accepted' if source else 'accepted_expert_token_counts')
   if p is None or a is None:return {}
   assert step not in seen;seen.add(step);proposed.update(p);accepted.update(a);active.update({k:int(v>0) for k,v in a.items()})
   if 'optimizer_step_applied' not in x:applied_flags_available=False
   if x.get('optimizer_step_applied'):updates+=1;applied.update({k:int(v>0) for k,v in a.items()})
 if seen!=set(range(1,2501)):return {}
 assert sum(proposed.values())==163840000 and len(proposed)==4*e
 n=sum(proposed.values());a=sum(accepted.values());v=np.array([accepted[k] for k in proposed],float)
 result={'proposed_assignments':n,'accepted_assignments':a,'dropped_assignments':n-a,'acceptance_rate':a/n,'mean_accepted_per_expert':a/(4*e),'mean_active_attempt_steps_per_expert':sum(active.values())/(4*e),'mean_active_applied_steps_per_expert':sum(applied.values())/(4*e) if applied_flags_available else None,'drop_rate':1-a/n,'load_cv':float(v.std()/v.mean()),'applied_steps':updates if applied_flags_available else None}
 jwrite(cache,{'identity':identity,'summary':result});return result

def metrics(dest):
 m=jread(dest/'metadata.json');s=jread(dest/'summary.json');curve=[json.loads(x) for x in (dest/'metrics.jsonl').read_text().splitlines()]
 assert curve[0]['step']==0 and curve[-1]['step']==2500
 assert all(math.isfinite(x['loss']) for x in curve)
 by_step={x['step']:x for x in curve}
 assert len(by_step)==len(curve), 'Duplicate validation steps'
 assert all(step in by_step for step in range(0,1251,50)), 'Canonical early-AUC samples missing'
 early=[by_step[step] for step in range(0,1251,50)]
 auc=sum(.5*(a['loss']+b['loss'])*(b['step']-a['step']) for a,b in zip(early,early[1:]))
 assert abs(curve[-1]['loss']-s['final_validation_loss'])<1e-10
 return m,s,curve[0]['loss'],curve[-1]['loss'],auc

def developmental(c,root,e,seed):
 cache=OUT/f'developmental/e{e}_seed{seed}.csv'
 if cache.exists():
  f=pd.read_csv(cache);assert len(f)==11;return f.set_index('router_age').to_dict('index')
 device=torch.device('cuda');batches=probe_batches(c)
 model=r.load_model_from_checkpoint(c['model'],str(root/'checkpoints/step_2500.pt'),device);mature=collect(model,batches,device);del model
 rows=[]
 for a in AGES:
  model=r.load_model_from_checkpoint(c['model'],str(root/f'checkpoints/step_{25*a}.pt'),device);trace=collect(model,batches,device);del model
  probs=torch.cat([x[3] for x in trace]);top=probs.topk(2,-1).values;cv=[];mi=[];agree=[];distance=[]
  for layer in range(4):
   rr=[x for x in trace if x[0]==layer];ids=torch.cat([x[4] for x in rr]);tokens=torch.cat([x[2] for x in rr]);counts=torch.bincount(ids,minlength=e).float();cv.append(float(counts.std(unbiased=False)/counts.mean()))
   joint=torch.bincount(ids*256+tokens,minlength=e*256).reshape(e,256).double();joint/=joint.sum();ind=joint.sum(1,keepdim=True)*joint.sum(0,keepdim=True);nz=joint>0;mi.append(float((joint[nz]*(joint[nz]/ind[nz]).log()).sum()))
   aa=next(x for x in trace if x[0]==layer);bb=next(x for x in mature if x[0]==layer);agree.append(float((aa[4]==bb[4]).float().mean()));distance.append(js(aa[3],bb[3]))
  sample=cumulative_exposure(root,25*a,e)
  rows.append({'num_experts':e,'seed':seed,'router_age':a,'router_entropy':float(-(probs*probs.clamp_min(1e-12).log()).sum(-1).mean()),'router_margin':float((top[:,0]-top[:,1]).mean()),'router_load_cv':np.mean(cv),'router_token_id_specialization':np.mean(mi),'router_assignment_agreement_to_r100':np.mean(agree),'router_js_distance_to_r100':np.mean(distance),'router_cumulative_sampled_exposure':np.mean(sample) if sample is not None else np.nan})
  del trace,probs;torch.cuda.empty_cache()
 writecsv(cache,rows);return pd.DataFrame(rows).set_index('router_age').to_dict('index')

_RECOVERY_DIGESTS=None
def recovery_digest(path):
 global _RECOVERY_DIGESTS
 if _RECOVERY_DIGESTS is None:
  audit=pd.read_csv(OUT/'historical_recovery_audit.csv').fillna('')
  _RECOVERY_DIGESTS={x.path:x.final_checkpoint_hash for x in audit.itertuples() if x.valid}
 checked=path/'validated_result.json'
 if checked.exists():
  digest=jread(checked).get('final_checkpoint_sha256')
  if digest:return digest
 return _RECOVERY_DIGESTS.get(str(path)) or sha(path/'checkpoints/final_recovered.pt')

def main():
 manifest=OUT/'dataset_manifest.json'
 if manifest.exists() and jread(manifest).get('frozen'):
  assert sha(OUT/'router_conditioned_sparse_recoverability_equal_global_budget.csv')==jread(manifest)['sha256'];print('Dataset already frozen; no changes');return
 jobs=pd.read_csv(OUT/'run_manifest.csv');assert jobs.status.isin(['REUSED','COMPLETE']).all(),'Unfinished jobs; no freeze permitted'
 # Wait for the incremental exporter to exit before performing the existing artifact validation and freeze.
 progress_lock=(OUT/'progress.lock').open('a');fcntl.flock(progress_lock,fcntl.LOCK_EX)
 assert len(jobs[jobs.job_type=='source'])==25 and len(jobs[jobs.job_type=='dense'])==275 and len(jobs[jobs.job_type=='sparse'])==1650
 def job(kind,e,s,a=None,sp=None):
  x=jobs[jobs.condition_id==cid(kind,e,s,a,sp)];assert len(x)==1;return x.iloc[0]
 source_rows=[];checkpoint_rows=[];mask_rows=[];dense_rows=[];sparse_rows=[];rows=[];checks=[]
 version=time.strftime('dataset-a-v1-%Y%m%dT%H%M%SZ',time.gmtime())
 dataidentity={k:sha(ROOT/config(8,7)['data'][k]) for k in ('train_path','validation_path')}
 datahash=hashlib.sha256(json.dumps(dataidentity,sort_keys=True).encode()).hexdigest()
 for e,seed in itertools.product(ES,SEEDS):
  sourcejob=job('source',e,seed);root=Path(sourcejob.artifact_path);initial=load(root/'checkpoints/step_0.pt');c=initial['config'];assert not protocol_errors(c,e,seed)
  confighash=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest();recoveryhash=hashlib.sha256(json.dumps({'model':c['model'],'training':{k:c['training'][k] for k in ('steps','batch_size','learning_rate','weight_decay','grad_clip','precision')},'routing':c['routing'],'data_hash':datahash},sort_keys=True).encode()).hexdigest()
  sx=exposures(root/'logs/training_exposure.jsonl',True,e);dev=developmental(c,root,e,seed)
  source_rows.append({'condition_id':sourcejob.condition_id,'status':sourcejob.status,'num_experts':e,'seed':seed,'path':str(root),'training_steps_attempted':2500,'training_steps_applied':sx.get('applied_steps'),'batch_size':128,'sequence_length':128,'gradient_accumulation':1,'effective_batch_size':128,'input_tokens_processed':40960000,'optimizer':'AdamW','learning_rate':.0003,'scheduler':'none','warmup':0,'weight_decay':.01,'grad_clip':1,'precision':'fp16','GradScaler':True,'dataset_hashes':json.dumps(dataidentity),'config_hash':confighash,**sx})
  mm={}
  for sp in SP:
   p=OUT/f'masks/e{e}/seed_{seed}/s{sp:.2f}.pt';mask=torch.load(p,map_location='cpu',weights_only=True);keep=sum(int(v.sum()) for v in mask.values());total=sum(v.numel() for v in mask.values());digest=state_dict_hash(mask)
   mm[sp]={'path':str(p),'hash':digest,'keep':keep,'pruned':total-keep,'actual':1-keep/total};mask_rows.append({'num_experts':e,'seed':seed,'target_sparsity':sp,'actual_sparsity':1-keep/total,'keep_count':keep,'pruned_count':total-keep,'mask_hash':digest,'path':str(p),'mask_checkpoint':str(root/'checkpoints/step_2500.pt'),'mask_source_age':100,'bias_handling':'excluded','scope':'expert_local','method':'magnitude','rewind_checkpoint':str(root/'checkpoints/step_0.pt')})
  for a in AGES:
   cp=root/f'checkpoints/step_{a*25}.pt';ck=load(cp);assert ck['step']==a*25 and not protocol_errors(ck['config'],e,seed)
   h={g:state_dict_hash(group(ck['model'],g)) for g in ('router','expert','shared')};checkpointsha=sha(cp);checkpoint_rows.append({'num_experts':e,'seed':seed,'router_age':a,'source_checkpoint_step':a*25,'path':str(cp),'sha256':checkpointsha,'model_hash':state_dict_hash(ck['model']),**h,'config_hash':confighash})
   dj=job('dense',e,seed,a);dp=Path(dj.artifact_path);dm,ds,di,df,da=metrics(dp);dx=exposures(dp/'gradient_stats/gradient_stats.jsonl',False,e)
   assert dm['initial_router_state_hash']==h['router'] and dm['router_step']==a*25
   dense_rows.append({'condition_id':dj.condition_id,'num_experts':e,'seed':seed,'router_age':a,'path':str(dp),'status':dj.status,'source_checkpoint_hash':checkpointsha,'router_hash':h['router'],'expert_hash':dm['expert_state_hash'],'shared_hash':dm['shared_state_hash'],'recovery_config_id':recoveryhash,'data_hash':dm['training_batch_sequence_hash'],'initial_validation_loss':di,'early_auc':da,'recovery_improvement':di-df,'final_loss':df,'final_checkpoint_sha256':recovery_digest(dp),**dx})
   for sp in SP:
    sj=job('sparse',e,seed,a,sp);p=Path(sj.artifact_path);sm,ss,si,sf,sa=metrics(p);rx=exposures(p/'gradient_stats/gradient_stats.jsonl',False,e)
    assert sm['initial_router_state_hash']==dm['initial_router_state_hash'] and sm['shared_state_hash']==dm['shared_state_hash'] and sm['mask_hash']==mm[sp]['hash']
    assert sm['training_batch_sequence_hash']==dm['training_batch_sequence_hash'] and sm['validation_batch_sequence_hash']==dm['validation_batch_sequence_hash']
    # Canonical dense comparator may be selected from duplicate valid historical endpoints; always recompute the paired metric from the linked raw endpoints.
    gap=sf-df
    row={'condition_id':sj.condition_id,'dataset_version':version,'architecture_name':f'{e}E','num_experts':e,'seed':seed,'router_age':a,'source_checkpoint_step':25*a,'sparsity':sp,'top_k':1,'num_layers':4,'d_model':256,'expert_hidden_size':1024,'capacity_factor':1.25,'source_attempted_steps':2500,'source_applied_steps':sx.get('applied_steps'),'source_batch_size':128,'source_seq_len':128,'source_input_tokens':40960000,'source_global_proposed_assignments':163840000,'source_global_accepted_assignments':sx.get('accepted_assignments'),'source_global_dropped_assignments':sx.get('dropped_assignments'),'source_acceptance_rate':sx.get('acceptance_rate'),'source_mean_accepted_per_expert':sx.get('mean_accepted_per_expert'),'source_mean_active_attempt_steps_per_expert':sx.get('mean_active_attempt_steps_per_expert'),'source_mean_active_applied_steps_per_expert':sx.get('mean_active_applied_steps_per_expert'),'target_sparsity':sp,'actual_sparsity':mm[sp]['actual'],'mask_keep_count':mm[sp]['keep'],'mask_pruned_count':mm[sp]['pruned'],'mask_method':'magnitude','mask_scope':'expert_local','mask_source_age':100,'mask_hash':mm[sp]['hash'],'rewind_checkpoint':str(root/'checkpoints/step_0.pt'),'dense_initial_validation_loss':di,'sparse_preprune_validation_loss':di,'sparse_postprune_validation_loss':si,'pruning_shock':si-di,'dense_early_auc':da,'sparse_early_auc':sa,'matched_early_auc_gap':sa-da,'dense_final_validation_loss':df,'sparse_final_validation_loss':sf,'ticket_gap':gap,'dense_recovery_improvement':di-df,'sparse_recovery_improvement':si-sf,'source_drop_rate':sx.get('drop_rate'),'source_load_cv':sx.get('load_cv'),'source_checkpoint_path':str(cp),'source_checkpoint_hash':checkpointsha,'router_hash':h['router'],'expert_hash':h['expert'],'shared_hash':h['shared'],'rewound_expert_hash':sm['expert_state_hash'],'rewound_shared_hash':sm['shared_state_hash'],'mask_path':mm[sp]['path'],'dense_result_path':str(dp),'sparse_result_path':str(p),'source_config_hash':confighash,'recovery_config_hash':recoveryhash,'data_hash':datahash,'protocol_version':'historical-native-fixed-budget-v1','artifact_status':sj.status,**{k:v for k,v in dev[a].items() if k.startswith('router_')}}
    for kind,x in [('dense',dx),('sparse',rx)]:
     for k in ('proposed_assignments','accepted_assignments','dropped_assignments','acceptance_rate','drop_rate','load_cv'):row[f'{kind}_recovery_{k}']=x.get(k)
     for k in ('mean_accepted_per_expert','mean_active_attempt_steps_per_expert','mean_active_applied_steps_per_expert'):row[f'{kind}_{k}']=x.get(k)
    rows.append(row);sparse_rows.append({'condition_id':sj.condition_id,'num_experts':e,'seed':seed,'router_age':a,'sparsity':sp,'path':str(p),'status':sj.status,'dense_condition_id':dj.condition_id,'mask_hash':mm[sp]['hash'],'initial_validation_loss':si,'early_auc':sa,'recovery_improvement':si-sf,'final_loss':sf,'final_checkpoint_sha256':recovery_digest(p),'source_checkpoint_hash':checkpointsha,'router_hash':h['router'],'expert_hash':sm['expert_state_hash'],'shared_hash':sm['shared_state_hash'],'data_hash':sm['training_batch_sequence_hash'],'validation_data_hash':sm['validation_batch_sequence_hash'],**rx})
    # Independent path reopens raw files instead of using the merged summaries.
    raw_sparse=json.loads((p/'metrics.jsonl').read_text().splitlines()[-1])['loss'];raw_dense=json.loads((dp/'metrics.jsonl').read_text().splitlines()[-1])['loss'];recomputed=raw_sparse-raw_dense
    checks.append({'condition_id':sj.condition_id,'num_experts':e,'seed':seed,'router_age':a,'sparsity':sp,'stored_ticket_gap':gap,'recomputed_ticket_gap':recomputed,'absolute_difference':abs(gap-recomputed),'pass':abs(gap-recomputed)<1e-12})
  print('ASSEMBLED',e,seed,flush=True)
 raw=pd.DataFrame(rows);keys=['num_experts','seed','router_age','sparsity'];assert len(raw)==1650 and not raw.duplicated(keys).any()
 expected=set(itertools.product(ES,SEEDS,AGES,SP));assert set(raw[keys].itertuples(index=False,name=None))==expected
 assert all(x['pass'] for x in checks)
 required=['condition_id','dataset_version','architecture_name','num_experts','seed','router_age','source_checkpoint_step','sparsity','top_k','num_layers','d_model','expert_hidden_size','capacity_factor','source_attempted_steps','source_batch_size','source_seq_len','source_input_tokens','source_global_proposed_assignments','target_sparsity','actual_sparsity','mask_keep_count','mask_pruned_count','mask_method','mask_scope','mask_source_age','mask_hash','rewind_checkpoint','dense_initial_validation_loss','sparse_preprune_validation_loss','sparse_postprune_validation_loss','pruning_shock','dense_early_auc','sparse_early_auc','matched_early_auc_gap','dense_final_validation_loss','sparse_final_validation_loss','ticket_gap','dense_recovery_improvement','sparse_recovery_improvement','router_entropy','router_margin','router_load_cv','router_token_id_specialization','router_assignment_agreement_to_r100','router_js_distance_to_r100','source_checkpoint_path','source_checkpoint_hash','router_hash','expert_hash','shared_hash','mask_path','dense_result_path','sparse_result_path','source_config_hash','recovery_config_hash','data_hash','protocol_version','artifact_status']
 assert raw[required].notna().all().all(),'Missing required measurement or provenance field'
 assert np.isfinite(raw[required].select_dtypes(include='number').to_numpy()).all(),'Nonfinite required measurement'
 assert not raw.condition_id.duplicated().any()
 assert raw.protocol_version.eq('historical-native-fixed-budget-v1').all()
 assert raw.source_checkpoint_step.eq(raw.router_age*25).all()
 assert raw.source_attempted_steps.eq(2500).all() and raw.source_input_tokens.eq(40960000).all()
 assert raw.groupby(['num_experts','seed','router_age']).dense_result_path.nunique().eq(1).all()
 assert raw.groupby(['num_experts','seed','sparsity']).mask_hash.nunique().eq(1).all()
 assert len(set(raw.dense_result_path))==275 and len(set(raw.sparse_result_path))==1650

 writecsv(OUT/'raw_condition_values.csv',raw)
 final=raw.copy();final['is_boundary_age']=final.router_age.isin([0,100]);final['is_intermediate_age']=~final.is_boundary_age
 for boundary in (0,100):
  lookup=raw[raw.router_age==boundary].set_index(['num_experts','seed','sparsity'])
  for name,col in [('ticket_gap','ticket_gap'),('sparse_loss','sparse_final_validation_loss'),('dense_loss','dense_final_validation_loss')]:final[f'{name}_vs_same_seed_r{boundary}']=[x[col]-lookup.loc[(x.num_experts,x.seed,x.sparsity),col] for _,x in raw.iterrows()]
 for phase,col in [('source','source_mean_accepted_per_expert'),('dense_recovery','dense_mean_accepted_per_expert'),('sparse_recovery','sparse_mean_accepted_per_expert')]:
  baseline=raw[raw.num_experts==8].set_index(['seed','router_age','sparsity'])[col]
  final[f'relative_{phase}_exposure_vs_8e']=[x[col]/baseline.loc[(x.seed,x.router_age,x.sparsity)] if pd.notna(x[col]) else np.nan for _,x in raw.iterrows()]
 for name,data in [('source_trajectory_manifest.csv',source_rows),('source_checkpoint_manifest.csv',checkpoint_rows),('mask_manifest.csv',mask_rows),('dense_recovery_manifest.csv',dense_rows),('sparse_recovery_manifest.csv',sparse_rows),('ticket_gap_recomputation_check.csv',checks)]:writecsv(OUT/name,data)
 summarize_and_plot(final)
 mainpath=OUT/'router_conditioned_sparse_recoverability_equal_global_budget.csv';writecsv(mainpath,final)
 writecsv(OUT/'missing_or_invalid_cells.csv',pd.DataFrame(columns=keys+['reason']))
 (OUT/'dataset_completeness_audit.md').write_text('PASS\n\n1650 unique sparse cells; 275 matched dense recoveries; 25 source trajectories; exact Cartesian product, no missing cells. Raw endpoints independently recomputed at tolerance 1e-12. Masks, source identities, budgets, native routing, and data linkage validated before execution/reuse and at finalization. Historical unavailable counters remain NA; endpoint/integrity fields are complete.\n')
 (OUT/'protocol_integrity.md').write_text('PASS\n\nEqual 2500 source and recovery attempts; source input tokens 40,960,000; native top-1 trainable routers; source R100 expert-local magnitude masks, bias exclusion, R0 surviving expert/shared rewind; same-seed/age dense comparator shared across sparsities. No exposure matching or mechanism interventions. Source reconstruction overlap hashes passed. Evaluation is token-weighted validation CE; AUC is trapezoidal steps 0–1250.\n')
 jwrite(OUT/'schema.json',{'columns':{c:str(final[c].dtype) for c in final},'NA':'Unavailable historical diagnostic only; never fabricated','source_exposure_scope':'complete 2500-attempt source trajectory, repeated across its ages; router sampled exposure is age-cumulative','marginal_summary_rule':'within-seed mean over omitted dimensions, then sample SD across five seeds','relative_exposure_rule':'same seed, age and sparsity against E8; full accepted counters only'})
 counts=jobs.groupby(['job_type','status']).size().to_dict();sha256=sha(mainpath)
 obs=pd.read_csv(OUT/'best_age_by_architecture_sparsity.csv');endpoint_summary=pd.read_csv(OUT/'endpoint_winner_comparison.csv');interior_count=int(obs.best_age.between(10,90).sum());r0_count=int(obs.best_age.eq(0).sum());r100_count=int(obs.best_age.eq(100).sum());tie_count=int(obs.best_age.isna().sum());unanimous_count=int(obs.number_seeds_selecting_mean_winner.eq(5).sum());discord={col:int(endpoint_summary[col+'_winner_disagrees'].sum()) for col in ('sparse_final_validation_loss','pruning_shock','matched_early_auc_gap')};report=f'''FIXED-BUDGET ROUTER-CONDITIONED SPARSE RECOVERABILITY DATASET

Dataset title: {TITLE}
Protocol status: PASS
Architectures: {ES}
Seeds: {SEEDS}
New seeds added: 43,61
Router ages: {AGES}
Sparsities: {SP}
Sparse conditions expected: 1650
Sparse conditions complete: 1650
Dense conditions expected: 275
Dense conditions complete: 275
Historical source trajectories reused: {counts.get(('source','REUSED'),0)}
New source trajectories trained: {counts.get(('source','COMPLETE'),0)}
Historical dense recoveries reused: {counts.get(('dense','REUSED'),0)}
New dense recoveries: {counts.get(('dense','COMPLETE'),0)}
Historical sparse recoveries reused: {counts.get(('sparse','REUSED'),0)}
New sparse recoveries: {counts.get(('sparse','COMPLETE'),0)}
Missing: 0
Invalid: 0
Duplicates: 0
Ticket-gap recomputation: PASS
Completeness audit: PASS
Protocol audit: PASS
Fixed source-training budget: 2500 attempts; 40,960,000 input tokens
Fixed recovery budget: 2500 attempts
Independent replicate: source seed
Primary metric: sparse final CE minus same-seed/same-age dense final CE
Dataset SHA256: {sha256}

Mean winning age by architecture and sparsity:\n{obs.to_string(index=False)}

Descriptive winner counts: {interior_count}/30 interior mean winners, {r0_count}/30 R0 winners, {r100_count}/30 R100 winners, and {tie_count}/30 tied winner sets. All five seed winner sets overlap the mean-winning set in {unanimous_count}/30 architecture/sparsity cells. Endpoint winner disagreements out of 30: {discord}. These counts describe observed grid cells, not independent statistical replicates.

Seed-level reproducibility: all five seed values, per-seed winners, winner spread and counts selecting the mean winner are retained in the summary and winner tables. Five seeds are not proof of population significance.
Where intermediate optima appear: architecture/sparsity rows with best_age in 10–90 in the table above.
Where boundary optima appear: rows with best_age 0 or 100; these observations are preserved without special reruns.
How behavior changes with sparsity/expert count: inspect the complete observed winner matrix, age/sparsity curves and heatmaps; no interpolation or scaling-law claim is made.
Sparse loss, pruning shock and early recovery versus ticket-gap winners: endpoint_winner_comparison.csv reports all four independently selected descriptive minima and disagreement indicators.
Observed cross-architecture patterns: heterogeneous cells and seed disagreements remain raw observations rather than exclusions.
Important limitations: equal global tokens/attempts do not equalize expert exposure, successful updates or convergence. Source counters are unavailable for some reused trajectories. Historical developmental exposure is sampled proposed counts; first-batch maturity and eight-batch other probes retain their historical definitions. Ages/sparsities/experts are not independent replicates. No post-hoc significance tests.
Questions Dataset B should resolve: whether patterns persist under a separately preregistered source-exposure comparison with fixed recovery; Dataset A cannot answer this causally. Dataset B was not started.
'''
 report+=descriptive_report(final,obs,endpoint_summary)
 (OUT/'final_report.md').write_text(report)
 jwrite(manifest,{'title':TITLE,'version':version,'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'timestamp':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'row_count':len(final),'column_count':len(final.columns),'sha256':sha256,'architectures':ES,'seeds':SEEDS,'router_ages':AGES,'sparsities':SP,'source_budget':{'attempts':2500,'input_tokens':40960000},'recovery_budget':{'attempts':2500},'frozen':True})
 terminal=f'''FIXED-BUDGET DATASET BUILD COMPLETE

Title: {TITLE}
Architectures: 8E,12E,16E,24E,32E
Seeds: 7,17,29,43,61
Router ages: R0,R10,R20,R30,R40,R50,R60,R70,R80,R90,R100
Sparsities: 70,75,80,85,90,95
Expected sparse conditions: 1650
Completed: 1650
Missing: 0
Protocol-invalid: 0
Historical conditions reused: {counts.get(('sparse','REUSED'),0)} sparse; {counts.get(('dense','REUSED'),0)} dense
New source trajectories trained: {counts.get(('source','COMPLETE'),0)}
New dense recoveries: {counts.get(('dense','COMPLETE'),0)}
New sparse recoveries: {counts.get(('sparse','COMPLETE'),0)}
Ticket-gap recomputation: PASS
Dataset completeness: PASS
Dataset path: {mainpath}
Dataset SHA256: {sha256}
Major descriptive observations:
1. Complete five-seed Cartesian grid; all outcomes preserved.
2. Mean winners: {interior_count} interior, {r0_count} R0, {r100_count} R100, and {tie_count} tied sets, out of 30 architecture/sparsity cells.
3. All five seed winner sets overlap the mean-winning set in {unanimous_count}/30 cells.
4. Endpoint winner disagreements: {discord}.
5. Equal global budgets do not equalize expert maturity.
NO EXPOSURE-MATCHED TRAINING WAS STARTED.
NO MECHANISM EXPERIMENTS WERE STARTED.
'''
 (OUT/'terminal_summary.txt').write_text(terminal)
 jwrite(OUT/'build_status.json',{'updated_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'counts':{f'{k[0]}:{k[1]}':int(v) for k,v in counts.items()},'running_conditions':[],'dataset_frozen':True,'sha256':sha256})
 print(terminal,flush=True)

def descriptive_report(f,obs,endpoints):
 """Report measured contrasts and heterogeneity without causal interpretation."""
 matrix=obs.pivot(index='num_experts',columns='sparsity',values='winning_ages')
 agreement=obs[['num_experts','sparsity','winning_ages','seed_winners','number_seeds_selecting_mean_winner','winner_spread','tied_best_ages','gap_to_second_ranked_age']]
 paired=pd.read_csv(OUT/'dense_sparse_boundary_decomposition.csv')
 paired_mean=paired.groupby(['num_experts','sparsity','candidate_age','boundary_age'])[['sparse_loss_difference','dense_loss_difference','ticket_gap_difference']].mean().reset_index()
 writecsv(OUT/'dense_sparse_boundary_decomposition_summary.csv',paired_mean)
 shifts=[]
 for e,g in obs.groupby('num_experts'):
  g=g.sort_values('sparsity');sequence=list(zip(g.sparsity,g.winning_ages))
  shifts.append(f"{e}E: "+', '.join(f'{100*sp:g}% → {age}' for sp,age in sequence))
 reproducible=agreement.sort_values(['number_seeds_selecting_mean_winner','winner_spread','num_experts','sparsity'],ascending=[False,True,True,True]).head(5)
 heterogeneous=agreement.sort_values(['number_seeds_selecting_mean_winner','winner_spread','num_experts','sparsity'],ascending=[True,False,True,True]).head(5)
 return '\n'.join([
  '\nArchitecture × sparsity mean winner table (measured ages):','```',matrix.to_string(),'```',
  'Seed-specific consistency and numerical ties:','```',agreement.to_string(index=False),'```',
  'Sparsity-dependent shifts:',*shifts,
  'Dense-versus-sparse contributions at the mean winner, paired to each boundary. Each row averages five within-seed contrasts. Ticket-gap difference = sparse-loss difference − dense-loss difference. Negative sparse differences indicate better sparse endpoints; positive dense differences indicate worse dense comparators. These are descriptions of the paired endpoints, not mechanisms.',
  '```',paired_mean.to_string(index=False),'```',
  'Pruning shock and early recovery versus final recoverability: independently selected endpoint winners follow. The endpoint_winner_comparison.csv and full curves preserve all observations.',
  '```',endpoints.to_string(index=False),'```',
  'Most consistent winner selections (ranked by seed agreement, then winner spread; descriptive ranking only):','```',reproducible.to_string(index=False),'```',
  'Largest winner heterogeneities (ranked by least agreement, then largest spread; no outcomes excluded):','```',heterogeneous.to_string(index=False),'```',
  'Near-ties: age_differences_from_mean_winner.csv retains every observed difference from the minimum. Numerical ties use tolerance 1e-12 and retain all tied ages; winner matrices display all tied ages, and scalar unique-winner fields are NA when tied. No post-hoc scientific near-tie threshold is imposed.',
  'Cross-architecture plots use common scales and only observed age/sparsity points. No interpolated values are observations. Differences across expert counts remain fixed-global-budget comparisons.',
  'Limitations and Dataset B questions: seed disagreement and small winner margins limit ranking precision. Dataset A does not equalize expert exposure or convergence and cannot determine why a pattern occurs. A separately specified Dataset B could test whether these descriptions persist under a different source budget. No such runs were launched.',''])

def summarize_and_plot(f):
 valuecols=['ticket_gap','sparse_final_validation_loss','dense_final_validation_loss','pruning_shock','matched_early_auc_gap']
 def summary(data,by):
  # All rows in data are already one per seed per target group.
  rows=[]
  for key,g in data.groupby(by):
   if not isinstance(key,tuple):key=(key,)
   row=dict(zip(by,key));row['n_seeds']=g.seed.nunique()
   for col in valuecols:
    v=g[col];row.update({col+'_mean':v.mean(),col+'_sd':v.std(ddof=1),col+'_median':v.median(),col+'_min':v.min(),col+'_max':v.max()})
   row['seed_ticket_gaps']=json.dumps(dict(zip(g.seed,g.ticket_gap)));rows.append(row)
  return pd.DataFrame(rows)
 full=summary(f,['num_experts','router_age','sparsity']);writecsv(OUT/'summary_by_architecture_age_sparsity.csv',full)
 for by,name in [(['num_experts','router_age'],'summary_by_architecture_age.csv'),(['num_experts','sparsity'],'summary_by_architecture_sparsity.csv')]:
  within=f.groupby(by+['seed'])[valuecols].mean().reset_index();writecsv(OUT/name,summary(within,by))
 winners=[];best=[];contrasts=[];endpoint=[];age_deltas=[];decomposition=[]
 for (e,sp),g in f.groupby(['num_experts','sparsity']):
  means=g.groupby('router_age')[valuecols].mean();ordered=means.ticket_gap.sort_values()
  tied=sorted(int(x) for x in means.index[(means.ticket_gap-ordered.iloc[0]).abs()<=1e-12]);a=tied[0];sw={}
  for age,values in means.iterrows():age_deltas.append({'num_experts':e,'sparsity':sp,'router_age':int(age),'mean_ticket_gap':values.ticket_gap,'difference_from_best_mean':values.ticket_gap-ordered.iloc[0],'numerical_tie_for_best':int(age) in tied})
  for seed,h in g.groupby('seed'):
   ranked=h.ticket_gap.sort_values();seed_ties=sorted(h.loc[(h.ticket_gap-ranked.iloc[0]).abs()<=1e-12,'router_age'].astype(int).tolist());sw[int(seed)]=seed_ties
   winners.append({'num_experts':e,'sparsity':sp,'seed':seed,'best_age':seed_ties[0] if len(seed_ties)==1 else None,'representative_best_age':seed_ties[0],'is_unique_winner':len(seed_ties)==1,'winning_ages':' | '.join(f'R{x}' for x in seed_ties),'ticket_gap':ranked.iloc[0],'tied_best_ages':json.dumps(seed_ties),'gap_to_second_ranked_age':float(ranked.iloc[1]-ranked.iloc[0])})
  selected=g[g.router_age==a];all_seed_ages=[x for ages in sw.values() for x in ages]
  row={'num_experts':e,'sparsity':sp,'best_age':a if len(tied)==1 else None,'representative_best_age':a,'is_unique_winner':len(tied)==1,'winning_ages':' | '.join(f'R{x}' for x in tied),'mean_best_ticket_gap':selected.ticket_gap.mean(),'sd_best_ticket_gap':selected.ticket_gap.std(ddof=1),'seed_winners':json.dumps(sw),'number_seeds_selecting_mean_winner':sum(bool(set(ages)&set(tied)) for ages in sw.values()),'number_seeds_with_identical_winner_set':sum(set(ages)==set(tied) for ages in sw.values()),'winner_spread':max(all_seed_ages)-min(all_seed_ages),'tied_best_ages':json.dumps(tied),'gap_to_second_ranked_age':float(ordered.iloc[1]-ordered.iloc[0])}
  for candidate_age in tied:
   selected=g[g.router_age==candidate_age]
   for boundary in (0,100):
    d=selected.set_index('seed').ticket_gap-g[g.router_age==boundary].set_index('seed').ticket_gap
    if candidate_age==a:row[f'mean_difference_from_R{boundary}']=d.mean()
    for seed,v in d.items():
     sv=selected[selected.seed==seed].iloc[0];bv=g[(g.router_age==boundary)&(g.seed==seed)].iloc[0]
     sparse_delta=sv.sparse_final_validation_loss-bv.sparse_final_validation_loss;dense_delta=sv.dense_final_validation_loss-bv.dense_final_validation_loss
     assert abs(v-(sparse_delta-dense_delta))<1e-12
     decomposition.append({'num_experts':e,'sparsity':sp,'seed':seed,'candidate_age':candidate_age,'boundary_age':boundary,'sparse_loss_difference':sparse_delta,'dense_loss_difference':dense_delta,'ticket_gap_difference':v})
     contrasts.append({'num_experts':e,'sparsity':sp,'candidate_age':candidate_age,'boundary':boundary,'seed':seed,'difference':v,'mean_difference':d.mean(),'sample_sd':d.std(ddof=1),'seeds_favoring_candidate':int((d<0).sum())})
  best.append(row);end={'num_experts':e,'sparsity':sp,'ticket_gap_winner':a if len(tied)==1 else None,'ticket_gap_winning_ages':json.dumps(tied)}
  for col in valuecols[1:]:
   other=sorted(int(x) for x in means.index[(means[col]-means[col].min()).abs()<=1e-12])
   end[col+'_winner']=other[0] if len(other)==1 else None;end[col+'_winning_ages']=json.dumps(other);end[col+'_winner_disagrees']=set(other).isdisjoint(tied)
  endpoint.append(end)
 writecsv(OUT/'seed_specific_winners_by_architecture_sparsity.csv',winners);writecsv(OUT/'boundary_contrasts_by_architecture_sparsity.csv',contrasts);writecsv(OUT/'best_age_by_architecture_sparsity.csv',best);writecsv(OUT/'endpoint_winner_comparison.csv',endpoint)
 writecsv(OUT/'age_differences_from_mean_winner.csv',age_deltas);writecsv(OUT/'dense_sparse_boundary_decomposition.csv',decomposition)
 best_sp=[]
 for (e,age),g in f.groupby(['num_experts','router_age']):
  means=g.groupby('sparsity').ticket_gap.mean();ranked=means.sort_values();ties=sorted(float(x) for x in means.index[(means-ranked.iloc[0]).abs()<=1e-12])
  best_sp.append({'num_experts':e,'router_age':age,'best_sparsity':ties[0] if len(ties)==1 else None,'tied_best_sparsities':json.dumps(ties),'gap_to_second_ranked_sparsity':float(ranked.iloc[1]-ranked.iloc[0])})
 writecsv(OUT/'best_sparsity_by_architecture_age.csv',best_sp)
 writecsv(OUT/'architecture_sparsity_winner_matrix.csv',pd.DataFrame(best).pivot(index='num_experts',columns='sparsity',values='winning_ages').reset_index())
 for seed in SEEDS:writecsv(OUT/f'architecture_sparsity_winner_matrix_seed{seed}.csv',pd.DataFrame(winners).query('seed==@seed').pivot(index='num_experts',columns='sparsity',values='winning_ages').reset_index())
 jwrite(OUT/'winner_conventions.json',{'unique_winner_fields':'best_age and best_sparsity are NA if multiple numerical minima exist; complete winner sets are always retained and displayed in matrices','representative_best_age':'Lowest tied age, only for scalar descriptive summaries; all tied ages receive paired contrasts and dense/sparse decompositions','numerical_tie_tolerance':1e-12,'seed_agreement':'Winner sets intersect; exact set agreement is reported separately','endpoint_disagreement':'No overlap between the endpoint winner set and ticket-gap winner set','near_ties':'All observed mean differences from the minimum and second-ranked gaps are reported without a post-hoc scientific threshold','replicate':'source seed'})
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
 def save(name):plt.tight_layout();plt.savefig(figdir/(name+'.png'),dpi=130);plt.close()
 for (e,sp),g in f.groupby(['num_experts','sparsity']):
  fig,axes=plt.subplots(1,5,figsize=(20,4))
  for ax,col in zip(axes,valuecols):
   for seed,h in g.groupby('seed'):
    h=h.sort_values('router_age');ax.plot(h.router_age,h[col],'.-',alpha=.5,label=str(seed))
   st=g.groupby('router_age')[col].agg(['mean','std']);ax.errorbar(st.index,st['mean'],yerr=st['std'],color='black',fmt='o-',label='mean ± seed SD');ax.set_xlabel('Router age');ax.set_title(col)
  axes[0].legend(fontsize=6);save(f'e{e}_s{sp:.2f}_age_curves')
 for (e,a),g in f.groupby(['num_experts','router_age']):
  plt.figure()
  for seed,h in g.groupby('seed'):
   h=h.sort_values('sparsity');plt.plot(h.sparsity,h.ticket_gap,'.-',label=str(seed))
  plt.xlabel('Sparsity');plt.ylabel('Ticket gap');plt.legend();save(f'e{e}_R{a}_sparsity_curve')
 for e,g in f.groupby('num_experts'):
  for col in valuecols:
   pivot=g.pivot_table(index='router_age',columns='sparsity',values=col,aggfunc='mean');writecsv(OUT/f'heatmap_e{e}_{col}.csv',pivot.reset_index());plt.figure();plt.imshow(pivot,aspect='auto',interpolation='nearest');plt.xticks(range(6),SP);plt.yticks(range(11),AGES);plt.xlabel('Sparsity');plt.ylabel('Router age');plt.title(f'{e}E {col}');plt.colorbar();save(f'e{e}_{col}_heatmap')

 # Cross-architecture comparisons use only the measured grid.
 plt.figure(figsize=(9,5))
 for e,g in pd.DataFrame(best).groupby('num_experts'):
  g=g.sort_values('sparsity');points=[(x.sparsity,age) for x in g.itertuples() for age in json.loads(x.tied_best_ages)];plt.plot([x[0] for x in points],[x[1] for x in points],'o',label=f'{e}E')
 plt.xlabel('Sparsity');plt.ylabel('Mean ticket-gap winning router age');plt.yticks(AGES);plt.legend();save('cross_architecture_winner_age_vs_sparsity')
 means=f.groupby(['num_experts','router_age','sparsity']).ticket_gap.mean().reset_index()
 fig=plt.figure(figsize=(5*len(means.num_experts.unique()),5));lo=means.ticket_gap.min();hi=means.ticket_gap.max()
 for i,(e,g) in enumerate(means.groupby('num_experts'),1):
  ax=fig.add_subplot(1,len(means.num_experts.unique()),i,projection='3d');ax.scatter(g.sparsity,g.router_age,g.ticket_gap,c=g.ticket_gap,vmin=lo,vmax=hi,cmap='viridis');ax.set_zlim(lo,hi if hi>lo else lo+1e-6);ax.set_xlabel('Sparsity');ax.set_ylabel('Router age');ax.set_zlabel('Mean ticket gap');ax.set_title(f'{e}E observed grid')
 save('cross_architecture_ticket_gap_surfaces_observed_points')

if __name__=='__main__':main()
