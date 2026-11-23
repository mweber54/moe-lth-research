#!/usr/bin/env python3
"""Finalize authorized fixed-budget 32E localization. Analysis only; never launches experiments."""
import json, math, hashlib
from pathlib import Path
from collections import Counter
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_32e_router_age_localization import ROOT,OUT,SEEDS,sha256_file,write_json
from moe_lth.pruning.router_age import state_dict_hash,parameter_group
from analyze_router_age_cross_architecture_map import load_architecture,summarize,METRICS

def csv(name,rows):
 f=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);f.to_csv(OUT/name,index=False,na_rep='NA');return f

def readlines(path):return [json.loads(x) for x in path.read_text().splitlines() if x]

def exposure(path,e,source):
 rows=readlines(path);rows=[x for x in rows if x['step']>0];assert sorted(x['step'] for x in rows)==list(range(1,2501))
 proposed=Counter();accepted=Counter();active=Counter();updates=Counter();snapshots=[]
 for x in rows:
  p=x['proposed'] if source else x['expert_token_counts'];a=x['accepted'] if source else x['accepted_expert_token_counts']
  assert len(p)==len(a)==4*e and sum(p.values())==65536
  assert all(0<=a[k]<=p[k] for k in p)
  proposed.update(p);accepted.update(a);active.update({k:int(v>0) for k,v in a.items()})
  if x['optimizer_step_applied']:updates.update({k:int(v>0) for k,v in a.items()})
  if source and x['step']%125==0:
   snapshots.append({'source_step':x['step'],'router_age':x['step']/25,'proposed':sum(proposed.values()),'accepted':sum(accepted.values()),'accepted_per_expert':sum(accepted.values())/(4*e),'dropped':sum(proposed.values())-sum(accepted.values())})
 n=sum(proposed.values());a=sum(accepted.values());vals=np.array([accepted[k] for k in proposed])
 summary={'total_proposed':n,'total_accepted':a,'capacity_drops':n-a,'accepted_proposed_ratio':a/n,'mean_drop_rate':1-a/n,'accepted_per_expert':a/(4*e),'load_cv':float(vals.std()/vals.mean()),'active_attempt_steps_mean':np.mean([active[k] for k in proposed]),'active_applied_steps_mean':np.mean([updates[k] for k in proposed]),'active_applied_steps_min':min(updates[k] for k in proposed),'optimizer_attempts':2500,'optimizer_applied_steps':sum(x['optimizer_step_applied'] for x in rows)}
 detail=[{'expert':k,'proposed':proposed[k],'accepted':accepted[k],'fraction_of_accepted_assignments':accepted[k]/a,'active_attempt_steps':active[k],'active_applied_update_steps':updates[k]} for k in proposed]
 return summary,detail,snapshots

def main():
 torch.set_num_threads(4)
 decision=json.loads((OUT/'localization_decision.json').read_text());assert decision['status']=='complete'
 outcomes=pd.read_csv(OUT/'outcomes.csv');runs=pd.read_csv(OUT/'run_manifest.csv');dev=pd.read_csv(OUT/'developmental_metrics.csv')
 manifest=[];trajectory=[];sources=[];capacities=[];details=[];age_exposure=[];integrity=[];mask_checks=[]
 for seed in SEEDS:
  root=OUT/f'references/seed_{seed}'
  for p in sorted((root/'checkpoints').glob('step_*.pt')):
   ck=torch.load(p,map_location='cpu',weights_only=False);c=ck['config'];step=ck['step'];assert c['seed']==seed and c['model']['num_experts']==32 and step==int(p.stem.split('_')[1])
   assert c['training']['steps']==2500 and c['training']['batch_size']==128 and c['data']['seq_len']==128
   config_hash=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
   manifest.append({'seed':seed,'experts':32,'router_age':step/25,'source_step':step,'checkpoint_file':str(p),'checkpoint_sha256':sha256_file(p),'training_config_hash':config_hash,'data_identity':json.dumps({k:sha256_file(ROOT/c['data'][k]) for k in ('train_path','validation_path')},sort_keys=True),**{g+'_hash':state_dict_hash({n:v for n,v in ck['model'].items() if parameter_group(n)==g}) for g in ('router','expert','shared')}})
  from moe_lth.models.transformer import TinyMoELanguageModel
  from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
  final=torch.load(root/'checkpoints/step_2500.pt',map_location='cpu',weights_only=False)
  model=TinyMoELanguageModel(final['config']['model']);model.load_state_dict(final['model'])
  expected=expert_local_magnitude_masks(model,.85)
  mask=torch.load(OUT/f'masks/seed_{seed}/pruning_mask.pt',map_location='cpu',weights_only=True)
  assert set(expected)==set(mask) and all(torch.equal(expected[n],mask[n]) for n in mask)
  digest=state_dict_hash(mask);seedruns=runs[(runs.seed==seed)&(runs.kind=='sparse')]
  assert seedruns.mask_hash.eq(digest).all()
  mask_checks.append({'seed':seed,'mask_hash':digest,'target_sparsity':.85,'actual_weight_sparsity':1-sum(int(v.sum()) for v in mask.values())/sum(v.numel() for v in mask.values()),'R100_magnitude_reconstruction_matches':True,'bias_pruned':False,'reused_across_ages':True,'rewind':'R0'})
  del model,expected,final,mask
  vals=readlines(root/'logs/validation_metrics.jsonl');vm={x['step']:x['loss'] for x in vals}
  for a in range(0,101,10):trajectory.append({'seed':seed,'router_age':a,'source_step':a*25,'validation_loss':vm.get(a*25,np.nan),'R80_R100_absolute_improvement':vm[2000]-vm[2500],'R80_R100_relative_improvement':(vm[2000]-vm[2500])/vm[2000]})
  summary,detail,snaps=exposure(root/'logs/training_exposure.jsonl',32,True);sources.append({'seed':seed,'experts':32,**summary});capacities.append({'seed':seed,'kind':'source','router_age':100,'nominal_capacity_per_expert':640,**summary});details.extend({'seed':seed,'kind':'source','router_age':100,**x} for x in detail);age_exposure.extend({'seed':seed,**x} for x in snaps)
 csv('pruning_mask_integrity.csv',mask_checks)
 assert len({x['mask_hash'] for x in mask_checks})==3
 m=csv('source_checkpoint_manifest.csv',manifest)
 for seed,g in m.groupby('seed'):
  for col in ('router_hash','expert_hash','shared_hash'):assert not g[col].duplicated().any()
 csv('source_training_trajectory.csv',trajectory);csv('source_exposure_summary.csv',sources);csv('source_exposure_by_age.csv',age_exposure)
 for row in runs.to_dict('records'):
  dest=Path(row['condition_dir']);meta=json.loads((dest/'metadata.json').read_text());curve=readlines(dest/'metrics.jsonl')
  assert curve[-1]['step']==2500 and abs(curve[-1]['loss']-row['final_validation_loss'])<1e-12
  assert meta['recovery_steps']==2500 and meta['optimizer_state_entries_at_start']==0 and meta['optimizer']=='fresh_AdamW' and meta['scheduler']=='none' and meta['router_mode']=='trainable' and not meta['confidence_control']
  assert row['native_routing'] and pd.isna(row['forward_callbacks'])
  assert sha256_file(dest/'checkpoints/final_recovered.pt')==row['final_checkpoint_sha256']
  summary,detail,_=exposure(dest/'gradient_stats/gradient_stats.jsonl',32,False)
  capacities.append({'seed':row['seed'],'kind':row['kind'],'router_age':row['router_age'],'nominal_capacity_per_expert':640,**summary});details.extend({'seed':row['seed'],'kind':row['kind'],'router_age':row['router_age'],**x} for x in detail)
  integrity.append({'condition':row['condition'],'passed':True})
 for (seed,age),g in runs.groupby(['seed','router_age']):
  assert len(g)==2
  for col in ('initial_router_state_hash','shared_state_hash','training_batch_sequence_hash','validation_batch_sequence_hash'):assert g[col].nunique()==1
  d=g[g.kind=='dense'].iloc[0];s=g[g.kind=='sparse'].iloc[0];assert s.sparsity==.85 and abs(s.dense_reference_loss-d.final_validation_loss)<1e-12
 csv('capacity_summary.csv',capacities);csv('expert_exposure.csv',details)
 csv('dense_recovery_results.csv',runs[runs.kind=='dense']);csv('sparse_recovery_results.csv',runs[runs.kind=='sparse'])
 outcomes['matched_early_auc']=outcomes.sparse_early_auc-outcomes.dense_early_auc;csv('matched_ticket_results.csv',outcomes)
 summary=[]
 for age,g in outcomes.groupby('router_age'):
  row={'router_age':age,'num_seeds':3}
  for col in ('sparse_final_loss','dense_final_loss','ticket_gap','immediate_pruning_shock','sparse_early_auc','dense_early_auc','matched_early_auc'):
   row['mean_'+col]=g[col].mean();row['sd_'+col]=g[col].std(ddof=1)
  summary.append(row)
 summary=csv('age_summary.csv',summary);csv('stage_a_age_summary.csv',summary[summary.router_age.isin([0,20,40,60,80,100])])
 csv('developmental_stats_by_seed_age.csv',dev);csv('developmental_age_means.csv',dev.groupby('router_age')[METRICS].mean().reset_index())
 best=int(summary.loc[summary.mean_ticket_gap.idxmin(),'router_age']);pivot=outcomes.pivot(index='seed',columns='router_age',values='ticket_gap');contrasts=[]
 for boundary in (0,100):
  diff=pivot[best]-pivot[boundary]
  for seed,v in diff.items():contrasts.append({'candidate':best,'boundary':boundary,'seed':seed,'difference':v,'mean_paired_difference':diff.mean(),'sample_sd_paired_difference':diff.std(ddof=1),'seeds_favoring_candidate':int((diff<0).sum())})
 csv('paired_boundary_contrasts.csv',contrasts)
 local=pd.read_csv(OUT/'local_uncertainty.csv')
 for seed in SEEDS:local[f'seed_{seed}_gap_above_best']=[pivot.loc[seed,a]-pivot.loc[seed,best] for a in local.router_age]
 csv('local_uncertainty.csv',local)
 frames=[load_architecture(e) for e in (8,12,16,24,32)];frame=pd.concat(frames,ignore_index=True);cross,winners,endpoints=summarize(frame)
 csv('cross_architecture_summary.csv',cross);csv('cross_architecture_seed_winners.csv',winners);csv('cross_architecture_endpoints.csv',endpoints);csv('cross_architecture_outcomes.csv',frame)
 # Historical helper merges only persisted developmental ages. Assert no lost 32E conditions.
 assert len(frame[frame.experts==32])==len(outcomes)
 ranges=[]
 for e,g in frame.groupby('experts'):
  mean=g.groupby('router_age').ticket_gap.mean();sd=g.groupby('router_age').ticket_gap.std(ddof=1)
  ranges.append({'experts':e,'mean_curve_range':mean.max()-mean.min(),'mean_seed_sd':sd.mean(),'range_to_mean_sd':(mean.max()-mean.min())/sd.mean()})
 ranges=csv('cross_architecture_flatness.csv',ranges)
 # Preserves fixed historical feature scale; reports all optimum-pair distances without causal claims.
 from analyze_router_age_cross_architecture_map import FEATURES
 old=frame[frame.experts.isin([8,12])];scale=(old[FEATURES].max()-old[FEATURES].min()).clip(lower=1e-12);dist=[]
 import itertools
 for seed in SEEDS:
  for e1,e2 in itertools.combinations((8,12,16,24,32),2):
   a1=int(cross.set_index('experts').loc[e1,'mean_winner']);a2=int(cross.set_index('experts').loc[e2,'mean_winner'])
   a=frame[(frame.experts==e1)&(frame.seed==seed)&(frame.router_age==a1)].iloc[0];b=frame[(frame.experts==e2)&(frame.seed==seed)&(frame.router_age==a2)].iloc[0]
   dist.append({'seed':seed,'first_experts':e1,'second_experts':e2,'first_age':a1,'second_age':a2,'historical_scaled_distance':float(np.sqrt(np.mean(((a[FEATURES].to_numpy(float)-b[FEATURES].to_numpy(float))/scale.to_numpy())**2)))})
 csv('cross_architecture_developmental_distances.csv',dist)
 figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
 def save(name):
  plt.tight_layout()
  for ext in ('png','pdf'):plt.savefig(figdir/f'{name}.{ext}',dpi=160)
  plt.close()
 for metric in ('token_specialization','assignment_agreement_to_r100','js_distance_to_r100'):
  plt.figure()
  for seed,g in dev.groupby('seed'):plt.plot(g[metric],g.ticket_gap,'.',label=f'seed {seed}')
  plt.xlabel(metric);plt.ylabel('Matched ticket gap');plt.legend();save('gap_vs_'+metric)
 plt.figure()
 for seed,g in outcomes.groupby('seed'):plt.plot(g.router_age,g.matched_early_auc,'.-',label=f'seed {seed}')
 plt.xlabel('Router age (%)');plt.ylabel('Sparse minus dense AUC, steps 0–1250');plt.legend();save('matched_early_auc')
 plt.figure()
 for seed,g in pd.DataFrame(trajectory).groupby('seed'):plt.plot(g.router_age,g.validation_loss,'.-',label=f'seed {seed}')
 plt.xlabel('Source age (%)');plt.ylabel('Source validation loss');plt.legend();save('source_validation_trajectory')
 plt.figure()
 for e,g in frame.groupby('experts'):
  st=g.groupby('router_age').ticket_gap.agg(['mean','std']);plt.errorbar(st.index,st['mean'],yerr=st['std'],fmt='.-',label=f'{e}E')
 plt.xlabel('Router age (%)');plt.ylabel('Mean ticket gap ± sample seed SD');plt.legend();save('five_architecture_ticket_curves')
 plt.figure();plt.plot(cross.experts,cross.mean_winner,'o-');plt.xlabel('Expert count');plt.ylabel('Best observed mean router age (%)');save('architecture_vs_best_age')
 # Exposure comparisons: source 24E is sampled; recovery is complete.
 audit=ROOT/'results/architecture_scaling_audit';oldrec=pd.read_csv(audit/'recovery_exposure.csv');cur=pd.DataFrame(capacities);comparisons=[]
 for kind in ('dense','sparse'):
  new=cur[(cur.kind==kind)&cur.router_age.isin([0,20,100])].accepted_per_expert.mean()
  for e in (8,24):
   old=oldrec[(oldrec.experts==e)&(oldrec.kind==kind)&oldrec.router_age.isin([0,20,100])].accepted_per_expert.mean();comparisons.append({'phase':kind,'reference_experts':e,'relative_32E_exposure':new/old,'coverage':'complete recovery; common R0/R20/R100'})
 oldsource=pd.read_csv(audit/'sampled_source_exposure.csv');new_sample=[]
 for seed in SEEDS:
  u=readlines(OUT/f'references/seed_{seed}/logs/expert_usage.jsonl');new_sample.append(sum(min(x['token_count'],640) for x in u)/128)
 for e in (8,24):
  old=oldsource[(oldsource.experts==e)&(oldsource.router_age==100)].sampled_accepted_mean.mean();comparisons.append({'phase':'source','reference_experts':e,'relative_32E_exposure':np.mean(new_sample)/old,'coverage':'matched sampled batches only; complete 24E source unavailable'})
 csv('exposure_comparison.csv',comparisons)
 verdict=decision['verdict'];row=summary.set_index('router_age').loc[best];basin=decision['basin'];wins=pd.read_csv(OUT/'seed_winners.csv');d0=pivot[best]-pivot[0];d100=pivot[best]-pivot[100];both=(d0<0)&(d100<0)
 shock=int(summary.loc[summary.mean_immediate_pruning_shock.idxmin(),'router_age']);early=int(summary.loc[summary.mean_matched_early_auc.idxmin(),'router_age']);sparsebest=int(summary.loc[summary.mean_sparse_final_loss.idxmin(),'router_age'])
 monotonic=bool((np.diff(cross.sort_values('experts').mean_winner)>=0).all());target=f'R{best}' if both.all() and 0<best<100 and 'SEED-CONDITIONED' not in verdict else 'none'
 c24=cross.set_index('experts').loc[24];c32=cross.set_index('experts').loc[32];flat=ranges.set_index('experts');late=pd.DataFrame(trajectory).drop_duplicates('seed')[['seed','R80_R100_absolute_improvement','R80_R100_relative_improvement']]
 sequence='\n'.join(f'{int(x.experts)}E -> R{int(x.mean_winner)}; basin {x.basin}' for x in cross.itertuples())
 interpretation=f'''These results characterize router-age-conditioned sparse recoverability under a fixed global source-training and recovery budget.

Raw normalized router age is a fraction of 2500 source attempts, not expert developmental exposure. The experiment controls global source tokens, source attempted steps and recovery attempted steps. It does not equalize accepted tokens per expert, successful expert updates, or maturity. Per-expert source and recovery exposure decrease as expert count increases, so normalized router age should not be interpreted as equal developmental maturity across architectures.

Observed mean-winner sequence:\n{sequence}

Monotonically nondecreasing observed ages: {monotonic}; no scaling law follows. Feature distances in cross_architecture_developmental_distances.csv use the historical coordinate ranges; neither proximity nor a nearest age establishes one common developmental optimum. Existing normalization includes sampled exposure, which itself scales with expert count.

Seed-level disagreement is reported directly rather than absorbed into the architecture mean. 24E versus 32E winner spread: {c24.seed_winner_spread} versus {c32.seed_winner_spread}; mean across-age seed SD: {c24.mean_age_seed_sd:.6f} versus {c32.mean_age_seed_sd:.6f}. These are descriptive, with unequal adaptive grids and only three seeds. They do not establish a population trend in seed dependence. Mean curve ranges: {flat.loc[24,'mean_curve_range']:.6f} versus {flat.loc[32,'mean_curve_range']:.6f}; relative-to-SD measures are saved, without a post-hoc significance/equivalence threshold.

No exposure matching, longer recovery, old-architecture retraining or mechanism experiments were performed. Future mechanistic target: {target}; recommendation only, not a launched experiment.
'''
 (OUT/'cross_architecture_interpretation.md').write_text(interpretation)
 (OUT/'protocol_integrity.md').write_text('PASS\n\n32E, seeds 7/17/29, 2500 source attempts × 128 × 128 = 40,960,000 token occurrences; 2500 attempts for every paired recovery. Learned top-1 native trainable routers, four layers, d_model 256, hidden 1024, heads 4, capacity factor 1.25, dropout 0, auxiliary weight 0.1, fresh AdamW lr 0.0003/wd 0.01/clip 1, fp16 GradScaler, no scheduler/warmup. Historical PyTorch 2.7/CUDA12.8 runtime. R100 magnitude masks are expert-local across the two weight matrices, exclude biases, and retain rounded 15%; one mask per seed shared across ages. Surviving expert/shared values rewind to R0. Initial hash checks and final SHA checks execute in the runner; matched metadata/raw final-loss and complete exposure checks execute in this finalizer. No training-forward overrides, confidence controls, replay, or frozen routers. Source logging is opt-in passive post-update integer counting; CUDA invariance test passed. Stage A completed before historical paired-SD Stage B selection; Stage C omitted. R0 source validation was not evaluated by the historical trainer and is NA; evaluating it during training would change historical evaluation cadence.\n')
 header=f'''32-EXPERT ROUTER-AGE TICKET LOCALIZATION

Verdict: {verdict}

Mean winner: R{best}
Mean ticket gap ± seed SD: {row.mean_ticket_gap:.8f} ± {row.sd_ticket_gap:.8f}
Local uncertainty basin: {basin}

Seed winners:\n'''+ '\n'.join(f'- seed {int(x.seed)}: R{int(x.best_router_age)} ({x.best_ticket_gap:.8f}); second R{int(x.second_best_router_age)} ({x.second_best_ticket_gap:.8f}), difference {x.difference:.8f}' for x in wins.itertuples())+f'''

Candidate vs R0:
- mean paired difference: {d0.mean():+.8f}
- seed differences: {d0.to_dict()}

Candidate vs R100:
- mean paired difference: {d100.mean():+.8f}
- seed differences: {d100.to_dict()}

Candidate beats both boundaries in all seeds?: {bool(both.all())}; seeds not favoring both: {list(both.index[~both])}
Pruning-shock winner: R{shock}
Early-recovery winner: R{early}
Sparse-final-loss winner: R{sparsebest}
Matched-ticket-gap winner: R{best}

Late source-training improvement:\n{late.to_string(index=False)}

Per-expert exposure note:\n{pd.DataFrame(comparisons).to_string(index=False)}

Developmental-state comparison: Descriptive only; full age means and historical scaled optimum distances are persisted. No common equal-maturity optimum established.

Cross-architecture sequence:\n{sequence}

Does optimum move monotonically later with expert count?: {monotonic} descriptively; no general law established.
Does a single coarse developmental state explain the optima?: Not established by historical descriptive distances.
Does seed dependence increase with expert count?: Compare winner spread and SD below; three seeds and unequal adaptive grids do not establish a trend.
Is the current fixed-budget phenomenon still present at 32E?: {verdict}; favorable interior evidence requires the paired-boundary comparisons above.
Recommended future mechanistic target, if any: {target}

Scientific interpretation:\n{interpretation}

Limitations: n=3 independent source seeds; adaptively sampled ages; no statistical significance or equivalence claim; R100 is not a convergence criterion. Historical source exposure is sampled, new source counters complete. Gate amplitudes, capacities, exposure and successful expert updates are not matched across architectures. Sparse loss alone never selects the winner. Immediate shock and early AUC are separate historical readouts. Seed SD uses ddof=1; load CV uses population SD. Maturity assignment/JS probes use first validation batch per layer; other historical probes use first eight batches. Exposure developmental metric remains sampled proposed counts, not new accepted counters.\n'''
 qanswers=[
  ('Does 32E exhibit an intermediate router-age ticket optimum?',verdict),
  ('Which age minimizes mean ticket gap?',f'R{best}'),
  ('What is mean ± seed SD?',f'{row.mean_ticket_gap:.8f} ± {row.sd_ticket_gap:.8f}'),
  ('Does the candidate beat R0 on average?',str(bool(d0.mean()<0))),
  ('Does it beat R100 on average?',str(bool(d100.mean()<0))),
  ('Does it beat both in all three seeds?',str(bool(both.all()))),
  ('Which seeds disagree?',str(list(both.index[~both]))),
  ('What are the seed-specific winners?',str(dict(zip(wins.seed,wins.best_router_age)))),
  ('Is the optimum sharp or broad?',f'Descriptive connected local basin {basin}; no unsampled point precision.'),
  ('What is the local uncertainty basin?',str(basin)),
  ('Does immediate pruning shock identify the same age?',f'R{shock}; same={shock==best}'),
  ('Does early matched recovery identify the same age?',f'R{early}; same={early==best}'),
  ('Does absolute sparse loss identify the same age?',f'R{sparsebest}; same={sparsebest==best}'),
  ('Is the result driven by sparse loss or matched recoverability?','Winner uses matched sparse-minus-dense loss. Comparing the readouts above does not identify a causal mechanism.'),
  ('How do developmental statistics evolve?','Full historical metric age means and seed-age rows follow and are saved in developmental_age_means.csv. R100 agreement/JS are references, not proof of convergence.'),
  ('Is R100 still improving late in source training?',str(late.to_dict('records'))),
  ('How does per-expert exposure compare with 24E?',str([x for x in comparisons if x['reference_experts']==24])),
  ('Does 32E show more seed dependence than 24E?',f'Winner spread {c32.seed_winner_spread} versus {c24.seed_winner_spread}; mean seed SD {c32.mean_age_seed_sd:.6f} versus {c24.mean_age_seed_sd:.6f}. Mixed indicators are not forced into a trend claim.'),
  ('Is the mean curve flatter than 24E?',f'Mean curve range {flat.loc[32,"mean_curve_range"]:.6f} versus {flat.loc[24,"mean_curve_range"]:.6f}; range/mean-SD {flat.loc[32,"range_to_mean_sd"]:.6f} versus {flat.loc[24,"range_to_mean_sd"]:.6f}. Unequal adaptive grids limit this comparison.'),
  ('Does 32E exhibit a boundary optimum?',f'Aggregate winner at boundary: {best in (0,100)}; final hierarchy verdict {verdict}'),
  ('Is there evidence for monotonic shift with expert count?',f'Observed mean age sequence nondecreasing={monotonic}; no scaling law.'),
  ('Is there a common coarse developmental-state optimum?','Not established; historical-scaled optimum distances are descriptive and exposure differs.'),
  ('Does the fixed-budget phenomenon survive at 32E?',verdict+'; see paired boundary evidence rather than forcing an interior interpretation.'),
  ('Should 32E enter a future mechanism experiment?',target+'; recommendation only; no mechanism run launched.'),
  ('What claim is justified without equal development?',interpretation.split('\n')[0])]
 header+='\n\nRequested questions\n\n'+'\n\n'.join(f'{i}. **{q}**\n\n{a}' for i,(q,a) in enumerate(qanswers,1))
 prior=(OUT/'final_report.md').read_text();(OUT/'historical_analysis_details.md').write_text(prior)
 header+='\n\nDevelopmental age means:\n\n```text\n'+dev.groupby('router_age')[METRICS].mean().to_string()+'\n```\n\nMean curve and seed-dependence comparison:\n\n```text\n'+ranges.to_string(index=False)+'\n```\n\nAll 25 requested questions are addressed by the winner/boundary/shock/AUC fields, developmental tables, late source table, exposure comparisons, cross-architecture interpretation and limitations above. Raw values remain in CSV files.\n\n'+verdict+'\n'
 (OUT/'final_report.md').write_text(header)
 terminal=f'''32E ROUTER-AGE LOCALIZATION COMPLETE

Protocol:
- source global budget matched: YES, 40,960,000 token occurrences
- recovery global budget matched: YES, 2500 attempts each
- architecture: 32E, top-1, 4 layers, d_model256, hidden1024
- seeds: 7,17,29
- sparsity: 85% expert weights with historical rounding
- routing: native learned trainable
- protocol integrity: PASS

Mean winner: R{best}
Mean ticket gap ± SD: {row.mean_ticket_gap:.8f} ± {row.sd_ticket_gap:.8f}
Basin: {basin}

Seed winners:\n'''+ '\n'.join(f'- {int(x.seed)}: R{int(x.best_router_age)}' for x in wins.itertuples())+f'''

Candidate vs R0: {d0.mean():+.8f}; {d0.to_dict()}
Candidate vs R100: {d100.mean():+.8f}; {d100.to_dict()}
Beats both boundaries all seeds?: {bool(both.all())}

Pruning-shock winner: R{shock}
Early-recovery winner: R{early}
Sparse-loss winner: R{sparsebest}
Ticket-gap winner: R{best}

Late source development:\n{late.to_string(index=False)}
32E relative per-expert exposure:\n{pd.DataFrame(comparisons).to_string(index=False)}

Cross-architecture winners:\n{sequence}

Monotonic age scaling supported?: {monotonic} for observed ages only
Common developmental-state hypothesis supported?: Not established
Seed-conditioning trend supported?: Descriptive comparison only; not established as a trend

32E verdict:
{verdict}

Recommended mechanism target:
{target}

Path to final_report.md:
{OUT/'final_report.md'}
'''
 (OUT/'terminal_summary.txt').write_text(terminal)
 decision['extended_report_complete']=True;write_json(OUT/'localization_decision.json',decision)
 write_json(OUT/'artifact_integrity.json',{'status':'validated','run_checks':integrity,'files':{str(p.relative_to(OUT)):sha256_file(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ('.csv','.md','.json') and p.name!='artifact_integrity.json'}})
 print(terminal,flush=True)

if __name__=='__main__':
 try:main()
 except Exception as error:
  (OUT/'protocol_integrity.md').write_text('PROTOCOL INVALID\n\nFinalization/integrity gate failed: '+repr(error)+'\nInterpretation stopped pending investigation; no new runs launched.\n')
  raise
