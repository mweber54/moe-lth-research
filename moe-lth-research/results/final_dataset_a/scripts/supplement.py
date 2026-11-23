"""Additional existing-data diagnostics, appended transparently to the report."""
from pathlib import Path
import json,datetime,hashlib
import numpy as np,pandas as pd
from build import OUT,DATA,ROOT,csv,atomic,jsave

def augment():
 from analyze import mdtable
 f=pd.read_csv(OUT/'canonical/dataset_a_canonical.csv');m=pd.read_csv(OUT/'summaries/dataset_a_master_results.csv');dev=pd.read_csv(OUT/'canonical/dataset_a_router_development.csv');audit=pd.read_csv(OUT/'audits/dataset_a_completion_audit.csv');src=pd.read_csv(OUT/'canonical/dataset_a_sources.csv');old=DATA/'interim_analysis/20260921T133833582668Z'
 inp=json.loads((OUT/'dataset_a_input_hashes.json').read_text())
 for name in ['interim_developmental_distances.csv','interim_development_normalization.csv','interim_development_nearest_observed_states.csv','interim_architecture_sparsity_winner_map.csv','interim_analysis_report.md']:
  p=old/name;b=p.read_bytes();dest=OUT/'inputs/september_21'/name;atomic(dest,b);inp[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'mtime_ns':p.stat().st_mtime_ns}
 jsave(OUT/'dataset_a_input_hashes.json',inp)
 keys=['router_entropy','router_margin','router_load_cv','router_token_id_specialization','router_assignment_agreement_to_r100','router_js_distance_to_r100'];dv=[]
 for r in m.itertuples():
  if not isinstance(r.mean_winning_ages,str):continue
  for a in r.mean_winning_ages.split(';'):
   age=int(a[1:]);g=dev[(dev.num_experts==r.num_experts)&(dev.router_age==age)]
   for d in g.to_dict('records'):dv.append({'architecture':r.architecture,'sparsity':r.sparsity,'complete_five_seed_curve':r.complete_five_seed_curve,**{k:d[k] for k in ['num_experts','seed','router_age']+keys},'development_artifact':d.get('development_artifact')})
 dv=csv(OUT/'summaries/dataset_a_development_at_mean_winners.csv',dv)
 changes=[]
 for e,g in dev.groupby('num_experts'):
  for k in keys:
   p=g.pivot(index='seed',columns='router_age',values=k);delta=p[100]-p[0];changes.append({'num_experts':e,'metric':k,'n_paired_seeds':delta.count(),'r100_minus_r0_mean':delta.mean(),'r100_minus_r0_sample_sd':delta.std(ddof=1)})
 changes=csv(OUT/'summaries/dataset_a_development_boundary_changes.csv',changes)
 distances=pd.read_csv(old/'interim_developmental_distances.csv');same=distances[distances.first_age==distances.second_age];csv(OUT/'summaries/dataset_a_same_age_development_distances.csv',same)
 exp=[]
 base=src[src.num_experts==8].set_index('seed').mean_accepted_per_expert
 for e,g in src.groupby('num_experts'):
  ratios=g.set_index('seed').mean_accepted_per_expert/base
  exp.append({'num_experts':e,'n_paired_source_seeds':ratios.count(),'mean_paired_exposure_ratio_vs_8e':ratios.mean(),'sd_paired_exposure_ratio_vs_8e':ratios.std(ddof=1)})
 exp=csv(OUT/'summaries/dataset_a_source_exposure_ratios.csv',exp)
 historical=[]
 for name,e in [('expert_count_replication/ticket_gap_seed_level.csv',None),('router_development_12expert_retry4_20260902/ticket_gap_seed_level.csv',12),('16e_router_age_localization/outcomes.csv',16),('24e_router_age_localization/outcomes.csv',24),('32e_router_age_localization/matched_ticket_results.csv',32)]:
  p=ROOT/'results'/name
  if not p.exists():continue
  for z in pd.read_csv(p).to_dict('records'):
   ee=int(z.get('expert_count',z.get('experts',e)));a=int(z.get('router_age',z.get('router_age_percent')));sp=float(z.get('sparsity',.85));seed=int(z['seed']);g=f[(f.num_experts==ee)&(f.seed==seed)&(f.router_age==a)&(f.sparsity==sp)]
   if len(g)!=1:continue
   r=g.iloc[0];sl=z.get('sparse_final_loss',z.get('final_validation_loss'));dl=z.get('dense_final_loss',z.get('matched_dense_final_loss'));match=abs(sl-r.sparse_final_validation_loss)<=1e-10 and abs(dl-r.dense_final_validation_loss)<=1e-10
   historical.append({'condition_id':r.condition_id,'historical_table':str(p),'stored_gap':z['ticket_gap'],'recomputed_gap':r.ticket_gap,'absolute_difference':abs(z['ticket_gap']-r.ticket_gap),'sparse_endpoint_matches':abs(sl-r.sparse_final_validation_loss)<=1e-10,'dense_endpoint_matches':abs(dl-r.dense_final_validation_loss)<=1e-10,'comparison_status':'MATCH' if match and abs(z['ticket_gap']-r.ticket_gap)<=1e-10 else 'MISMATCH' if match else 'DIFFERENT_ENDPOINTS_NOT_ARITHMETIC_FAILURE'})
 csv(OUT/'audits/dataset_a_historical_gap_candidates.csv',historical)
 # Classify stale statuses without treating status text as artifact validation.
 inventory=[]
 for r in audit.itertuples():
  if r.job_type=='source':continue
  p=Path(r.artifact_path) if isinstance(r.artifact_path,str) else Path('/nonexistent');curve=p/'metrics.jsonl';last=None;mtime=None
  if curve.exists():
   lines=curve.read_text().splitlines();last=json.loads(lines[-1]).get('step') if lines else None;mtime=datetime.datetime.fromtimestamp(curve.stat().st_mtime,datetime.timezone.utc).isoformat()
  assessment='COMPLETE_ARTIFACT' if r.condition_valid else 'MISSING_ARTIFACT' if not r.artifact_exists else 'INCOMPLETE_OR_INVALID_ARTIFACT'
  if r.manifest_status=='RUNNING':assessment='RUNNING_STATUS_WITH_COMPLETE_ARTIFACT' if r.condition_valid else 'RUNNING_STATUS_WITHOUT_VALID_ENDPOINT; remote liveness unverified'
  if r.manifest_status=='PENDING' and r.condition_valid:assessment='STALE_PENDING_COMPLETE_ARTIFACT'
  inventory.append({'condition_id':r.condition_id,'architecture':r.architecture,'seed':r.seed,'router_age':r.router_age,'sparsity':r.sparsity,'manifest_status':r.manifest_status,'artifact_assessment':assessment,'last_observed_validation_step':last,'raw_curve_mtime_utc':mtime})
 csv(OUT/'audits/dataset_a_manifest_reconciliation.csv',inventory)
 diagnostics=[]
 for kind in ['dense','sparse']:
  for r in f.drop_duplicates(kind+'_artifact_path').to_dict('records'):
   p=Path(r[kind+'_artifact_path']);row={'condition_id':r['dense_condition_id'] if kind=='dense' else r['condition_id'],'model_type':kind,'artifact_path':str(p)}
   for filename,prefix in [('metadata.json','metadata_'),('summary.json','summary_')]:
    for k,v in json.loads((p/filename).read_text()).items():row[prefix+k]=json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v
   diagnostics.append(row)
 csv(OUT/'canonical/dataset_a_recovery_diagnostics.csv',diagnostics)
 # Observed within-seed update accounting must agree with metadata if both exist.
 comparisons=[]
 for kind in ['dense','sparse']:
  for r in f.drop_duplicates(kind+'_artifact_path').itertuples():
   d=r._asdict();logged=d.get(kind+'_applied_steps');meta=d.get(kind+'_optimizer_steps_applied');comparisons.append({'condition_id':d['dense_condition_id'] if kind=='dense' else d['condition_id'],'model_type':kind,'logged_applied_updates':logged,'metadata_applied_updates':meta,'comparison_available':pd.notna(logged) and pd.notna(meta),'match':abs(logged-meta)<.5 if pd.notna(logged) and pd.notna(meta) else None})
 update=csv(OUT/'audits/dataset_a_optimizer_update_audit.csv',comparisons)
 mismatches=update[update.comparison_available & ~update['match'].fillna(True).astype(bool)]
 if len(mismatches):
  status=json.loads((OUT/'dataset_a_finalization_status.json').read_text());status.update(status='BLOCKED',frozen=False);status['failed_tests'].append('raw_optimizer_update_accounting');jsave(OUT/'dataset_a_finalization_status.json',status)
  marker=OUT/'DATASET_A_FROZEN.json'
  if marker.exists():marker.unlink()
  raise AssertionError('Optimizer-update accounting mismatches require report regeneration')
 comp=pd.read_csv(OUT/'summaries/dataset_a_interim_comparison.csv');special=comp[comp.num_experts.isin([24,32])][['architecture','sparsity','september_21_winning_ages','mean_winning_ages','complete_five_seed_curve','min_seeds_per_age','max_seeds_per_age']]
 addon='''\n## Additional measured diagnostics\n\nOBSERVATION: the current 24E/32E provisional winners and coverage are explicitly compared below. Neither changing nor unchanged available-seed winners establish completion.\n\n'''+mdtable(special)+'''\n\nOBSERVATION: accepted source exposure ratios are paired to 8E within the same seed. Missing source counters are not replaced with sampled proposed exposure.\n\n'''+mdtable(exp)+'\n\nOBSERVATION: measured R100−R0 source developmental changes (paired source seeds only):\n\n'+mdtable(changes)+f'''\n\nAt mean-winning ages, source entropy is measured in {dv.router_entropy.notna().sum()}/{len(dv)} architecture/sparsity/seed rows. The remaining winner-linked developmental values stay NA. This prevents a complete developmental characterization of favorable regions. `dataset_a_development_at_mean_winners.csv` retains every selected winning age and available source probe. These duplicated architecture/sparsity links are not independent developmental observations.\n\nThe preserved September 21 normalization and pairwise-distance tables allow reproducible state comparisons without redefining the metric. Same-age cross-architecture measured-state distance has median {same.historical_range_normalized_rms.median():.5f} and maximum {same.historical_range_normalized_rms.max():.5f} over {len(same)} available pairs; these pairs are descriptive, not independent replicates. Normalized chronological age is therefore not an identity of measured developmental state. No interpolation or extrapolation is used.\n\nRaw gradient-log applied-update counts matched metadata for {update.comparison_available.sum()} distinct recoveries with both available; {len(update)-update.comparison_available.sum()} lack an available comparison. See the update audit for AMP accounting and the manifest reconciliation for the final observed step of incomplete runs.\n'''
 p=OUT/'reports/dataset_a_final_analysis.md';atomic(p,p.read_text()+addon)
if __name__=='__main__':augment()
