"""Normalize historical supplemental protocols after audit.py; no training."""
from pathlib import Path
import csv,json,hashlib,datetime
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/imp_ticket_audit_v1'
def read(p):return list(csv.DictReader(open(ROOT/p)))
def write(name,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def js(p):return json.load(open(p))
raw=read('results/imp_80_85/final_recovery_metrics.csv');by={(int(r['num_experts']),int(r['seed']),float(r['sparsity'])):r for r in raw}
allrows=[]; checks=[]
for r in read('results/imp_80_85/imp_round_metrics.csv'):
 e=int(r['num_experts']);seed=int(r['seed']);base=by[e,seed,.8];dense=float(base['dense_validation_ce']);ce=float(r['final_validation_loss'])
 allrows.append(dict(architecture=f'E{e}',source_seed=seed,mask_id=r['mask_hash'],initialization_id=r['initial_sha256'],sparsity=r['sparsity'],discovery_run=base['protocol']+f'/E{e}/seed{seed}',recovery_run=r['curves_path'],protocol=base['protocol'],role='discovery_stage_not_independent_final_replicate',budget=2500,absolute_sparse_ce=ce,absolute_dense_ce=dense,sparse_minus_dense_ce=ce-dense,classification='discovery_only',mask_path=str(Path(r['trained_checkpoint']).parents[2]/f"mask_{int(r['round']):02d}.pt") if int(r['round']) else 'implicit_all_ones',initialization_path=r['original_initialization_checkpoint']))
extension_table={(int(r['num_experts']),int(r['seed']),float(r['sparsity'])):r for r in read('results/imp_extended_recovery/sparse_dense_comparison.csv')}
for (e,seed,s),base in sorted(by.items()):
 r=extension_table.get((e,seed,s))
 root=ROOT/f'results/imp_extended_recovery/e{e}/seed_{seed}'
 sp=root/f'final_s{s:.2f}/extended_summary.json';dp=root/'dense/extended_summary.json';sm=js(sp);dm=js(dp)
 match=sm['target_step']==dm['target_step'] and all(sm['trainer_record'].get(k)==dm['trainer_record'].get(k) and sm['trainer_record'].get(k) is not None for k in ('training_batch_sequence_hash','validation_batch_sequence_hash','optimizer','scheduler','precision','recovery_steps')) and sm['mask_hash']==base['mask_hash']
 ce=float(sm['extension_final_validation_cross_entropy']);dense=float(dm['extension_final_validation_cross_entropy'])
 checks.append(dict(kind='extension',source=str(sp),dense_source=str(dp),matched=match,table_gap_difference=ce-dense-float(r['sparse_minus_dense_gap_5000']) if r else '',present_in_stale_summary=r is not None))
 allrows.append(dict(architecture=f'E{e}',source_seed=seed,mask_id=base['mask_hash'],initialization_id=base['initial_sha256'],sparsity=s,discovery_run=base['protocol']+f'/E{e}/seed{seed}',recovery_run=str(sp),protocol='5000_attempts_fresh_optimizer_at2500',role='same_mask_continuation_not_independent_initialization',budget=5000,absolute_sparse_ce=ce,absolute_dense_ce=dense,sparse_minus_dense_ce=ce-dense if match else '',dense_pair_hashes_match=match,validation_endpoint_at_or_better_dense=ce<=dense if match else '',strict_original_lth_status='unestablished',mask_path=str(Path(base['trained_checkpoint']).parents[2]/f'mask_{8 if s==.8 else 9:02d}.pt'),initialization_path=base['original_initialization_checkpoint']))
for r in read('results/imp_mask_information_v2/results.csv'):
 seed=int(r['seed']);base=by[16,seed,.8];stage=js(r['stage_path']);densemeta=js(Path(base['matched_dense_checkpoint']).parent.parent/'metadata.json')
 match=all(stage.get(k)==densemeta.get(k) for k in ('training_batch_sequence_hash','validation_batch_sequence_hash','recovery_steps','optimizer','scheduler','precision'))
 ce=float(r['final_ce']);dense=float(r['historical_dense_ce'])
 allrows.append(dict(architecture='E16',source_seed=seed,mask_id=r['mask_hash'],initialization_id=base['initial_sha256'],sparsity=.8,discovery_run='original_IMP' if r['condition']=='IMP' else 'conditional_randomization/'+r['job'],recovery_run=r['curves_path'],protocol='mask_information_v2_2500',role='mask_randomization' if r['condition']!='IMP' else 'same_IMP_recovery_not_independent_source',budget=2500,absolute_sparse_ce=ce,absolute_dense_ce=dense,sparse_minus_dense_ce=ce-dense if match else '',dense_pair_hashes_match=match,historical_dense_gap_unverified_if_unmatched=ce-dense,validation_endpoint_at_or_better_dense=ce<=dense if match else '',strict_original_lth_status='unestablished',condition=r['condition'],replicate=r['replicate'],mask_path=r['mask_path'],initialization_path=base['original_initialization_checkpoint']))
# Snapshot complete common-evaluation pairs, exclude running statuses and never mutate them.
status=[];order={}
for p in sorted((ROOT/'results/imp_recovery_order_stability_v1/status').glob('*.json')):
 d=js(p);status.append(dict(path=str(p),**{k:d.get(k) for k in ('job_id','status','updated_utc','attempted_updates','applied_updates','attempt')}))
 if d.get('status')!='COMPLETE':continue
 ev=Path(d['attempt'])/'common_evaluations.jsonl'
 for line in ev.read_text().splitlines():
  v=json.loads(line)
  if int(v['step'])==5000:order[(int(v['seed']),v['order'],v['model_type'])]=(v,d)
for (seed,o,m),(v,d) in order.items():
 if m!='imp80' or (seed,o,'unpruned') not in order:continue
 dense,dd=order[seed,o,'unpruned'];base=by[32,seed,.8]
 match=all(d.get(k)==dd.get(k) for k in ('permuted_stream_hash_segment_0','permuted_stream_hash_segment_1')) and v['validation_stream_hash']==dense['validation_stream_hash']
 allrows.append(dict(architecture='E32',source_seed=seed,mask_id=base['mask_hash'],initialization_id=base['initial_sha256'],sparsity=.8,discovery_run='original_IMP/E32/seed'+str(seed),recovery_run=d['job_id'],protocol='5000_order_'+o+'_FP32_readout_reset2500',role='recovery_stream_replicate_not_independent_initialization',budget=5000,absolute_sparse_ce=v['cross_entropy'],absolute_dense_ce=dense['cross_entropy'],sparse_minus_dense_ce=v['cross_entropy']-dense['cross_entropy'] if match else '',dense_pair_hashes_match=match,validation_endpoint_at_or_better_dense=v['cross_entropy']<=dense['cross_entropy'] if match else '',strict_original_lth_status='unestablished',order=o))
write('supplemental_comparison.csv',allrows);write('supplemental_checks.csv',checks);write('order_status_snapshot.csv',status)
# Recompute mask randomization effects with seed-level t intervals.
fx=[]
for cond in ('matrix','degree'):
 g=[r for r in read('results/imp_mask_information_v2/paired_effects.csv') if r['condition']==cond];x=np.array([float(r['mean_minus_imp']) for r in g]);ci=stats.t.ppf(.975,len(x)-1)*stats.sem(x)
 fx.append(dict(property=cond+'_control_minus_IMP',n_independent_seeds=len(x),n_nested_masks=sum(int(r['n_masks']) for r in g),mean_effect=x.mean(),sd=x.std(ddof=1),ci95_low=x.mean()-ci,ci95_high=x.mean()+ci))
write('mask_randomization_summary.csv',fx)
(OUT/'snapshot.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),statuses=len(status),completed_statuses=sum(r['status']=='COMPLETE' for r in status),matched_order_pairs=sum(r['role']=='recovery_stream_replicate_not_independent_initialization' for r in allrows),supplemental_rows=len(allrows)),indent=2))
print(json.dumps(fx,indent=2));print((OUT/'snapshot.json').read_text())
canonical=[];contrasts=[]
for r in read('results/final_dataset_a/canonical/dataset_a_canonical.csv'):
 if r['router_age']!='0':continue
 ce=float(r['sparse_final_validation_loss']);dense=float(r['dense_final_validation_loss']);valid=r['condition_valid']=='True'
 canonical.append(dict(architecture='E'+r['num_experts'],source_seed=r['seed'],mask_id=r['mask_hash'],initialization_id=r['source_checkpoint_hash'],sparsity=r['sparsity'],discovery_run=r['source_artifact_path']+'/OMP_step2500',recovery_run=r['sparse_artifact_path'],protocol=r['protocol_version']+'_OMP_R0',role='canonical_OMP_R0_manifest_comparator',budget=r['recovery_attempt_budget'],absolute_sparse_ce=ce,absolute_dense_ce=dense,sparse_minus_dense_ce=ce-dense if valid else '',dense_pair_hashes_match=valid,verification_scope='canonical_manifest_not_tensor_reverified_in_this_audit',mask_path=r['mask_artifact_path'],mask_exists=Path(r['mask_artifact_path']).is_file(),initialization_path=r['source_checkpoint_path'],initialization_exists=Path(r['source_checkpoint_path']).is_file(),validation_endpoint_at_or_better_dense=ce<=dense if valid else '',strict_original_lth_status='unestablished'))
 key=(int(r['num_experts']),int(r['seed']),float(r['sparsity']))
 if key in by:
  imp=by[key];matched=imp['initial_sha256']==r['source_checkpoint_hash'] and imp['training_batch_sequence_hash']==r['data_hash'] and imp['validation_batch_sequence_hash']==r['validation_hash'] and abs(float(imp['dense_validation_ce'])-dense)<1e-9
  contrasts.append(dict(architecture='E'+r['num_experts'],source_seed=r['seed'],sparsity=r['sparsity'],IMP_mask_id=imp['mask_hash'],OMP_mask_id=r['mask_hash'],same_init_data_dense=matched,IMP_ce=imp['final_validation_loss'],OMP_ce=ce,OMP_minus_IMP=ce-float(imp['final_validation_loss']) if matched else '',confound='different_discovery_algorithms_and_compute; not independent_IMP_discovery_replicates'))
write('canonical_OMP_R0.csv',canonical);write('within_seed_IMP_vs_OMP.csv',contrasts)
# Preserve complete provenance manifests, including all nine masks per learned trajectory.
for name in ['imp_mask_manifest','checkpoint_manifest']:
 write(name+'.csv',[dict(r,verification_scope='historical_manifest; primary final masks independently rehashed in provenance.csv') for r in read('results/imp_80_85/'+name+'.csv')])
