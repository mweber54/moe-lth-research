"""Verify all 15 own-initialization/IMP variants and register 60 factorial slots."""
from af_common import *
import statistics

def main():
 torch.set_num_threads(4)
 if (OUT/'manifest.json').exists():raise RuntimeError('Preserve existing manifest')
 cohort=read(HIST/'cohort.json');registered={x['path']:x['sha256'] for x in cohort['files']};variants=[];jobs=[];files={};groups=[];decisions=[];timings=[]
 for c in cohort['conditions']:
  seed=c['seed'];variant='dense' if c['sparsity'] is None else f"imp{round(c['sparsity']*100)}";id=f'e32_seed{seed}_{variant}'
  if seed not in SEEDS or c['config']['model']['num_experts']!=32:raise RuntimeError('Cohort mismatch')
  for p in [c['initial'],c['mask'],c['boundary']]:
   if p:
    h=sha(p)
    if p not in registered or h!=registered[p]:raise RuntimeError('Source hash mismatch '+p)
    files[p]=h
  initial=load_model_from_checkpoint(c['config']['model'],c['initial'],torch.device('cpu'));masks=load_masks(c['mask']) if c['mask'] else {};apply_masks_(initial,masks)
  for r in inventory(initial,masks):groups.append(dict(variant=id,**r))
  done_path=HIST/'reruns'/c['id']/'completed.json';reason=[];done=read(done_path) if done_path.exists() else None
  attempt=Path(done['attempt']) if done else None
  if done is None:reason.append('No completed compatible full rerun')
  else:
   if sha(done['checkpoint'])!=done['sha256']:reason.append('Final hash mismatch')
   else:files[done['checkpoint']]=done['sha256']
   if done['cohort_sha256']!=sha(HIST/'cohort.json') or not done['reset_weights_unchanged']:reason.append('Cohort/reset identity mismatch')
   for segmentname in ('segment_0','segment_2500'):
    record=read(attempt/segmentname/'record.json')
    if record['optimizer_step_attempts']!=2500 or record['optimizer_state_entries_at_start']!=0 or record['scheduler']!='none' or record['precision']!='fp16' or record['router_mode']!='trainable':reason.append('Recovery protocol differs')
    if record['validation_batch_sequence_hash']!=c['canonical_pair']['validation_hash']:reason.append('Validation mismatch')
    if record['training_batch_sequence_hash']!=(c['canonical_pair']['data_hash'] if segmentname=='segment_0' else c['extension_identity']['training_batch_hash']):reason.append('Training sequence mismatch')
    files[str(attempt/segmentname/'record.json')]=sha(attempt/segmentname/'record.json');timings.append(record['wall_seconds'])
   a=read(attempt/'snapshots/step_2500.json');b=read(attempt/'snapshots/step_2500_after_reset.json')
   if a['model_hash']!=b['model_hash'] or b['optimizer_entries']!=0:reason.append('Reset changed weights or optimizer nonfresh')
   snapshots={}
   for step in STEPS:
    p=attempt/f'snapshots/step_{step:04d}.pt';meta=p.with_suffix('.json')
    if not p.exists() or not meta.exists():reason.append(f'Missing diagnostic snapshot {step}')
    elif sha(p)!=read(meta)['snapshot_sha256']:reason.append(f'Snapshot hash mismatch {step}')
    else:snapshots[str(step)]=str(p);files[str(p)]=sha(p)
   expected=state_dict_hash(initial.state_dict())
   if read(attempt/'snapshots/step_0000.json')['model_hash']!=expected:reason.append('Initialization differs')
  v=dict(id=id,seed=seed,variant=variant,sparsity=c['sparsity'] or 0.,config=c['config'],initial=c['initial'],mask=c['mask'],mask_hash=R._mask_hash(masks),initial_model_hash=state_dict_hash(initial.state_dict()),canonical_pair=c['canonical_pair'],extension_identity=c['extension_identity'],full_reuse=not reason,full_attempt=str(attempt) if attempt else None,full_snapshots=snapshots if done else {},historical_full_ce=done['final_validation_ce'] if done else None)
  variants.append(v);decisions.append(dict(variant=id,reuse=not reason,reason=reason or ['Exact own-IMP/R0/5000 two-call provenance, checkpoint/reset/data and diagnostic snapshots verified'],clipping='reconstructed historical coefficient; direct native returned norm unavailable' if not reason else 'new native coefficient'))
  for condition in CONDITIONS:
   jid=id+'_'+condition;job=dict(id=jid,variant_id=id,condition=condition,seed=seed,sparsity=v['sparsity'],reuse=condition=='full' and v['full_reuse'],depends_on=[] if condition=='full' else [id+'_full']);jobs.append(job)
   save(OUT/'status'/f'{jid}.json',dict(status='REUSED_PENDING_DIAGNOSTICS' if job['reuse'] else 'PENDING',updated=time.time()))
  print('Audited',id,'full reuse=',not reason,flush=True)
 table(OUT/'parameter_groups.csv',groups);save(OUT/'reuse_decisions.json',decisions);save(OUT/'input_hashes.json',files)
 save(OUT/'manifest.json',dict(variants=variants,jobs=jobs,registration_sha256=sha(OUT/'registration.json'),protocol_sha256=sha(OUT/'protocol.md'),seeds=list(SEEDS),conditions=list(CONDITIONS),new_recoveries=sum(not j['reuse'] for j in jobs),reused_full=sum(j['reuse'] for j in jobs)))
 new=sum(not j['reuse'] for j in jobs);save(OUT/'resource_estimate.json',dict(new_recoveries=new,reused_conditions=60-new,total_registered=60,segments_each=2,attempts_each=5000,median_measured_segment_seconds=statistics.median(timings),estimated_gpu_hours=2*new*statistics.median(timings)*1.15/3600,diagnostic_overhead_allowance=.15,retained_bytes_estimate=new*(5*274000000+2*274000000+2*825000000),rolling_active_resume_bytes_estimate=2*825000000,suggested_free_storage_gib=220,free_storage_bytes=__import__('shutil').disk_usage(OUT).free,limitations='Same-architecture historical instrumented throughput; contention and frozen-group checks can change timing. Model snapshots plus segment-end model/optimizer/scaler/RNG states; no SVD outputs.'))
 print('REGISTERED',new,'new',60-new,'reused',flush=True)
if __name__=='__main__':main()
