from common import *
from moe_lth.pruning.masks import load_masks
from moe_lth.pruning.router_age import parameter_group
manifest=rows(OLD/'variant_manifest.csv');reg=read(OLD/'registration.json');orders=read(OLD/'recovery_order_manifest.json');stream=read(OLD/'canonical_stream_manifest.json')
source=[];orderrows=[];out=[];trainability=[];parameter_checks=[]
for file,h in reg['sealed_files'].items():
 if '/src/' in file or '/scripts/' in file:
  actual=sha(file);source.append(dict(path=file,sealed_sha256=h,actual_sha256=actual,matches=actual==h))
write('sealed_source_audit.csv',source)
assert all(r['matches'] for r in source)
for seed in SEEDS:
 config=stream['seeds'][str(seed)]['config'];p.native.recovery.seed_everything(seed)
 train,val=p.native._source_stream(seed,config)
 th=p.native.recovery._batch_sequence_hash(train);vh=p.native.recovery._batch_sequence_hash(val)
 assert th==stream['seeds'][str(seed)]['training_batch_sequence_hash'];assert vh==stream['seeds'][str(seed)]['validation_batch_sequence_hash']
 expected={}
 for order in ('A','B'):
  expected[order]={}
  for i in (0,1):
   info=orders['segments'][order][f'segment_{i}'];path=OLD/info['path'];arr=np.load(path);sh=sha(path)
   assert sh==info['sha256'];assert np.array_equal(np.sort(arr),np.arange(i*2500,(i+1)*2500))
   batchhash=p.native.recovery._batch_sequence_hash([train[int(j)] for j in arr]);expected[order][i]=batchhash
   orderrows.append(dict(seed=seed,order=order,segment=i,path=str(path),array_file_sha256=sh,array_matches=True,reconstructed_training_hash=th,reconstructed_permuted_segment_hash=batchhash,validation_hash=vh,target_tokens=sum(y.numel() for _,y in val)))
 with p.native._preserved_rng():model=p.native.TinyMoELanguageModel(config['model'])
 for group in ('expert','router','shared'):
  pars=[v for n,v in model.named_parameters() if parameter_group(n)==group]
  trainability.append(dict(seed=seed,group=group,parameter_count=sum(v.numel() for v in pars),all_requires_grad=all(v.requires_grad for v in pars),optimizer_selection='all model parameters with requires_grad; source-sealed trainer'))
 del model
 for r in [r for r in manifest if int(r['seed'])==seed]:
  if r['reused']=='True':attempt=Path(r['reused_attempt'])
  else:attempt=Path(read(OLD/r['status_path'])['attempt'])
  snap=attempt/'snapshots/step_0000.pt';payload=torch.load(snap,map_location='cpu',weights_only=False,mmap=True)
  starting_hash=p.native._tensor_state_hash(payload['model']);cfg=payload['config'];ev=[json.loads(l) for l in (attempt/'common_evaluations.jsonl').read_text().splitlines()]
  metadata=[read(attempt/s/'metadata.json') for s in ('segment_0','segment_2500')]
  flags=dict(config_model_match=cfg['model']==config['model'],config_data_match=cfg['data']==config['data'],learning_rate_match=cfg['training']['learning_rate']==config['training']['learning_rate'],weight_decay_match=cfg['training']['weight_decay']==config['training']['weight_decay'],clipping_match=cfg['training']['grad_clip']==config['training']['grad_clip'],precision_match=all(m['precision']==config['training']['precision'] for m in metadata),optimizer_match=all(m['optimizer']=='fresh_AdamW' and m['optimizer_state_entries_at_start']==0 for m in metadata),scheduler_match=all(m['scheduler']=='none' for m in metadata),router_trainable=all(m['router_mode']=='trainable' and m['trainable_router_gradient_observed'] for m in metadata),data_order_match=all(metadata[i]['training_batch_sequence_hash']==expected[r['order']][i] for i in (0,1)),eval_hash_match=all(e['validation_stream_hash']==vh and e['target_tokens']==sum(y.numel() for _,y in val) and e['precision']=='FP32' and e['tf32']==False for e in ev),step_schedule_match=tuple(e['step'] for e in ev)==STEPS,starting_state_hash_match=starting_hash==r['initial_model_state_hash'],mask_hash_match=all(m['mask_hash']==r['mask_sha256'] for m in metadata),nominal_sparsity_match=all(m['sparsity']==.8 for m in metadata),reset_model_preserved=read(attempt/'reset_boundary.json')['model_parameters_unchanged'])
  assert all(flags.values()),(r['job_id'],flags)
  placement=r['placement'];number=0 if placement=='original' else (2 if r['family']=='imp' else 1)
  out.append(dict(job_id=r['job_id'],seed=seed,order=r['order'],variant=r['variant'],family=r['family'],placement=placement,realization=r['realization'],attempt=str(attempt),**flags,protocol_compatible=True,legacy_draws_per_mask=number,required_draws=3,approved_for_reuse=placement=='original',disposition='reuse original' if placement=='original' else 'rerun all full-shuffle draws per user instruction'))
  print(r['job_id'],'protocol compatible; disposition',out[-1]['disposition'],flush=True)
 write('recovery_compatibility.csv',out);write('reconstructed_data_orders.csv',orderrows);write('parameter_trainability.csv',trainability)
 del train,val
write('compatibility_status.csv',[dict(status='complete',runs_checked=len(out),original_runs_compatible=sum(r['placement']=='original' for r in out),full_shuffle_runs_not_reused=sum(r['placement']=='shuffled' for r in out),unresolved_protocol_mismatches=0,draw_mismatch_resolution='full-shuffle reruns explicitly accepted by user; Phase2 not launched')])
