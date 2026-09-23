"""Read-only historical audit gate before any Dataset A training."""
from common import *
import yaml,itertools,collections,time
import pandas as pd
from moe_lth.models.transformer import TinyMoELanguageModel
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks

def main():
 torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'audit_complete.json').exists():raise RuntimeError('Audit already frozen; use existing manifest instead of overwriting')
 # Search every source config, not just preferred paths, and all seed-bearing metadata.
 source_paths=sorted((ROOT/'results').rglob('resolved_config.yaml'));seed_hits=[]
 for p in list((ROOT/'configs').rglob('*.yaml'))+source_paths+list((ROOT/'results').rglob('metadata.json')):
  try:d=yaml.safe_load(p.read_text()) if p.suffix=='.yaml' else jread(p)
  except Exception:continue
  if isinstance(d,dict) and d.get('seed') in (43,61):seed_hits.append(str(p))
 assert not seed_hits,seed_hits
 jwrite(OUT/'new_seed_audit.json',{'chosen_new_seeds':[43,61],'searched_source_configs':len(source_paths),'seed_hits':seed_hits,'search_scope':'all configs/*.yaml, all results resolved_config.yaml and metadata.json, including ignored files','reason':'preferred defaults unused; never chosen from outcomes'})
 source_inventory=[];source_selection={};checkpoint_info={};masks={};base_hashes={};mask_rows=[];jobs=[]
 for p in source_paths:
  try:c=yaml.safe_load(p.read_text());e=c['model']['num_experts'];seed=c['seed'];err=protocol_errors(c,e,seed) if e in ES and seed in SEEDS else ['outside canonical E/seed']
  except Exception as ex:source_inventory.append({'path':str(p.parent),'valid':False,'reason':repr(ex)});continue
  steps=sorted(int(x.stem.split('_')[1]) for x in (p.parent/'checkpoints').glob('step_*.pt'))
  source_inventory.append({'path':str(p.parent),'experts':e,'seed':seed,'valid':not err,'reason':';'.join(err),'available_steps':json.dumps(steps),'complete_age_grid':all(a*25 in steps for a in AGES)})
 for e,seed in itertools.product(ES,SEEDS):
  root=primary_ref(e,seed);valid=False;complete=False
  if root and root.exists():
   c=yaml.safe_load((root/'resolved_config.yaml').read_text());assert not protocol_errors(c,e,seed)
   initial=load(root/'checkpoints/step_0.pt');final=load(root/'checkpoints/step_2500.pt');assert not protocol_errors(final['config'],e,seed)
   valid=True;complete=all((root/f'checkpoints/step_{a*25}.pt').exists() for a in AGES)
   # Exact identities at available ages; collect same-trajectory alternate checkpoints only by R0/R100 identity.
   start_hash=state_dict_hash(initial['model']);end_hash=state_dict_hash(final['model'])
   paths=[root]
   for item in source_inventory:
    alt=Path(item['path'])
    if alt==root or not item.get('valid') or item.get('experts')!=e or item.get('seed')!=seed:continue
    aa=alt/'checkpoints/step_0.pt';zz=alt/'checkpoints/step_2500.pt'
    if aa.exists() and zz.exists() and state_dict_hash(load(aa)['model'])==start_hash and state_dict_hash(load(zz)['model'])==end_hash:paths.append(alt)
   for parent in paths:
    for a in AGES:
     p=parent/f'checkpoints/step_{a*25}.pt'
     if not p.exists():continue
     ck=load(p);assert ck['step']==a*25 and not protocol_errors(ck['config'],e,seed)
     entry={'path':str(p),'sha256':sha(p),'model_hash':state_dict_hash(ck['model']),**{g+'_hash':state_dict_hash(group(ck['model'],g)) for g in ('router','expert','shared')}}
     if (e,seed,a) in checkpoint_info:assert checkpoint_info[e,seed,a]['model_hash']==entry['model_hash']
     else:checkpoint_info[e,seed,a]=entry
   base_hashes[e,seed]={'shared':state_dict_hash(group(initial['model'],'shared')),'dense_expert':state_dict_hash(group(initial['model'],'expert')),'initial_model':start_hash,'final_model':end_hash}
   model=TinyMoELanguageModel(c['model']);model.load_state_dict(final['model'])
   for sp in SP:
    mask=expert_local_magnitude_masks(model,sp);digest=state_dict_hash(mask);dest=OUT/f'masks/e{e}/seed_{seed}/s{sp:.2f}.pt';dest.parent.mkdir(parents=True,exist_ok=True);torch.save(mask,dest)
    rewound={n:v*mask[n] if n in mask else v for n,v in initial['model'].items()};masks[e,seed,sp]={'hash':digest,'expert_hash':state_dict_hash(group(rewound,'expert')),'path':str(dest)}
    keep=sum(int(v.sum()) for v in mask.values());total=sum(v.numel() for v in mask.values());mask_rows.append({'experts':e,'seed':seed,'target_sparsity':sp,'actual_sparsity':1-keep/total,'keep_count':keep,'pruned_count':total-keep,'mask_hash':digest,'path':str(dest),'source_checkpoint':str(root/'checkpoints/step_2500.pt'),'source_age':100,'bias_handling':'excluded','scope':'expert_local','method':'magnitude','rewind':'R0'})
   del model,initial,final
  source_selection[e,seed]={'historical_path':str(root) if root else None,'source_valid':valid,'complete':complete,'run_path':str(root) if complete else str(OUT/f'sources/e{e}/seed_{seed}')}
  jobs.append({'condition_id':cid('source',e,seed),'architecture':e,'seed':seed,'router_age':None,'sparsity':None,'job_type':'source','status':'REUSED' if complete else 'PENDING','attempt_count':0,'artifact_path':source_selection[e,seed]['run_path'],'last_error':''})
  print('SOURCE AUDITED',e,seed,'complete',complete,flush=True)
 writecsv(OUT/'source_inventory.csv',source_inventory);writecsv(OUT/'mask_manifest.csv',mask_rows)
 jwrite(OUT/'historical_source_identities.json',{'sources':{f'{e}:{s}':x for (e,s),x in source_selection.items()},'checkpoints':{f'{e}:{s}:{a}':x for (e,s,a),x in checkpoint_info.items()},'base_hashes':{f'{e}:{s}':x for (e,s),x in base_hashes.items()},'masks':{f'{e}:{s}:{sp:.2f}':x for (e,s,sp),x in masks.items()}})
 # Materialize only data, with the historical generator, to authenticate recovery schedule hashes.
 from moe_lth.data import build_dataloaders
 from moe_lth.experiments import run_router_age_recovery as rec
 datahashes={}
 for seed in SEEDS:
  c=config(8,seed);tr,va=build_dataloaders(c['data'],128,seed);tb=rec._materialize_batches(tr,2500);vb=rec._materialize_validation_batches(va,32)
  datahashes[seed]={'train':rec._batch_sequence_hash(tb),'validation':rec._batch_sequence_hash(vb)};del tb,vb,tr,va
 jwrite(OUT/'canonical_data_hashes.json',datahashes)
 candidates=[];chosen={};meta_paths=sorted((ROOT/'results').rglob('metadata.json'))
 for i,p in enumerate(meta_paths):
  if OUT in p.parents:continue
  d=jread(p)
  if not all(k in d for k in ('seed','router_age_percent','sparsity','recovery_steps','router_checkpoint')):continue
  seed=d['seed'];a=d['router_age_percent'];sp=float(d['sparsity']);dest=p.parent;ckp=dest/'checkpoints/final_recovered.pt';err=[];e=None
  if seed not in SEEDS or a not in AGES or sp not in (0.,)+SP:continue
  if not ckp.exists():err.append('missing final checkpoint')
  else:
   ck=load(ckp);c=ck.get('config',{});e=c.get('model',{}).get('num_experts');err+=protocol_errors(c,e,seed) if e in ES else ['noncanonical architecture'];
   if ck.get('step')!=2500:err.append('final step')
  if e not in ES:candidates.append({'path':str(dest),'experts':e,'seed':seed,'age':a,'sparsity':sp,'valid':False,'reason':';'.join(err)});continue
  # Reject interventions; legacy native rows lacking router_mode must demonstrate a changing learned router.
  for k,v in {'recovery_steps':2500,'optimizer':'fresh_AdamW','optimizer_state_entries_at_start':0,'scheduler':'none','precision':'fp16','recovery_eval_interval':50,'pruning_method':'expert_local_magnitude','confidence_control':False,'integrity_checks_passed':True}.items():
   if d.get(k)!=v:err.append(k)
  if d.get('router_mode','trainable')!='trainable' or d.get('temperature',1)!=1:err.append('non-native router mode')
  if any(x in str(dest) for x in ('gate_amplitude','gate_replay','optimization_geometry','coadaptation_mechanism','fixed_schedule','amplitude_pattern')):err.append('intervention family')
  if d.get('initial_router_state_hash')==d.get('final_router_state_hash'):err.append('unchanged/frozen router')
  info=checkpoint_info.get((e,seed,a));base=base_hashes.get((e,seed));mask=masks.get((e,seed,sp)) if sp else None
  if not info:err.append('source age identity unavailable')
  elif d.get('initial_router_state_hash')!=info['router_hash'] or d.get('router_step')!=a*25:err.append('source router mismatch')
  if base:
   if d.get('shared_state_hash')!=base['shared']:err.append('shared rewind mismatch')
   expected=mask['expert_hash'] if mask else base['dense_expert']
   if d.get('expert_state_hash')!=expected:err.append('expert rewind mismatch')
  else:err.append('source identity unavailable')
  if sp and (not mask or d.get('mask_hash')!=mask['hash']):err.append('mask mismatch')
  if d.get('training_batch_sequence_hash')!=datahashes[seed]['train'] or d.get('validation_batch_sequence_hash')!=datahashes[seed]['validation']:err.append('data sequence mismatch')
  curve=dest/'metrics.jsonl';summary=dest/'summary.json'
  if not curve.exists() or not summary.exists():err.append('missing raw metrics/summary')
  else:
   rows=[json.loads(x) for x in curve.read_text().splitlines()];summ=jread(summary)
   if not rows or rows[-1]['step']!=2500 or abs(rows[-1]['loss']-summ['final_validation_loss'])>1e-10:err.append('raw loss mismatch')
  if not err:
   if sp:
    mm=torch.load(mask['path'],map_location='cpu',weights_only=True)
    if any(bool(ck['model'][n][~v.bool()].any()) for n,v in mm.items()):err.append('nonzero pruned weight')
   validpath=dest/'validated_result.json'
   filehash=sha(ckp)
   if validpath.exists() and jread(validpath).get('final_checkpoint_sha256',filehash)!=filehash:err.append('final hash mismatch')
  item={'path':str(dest),'experts':e,'seed':seed,'age':a,'sparsity':sp,'valid':not err,'reason':';'.join(err),'final_checkpoint_hash':filehash if not err else None}
  candidates.append(item)
  if not err:
   key=(e,seed,a,sp)
   # Prefer original primary localization over reproducibility repeats, then stable path order.
   priority=(('seed7_reproducibility' in str(dest)),('expert_count_replication' not in str(dest) and '_router_age_localization' not in str(dest)),str(dest))
   if key not in chosen or priority<chosen[key][0]:chosen[key]=(priority,str(dest))
  if i%25==0:print('RECOVERY CANDIDATES',i,'/',len(meta_paths),'valid',sum(x['valid'] for x in candidates),flush=True)
 writecsv(OUT/'historical_recovery_audit.csv',candidates)
 grid=[]
 for e,seed,a in itertools.product(ES,SEEDS,AGES):
  source=source_selection[e,seed];dense=chosen.get((e,seed,a,0.),(None,None))[1]
  jobs.append({'condition_id':cid('dense',e,seed,a),'architecture':e,'seed':seed,'router_age':a,'sparsity':0.,'job_type':'dense','status':'REUSED' if dense else 'PENDING','attempt_count':0,'artifact_path':dense or str(OUT/f'recoveries/e{e}/seed_{seed}/R{a}/dense'),'last_error':''})
  for sp in SP:
   sparse=chosen.get((e,seed,a,sp),(None,None))[1]
   jobs.append({'condition_id':cid('sparse',e,seed,a,sp),'architecture':e,'seed':seed,'router_age':a,'sparsity':sp,'job_type':'sparse','status':'REUSED' if sparse else 'PENDING','attempt_count':0,'artifact_path':sparse or str(OUT/f'recoveries/e{e}/seed_{seed}/R{a}/s{sp:.2f}'),'last_error':''})
   grid.append({'architecture':e,'seed':seed,'router_age':a,'sparsity':sp,'source_exists':bool(source['historical_path']),'source_valid':source['source_valid'],'source_complete_age_grid':source['complete'],'dense_exists':any(x['experts']==e and x['seed']==seed and x['age']==a and x['sparsity']==0 for x in candidates),'dense_valid':bool(dense),'sparse_exists':any(x['experts']==e and x['seed']==seed and x['age']==a and x['sparsity']==sp for x in candidates),'sparse_valid':bool(sparse),'needs_source_training':not source['complete'],'needs_dense_recovery':not dense,'needs_sparse_recovery':not sparse,'reuse_source_path':source['historical_path'],'reuse_dense_path':dense,'reuse_sparse_path':sparse,'invalid_reason':'missing source checkpoints; one reconstruction and overlap-hash gate required' if source['source_valid'] and not source['complete'] else ''})
 writecsv(OUT/'canonical_grid_audit.csv',grid);writecsv(OUT/'run_manifest.csv',jobs)
 counts=collections.Counter((x['job_type'],x['status']) for x in jobs)
 jwrite(OUT/'audit_complete.json',{'status':'COMPLETE','completed_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'grid_rows':len(grid),'seed_check_passed':True,'counts':{str(k):v for k,v in counts.items()},'grid_sha256':sha(OUT/'canonical_grid_audit.csv'),'training_started':False})
 print('AUDIT COMPLETE',counts,flush=True)

if __name__=='__main__':main()
