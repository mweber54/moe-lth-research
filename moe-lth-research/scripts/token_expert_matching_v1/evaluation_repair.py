"""Post hoc evaluation only. Never invokes training or changes historical artifacts."""
import sys, copy, statistics, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'imp_adaptation_factorial_v1'))
from af_common import *
from af_engine import batches
import replay
D=ROOT/'results/token_expert_matching_v1'; O=D/'evaluation_repair_v1'
SEEDS=list(range(91001,91011)); NAMESPACE='evaluation_repair_v1/validation-only'

def summarize(xs):
 return dict(mean=statistics.mean(xs),sd=statistics.stdev(xs) if len(xs)>1 else 0.,minimum=min(xs),maximum=max(xs))

def shuffled(payload,rep,batch):
 q=dict(payload);q['layers']=[replay.permute(x,SEEDS[rep],NAMESPACE,batch) for x in payload['layers']]
 for a,b in zip(payload['layers'],q['layers']):
  assert torch.equal(b['token_positions'].sort().values,torch.arange(len(b['ids'])))
  for e in range(len(a['order'])):
   for accepted in (False,True):
    ma=a['ids']==e;mb=b['ids']==e
    if accepted:ma &= a['accepted'];mb &= b['accepted']
    assert int(ma.sum())==int(mb.sum())
    assert torch.equal(a['gates'][ma].sort().values,b['gates'][mb].sort().values)
 return q

def main():
 O.mkdir(exist_ok=False)
 config=dict(post_hoc=True,checkpoint_step=5000,checkpoint_local_step=2500,training_trace_step=None,validation_permutation_seeds=SEEDS,permutation_namespace=NAMESPACE,permutation_key=['validation_replicate_seed','namespace','validation_batch_index','layer_index'],source_seed=17,training_permutation_id=101,precision='FP32; TF32 disabled; eval(); inference_mode()',token_weighting='task CE mean times actual target-token count; sum divided by total tokens',sd='sample SD over evaluation permutations; no training replication',logit_atol=0.,logit_rtol=0.,historical_artifacts_preserved=True)
 save(O/'analysis_config.json',config)
 torch.set_num_threads(4);R.configure_device(torch.device('cuda'));torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 manifest=read(OUT/'manifest.json');rows=[];provenance=[];checks=[];mapping={};start=time.monotonic()
 for variant in ('e32_seed17_dense','e32_seed17_imp80'):
  v=next(x for x in manifest['variants'] if x['id']==variant);_,val=batches(v);masks=load_masks(v['mask']) if v['mask'] else {};models={};base={};record=[]
  old=read(next((D/'posthoc_validation').glob('*/'+variant+'_batches.json')))
  for kind in ('donor','identity','shuffle_101'):
   a=D/'attempts'/f'{variant}_{kind}';result=read(a/'result.json');p=Path(result['checkpoint']);cp=load(p)
   assert p==a/'segment_2500/checkpoints/final_recovered.pt' and cp['step']==2500
   assert cp['config']['model']==v['config']['model']
   recs=[read(a/f'segment_{o}/record.json') for o in (0,2500)]
   assert sum(r['optimizer_step_attempts'] for r in recs)==5000
   assert result['applied_updates']+result['amp_skips']==5000
   assert all(r['validation_batch_sequence_hash']==R._batch_sequence_hash(val) for r in recs)
   assert read(a/'provenance.json')['mask_hash']==v['mask_hash']
   m=load_model_from_checkpoint(v['config']['model'],str(p),torch.device('cuda')).float().eval()
   assert m.token_embedding.weight is m.lm_head.weight
   for n,mask in masks.items():assert torch.count_nonzero(m.state_dict()[n][~mask.to('cuda')])==0
   base[kind]=state_dict_hash(m.state_dict());replay.install(m);models[kind]=m
   record.append(dict(training_condition=kind,path=str(p),sha256=sha(p),state_hash=base[kind],checkpoint_step=5000,checkpoint_local_step=2500,segment_offset=2500,model_config=cp['config']['model'],budget=result,segment_records=recs))
  for kind in ('identity','shuffle_101'):
   m=copy.deepcopy(models[kind]);before={n:t.clone() for n,t in m.state_dict().items()};routers={n:p for n,p in models['donor'].named_parameters() if group(n)=='R'}
   with torch.no_grad():
    for n,p in m.named_parameters():
     if n in routers:p.copy_(routers[n])
   assert all(torch.equal(t,m.state_dict()[n]) for n,t in before.items() if group(n)!='R')
   assert m.token_embedding.weight is m.lm_head.weight
   models[kind+'_transplant']=m;base[kind+'_transplant']=state_dict_hash(m.state_dict())
  provenance.append(dict(variant=variant,mask_path=v['mask'],mask_hash=v['mask_hash'],mask_file_sha256=sha(v['mask']) if v['mask'] else None,checkpoints=record,validation_sequence_hash=R._batch_sequence_hash(val),batches=[dict(validation_batch_index=i,batch_hash=R._batch_sequence_hash([b]),token_hash=replay.digest(b[0]),target_hash=replay.digest(b[1]),shape=list(b[0].shape),target_tokens=b[1].numel()) for i,b in enumerate(val)]))
  with torch.inference_mode():
   for bi,(x,y) in enumerate(val):
    x=x.cuda();y=y.cuda();bh=R._batch_sequence_hash([val[bi]]);assert bh==old[bi]['batch_hash'] and y.numel()==old[bi]['target_tokens']
    donor=models['donor'](x);payload=replay.capture(models['donor'],donor,x,5000,bh)
    def evaluate(kind,policy,rep=-1,p=None):
     m=models[kind];consumed=[];handles=[]
     if p is not None:
      for block in m.blocks:handles.append(block.moe.register_forward_hook(lambda mod,inp,out:consumed.append(mod._slot is None)))
     try:out=m(x) if p is None else replay.forward(m,x,p,5000,bh)
     finally:
      for h in handles:h.remove()
     if p is not None:assert len(consumed)==len(m.blocks) and all(consumed)
     ce=float(torch.nn.functional.cross_entropy(out.logits.flatten(0,1),y.flatten()))
     rows.append(dict(variant=variant,recipient_training_condition=kind.replace('_transplant',''),evaluation_policy=policy,donor_identity=variant+'_donor',checkpoint_step=5000,checkpoint_local_step=2500,training_trace_step='',training_permutation_id=101 if kind.startswith('shuffle') else '',validation_batch_index=bi,batch_hash=bh,permutation_replicate=rep,validation_permutation_seed=SEEDS[rep] if rep>=0 else '',target_tokens=y.numel(),ce=ce))
     return out
    # Reuse verified historical contextual/identity values, after checking identity hashes.
    summary=read(next((D/'posthoc_validation').glob('*/'+variant+'_summary.json')))
    assert summary['donor_sha256']==record[0]['sha256'] and summary['recipient_sha256']==record[1]['sha256']
    for policy,key in [('native','donor_native'),('I','donor_native')]:
     out=evaluate('donor',policy,p=payload if policy=='I' else None)
     assert torch.equal(out.logits,donor.logits)
     for a,b in zip(out.routes,donor.routes):assert torch.equal(a.accepted_mask,b.accepted_mask)
    for kind in ('identity','shuffle_101'):
     for policy,p in [('I',payload),('native',None)]:
      if kind=='identity':
       key='identity_donor_generated_slots' if policy=='I' else 'identity_native'
       template=dict(rows[-1]);template.update(recipient_training_condition=kind,evaluation_policy=policy,ce=old[bi]['ce'][key]);rows.append(template)
      else:evaluate(kind,policy,p=p)
     if kind=='identity':
      template=dict(rows[-1]);template.update(evaluation_policy='donor_router_weights',ce=old[bi]['ce']['identity_donor_router_weights']);rows.append(template)
     else:evaluate(kind+'_transplant','donor_router_weights')
    for rep in range(10):
     q=shuffled(payload,rep,bi)
     for li,l in enumerate(q['layers']):
      key=(rep,bi,li)
      if key in mapping:assert mapping[key]==l['permutation_hash']
      mapping[key]=l['permutation_hash']
     for kind in ('donor','identity','shuffle_101'):
      out=evaluate(kind,'S',rep,q)
      if bi==0 and rep==0:
       again=replay.forward(models[kind],x,shuffled(payload,rep,bi),5000,bh);assert torch.equal(out.logits,again.logits)
    if bi==0:
     identity=dict(payload);identity['layers']=[dict(l,**{k:l[k][torch.arange(len(l['ids']))] for k in ('ids','gates','accepted','token_positions')}) for l in payload['layers']]
     assert torch.equal(replay.forward(models['donor'],x,identity,5000,bh).logits,donor.logits)
    print(variant,bi,'complete',flush=True)
  for k,m in models.items():assert state_dict_hash(m.state_dict())==base[k]
  checks.append(dict(variant=variant,passed=True,native_identity_max_logit_difference=0.,route_masks_exact=True,identity_permutation_exact=True,permutation_bijective=True,counts_and_gate_multisets_exact=True,payload_identity_checked=True,each_layer_consumed_once=True,repeat_exact=True,weights_unchanged=True,transplant_only_router=True,tied_weights_preserved=True,masks_preserved=True,reused_historical_identity_context=True))
  del models;torch.cuda.empty_cache()
  table(O/'per_batch_results.csv',rows);save(O/'checkpoint_and_batch_provenance.json',provenance);save(O/'numerical_checks.json',checks)
 aggregate=[]
 for key in sorted(set((r['variant'],r['recipient_training_condition'],r['evaluation_policy'],r['permutation_replicate']) for r in rows)):
  rs=[r for r in rows if (r['variant'],r['recipient_training_condition'],r['evaluation_policy'],r['permutation_replicate'])==key];assert len(rs)==31
  aggregate.append(dict(variant=key[0],recipient_training_condition=key[1],evaluation_policy=key[2],permutation_replicate=key[3],checkpoint_step=5000,training_trace_step='',donor_identity=key[0]+'_donor',target_tokens=sum(r['target_tokens'] for r in rs),ce=sum(r['ce']*r['target_tokens'] for r in rs)/sum(r['target_tokens'] for r in rs)))
 table(O/'per_permutation_results.csv',aggregate)
 summaries=[];contrasts=[];penalties={}
 for variant in ('e32_seed17_dense','e32_seed17_imp80'):
  def loss(k,p,r=-1):return next(a['ce'] for a in aggregate if (a['variant'],a['recipient_training_condition'],a['evaluation_policy'],a['permutation_replicate'])==(variant,k,p,r))
  for k in ('identity','shuffle_101','donor'):
   for p in ('I','S','native','donor_router_weights'):
    xs=[a['ce'] for a in aggregate if (a['variant'],a['recipient_training_condition'],a['evaluation_policy'])==(variant,k,p)]
    if xs:summaries.append(dict(variant=variant,recipient_training_condition=k,evaluation_policy=p,checkpoint_step=5000,donor_identity=variant+'_donor',**summarize(xs)))
  vals={k:[] for k in ('A','B','C','D')}
  for rep in range(10):
   ii=loss('identity','I');si=loss('shuffle_101','I');is_=loss('identity','S',rep);ss=loss('shuffle_101','S',rep)
   for k,value in dict(A=is_-ii,B=si-ss,C=ss-ii,D=(is_-ii)-(ss-si)).items():vals[k].append(value);contrasts.append(dict(variant=variant,contrast=k,permutation_replicate=rep,value=value))
  penalties[variant]=vals['C']
  for k,xs in vals.items():contrasts.append(dict(variant=variant,contrast=k,permutation_replicate='summary',**summarize(xs)))
 diff=[s-d for s,d in zip(penalties['e32_seed17_imp80'],penalties['e32_seed17_dense'])]
 for r,x in enumerate(diff):contrasts.append(dict(variant='sparse_minus_dense',contrast='differential_penalty',permutation_replicate=r,value=x))
 contrasts.append(dict(variant='sparse_minus_dense',contrast='differential_penalty',permutation_replicate='summary',**summarize(diff)))
 table(O/'crossed_policy_summary.csv',summaries);table(O/'contrasts.csv',contrasts)
 save(O/'completion.json',dict(status='EVALUATIONS_COMPLETE',training_updates=0,wall_seconds=time.monotonic()-start,analysis_config_sha256=sha(O/'analysis_config.json'),per_batch_rows=len(rows),checks_passed=True))
if __name__=='__main__':main()
