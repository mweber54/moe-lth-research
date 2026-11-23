"""Additional read-only provenance and replay rejection checks for the amendment."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'imp_adaptation_factorial_v1'))
from af_common import *
from af_engine import batches
import replay
D=ROOT/'results/token_expert_matching_v1';O=D/'evaluation_repair_v1'
def main():
 torch.set_num_threads(4)
 manifest=read(OUT/'manifest.json');records=[]
 for variant in ('e32_seed17_dense','e32_seed17_imp80'):
  v=next(v for v in manifest['variants'] if v['id']==variant);train,val=batches(v)
  donor=D/'attempts'/f'{variant}_donor';inv=read(donor/'trace_inventory.json');assert [i['step'] for i in inv]==list(range(1,5001))
  trace=load(donor/'traces/5000.pt');assert sha(donor/'traces/5000.pt')==inv[-1]['sha256']
  assert trace['step']==5000 and trace['token_hash']==replay.digest(train[-1][0]) and trace['batch_hash']==R._batch_sequence_hash([train[-1]])
  for kind in ('donor','identity','shuffle_101'):
   a=D/'attempts'/f'{variant}_{kind}';p=a/'segment_2500/checkpoints/final_recovered.pt';c=load(p);snap=load(a/'snapshots/5000.pt')
   assert state_dict_hash(c['model'])==state_dict_hash(snap['model'])
   prov=read(a/'provenance.json');assert prov['trace_inventory_sha256']==sha(donor/'trace_inventory.json')
   assert prov['initial_hash']==v['initial_model_hash'] and prov['mask_hash']==v['mask_hash']
   for off in (0,2500):
    r=read(a/f'segment_{off}/record.json');assert r['training_batch_sequence_hash']==R._batch_sequence_hash(train[off:off+2500]);assert r['mask_hash']==v['mask_hash']
   records.append(dict(variant=variant,training_condition=kind,checkpoint_step=5000,checkpoint_local_step=c['step'],training_trace_step=5000,validation_batch_index=None,training_permutation_id=101 if kind=='shuffle_101' else None,final_checkpoint_matches_absolute_step5000_snapshot=True,training_batches_and_mask_hash_verified=True,trace_inventory_sha256=sha(donor/'trace_inventory.json'),final_checkpoint_sha256=sha(p)))
 # Explicit negative replay checks on the exact final donor and validation tokens.
 v=next(v for v in manifest['variants'] if v['id']=='e32_seed17_dense');_,val=batches(v)
 m=load_model_from_checkpoint(v['config']['model'],str(D/'attempts/e32_seed17_dense_donor/segment_2500/checkpoints/final_recovered.pt'),torch.device('cuda')).float().eval();replay.install(m)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 with torch.inference_mode():
  x=val[0][0].cuda();bh=R._batch_sequence_hash([val[0]]);p=replay.capture(m,m(x),x,5000,bh);negative={}
  cases={'wrong_batch':dict(p,batch_hash='invalid'),'wrong_step':dict(p,step=4999),'wrong_tokens':dict(p,token_hash='invalid'),'missing_layer':dict(p,layers=p['layers'][:-1])}
  for label,q in cases.items():
   try:replay.forward(m,x,q,5000,bh)
   except (ValueError,RuntimeError):negative[label]=True
   else:raise AssertionError(label)
  block=m.blocks[0].moe;block._replay_required=True;block._slot=None
  try:
   try:m(x)
   except RuntimeError as e:assert 'no native fallback' in str(e);negative['missing_payload_no_native_fallback']=True
   else:raise AssertionError('fallback')
  finally:block._replay_required=False
 save(O/'additional_provenance_checks.json',dict(records=records,negative_payload_checks=negative,passed=True))
if __name__=='__main__':main()
