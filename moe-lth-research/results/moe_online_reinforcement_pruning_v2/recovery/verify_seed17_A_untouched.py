"""Read-only verification of the durable seed17/A untouched checkpoint."""
import hashlib,json,math,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import branch
from study_common import TinyMoELanguageModel,nested_equal,material,N

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for data in iter(lambda:f.read(1048576),b''):h.update(data)
 return h.hexdigest()
p=ROOT/'branches/seed17_A/untouched/checkpoint.pt';gate=json.loads((ROOT/'engineering_gates.json').read_text())
assert gate['branch_sha256']==sha(ROOT/'branch.py') and gate['sealed_inputs_sha256']==sha(ROOT/'sealed_inputs.json')
ck=torch.load(p,map_location='cpu',weights_only=False);i=ck['identity'];assert (i['seed'],i['order'],i['arm'])==(17,'A','untouched')
assert ck['global_step']==ck['next_batch_index'] and 250<=ck['global_step']<=2500
assert 0<=ck['applied_since_branch']<=ck['global_step']-250 and ck['scheduler'] is None
assert all(math.isfinite(v) for v in ck['losses'].values())
assert all(k in ck for k in ('python_rng','numpy_rng','torch_rng','cuda_rng','model','optimizer','scaler')) and ck['cuda_rng'] and ck['scaler']['scale']>0
selection,mask_hash=branch.get_selection(17,'A','untouched');assert not selection and mask_hash==i['mask_sha256']
prefix=branch.V1/'prefixes/seed17_A/native/interruption.pt';assert sha(prefix)==i['prefix_sha256']
saved=torch.load(prefix,map_location='cpu',weights_only=False);config,batches,validation=material(17,'A')
assert N.recovery._batch_sequence_hash(batches[250:])==i['train_hash'] and N.recovery._batch_sequence_hash(validation)==i['validation_hash']
model=TinyMoELanguageModel(config['model']);model.load_state_dict(ck['model'],strict=True)
opt=torch.optim.AdamW([q for q in model.parameters() if q.requires_grad]);opt.load_state_dict(ck['optimizer']);assert nested_equal(opt.state_dict(),ck['optimizer'])
for p,state in opt.state.items():
 for name in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
  if name in state:assert state[name].shape==p.shape
out={'verified_unix':time.time(),'checkpoint_sha256':sha(ROOT/'branches/seed17_A/untouched/checkpoint.pt'),'identity':i,'global_step':ck['global_step'],'next_batch_index':ck['next_batch_index'],'applied_since_branch':ck['applied_since_branch'],'scheduler':'none','pruning':'untouched: empty mask','optimizer_state_entries':len(opt.state),'scaler_present':True,'python_numpy_torch_cuda_rng_present':True,'note':'Existing production runner restores and validates full state on GPU before the next batch.'}
path=ROOT/'recovery/seed17_A_untouched_checkpoint_verification.json';path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
