import hashlib,json,math,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import branch
from study_common import TinyMoELanguageModel,nested_equal

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for data in iter(lambda:f.read(1048576),b''):h.update(data)
 return h.hexdigest()
p=ROOT/'branches/seed7_B/high-reinforcement-small/checkpoint.pt'; backup=ROOT/'recovery/seed7_B_high_step2000.pt'
assert sha(p)==sha(backup)=='f4501eec5c09f29f21e7ab6143ab1b5b2f7f27614e326c3b9eb1c793301b5fd1'
gate=json.loads((ROOT/'engineering_gates.json').read_text());assert gate['branch_sha256']==sha(ROOT/'branch.py') and gate['sealed_inputs_sha256']==sha(ROOT/'sealed_inputs.json')
ck=torch.load(p,map_location='cpu',weights_only=False); i=ck['identity']; assert (i['seed'],i['order'],i['arm'])==(7,'B','high-reinforcement-small')
assert ck['global_step']==ck['next_batch_index']==2000 and ck['applied_since_branch']==1750 and ck['scheduler'] is None
assert set(ck['losses'])=={'pre_mask_250','post_mask_250','251','275','500','1000'} and all(math.isfinite(x) for x in ck['losses'].values())
assert all(k in ck for k in ('python_rng','numpy_rng','torch_rng','cuda_rng','model','optimizer','scaler')) and len(ck['cuda_rng'])==1 and ck['scaler']['scale']>0
selection,mask_hash=branch.get_selection(7,'B','high-reinforcement-small');assert mask_hash==i['mask_sha256']
prefix=branch.V1/'prefixes/seed7_B/native/interruption.pt';assert sha(prefix)==i['prefix_sha256']
saved=torch.load(prefix,map_location='cpu',weights_only=False);model=TinyMoELanguageModel(saved['identity']['config']['model']);model.load_state_dict(ck['model'],strict=True)
opt=torch.optim.AdamW([q for q in model.parameters() if q.requires_grad]);opt.load_state_dict(ck['optimizer']);assert nested_equal(opt.state_dict(),ck['optimizer'])
params=dict(model.named_parameters());num=0
for name,indices in selection.items():
 q=params[name];idx=indices.long();assert torch.count_nonzero(q.detach().flatten()[idx])==0
 for k in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
  if k in opt.state[q]:assert torch.count_nonzero(opt.state[q][k].flatten()[idx])==0
 num+=idx.numel()
assert num==3355392
out={'verified_unix':time.time(),'checkpoint_sha256':sha(p),'checkpoint_backup':str(backup),'identity':i,'global_step':2000,'next_batch_index':2000,'applied_since_branch':1750,'scheduler':'none','selected_weights_and_moments_zero':num,'optimizer_state_entries':len(opt.state),'scaler_present':True,'python_numpy_torch_cuda_rng_present':True,'note':'Production runner rechecks identity and restores model, optimizer, scaler, RNG and masks on GPU before advancing.'}
(ROOT/'recovery/step2000_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
