"""Study-local full-state continuation. Not enabled until engineering gates pass."""
from __future__ import annotations
import argparse,copy,fcntl,hashlib,json,os,random,socket,sys,time,traceback,uuid
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent; V1=HERE.parent/'moe_online_reinforcement_pruning_v1'
sys.path.insert(0,str(V1))
from study_common import material, TinyMoELanguageModel, N, nested_equal, sha, save_json
from moe_lth.utils import create_grad_scaler,resolve_autocast_dtype
from moe_lth.pruning.router_age import grad_norms_by_group
ARMS={'untouched':None,'random-small':'random','high-reinforcement-small':'high'}

def rng_get(): return {'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
def rng_equal(a,b):
 return (a['python_rng']==b['python_rng'] and a['numpy_rng'][0]==b['numpy_rng'][0] and np.array_equal(a['numpy_rng'][1],b['numpy_rng'][1]) and a['numpy_rng'][2:]==b['numpy_rng'][2:] and torch.equal(a['torch_rng'].cpu(),b['torch_rng'].cpu()) and len(a['cuda_rng'])==len(b['cuda_rng']) and all(torch.equal(x.cpu(),y.cpu()) for x,y in zip(a['cuda_rng'],b['cuda_rng'])))
def rng_set(s):
 random.setstate(s['python_rng']);np.random.set_state(s['numpy_rng']);torch.set_rng_state(s['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda_rng']])
def save_pt(path,value):
 tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
 try:
  with tmp.open('wb') as f: torch.save(value,f);f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if tmp.exists():tmp.unlink()

def load_state(seed,order,device):
 prefix=V1/'prefixes'/f'seed{seed}_{order}'/'native/interruption.pt'
 manifest=json.loads((HERE/'input_manifest.json').read_text())['cells'][f'seed{seed}_{order}']
 if sha(prefix)!=manifest['files']['checkpoint']['sha256']:raise RuntimeError('prefix hash mismatch')
 saved=torch.load(prefix,map_location='cpu',weights_only=False)
 config,batches,validation=material(seed,order)
 verification=json.loads((V1/'prefixes'/f'seed{seed}_{order}'/'verification.json').read_text())
 if N.recovery._batch_sequence_hash(batches[:251])!=verification['train_hash_1_251'] or N.recovery._batch_sequence_hash(validation)!=verification['validation_hash']:raise RuntimeError('prefix training or validation stream hash mismatch')
 model=TinyMoELanguageModel(copy.deepcopy(config['model'])).to(device);model.load_state_dict(saved['model'],strict=True)
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=float(config['training']['learning_rate']),weight_decay=float(config['training']['weight_decay']))
 opt.load_state_dict(saved['optimizer']);dtype=resolve_autocast_dtype(config['training'].get('precision','fp32'),device)
 scaler=create_grad_scaler(device.type=='cuda' and dtype==torch.float16);scaler.load_state_dict(saved['scaler'])
 # Check that the serialized state is attached in the same parameter order and shape.
 named=list(model.named_parameters());live=[p for _,p in named if p.requires_grad]
 if len(opt.param_groups)!=1 or list(opt.param_groups[0]['params'])!=live:raise RuntimeError('optimizer parameter mapping mismatch')
 for name,p in named:
  state=opt.state.get(p,{})
  for k in ('exp_avg','exp_avg_sq'):
   if k in state and state[k].shape!=p.shape:raise RuntimeError('optimizer tensor shape mismatch: '+name)
 if not nested_equal(model.state_dict(),saved['model']) or not nested_equal(opt.state_dict(),saved['optimizer']) or not nested_equal(scaler.state_dict(),saved['scaler']):raise RuntimeError('full-state restore mismatch')
 if saved['step']!=250 or len(batches)!=2500:raise RuntimeError('batch position mismatch')
 rng_set(saved)
 if not rng_equal(rng_get(),saved):raise RuntimeError('prefix RNG restore mismatch')
 model.train()
 return saved,config,batches,validation,model,opt,scaler,dtype

def build_masks(model,opt,selection):
 params=dict(model.named_parameters());masks={}
 for name,indices in selection.items():
  p=params[name];idx=indices.to(p.device,dtype=torch.long)
  if idx.numel()!=13107 or torch.unique(idx).numel()!=idx.numel() or int(idx.min())<0 or int(idx.max())>=p.numel():raise RuntimeError('invalid selection '+name)
  mask=torch.ones_like(p,dtype=torch.bool);mask.view(-1)[idx]=False;masks[name]=mask
 return masks
@torch.no_grad()
def enforce(model,opt,masks,grad=False):
 params=dict(model.named_parameters())
 for name,mask in masks.items():
  p=params[name]
  if grad:
   if p.grad is not None:p.grad.masked_fill_(~mask,0)
  else:
   p.masked_fill_(~mask,0)
   for k in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
    if k in opt.state[p]:opt.state[p][k].masked_fill_(~mask,0)

def evaluate(model,validation,device):
 state=rng_get();modes={n:m.training for n,m in model.named_modules()};buffers={n:b.detach().clone() for n,b in model.named_buffers()}
 try:
  with torch.no_grad():return float(N.recovery._evaluate_condition_loss(model,validation,device,False,1.0))
 finally:
  for n,m in model.named_modules():m.training=modes[n]
  with torch.no_grad():
   for n,b in model.named_buffers():b.copy_(buffers[n])
  rng_set(state)

def update(model,opt,scaler,dtype,config,batch,masks,device):
 tokens,targets=(x.to(device) for x in batch);opt.zero_grad(set_to_none=True)
 with torch.autocast(device_type=device.type,dtype=dtype,enabled=dtype is not None):
  output=N.recovery._condition_forward(model,tokens,False,1.)
  ce=F.cross_entropy(output.logits.reshape(-1,output.logits.shape[-1]),targets.reshape(-1))
  loss=ce+float(config['routing']['aux_loss_weight'])*output.auxiliary_loss
 scaler.scale(loss).backward();scaler.unscale_(opt);enforce(model,opt,masks,grad=True)
 valid=all(v is not None for v in grad_norms_by_group(model).values())
 if not valid and not scaler.is_enabled():raise FloatingPointError('nonfinite gradient')
 torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(config['training']['grad_clip']))
 previous=float(scaler.get_scale());scaler.step(opt);scaler.update()
 applied=not (scaler.is_enabled() and float(scaler.get_scale())<previous)
 if not valid and applied:raise RuntimeError('overflow was not skipped')
 enforce(model,opt,masks)
 return applied

def verify_zeros(model,opt,masks):
 params=dict(model.named_parameters())
 for name,mask in masks.items():
  p=params[name]
  if torch.count_nonzero(p.detach()[~mask]):raise RuntimeError('weight regrowth '+name)
  for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
   if key in opt.state[p] and torch.count_nonzero(opt.state[p][key][~mask]):raise RuntimeError('moment regrowth '+name)

def verify_intervention(saved,model,opt,masks):
 # Compare every model entry and optimizer entry, allowing only selected
 # expert weights and Adam moment coordinates to become zero.
 current=model.state_dict()
 for name,original in saved['model'].items():
  actual=current[name].detach().cpu()
  if name in masks:
   mask=masks[name].detach().cpu()
   if not torch.equal(actual[mask],original[mask]) or torch.count_nonzero(actual[~mask]):raise RuntimeError('unexpected intervention change: '+name)
  elif not torch.equal(actual,original):raise RuntimeError('unexpected model change: '+name)
 if not nested_equal(opt.state_dict()['param_groups'],saved['optimizer']['param_groups']):raise RuntimeError('optimizer groups changed at intervention')
 live=[(n,p) for n,p in model.named_parameters() if p.requires_grad];groups=saved['optimizer']['param_groups']
 if len(groups)!=1 or len(groups[0]['params'])!=len(live):raise RuntimeError('unexpected optimizer layout')
 for (name,p),state_id in zip(live,groups[0]['params']):
  old=saved['optimizer']['state'].get(state_id,{});new=opt.state.get(p,{})
  if old.keys()!=new.keys():raise RuntimeError('optimizer state keys changed: '+name)
  for key,value in old.items():
   actual=new[key]
   if name in masks and key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
    mask=masks[name].detach().cpu();a=actual.detach().cpu()
    if not torch.equal(a[mask],value[mask]) or torch.count_nonzero(a[~mask]):raise RuntimeError('unexpected optimizer change: '+name)
   elif not nested_equal(actual,value):raise RuntimeError('unexpected optimizer change: '+name+' '+key)

def get_selection(seed,order,arm):
 if arm=='untouched':return {},hashlib.sha256(b'').hexdigest()
 path=HERE/'matching'/f'seed{seed}_{order}'/'selections.pt'
 seal=HERE/'sealed_inputs.json'
 if not seal.exists():raise RuntimeError('six-cell independent selection seal missing')
 data=json.loads(seal.read_text()); expected=data['cells'][f'seed{seed}_{order}']['selection_sha256']
 if sha(path)!=expected:raise RuntimeError('selection hash mismatch')
 selection=torch.load(path,map_location='cpu',weights_only=True)[ARMS[arm]]
 digest=hashlib.sha256()
 for name in sorted(selection):
  indices=selection[name].detach().cpu().to(torch.int64).contiguous().numpy()
  digest.update(name.encode('utf-8')+b'\0');digest.update(len(indices).to_bytes(8,'little'));digest.update(indices.astype('<i8',copy=False).tobytes())
 return selection,digest.hexdigest()

def checkpoint(model,opt,scaler,identity,step,applied,losses):
 return {'identity':identity,'global_step':step,'next_batch_index':step,'applied_since_branch':applied,'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'scheduler':None,'losses':losses,**rng_get()}

def run(seed,order,arm,mode,steps,phase):
 if mode=='production':
  gate=HERE/'engineering_gates.json'
  if not gate.exists():raise RuntimeError('production disabled: engineering gates not verified')
  proof=json.loads(gate.read_text())
  if proof['branch_sha256']!=sha(HERE/'branch.py') or proof['sealed_inputs_sha256']!=sha(HERE/'sealed_inputs.json') or len(proof['checks'])!=12:raise RuntimeError('engineering proof no longer matches code and selections')
 selection,mask_hash=get_selection(seed,order,arm)
 device=torch.device('cuda:0');saved,config,batches,validation,model,opt,scaler,dtype=load_state(seed,order,device)
 if mode=='oracle':
  if arm!='untouched':raise RuntimeError('oracle mode requires untouched')
  expected=torch.load(V1/'prefixes'/f'seed{seed}_{order}'/'expected_step251.pt',map_location='cpu',weights_only=False)
  before=rng_get();loss=evaluate(model,validation,device);after=rng_get()
  if not rng_equal(before,after):raise RuntimeError('evaluation changed RNG')
  applied=update(model,opt,scaler,dtype,config,batches[250],{},device)
  observed={'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict()}
  checks={k:nested_equal(observed[k],expected[k]) for k in observed}
  if not all(checks.values()):raise RuntimeError('step251 oracle mismatch: '+str(checks))
  print(json.dumps({'seed':seed,'order':order,'arm':arm,'global_step':251,'batch_index':250,'applied':applied,'pre_mask_ce':loss,'oracle_checks':checks}),flush=True);return
 if mode=='engineering':
  root=HERE/'engineering'/f'seed{seed}_{order}'/arm;root.mkdir(parents=True,exist_ok=True)
  masks=build_masks(model,opt,selection)
  identity={'seed':seed,'order':order,'arm':arm,'mask_sha256':mask_hash}
  if phase=='baseline':
   pre=evaluate(model,validation,device);enforce(model,opt,masks);verify_intervention(saved,model,opt,masks);verify_zeros(model,opt,masks);post=evaluate(model,validation,device)
   first=update(model,opt,scaler,dtype,config,batches[250],masks,device);verify_zeros(model,opt,masks)
   step251=evaluate(model,validation,device);measurements={'pre_mask_250':pre,'post_mask_250':post,'251':step251}
   save_pt(root/'step251.pt',checkpoint(model,opt,scaler,identity,251,int(first),measurements))
   second=update(model,opt,scaler,dtype,config,batches[251],masks,device);verify_zeros(model,opt,masks)
   save_pt(root/'uninterrupted252.pt',checkpoint(model,opt,scaler,identity,252,int(first)+int(second),measurements))
   print(json.dumps({'phase':phase,'global_step':252,'pre_mask_ce':pre,'post_mask_ce':post,'step251_ce':step251,'batches':[250,251]}),flush=True)
  elif phase=='resume':
   ck=torch.load(root/'step251.pt',map_location='cpu',weights_only=False)
   if ck['identity']!=identity or ck['global_step']!=251 or ck['next_batch_index']!=251:raise RuntimeError('engineering restart identity mismatch')
   model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);rng_set(ck);verify_zeros(model,opt,masks)
   second=update(model,opt,scaler,dtype,config,batches[251],masks,device);verify_zeros(model,opt,masks)
   ref=torch.load(root/'uninterrupted252.pt',map_location='cpu',weights_only=False)
   observed=checkpoint(model,opt,scaler,identity,252,ck['applied_since_branch']+int(second),ck['losses'])
   checks={k:nested_equal(observed[k],ref[k]) for k in ('model','optimizer','scaler','identity','global_step','next_batch_index','applied_since_branch')}
   checks['rng']=rng_equal(observed,ref)
   if not all(checks.values()):raise RuntimeError('restart mismatch: '+str(checks))
   print(json.dumps({'phase':phase,'global_step':252,'next_batch_index':252,'checks':checks}),flush=True)
  else:raise ValueError('engineering needs --phase baseline or resume')
  return
 if mode!='production':raise RuntimeError('unsupported mode')
 root=HERE/'branches'/f'seed{seed}_{order}'/arm;root.mkdir(parents=True,exist_ok=True)
 with (root/'lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  save_json(root/'running.json',{'pid':os.getpid(),'host':socket.getfqdn(),'started_unix':time.time()})
  identity={'seed':seed,'order':order,'arm':arm,'mask_sha256':mask_hash,'prefix_sha256':sha(V1/'prefixes'/f'seed{seed}_{order}'/'native/interruption.pt'),'train_hash':N.recovery._batch_sequence_hash(batches[250:]),'validation_hash':N.recovery._batch_sequence_hash(validation)}
  masks=build_masks(model,opt,selection);path=root/'checkpoint.pt';done=root/'complete.json'
  if done.exists():
   import status
   if status.status(root)=='COMPLETED':return
   done.rename(root/f'invalid_completion_{time.time_ns()}.json')
  if path.exists():
   ck=torch.load(path,map_location='cpu',weights_only=False)
   if ck['identity']!=identity or ck['global_step']!=ck['next_batch_index']:raise RuntimeError('checkpoint identity or batch position mismatch')
   model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);rng_set(ck);step=ck['global_step'];applied=ck['applied_since_branch'];losses=ck['losses'];verify_zeros(model,opt,masks)
  else:
   losses={'pre_mask_250':evaluate(model,validation,device)};enforce(model,opt,masks);verify_intervention(saved,model,opt,masks);verify_zeros(model,opt,masks);losses['post_mask_250']=evaluate(model,validation,device)
   step=250;applied=0
   save_pt(path,checkpoint(model,opt,scaler,identity,step,applied,losses))
  while step<2500:
   ok=update(model,opt,scaler,dtype,config,batches[step],masks,device);step+=1;applied+=int(ok)
   if step in (251,275,500,1000,2500):losses[str(step)]=evaluate(model,validation,device)
   if step%250==0 or step==2500:
    verify_zeros(model,opt,masks);save_pt(path,checkpoint(model,opt,scaler,identity,step,applied,losses))
   print(json.dumps({'pid':os.getpid(),'global_step':step,'applied_since_branch':applied,'arm':arm}),flush=True)
  verify_zeros(model,opt,masks)
  save_json(done,{'identity':identity,'global_step':step,'checkpoint_sha256':sha(path),'losses':losses,'applied_since_branch':applied})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('seed',type=int,choices=(7,17,29));p.add_argument('order',choices=('A','B'));p.add_argument('arm',choices=ARMS);p.add_argument('--mode',choices=('oracle','engineering','production'),default='oracle');p.add_argument('--steps',type=int,default=1);p.add_argument('--phase',choices=('baseline','resume'));a=p.parse_args()
 try:run(a.seed,a.order,a.arm,a.mode,a.steps,a.phase)
 except Exception:traceback.print_exc();sys.exit(1)
