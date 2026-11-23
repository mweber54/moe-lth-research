import os,sys,json,random
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
P=Path(__file__).resolve().parents[1]/'scripts/imp_adaptation_factorial_v1';sys.path.insert(0,str(P))
_previous=os.environ.get('TORCH_FORCE_WEIGHTS_ONLY_LOAD')
import af_common as C
import af_native as native
from af_diagnostics import grads,displacement
if _previous is None:os.environ.pop('TORCH_FORCE_WEIGHTS_ONLY_LOAD',None)
else:os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD']=_previous
from moe_lth.config import DEFAULT_CONFIG
from moe_lth.training.checkpoint import save_checkpoint

def setup(tmp_path,dropout=0.):
 c=deepcopy(DEFAULT_CONFIG);c.update(seed=7,device='cpu',output_dir=str(tmp_path))
 c['model'].update(num_layers=2,num_experts=2,d_model=8,num_heads=2,expert_hidden_size=16,vocab_size=32,max_seq_len=8,dropout=dropout,top_k=1)
 c['data'].update(seq_len=8,validation_blocks=1);c['training'].update(steps=5,batch_size=2,precision='fp32',weight_decay=.1)
 C.R.seed_everything(7);m=C.TinyMoELanguageModel(c['model']);p=tmp_path/'init.pt';save_checkpoint(p,m,None,0,None,c)
 masks={n:(torch.arange(v.numel()).reshape(v.shape)%3!=0) for n,v in C.expert_weight_parameters(m).items()};C.apply_masks_(m,masks)
 batches=[(torch.randint(0,32,(2,8)),torch.randint(0,32,(2,8))) for _ in range(5)]
 return c,p,masks,{n:v.clone() for n,v in m.state_dict().items()},batches

def test_groups_tying_and_identical_initial_forward(tmp_path):
 c,p,masks,state,b=setup(tmp_path);outputs=[]
 for cond in C.CONDITIONS:
  m=C.configure(C.TinyMoELanguageModel(c['model']),cond);m.load_state_dict(state);m.train()
  inv=C.inventory(m,masks);assert {r['group'] for r in inv}=={'E','R','S'}
  assert sum(r['parameter_count'] for r in inv)==sum(v.numel() for v in m.parameters())
  assert m.token_embedding.weight is m.lm_head.weight
  assert any('lm_head.weight' in r['aliases'] and r['group']=='S' for r in inv)
  with torch.no_grad():outputs.append(m(b[0][0]).logits)
 assert all(torch.equal(outputs[0],x) for x in outputs)

@pytest.mark.parametrize('condition',C.CONDITIONS)
def test_frozen_groups_weight_decay_and_gradient_flow(tmp_path,condition):
 c,p,masks,state,b=setup(tmp_path);m=C.configure(C.TinyMoELanguageModel(c['model']),condition);m.load_state_dict(state)
 opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001,weight_decay=.1)
 from moe_lth.pruning.rewind import register_mask_gradient_hooks
 hooks=register_mask_gradient_hooks(m,masks)
 for _ in range(2):
  opt.zero_grad();o=m(b[0][0]);loss=torch.nn.functional.cross_entropy(o.logits.flatten(0,1),b[0][1].flatten())+.1*o.auxiliary_loss;loss.backward()
  for g in C.TRAINABLE[condition]:assert any(v.grad is not None and v.grad.norm()>0 for n,v in m.named_parameters() if C.group(n)==g)
  opt.step();C.apply_masks_(m,masks)
 for n,v in m.named_parameters():
  if C.group(n) not in C.TRAINABLE[condition]:assert torch.equal(v,state[n]) and v.grad is None
 for n,v in C.expert_weight_parameters(m).items():assert torch.count_nonzero(v[~masks[n]])==0
 for h in hooks:h.remove()
 assert any(not torch.equal(v,state[n]) for n,v in m.named_parameters() if C.group(n)=='E')


def test_auxiliary_gradients_through_frozen_router_and_shared(tmp_path):
 c,p,masks,state,b=setup(tmp_path);m=C.configure(C.TinyMoELanguageModel(c['model']),'expert_only');m.load_state_dict(state)
 o=m(b[0][0]);o.auxiliary_loss.backward()
 assert any(p.grad is not None and p.grad.norm()>0 for n,p in m.named_parameters() if C.group(n)=='E')
 assert all(p.grad is None for n,p in m.named_parameters() if C.group(n)!='E')


def test_diagnostics_preserve_modes_rng_and_existing_gradients(tmp_path):
 c,p,masks,state,b=setup(tmp_path,.1);m=C.TinyMoELanguageModel(c['model']);m.load_state_dict(state);m.train();m.blocks[0].attention.eval()
 for v in m.parameters():v.grad=torch.ones_like(v)
 before=[v.grad.clone() for v in m.parameters()];rng=torch.get_rng_state().clone();modes=[x.training for x in m.modules()]
 with C.protected(m):
  m.eval();d=grads(m,b[0],masks,.1)
 assert torch.equal(rng,torch.get_rng_state()) and modes==[x.training for x in m.modules()]
 assert all(torch.equal(a,v.grad) for a,v in zip(before,m.parameters()))
 assert any(r['denominator']=='surviving expert-weight coordinates' for r in d['per_parameter'])
 assert all(r['task_over_weighted_aux'] is None or np.isfinite(r['task_over_weighted_aux']) for r in d['per_parameter'])


def test_native_resume_optimizer_scaler_rng_batches_and_reset(tmp_path):
 c,p,masks,state,b=setup(tmp_path,.1);torch.set_num_threads(1)
 kwargs=dict(config=c,initial=state,router_path=p,masks=masks,train=b,val=b[:1],condition='expert_router',seed=7,device=torch.device('cpu'),resume_interval=1)
 full=native.segment(dest=tmp_path/'full',**kwargs)
 def fail(m,o,s,step):
  if step==3:raise RuntimeError('simulated interruption')
 with pytest.raises(RuntimeError,match='simulated'):native.segment(dest=tmp_path/'resumed',callback=fail,**kwargs)
 resumed=native.segment(dest=tmp_path/'resumed',**kwargs)
 a=C.load(tmp_path/'full/interruption.pt');z=C.load(tmp_path/'resumed/interruption.pt')
 def equal(a,b):
  if torch.is_tensor(a):return torch.equal(a,b)
  if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
  if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
  return a==b
 for key in ('model','optimizer','scaler','python_rng','numpy_rng','torch_rng','step'):assert equal(a[key],z[key])
 assert full['final_validation_loss']==resumed['final_validation_loss']
 assert a['identity']['train_hash']==C.R._batch_sequence_hash(b)
 boundary=C.load(tmp_path/'full/checkpoints/final_recovered.pt')['model'];seen=[]
 native.segment(c,boundary,tmp_path/'full/checkpoints/final_recovered.pt',masks,b[:1],b[:1],tmp_path/'reset','expert_router',7,offset=2500,callback=lambda m,o,s,i:seen.append((i,len(o.state),C.state_dict_hash(m.state_dict()))) if i==0 else None,device=torch.device('cpu'))
 assert seen[0][1]==0 and seen[0][2]==C.state_dict_hash(boundary)


def test_native_full_matches_canonical_cpu(tmp_path):
 c,p,masks,state,b=setup(tmp_path);torch.set_num_threads(1)
 clips=[]
 a=native.segment(c,state,p,masks,b[:2],b[:1],tmp_path/'adapter','full',7,device=torch.device('cpu'),forward=lambda m,t,s:C.R._condition_forward(m,t,False,1.),clip_callback=lambda n,c:clips.append((n,c)))
 assert len(clips)==2 and all(0<=coef<=1 for _,coef in clips)
 def reference(*args,**kw):return C.R._run_recovery_condition(*args,**kw)
 old=native.native_function;native.native_function=lambda *a,**k:reference
 try:z=native.segment(c,state,p,masks,b[:2],b[:1],tmp_path/'canonical','full',7,device=torch.device('cpu'))
 finally:native.native_function=old
 aa=C.load(tmp_path/'adapter/checkpoints/final_recovered.pt')['model'];bb=C.load(tmp_path/'canonical/checkpoints/final_recovered.pt')['model']
 assert all(torch.equal(aa[k],bb[k]) for k in aa)
 assert a['final_validation_loss']==z['final_validation_loss']


def test_gap_decomposition_and_factorial_analysis(tmp_path,monkeypatch):
 import af_report as report
 monkeypatch.setattr(report,'OUT',tmp_path);monkeypatch.setattr(report,'figures',lambda *x:None)
 jobs=[];states={};results={};curves={}
 # Dense full=1, sparse full=1.1. Restriction harms dense by .3 and sparse by .1.
 for sp,tag in [(0.,'dense'),(.8,'sparse')]:
  for condition in C.CONDITIONS:
   jid=tag+'_'+condition;j=dict(id=jid,seed=7,sparsity=sp,condition=condition,reuse=condition=='full');jobs.append(j)
   attempt=tmp_path/jid;attempt.mkdir();ce=1.+(.1 if sp else 0)+(.1 if sp else .3)*(condition!='full')
   r=dict(job=jid,seed=7,sparsity=sp,condition=condition,final_ce=ce,source='historical' if condition=='full' else 'new')
   C.save(attempt/'result.json',r);C.save(attempt/'curves.json',[dict(step=0,loss=10.),dict(step=5000,loss=ce)])
   states[jid]=dict(status='REUSED' if j['reuse'] else 'COMPLETE',attempt=str(attempt),result_path=str(attempt/'result.json'),reuse=j['reuse'])
 monkeypatch.setattr(report,'status',lambda j:states[j['id']])
 effects,_=report.analyze(dict(jobs=jobs,reused_full=2,new_recoveries=6))
 effect=next(x for x in effects if x['condition']=='expert_only' and x['step']==5000)
 assert abs(effect['D']-(-.2))<1e-10
 assert abs(effect['sparse_loss_change']-.1)<1e-10 and abs(effect['dense_loss_change']-.3)<1e-10
 assert effect['same_condition_strict_match'] and not effect['sparse_at_or_below_original_full_dense']
 initial=next(x for x in effects if x['condition']=='expert_only' and x['step']==0)
 assert not initial['sparse_at_or_below_original_full_dense']


def test_detached_launcher_and_exclusive_fresh_claims(tmp_path,monkeypatch,capsys):
 import af_cluster as cluster
 import af_worker as worker
 import subprocess
 from types import SimpleNamespace
 calls=[]
 monkeypatch.setattr(subprocess,'Popen',lambda args,**kw:(calls.append((args,kw)) or SimpleNamespace(pid=12345)))
 exec(cluster.detach_program(tmp_path),{})
 assert json.loads(capsys.readouterr().out)['pid']==12345
 assert calls[0][1]['start_new_session'] and calls[0][1]['env']['TORCH_FORCE_WEIGHTS_ONLY_LOAD']=='1'
 monkeypatch.setattr(worker,'OUT',tmp_path)
 C.save(tmp_path/'launch.json',dict(admission_bytes=0))
 job=dict(id='dense_expert_only',reuse=False,depends_on=[])
 C.save(tmp_path/'status'/f"{job['id']}.json",dict(status='PENDING'))
 monkeypatch.setattr(worker,'status',lambda j:C.read(tmp_path/'status'/f"{j['id']}.json"))
 monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(100*1024**3,100*1024**3))
 assert worker.claim(dict(jobs=[job]))==job
 assert worker.claim(dict(jobs=[job])) is None
 C.save(tmp_path/'status'/f"{job['id']}.json",dict(status='PENDING'))
 with pytest.raises(RuntimeError,match='Existing attempt'):worker.claim(dict(jobs=[job]))


def test_zero_aux_gradient_ratio_is_undefined(tmp_path):
 c,p,masks,state,b=setup(tmp_path);m=C.TinyMoELanguageModel(c['model']);m.load_state_dict(state)
 d=grads(m,b[0],masks,0.)
 assert all(r['task_over_weighted_aux'] is None for r in d['per_parameter'])
 assert all(r['task_over_weighted_aux'] is None for r in d['groups'])
