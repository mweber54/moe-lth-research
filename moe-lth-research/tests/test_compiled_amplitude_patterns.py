from copy import deepcopy
import json
import numpy as np
import torch
from moe_lth.config import DEFAULT_CONFIG
from moe_lth.models import TinyMoELanguageModel
from moe_lth.experiments.fixed_schedule_intervention import Schedule
from moe_lth.experiments.amplitude_pattern_replay import PatternSchedule,coefficient_hash
from moe_lth.experiments.amplitude_pattern_transforms import accepted_means,materialize_static,token_permutation

def test_compiled_pair_replay(tmp_path):
 torch.set_num_threads(1);torch.manual_seed(7)
 cfg=deepcopy(DEFAULT_CONFIG['model']);cfg.update(vocab_size=256,max_seq_len=8,num_layers=2,num_heads=2,d_model=16,num_experts=2,expert_hidden_size=32,dropout=0.,top_k=1,capacity_factor=.6)
 donor=TinyMoELanguageModel(cfg);x=torch.randint(0,256,(4,8));cap=Schedule(tmp_path/'donor','capture');cap.train(donor,x,1);cap.validation(donor,x,0,0);cap.finish_capture({'fixture':True})
 conditions=['global_mean','layer_mean','expert_layer_mean','token_conditioned','temporal','temporal_shuffle','token_shuffle'];root=tmp_path/'compiled';root.mkdir();m={'status':'validated','conditions':conditions,'phase_entries':{},'amplitude_hashes':{}}
 for phase,step in [('train',1),('validation',0)]:
  key=cap._key(phase,step,0);v=torch.load(tmp_path/'donor'/key,weights_only=False);ids=[z['ids'].numpy() for z in v['layers']];ok=[z['accepted'].numpy().reshape(z['ids'].shape) for z in v['layers']];a=[z['capacity_scores'].numpy() for z in v['layers']];su,co=accepted_means(a,ids,ok,2)
  m['phase_entries'][phase]={key:dict(index=0,shape=v['shape'],input_hash=v['input_hash'],donor_file_hash=cap.entries[key])};m['amplitude_hashes'][phase]={}
  for condition in conditions:
   values=materialize_static(ids,ok,su,co,condition,np.float32) if condition in conditions[:3] else [np.where(mask,z,0).astype(np.float32) for z,mask in zip(a,ok)]
   if condition=='token_shuffle':
    for l,(i,mask,z) in enumerate(zip(ids,ok,values)):
     for e in range(2):
      pos=np.flatnonzero((mask&(i==e)).reshape(-1));z.reshape(-1)[pos]=z.reshape(-1)[pos][token_permutation(len(pos),7,step,l,e)]
   np.save(root/f'{condition}_{phase}.npy',np.stack(values)[None]);m['amplitude_hashes'][phase][condition]=[coefficient_hash(values)]
 (root/'manifest.json').write_text(json.dumps(m))
 for condition in conditions:
  outputs=[]
  for model in [TinyMoELanguageModel(cfg),TinyMoELanguageModel(cfg)]:
   reader=PatternSchedule(tmp_path/'donor',root,condition);out=reader.train(model,x,1);reader.verify_training()
   routers=[p for n,p in model.named_parameters() if '.router.' in n];g=torch.autograd.grad(out.logits.square().mean(),routers,allow_unused=True,retain_graph=True);assert all(v is None or v.count_nonzero()==0 for v in g)
   g=torch.autograd.grad(out.auxiliary_loss,routers,allow_unused=True);assert any(v is not None and v.count_nonzero()>0 for v in g)
   outputs.append([t.selected_probabilities.detach() for t in out.routes]);reader.validation(model,x,0,0)
  assert all(torch.equal(a,b) for a,b in zip(*outputs))
 # A changed compiled coefficient is rejected, rather than silently replayed.
 a=np.load(root/'global_mean_train.npy');a.flat[0]+=.01;np.save(root/'global_mean_train.npy',a)
 try:PatternSchedule(tmp_path/'donor',root,'global_mean').train(donor,x,1)
 except AssertionError as exc:assert 'amplitude changed' in str(exc)
 else:raise AssertionError('Corruption was accepted')
