"""Synthetic selection checks and short canonical engineering checks; not production branches."""
from common import *
from prepare import features,match
import argparse

def same(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 return a==b

def main():
 torch.set_num_threads(4);device=torch.device('cuda');R.configure_device(device);host=socket.getfqdn();assert host in HOSTS and torch.__version__.startswith('2.7.0') and torch.cuda.mem_get_info()[0]>24*1024**3
 # Independent adversarial trajectories: negative signed growth, tiny coordinate, fluctuating/net-zero.
 x=np.array([[-.1,1e-12,.1,.2],[-.2,2e-12,.5,.20001],[-.3,3e-12,.1,.20002]])
 f=features(x);assert f['sustained'][0] and f['slope'][0]>0 and f['slope'][1]<f['slope'][0]*1e-8;assert f['variation'][2]>f['variation'][3]*1000 and not f['sustained'][2]
 # A longer history is rejected, so no future checkpoint can silently enter the fit.
 try:features(np.vstack([x,x[-1]]))
 except ValueError:pass
 else:raise AssertionError('Future observations accepted')
 # Exact coordinate index checks under arbitrary signed histories and deterministic matching.
 generator=np.random.default_rng(9191);a=generator.normal(size=(3,4000));g,s,ff,c=match(a,'synthetic',registry());g2,s2,_,_=match(a.copy(),'synthetic',registry());assert np.array_equal(g,g2) and np.array_equal(s,s2);assert np.all(ff['sustained'][g]);assert len(set(g))==len(g) and not set(g)&set(s)
 train,val=data(17);train=train[1000:1003];cfg=variant(17)['config'];checks=[]
 # Verify each full-size initial branch against the source, without loss-based selection.
 for branch in BRANCHES:
  model,opt,scaler,indices,hs=assemble(17,branch,device);cp=load(source(17)/'checkpoints/step_1000.pt');params=dict(model.named_parameters())
  for n,t in model.state_dict().items():
   expected=cp['model'][n].clone()
   if n in indices:expected.view(-1)[indices[n].cpu()]=0
   assert torch.equal(t.cpu(),expected)
  assert not opt.state and all(p.requires_grad for p in model.parameters());checks.append(dict(branch=branch,only_intended_coordinates_changed=True,initial_optimizer_empty=True,optimizer_membership_all=True,mask_count=sum(len(t) for t in indices.values())))
  if branch=='growing':
   before_rng=copy.deepcopy(rng());before_opt=copy.deepcopy(opt.state_dict());before_weights=state_dict_hash(model.state_dict());evaluate(model,val[:1],device);assert same(before_rng,rng()) and same(before_opt,opt.state_dict()) and before_weights==state_dict_hash(model.state_dict())
   # Three native updates with a serializable interruption after the first.
   seed_everything(17);update(model,opt,scaler,train[0],cfg,indices,device);snapshot=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),scaler=copy.deepcopy(scaler.state_dict()),**rng())
   path=D/'validation'/f'{host}.resume_fixture.pt';atomic_tensor(path,snapshot)
   for b in train[1:]:update(model,opt,scaler,b,cfg,indices,device)
   final={n:t.detach().cpu().clone() for n,t in model.state_dict().items()};finalopt=copy.deepcopy(opt.state_dict());finalscale=copy.deepcopy(scaler.state_dict());finalrng=copy.deepcopy(rng())
   restored=decode_numpy(torch.load(path,map_location='cpu',weights_only=True));model.load_state_dict(restored['model']);opt.load_state_dict(restored['optimizer']);scaler.load_state_dict(restored['scaler']);set_rng(restored)
   for b in train[1:]:update(model,opt,scaler,b,cfg,indices,device)
   assert same(final,{n:t.cpu() for n,t in model.state_dict().items()}) and same(finalopt,opt.state_dict()) and same(finalscale,scaler.state_dict()) and same(finalrng,rng())
   # Surviving expert, router, shared and bias gradients remain present; removed gradients are zero.
   assert all(p.requires_grad for p in model.parameters())
   assert any(p.grad is not None and float(p.grad.norm())>0 for n,p in model.named_parameters() if group(n)=='E')
   for n,p in model.named_parameters():
    if n in indices and p.grad is not None:assert torch.count_nonzero(p.grad.view(-1)[indices[n]])==0
   assert any(float(p.grad.norm())>0 for n,p in model.named_parameters() if group(n)=='R')
   assert any(float(p.grad.norm())>0 for n,p in model.named_parameters() if group(n)=='S')
   assert any(float(p.grad.norm())>0 for n,p in model.named_parameters() if '.experts.' in n and n.endswith('.bias'))
   checks[-1].update(diagnostic_rng_optimizer_weights_preserved=True,serialized_resume_model_optimizer_scaler_rng_exact=True,masked_gradients_zero=True,router_shared_bias_gradient_flow=True)
  for h in hs:h.remove()
  del model,opt,scaler;torch.cuda.empty_cache()
 save(D/'validation'/f'{host}.json',dict(passed=True,host=host,torch=str(torch.__version__),synthetic_signed_vs_magnitude=True,no_individual_denominator=True,fluctuation_not_stability=True,no_future_observations=True,coordinate_matching_deterministic=True,branches=checks,peak_gpu_bytes=torch.cuda.max_memory_allocated(),free_gpu_bytes=torch.cuda.mem_get_info()[0],production_branches_run=0))
 print('VALIDATION PASSED',host,flush=True)
if __name__=='__main__':main()
