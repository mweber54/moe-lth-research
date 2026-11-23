"""Numerical schedule, state-surgery, attempted-clock and bookkeeping checks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/imp_mask_transition_v1'))
from common import *
from engine import evaluate,diagnostic
import pytest

def test_schedule_nesting_allocation_and_stability():
 final={'weight':torch.tensor([[True,False,False],[False,True,False]])}
 a=assignments(final,17);b=assignments(final,17)
 assert torch.equal(a['weight'],b['weight']) and torch.equal(a['weight']==0,final['weight'])
 prior=torch.ones_like(final['weight'])
 for k in range(1,11):
  current=mask_at(a,'gradual',100*k)['weight']
  assert not torch.any(current&~prior) and not torch.any(final['weight']&~current)
  assert int((~current).sum())==k*4//10;prior=current
 assert torch.equal(prior,final['weight'])
 assert stable_seed(17,'weight')!=stable_seed(17,'bias')

def test_event_surgery_retained_moments_steps_and_gradient_hooks():
 model=torch.nn.Linear(4,3,bias=False);final={'weight':torch.zeros_like(model.weight,dtype=torch.bool)};final['weight'][:,0]=True
 a=assignments(final,17);manager=MaskManager(a,'gradual');manager.register(model,manager.masks)
 opt=torch.optim.AdamW(model.parameters());model(torch.ones(2,4)).sum().backward();opt.step()
 p=model.weight;old=p.detach().clone();moments={k:v.clone() for k,v in opt.state[p].items()}
 event=manager.prune(model,opt,500);deleted=(a['weight']>0)&(a['weight']<=5)
 assert event['event_after']==5 and torch.equal(p[~deleted],old[~deleted]) and torch.count_nonzero(p[deleted])==0
 for k,v in opt.state[p].items():
  if v.shape==p.shape:
   assert torch.equal(v[~deleted],moments[k][~deleted]) and torch.count_nonzero(v[deleted])==0
  else:assert torch.equal(v,moments[k])
 opt.zero_grad();model(torch.ones(2,4)).sum().backward();assert torch.count_nonzero(p.grad[deleted])==0
 opt.step();apply_masks_(model,manager.masks);assert torch.count_nonzero(p[deleted])==0
 assert manager.prune(model,opt,500) is None
 manager2=MaskManager(a,'gradual');manager2.restore(manager.serial()['event']);assert manager2.serial()==manager.serial()

def test_attempted_clock_ignores_synthetic_amp_skips():
 applied=0;events=[];previous=0
 for attempt in range(1,1001):
  if attempt not in (100,200,1000):applied+=1
  k=event_count('gradual',attempt)
  if k>previous:events.append(attempt);previous=k
 assert events==list(range(100,1001,100)) and applied==997
 assert event_count('abrupt',999)==0 and event_count('abrupt',1000)==10

def test_component_partition_ties_counted_once():
 c=dict(vocab_size=32,max_seq_len=8,num_layers=2,num_heads=2,d_model=8,num_experts=2,expert_hidden_size=16,dropout=0.,top_k=1,capacity_factor=1.25)
 m=TinyMoELanguageModel(c);rows=inventory(m)
 assert sum(r['parameter_count'] for r in rows)==sum(p.numel() for p in m.parameters())
 assert any('lm_head.weight' in r['aliases'] and 'token_embedding.weight' in r['aliases'] for r in rows)

def test_common_readout_actual_target_weighting_and_state_preservation():
 torch.set_num_threads(2);R.seed_everything(17)
 c=dict(vocab_size=32,max_seq_len=8,num_layers=2,num_heads=2,d_model=8,num_experts=2,expert_hidden_size=16,dropout=.1,top_k=1,capacity_factor=1.25)
 m=TinyMoELanguageModel(c);m.train();probe=[(torch.randint(0,32,(3,8)),torch.randint(0,32,(3,8))),(torch.randint(0,32,(1,8)),torch.randint(0,32,(1,8)))]
 rng=torch.get_rng_state().clone();h=state_dict_hash(m.state_dict());value=evaluate(m,probe)
 assert value['target_tokens']==32 and m.training and torch.equal(torch.get_rng_state(),rng) and state_dict_hash(m.state_dict())==h
 with protected(m),torch.no_grad():
  m.eval();n=sum(float(torch.nn.functional.cross_entropy(m(x).logits.flatten(0,1),y.flatten(),reduction='none').double().sum()) for x,y in probe)
 assert value['ce']==pytest.approx(n/32,abs=1e-12)
 m(probe[0][0]).logits.sum().backward();grads={n:p.grad.clone() if p.grad is not None else None for n,p in m.named_parameters()};diagnostic(m,probe)
 assert all((p.grad is None and grads[n] is None) or torch.equal(p.grad,grads[n]) for n,p in m.named_parameters())
