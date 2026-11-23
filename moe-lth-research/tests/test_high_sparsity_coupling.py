"""Mechanism, selection, and preservation invariants for the isolated cohort."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/high_sparsity_coupling'))
from common import model,rng,state_dict_hash
from measure import associations,capture,assignments,representation
from gradient import diagnostic

def fixture():
 config={'model':dict(vocab_size=32,max_seq_len=8,num_layers=2,num_heads=2,d_model=8,num_experts=4,expert_hidden_size=16,dropout=0.,top_k=1,capacity_factor=.5)}
 from moe_lth.models import TinyMoELanguageModel
 with rng():
  torch.manual_seed(17);m=TinyMoELanguageModel(config['model']);tokens=torch.randint(0,32,(4,8));targets=torch.randint(0,32,(4,8))
 probe=dict(tokens=tokens,targets=targets,positions=torch.arange(32))
 with torch.no_grad():probe['source_full_inputs']=m(tokens).pre_router_hidden_states
 return m,probe

def test_actual_gradient_mask_and_full_softmax():
 m,p=fixture();m.eval();mask={}
 for n,w in m.named_parameters():
  if '.experts.' in n and n.endswith('fc1.weight'):
   v=torch.ones_like(w,dtype=torch.bool);v[:,::2]=False;mask[n]=v
 before=state_dict_hash(m.state_dict());rows=diagnostic(m,p,mask)
 assert state_dict_hash(m.state_dict())==before
 assert max(r['masked_identity_max_error'] for r in rows)<1e-6
 assert max(r['router_identity_max_error'] for r in rows)<1e-6
 assert max(r['task_aux_decomposition_error'] for r in rows)<1e-6
 assert max(r['softmax_derivative_error'] for r in rows)<1e-6
 assert max(r['unselected_nonzero_task_logit_gradients'] for r in rows)==3
 assert all(r['absolute_router_expert_gradient_cosine'] is None or abs(r['absolute_router_expert_gradient_cosine']-r['input_masked_cosine'])<1e-5 for r in rows)

def test_capacity_and_fixed_pair_identities():
 m,p=fixture();m.eval()
 with torch.no_grad():
  native=m(p['tokens']);a=assignments(m,p);first=a['native'][0].copy()
  for l,trace in enumerate(native.routes):
   ids=trace.selected_expert_indices.reshape(-1,1).numpy();assert np.array_equal(a['native'][l],(ids==np.arange(4)[None,:]))
   assert not np.any(trace.accepted_mask.numpy().sum(1)>1)
   assert trace.accepted_mask.sum()<p['tokens'].numel()
  with torch.no_grad():m.blocks[0].moe.router.projection.weight.neg_()
  b=assignments(m,p)
  assert np.array_equal(a['native'][0],first)
  assert not np.array_equal(first,b['native'][0])

def test_aggregation_population_and_undefined():
 x=np.arange(12.);scores=np.stack([x,x[::-1],np.ones(12)],1);acts=np.stack([2*x,-x,np.ones(12)],1);pairs=np.ones((12,3),dtype=bool)
 groups,pools,within=associations(scores,acts,{'fixed':pairs},0,'native')
 assert groups[-1]['pearson'] is None
 assert pools[0]['n_valid']==2 and pools[0]['n_excluded']==1
 # One constant unmatched row invalidates the all-pair intersection for every expert.
 assert pools[0]['matched_valid_experts']=='[]'
 assert abs(pools[0]['paper_pooled_standardized_pearson']-1)<1e-12
 assert representation(scores,scores)['cka']>1-1e-12


def test_full_geometry_identity_and_scaling():
 from measure import weight_geometry
 with rng():
  torch.manual_seed(124);w=torch.randn(80,40,dtype=torch.float64);routers=torch.randn(2,40,dtype=torch.float64)
 s={'blocks.0.moe.experts.0.fc1.weight':w,'blocks.0.moe.router.projection.weight':routers}
 scaled={k:(3*v if 'fc1' in k else v) for k,v in s.items()}
 rows,q=weight_geometry(s,{'self':s,'scaled':scaled},torch.device('cpu'))
 assert all(abs(r['normalized_gram_distance'])<1e-12 for r in rows)
 assert all(abs(r['gram_cosine']-1)<1e-12 for r in rows)
 assert all(r['subspace_overlap'] is None or abs(r['subspace_overlap']-1)<1e-10 for r in rows)
 from measure import spectrum
 _,values,basis=spectrum(w);_,singular,vh=torch.linalg.svd(w,full_matrices=False)
 assert torch.allclose(values,singular.square(),atol=1e-10,rtol=1e-10)
 for rank in (8,16,32):assert abs(float((basis[:,:rank].T@vh[:rank].T).square().sum()/rank)-1)<1e-10
 assert len(q)==2 and {r['router'] for r in q}=={'raw','centered'}
 assert all(abs(r['reference_trace']/r['trace']-9)<1e-6 for r in rows if r['reference']=='scaled')
