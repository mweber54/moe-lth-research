from gcommon import *
from metrics import *
from moe_lth.models import TinyMoELanguageModel
import copy

def checks():
 w=torch.randn(100,40);s,vb=basis(w);_,sd,vh=torch.linalg.svd(w,full_matrices=False);assert torch.allclose(s,sd,atol=2e-5,rtol=2e-5);assert overlap(vb[:,:16],vh.T[:,:16])>1-1e-5
 v=torch.eye(40)[:,:16];assert abs(alignment(v,torch.eye(40)[0])-1)<1e-6;assert alignment(v,torch.eye(40)[39])==0
 q,_=torch.linalg.qr(torch.randn(16,16));assert abs(overlap(v,v@q)-1)<1e-6;assert abs(overlap(v,-v)-1)<1e-6
 r=torch.randn(32,40);assert torch.allclose(centered(r).mean(0),torch.zeros(40),atol=1e-7);assert torch.allclose(centered(r+torch.ones(40)),centered(r),atol=5e-7)
 cfg=dict(vocab_size=32,max_seq_len=8,num_layers=1,num_heads=2,d_model=40,num_experts=4,expert_hidden_size=48,dropout=.1,top_k=1,capacity_factor=1.25)
 m=TinyMoELanguageModel(cfg);m.blocks[0].moe.experts[0].eval();o=torch.optim.AdamW(m.parameters());m(torch.ones((2,8),dtype=torch.long)).logits.sum().backward();o.step()
 with protected(m):
  inputs=[torch.randn(32,40)];ref=reference_responses(m,inputs);values,_=responses(m,inputs,ref)
  assert all(r['source_output_mse']==0 and abs(r['source_output_cosine']-1)<1e-6 for r in values)
  n=m(torch.ones((2,8),dtype=torch.long));pos=torch.arange(16).flip(0);xs=[x.reshape(-1,40)[pos] for x in n.pre_router_hidden_states];bases={(0,e):basis(m.blocks[0].moe.experts[e].fc1.weight)[1] for e in range(4)}
  vals,raw=responses(m,xs,native_traces=n.routes,positions=pos,expert_bases=bases)
  assert torch.equal(raw['0']['selected'],n.routes[0].selected_expert_indices.reshape(-1,1)[pos]);assert torch.equal(raw['0']['accepted'],n.routes[0].accepted_mask[pos])
  assert sum(r['accepted'] for r in vals)==int(n.routes[0].accepted_mask.sum())
 before=copy.deepcopy(m.state_dict());grads=[None if p.grad is None else p.grad.clone() for p in m.parameters()];opt=copy.deepcopy(o.state_dict());flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32);rng=torch.get_rng_state().clone();modes=[x.training for x in m.modules()]
 with protected(m):
  m(torch.ones((2,8),dtype=torch.long));torch.rand(2);g1=geometry(m.state_dict());g2=geometry(m.state_dict())
 assert flags==(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
 assert g1==g2 and torch.equal(rng,torch.get_rng_state());assert modes==[x.training for x in m.modules()]
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());assert all((p.grad is None if g is None else torch.equal(g,p.grad)) for g,p in zip(grads,m.parameters()))
 assert all(torch.equal(opt['state'][k][n],o.state_dict()['state'][k][n]) for k in opt['state'] for n in opt['state'][k])
 gpu_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
 if gpu_rng is not None:
  with protected(m):torch.rand(20,device='cuda')
  assert all(torch.equal(a,b) for a,b in zip(gpu_rng,torch.cuda.get_rng_state_all()))
 batches=[torch.arange(16).reshape(2,8)];batch_before=[x.clone() for x in batches]
 with protected(m):m(batches[0])
 assert all(torch.equal(a,b) for a,b in zip(batches,batch_before))
 assert len(grid())==75 and sum(x['job_type']=='dense' for x in grid())==15
 save(OUT/'validation/focused.json',dict(passed=True,checks=['inside/orthogonal alignment','sign/rotation invariance','router centering','identical geometry','mixed module modes','CPU and CUDA RNG','batch order','parameters/buffers','gradients','optimizer state','exact grid'],cuda_equivalence='pending',mask_and_probe_validation='prepare.py exact hashes/count/generation and shared probe files'))
 print('FOCUSED CHECKS PASS')
if __name__=='__main__':torch.set_num_threads(2);checks()
