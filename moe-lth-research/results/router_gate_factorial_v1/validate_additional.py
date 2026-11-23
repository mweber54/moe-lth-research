"""Compatibility checks using the original override implementation, including AMP."""
from support import *
from moe_lth.models import TinyMoELanguageModel
from moe_lth.models.router import RouteOverride
from moe_lth.utils import configure_device

def main():
 assert read(OUT/'validation_complete.json')['passed'];torch.set_num_threads(4);configure_device(torch.device('cuda'))
 cfg=dict(vocab_size=32,max_seq_len=8,num_layers=2,num_heads=2,d_model=16,num_experts=8,expert_hidden_size=32,dropout=0.,top_k=1,capacity_factor=1.25)
 m=TinyMoELanguageModel(cfg).cuda();x=torch.randint(0,32,(4,8),device='cuda');y=x.roll(1,1);records=[]
 for amp in (False,True):
  with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
   out=m(x);p={'layers':[dict(ids=t.selected_expert_indices.detach(),capacity_scores=t.selected_probabilities.detach()*.73,accepted=t.accepted_mask.detach()) for t in out.routes]}
   a=forward_payload(m,x,p,'A');b=forward_payload(m,x,p,'B');d=forward_payload(m,x,p,'D');c=forward_payload(m,x,p,'C')
   legacy=m(x,[RouteOverride(v['ids'],v['capacity_scores'],v['capacity_scores']) for v in p['layers']])
   dref=m(x,[RouteOverride(v['ids'],t.selected_probabilities.detach(),v['capacity_scores']) for v,t in zip(p['layers'],a.routes)])
   assert torch.equal(legacy.logits,b.logits) and torch.equal(a.logits,d.logits) and torch.equal(b.logits,c.logits) and torch.equal(dref.logits,d.logits)
   def grads(o,aux=False):return torch.autograd.grad(.1*o.auxiliary_loss if aux else F.cross_entropy(o.logits.flatten(0,1),y.flatten()),list(m.parameters()),allow_unused=True,retain_graph=True)
   for oo,rr in [(b,legacy),(d,dref)]:
    for aux in (False,True):
     for g,h in zip(grads(oo,aux),grads(rr,aux)):
      assert (g is None)==(h is None)
      if g is not None:torch.testing.assert_close(g,h,rtol=0,atol=0)
   ag=grads(a,True);dg_=grads(d,True)
   delta=sum(float((g-h).double().square().sum()) for g,h in zip(ag,dg_) if g is not None and h is not None)**.5
   records.append(dict(amp=amp,legacy_B_logits_and_all_gradients_exact=True,D_independent_reference_all_gradients_exact=True,aux_gradient_A_D_difference_norm=delta,all_forward_pairs_exact=True))
 # Freeze all executable source dependencies, immutable weights/configs/masks and audited reuse inputs.
 inputs=read(OUT/'protected_inputs.json')
 for p in (ROOT/'src/moe_lth').rglob('*.py'):inputs[str(p)]=file_hash(p)
 for row in rows():
  for kind in ('dense','sparse'):
   for name in ('metadata.json','summary.json'):
    p=Path(row[kind])/name;inputs[str(p.resolve())]=file_hash(p)
  meta=read(Path(row['dense'])/'metadata.json')
  for p in [Path(row['config']),Path(meta['router_checkpoint']),Path(meta['router_checkpoint']).parent/'step_0.pt',ROOT/f"results/expert_count_replication/e8/masks/s0.85/seed_{row['seed']}/pruning_mask.pt"]:inputs[str(p.resolve())]=file_hash(p)
 write(OUT/'execution_inputs.json',inputs)
 write(OUT/'additional_validation.json',dict(passed=True,controls=records,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),native_auxiliary_graph_consequence='A/D auxiliary scalars match but auxiliary gradients may differ through earlier gates; direct native-probability auxiliary gradients remain.'))
 # Freeze code only after all files are complete; worker checks both gates.
 write(OUT/'execution_ready.json',dict(passed=True,validation_hash=file_hash(OUT/'validation_complete.json'),additional_validation_hash=file_hash(OUT/'additional_validation.json'),input_manifest_hash=file_hash(OUT/'execution_inputs.json'),code_hashes={str(p):file_hash(p) for p in OUT.glob('*.py')}))
 print('ADDITIONAL LEGACY/AMP CHECKS PASS',flush=True)
if __name__=='__main__':main()
