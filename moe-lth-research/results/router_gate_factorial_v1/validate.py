from support import *
from copy import deepcopy
from moe_lth.models import TinyMoELanguageModel
from moe_lth.models.router import RouteOverride
from moe_lth.utils import configure_device
import time

def check_fixture(device,dtype):
 torch.manual_seed(3)
 cfg=dict(vocab_size=32,max_seq_len=8,num_layers=2,num_heads=2,d_model=16,num_experts=8,expert_hidden_size=32,dropout=0.,top_k=1,capacity_factor=1.25)
 old_dtype=torch.get_default_dtype();torch.set_default_dtype(dtype)
 m=TinyMoELanguageModel(cfg).to(device=device,dtype=dtype);x=torch.randint(0,32,(4,8),device=device);y=x.roll(1,1)
 o=m(x);payload={'layers':[dict(ids=t.selected_expert_indices.detach(),capacity_scores=t.selected_probabilities.detach()*.73,accepted=t.accepted_mask.detach()) for t in o.routes]}
 # Scaling priorities preserves all accepted sets.
 outs={};tasks={};norms={}
 for arm in ARMS:
  o,t,a,n,e=gradient(m,x,y,payload,arm);outs[arm]=o;tasks[arm]=t;norms[arm]=n
 assert torch.equal(outs['A'].logits,outs['D'].logits) and torch.equal(outs['B'].logits,outs['C'].logits)
 for a,b in [('A','D'),('B','C')]:assert torch.equal(outs[a].auxiliary_loss,outs[b].auxiliary_loss)
 for arm in ARMS:
  n=norms[arm];assert n['expert_task_norm']>0 and n['shared_task_norm']>0 and n['router_aux_norm']>0
  assert (n['router_task_norm']==0) == (arm in ('B','D'))
 # Full-model independent reference expression: c+(g-g.detach()) has exact donor values.
 import factorial as f
 original=f.transform
 try:
  f.transform=lambda g,c,a:c+(g-g.detach()) if a=='C' else original(g,c,a)
  _,ref,_,_,_=gradient(m,x,y,payload,'C')
 finally:f.transform=original
 for a,b in zip(tasks['C'],ref):
  assert (a is None)==(b is None)
  if a is not None:torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-8)
 # c=g reduces the entire computation and task gradient to A.
 p={'layers':[dict(ids=t.selected_expert_indices.detach(),capacity_scores=t.selected_probabilities.detach(),accepted=t.accepted_mask.detach()) for t in outs['A'].routes]}
 oa,ta,_,_,_=gradient(m,x,y,p,'A');oc,tc,_,_,_=gradient(m,x,y,p,'C')
 assert torch.equal(oa.logits,oc.logits)
 for a,b in zip(ta,tc):
  if a is not None:torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-8)
 torch.set_default_dtype(old_dtype)
 return dict(device=str(device),dtype=str(dtype),forward_pairs_exact=True,C_full_gradient_reference=True,C_equals_A_when_c_equals_g=True,norms=norms)

def main():
 torch.set_num_threads(4);assert not (OUT/'validation_complete.json').exists()
 protocol={'arms':ARMS,'experts':8,'seeds':[7,17,29],'ages':[20,100],'sparsity':.85,'new_recoveries':24,'steps':2500,'snapshots':STEPS,'primary':'donor assignments AND donor gate coefficients, token-weighted FP32 CE on identical 31 validation batches','age_contrast':'T=(CE_sparse_R20-CE_dense_R20)-(CE_sparse_R100-CE_dense_R100)','simple_value':'T_C-T_A','simple_detach':'T_D-T_A','interaction':'T_B-T_C-T_D+T_A','marginal_value':'.5*((T_C-T_A)+(T_B-T_D))','marginal_detach':'.5*((T_D-T_A)+(T_B-T_C))','attenuation':'1-abs(T_arm)/(abs(T_A)+1e-12); flag abs(T_A)<1e-5','stopping':'Stop after 24. Promote only if an isolated effect exceeds .02 nats in same direction in all three seeds, controls pass, and not solely evaluation-policy reversal. No fresh-seed confirmation authorized.','step25':'No archived donor validation at 25: use the fixed step0 diagnostic readout (explicitly labeled), native readout, save snapshot; omit 25 from primary timing comparisons.','auxiliary':'Native full-softmax probabilities and .1 objective remain. Gate detach also cuts downstream auxiliary paths through earlier gate coefficients, matching B. Direct router auxiliary gradients remain active.','precision':'Legacy FP16 AMP train; legacy TF32 FP32 readouts; fixed diagnostics FP32 TF32 disabled.','baseline_limitation':'A checkpoints and A/B intermediate snapshots unavailable. Reuse audited scalar readouts with provenance; B final checkpoints retained. Do not retrain controls.'}
 write(OUT/'protocol.json',protocol)
 fixtures=[check_fixture(torch.device('cpu'),torch.float64),check_fixture(torch.device('cuda'),torch.float32)]
 local=[]
 for dtype in (torch.float64,torch.float32,torch.float16):
  z=torch.randn(11,8,device='cuda',dtype=dtype,requires_grad=True);g=z.softmax(-1)[:,2];c=torch.rand_like(g);w=torch.randn_like(g)
  native=torch.autograd.grad((g*w).sum(),z,retain_graph=True)[0];v=transform(g,c,'C');assert torch.equal(v,c)
  torch.testing.assert_close(torch.autograd.grad((v*w).sum(),z)[0],native,rtol=0,atol=0)
  assert torch.equal(transform(g,c,'D'),g) and not transform(g,c,'D').requires_grad
  local.append(str(dtype))
 # Structural degeneracy checks, actual implementation uses full softmax before top1.
 for ne in (1,8):
  z=torch.randn(3,ne,requires_grad=True);p=z.softmax(-1);g=p[:,0];renorm=g/g
  assert torch.autograd.grad(renorm.sum(),z)[0].abs().max()==0
 z=torch.tensor([[1000.,-1000.]],requires_grad=True);assert torch.autograd.grad(z.softmax(-1)[0,0],z)[0].abs().max()==0
 protected={};checks=[];real=[]
 for root in [ROOT/'results/research_investigation_v1',BASE,ROOT/'results/cross_arch_optimization_geometry/fixed_schedule_intervention']:
  for p in root.glob('*'):
   if p.is_file():protected[str(p)]=file_hash(p)
 for seed in (7,17,29):
  selected=[r for r in rows() if r['seed']==seed];cfg,initial,ticket,masks=states(selected[0],'cuda');tr,va,th,vh=data(cfg,seed)
  schedule=FactorialSchedule(BASE/f'schedules/e8/seed{seed}','C');assert schedule.manifest['counts']=={'train':2500,'validation':1581}
  for key,h in schedule.entries.items():assert file_hash(schedule.path/key)==h,key
  for row in selected:
   for kind,base in [('dense',initial),('sparse',ticket)]:
    meta=read(Path(row[kind])/'metadata.json');assert meta['training_batch_sequence_hash']==th and meta['validation_batch_sequence_hash']==vh
    assert state_dict_hash(group(base,'expert'))==meta['expert_state_hash'];assert state_dict_hash(group(base,'shared'))==meta['shared_state_hash']
    assert kind=='dense' or R._mask_hash(masks)==meta['mask_hash']
    bmeta=read(BASE/f'recovery/e8/seed{seed}/R{row["age"]}/{kind}/metadata.json')
    for k in ('expert_state_hash','shared_state_hash','mask_hash','initial_router_state_hash','training_batch_sequence_hash','validation_batch_sequence_hash','recovery_steps','optimizer','scheduler'):assert meta[k]==bmeta[k],k
    m=R.assemble_router_age_model(cfg['model'],base,meta['router_checkpoint'],{} if kind=='dense' else masks,torch.device('cuda'))
    assert state_dict_hash(group(m.state_dict(),'router'))==meta['initial_router_state_hash']
    x,y=(t.to('cuda') for t in va[0]);_,_,p=schedule.payload(x,0,0,'validation');m.eval()
    pair={}
    for arm in ARMS:
     with torch.no_grad():pair[arm]=forward_payload(m,x,p,arm).logits
    assert torch.equal(pair['A'],pair['D']) and torch.equal(pair['B'],pair['C'])
    # All 12 actual initial states, both missing gate-gradient policies.
    for arm in ('C','D'):
     n,e=diagnostics(m,x,y,p,arm,OUT/f'validation_E_seed{seed}_R{row["age"]}_{kind}_{arm}.npz')
     assert n['expert_task_norm']>0 and n['shared_task_norm']>0 and n['router_aux_norm']>0
     assert (n['router_task_norm']==0)==(arm=='D');assert n['gate_nondegenerate_fraction']>0
     real.append(dict(seed=seed,age=row['age'],kind=kind,arm=arm,**n))
    checks.append(dict(seed=seed,age=row['age'],kind=kind,mask_hash=meta['mask_hash'],train_hash=th,validation_hash=vh,forward_pairs_exact=True))
    del m,pair;torch.cuda.empty_cache()
  del tr,va
  print('VALIDATED seed',seed,flush=True)
 write(OUT/'protected_inputs.json',protected)
 write(OUT/'validation_details.json',dict(fixtures=fixtures,local_exact_jacobian_dtypes=local,identity_checks=checks,real_state_diagnostics=real,structural_degeneracy='Single-expert and post-selection-normalized top1 Jacobians are zero; FP16 saturation may also zero derivatives. Actual E8 full-softmax selected gates have active Jacobians in all tested states.'))
 frozen_hash=file_hash(OUT/'provenance/native_trainer.py')
 for control in [BASE/'gate_replay_manifest.json',ROOT/'results/cross_arch_optimization_geometry/fixed_schedule_intervention/run_manifest.json']:
  if control.exists():assert frozen_hash in read(control)['code_hashes'].values()
 code={str(p):file_hash(p) for p in OUT.glob('*.py')};code[str(OUT/'protocol.json')]=file_hash(OUT/'protocol.json')
 write(OUT/'validation_complete.json',dict(passed=True,completed_utc=time.time(),code_hashes=code,checks=12,diagnostic_cells=24))
 (OUT/'validation_report.md').write_text('# Factorial implementation validation\n\nPASS: 12 historical initial-state/data/mask identities, 24 actual-state gradient diagnostics, CPU float64 and CUDA float32 full-model controls, and exact local float64/float32/float16 gate Jacobians. C returns donor gates bit-for-bit and supplies the native selected-gate Jacobian; D returns native gates bit-for-bit with zero gate Jacobian. A/D and B/C logits, assignments and accepted sets match exactly. C equals A when donor equals native.\n\nCE gradients: routers active in A/C and zero in B/D; expert branches and shared/upstream parameters remain active. Native auxiliary probabilities remain differentiable in all cells. The auxiliary objective stays at coefficient .1; later auxiliary paths through earlier gates can change under detach. It is not an exclusively CE-specific graph cut.\n\nTied embedding task gradient equals separate lookup plus output-head leaves (relative error <2e-5). Diagnostics preserve parameters, RNG, gradients and module modes. Single-expert, selected-only-normalized top1, and saturated softmax are structurally degenerate controls; the actual full-softmax E8 model is not structurally degenerate.\n\nAll 12,243 archived donor occurrences pass SHA256 integrity. Each recovery will check occurrence input hashes, assignments, accepted sets and ordering. Dense/sparse budgets use the same frozen legacy trainer, 2500 attempts, fresh AdamW, clipping 1, FP16 AMP, and primary FP32 token-weighted CE readout; applied updates/skips are reported separately.\n\nReused A intermediate/final checkpoints and B intermediate snapshots are missing. Audited A/B scalar readouts and execution provenance are reused; B final checkpoints remain. No reconstructed baseline training is presented. Donor validation at step25 does not exist: fixed step0 diagnostics and native readout are labeled separately, and primary timing comparisons exclude25. See protocol.json and validation_details.json.\n')
 print('ALL PREFLIGHT CHECKS PASS',flush=True)
if __name__=='__main__':main()
