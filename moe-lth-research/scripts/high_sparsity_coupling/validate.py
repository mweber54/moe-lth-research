"""Numerical identities and production-state equivalence gate."""
from common import *
from measure import corr,representation,weight_geometry,embedding,spectrum
from gradient import diagnostic
from runner import run_segment,batches
from preservation import capture as capture_state
import argparse

def numerical():
 with rng():
  torch.manual_seed(51);x=torch.randn(32,8);assert abs(cka(x.numpy(),x.numpy())-1)<1e-12;assert corr(np.ones(10),np.ones(10))[0] is None;assert corr(np.arange(7),np.arange(7))[0] is None
  g=x.double().T@x.double();h=9*g;assert float((g/g.norm()-h/h.norm()).norm())<1e-12
  w=torch.randn(12,8,requires_grad=True);mask=torch.rand(12,8)>.8;z=torch.randn(8);pre=(w*mask)@z;d=torch.randn(12);grad=torch.autograd.grad((pre*d).sum(),w)[0];assert torch.allclose(grad,d[:,None]*z*mask)
 return dict(cka_identity=True,gram_scale_invariance=True,undefined_correlations=True,masked_outer_product=True)
def provenance_checks():
 rows=[]
 for j in frozen()['conditions']:
  init=state(j['initial']);boundary=load(j['boundary']);cfg=boundary['config']
  for section in ('model','data','routing'):assert cfg[section]==j['config'][section]
  for key in ('learning_rate','weight_decay','grad_clip','precision','batch_size','steps'):assert cfg['training'][key]==j['config']['training'][key]
  masks=load_masks(j['mask']) if j['mask'] else {};ss=masked(init,masks)
  hashes={g:state_dict_hash({n:t for n,t in ss.items() if parameter_group(n)==g}) for g in ('expert','shared','router')}
  if masks:
   st=read(Path(j['boundary']).parents[1]/'stage.json')
   assert hashes['expert']==st['expert_state_hash'];assert hashes['shared']==st['shared_state_hash'];assert hashes['router']==st['initial_router_state_hash']
   actual=1-sum(int(m.sum()) for m in masks.values())/sum(m.numel() for m in masks.values());assert abs(actual-st['actual_sparsity'])<1e-12
   assert all('.experts.' in n and n.endswith('.weight') for n in masks)
   router=load(Path(j['boundary']).parents[1]/'component_checkpoints/initial_router.pt');assert state_dict_hash(router)==hashes['router']
  else:
   metadata_path=Path(j['boundary']).parents[1]/'metadata.json';st=read(metadata_path)
   assert hashes['expert']==st['expert_state_hash'] and hashes['shared']==st['shared_state_hash'] and hashes['router']==st['initial_router_state_hash']
   assert st['router_age_percent']==0 and st['router_step']==0 and st['router_mode']=='trainable'
   assert st['training_batch_sequence_hash']==j['canonical_pair']['data_hash']
   assert st['validation_batch_sequence_hash']==j['canonical_pair']['validation_hash']
  rows.append(dict(condition=j['id'],**{g+'_hash':v for g,v in hashes.items()},verified=True))
 table(OUT/'validation/initialization_components.csv',rows)
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--numerical-only',action='store_true');a=p.parse_args();torch.set_num_threads(2);R.configure_device(torch.device('cuda'));checks=numerical();save(OUT/'validation/numerical.json',checks)
 if a.numerical_only:return
 provenance_checks()
 import runpy
 tests=runpy.run_path(str(ROOT/'tests/test_high_sparsity_coupling.py'))
 for name,func in tests.items():
  if name.startswith('test_'):func()
 j=next(j for j in frozen()['conditions'] if j['seed']==7 and j['sparsity']==.8);masks=load_masks(j['mask']);initial=masked(state(j['initial']),masks);train,val=batches(j,total=25)
 assert R._batch_sequence_hash(val)==j['canonical_pair']['validation_hash']
 root=OUT/'validation'/f'attempt_{time.time_ns()}';root.mkdir(parents=True)
 # Same native trainer, full native batches, identical precision and diagnostic schedule.
 records=[]
 for enabled in (False,True):records.append(run_segment(j,initial,masks,j['initial'],train,val,root/('enabled' if enabled else 'disabled'),0,25,enabled))
 aa=state(root/'enabled/checkpoints/final_recovered.pt');bb=state(root/'disabled/checkpoints/final_recovered.pt');maxdiff=max(float((aa[k]-bb[k]).abs().max()) for k in aa);assert all(torch.allclose(aa[k],bb[k],atol=1e-6,rtol=1e-5) for k in aa)
 assert abs(records[0]['final_validation_loss']-records[1]['final_validation_loss'])<1e-6
 for field in ('optimizer_steps_applied','amp_overflow_step_count','loss_scale_final','training_batch_sequence_hash','validation_batch_sequence_hash'):assert records[0][field]==records[1][field]
 # Exercise the native fresh-optimizer second-call boundary on an abbreviated segment.
 cp=root/'enabled/checkpoints/final_recovered.pt'
 reset_record=run_segment(j,aa,masks,cp,train[:1],val,root/'reset',2500,1,True)
 pre=read(root/'snapshots/step_0025.json');post=read(root/'snapshots/step_2500_after_reset.json')
 assert pre['model_hash']==post['model_hash'] and post['optimizer_entries']==0
 # Check exact clone diagnostic preservation, including preexisting gradients and RNG.
 from analyze import write_probe
 probe=write_probe(j,val);m=model(initial,j['config']);opt=torch.optim.AdamW(m.parameters(),lr=.0003);scaler=torch.amp.GradScaler('cuda')
 m.train();next(m.parameters()).grad=torch.zeros_like(next(m.parameters()));before=capture_state(m,opt,scaler,train)
 clone=model(initial,j['config'])
 with rng():rows=diagnostic(clone,probe,masks)
 after=capture_state(m,opt,scaler,train);assert before==after
 assert max(r['masked_identity_max_error'] for r in rows)<1e-6;assert max(r['router_identity_max_error'] for r in rows)<1e-6;assert max(r['task_aux_decomposition_error'] for r in rows)<1e-6
 table(root/'gradient_identity.csv',rows)
 result=dict(passed=True,attempt=str(root),steps=25,max_parameter_absolute_difference=maxdiff,validation_loss_difference=abs(records[0]['final_validation_loss']-records[1]['final_validation_loss']),atol=1e-6,rtol=1e-5,production_state_preserved=True,optimizer_reset_weights_unchanged=True,gradient_identity_max_error=max(r['masked_identity_max_error'] for r in rows),unselected_nonzero_task_gradients=max(r['unselected_nonzero_task_logit_gradients'] for r in rows),unix=time.time(),benchmark_segments=[{k:r[k] for k in ('wall_seconds','peak_cuda_allocated_bytes','instrumented')} for r in records])
 save(OUT/'validation/equivalence.json',result);save(OUT/'validation'/f'equivalence_{os.uname().nodename}.json',result)
 print('VALIDATED',maxdiff,flush=True)
if __name__=='__main__':main()
