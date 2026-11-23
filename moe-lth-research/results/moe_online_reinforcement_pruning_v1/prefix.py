"""Capture exact dense step-250 state, reinforcement scores, and step-251 oracle."""
from __future__ import annotations
import argparse,copy,fcntl,math,random,shutil,tempfile,time
from pathlib import Path
from study_common import *
from torch.optim.optimizer import register_optimizer_step_pre_hook,register_optimizer_step_post_hook

class Stop251(Exception):pass

def native_args(seed,order,output,checkpoint,state,config,batches,validation,callback,forward=None):
 device=torch.device('cuda:0')
 with N._preserved_rng():
  model=TinyMoELanguageModel(copy.deepcopy(config['model'])).to(device);model.load_state_dict(state);model.eval()
  reference=selected_experts_per_batch(model,N.recovery._calibration_batches(validation),device)
 del model
 return dict(config=config,condition_name=f'online_reinforcement_prefix_{seed}_{order}',pruned_base_state=state,
  router_checkpoint=str(Path(entry(seed,order)['base']['initial_checkpoint'])),router_age_percent=0,router_step=0,masks={},
  expert_hash=state_dict_hash({n:t for n,t in state.items() if parameter_group(n)=='expert'}),
  shared_hash=state_dict_hash({n:t for n,t in state.items() if parameter_group(n)=='shared'}),mask_hash=N.recovery._mask_hash({}),
  reference_selected=reference,calibration_batches=N.recovery._calibration_batches(validation),train_batches=batches[:251],
  validation_batches=validation,train_batch_hash=N.recovery._batch_sequence_hash(batches[:251]),
  validation_batch_hash=N.recovery._batch_sequence_hash(validation),device=device,recovery_steps=251,dense_loss=1.,
  output_dir=output,confidence_control=False,target_confidence=None,seed=seed,sparsity=0.,router_mode='trainable',
  diagnostic_steps=(0,251),state_snapshot_callback=callback,training_forward_callback=forward,
  interruption_checkpoint=checkpoint,interruption_interval=250)

def run(seed,order):
 root=OUT/'prefixes'/f'seed{seed}_{order}';root.mkdir(parents=True,exist_ok=True)
 done=root/'verification.json'
 if done.exists():return
 with (root/'lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  adm=admission();save_json(root/'admission.json',adm)
  e=entry(seed,order);cp=Path(e['base']['initial_checkpoint']);assert sha(cp)==e['base']['initial_checkpoint_sha256']
  state=N._read_model_payload(cp)['model'];config,batches,validation=material(seed,order)
  failed=root/f'failed_{time.time_ns()}'
  if (root/'native').exists() and not done.exists():
   failed.mkdir();(root/'native').rename(failed/'native')
  names=expert_names(state);signed={n:torch.zeros_like(state[n],device='cuda',dtype=torch.float64) for n in names};absolute={n:x.clone() for n,x in signed.items()}
  initial={n:state[n].detach().cpu().clone() for n in names};parameters={};before={};applied=0;worst=0.
  expected={};current_attempt=0
  def pre(opt,args,kwargs):
   for n,q in parameters.items():
    if q.grad is not None:before[n]=q.detach().clone()
  @torch.no_grad()
  def post(opt,args,kwargs):
   nonlocal applied,worst
   group,=opt.param_groups;beta1,beta2=group['betas'];lr,wd,eps=(group[x] for x in ('lr','weight_decay','eps'));errors=[]
   for n,old in before.items():
    q=parameters[n];m=opt.state[q];step=int(m['step']);den=(m['exp_avg_sq'].sqrt()/math.sqrt(1-beta2**step)).add_(eps);scale=-lr/(1-beta1**step)
    update=(m['exp_avg'].double()/den.double()).mul_(scale);expected_q=old.mul(1-lr*wd).double()+update
    tol=(old.double().abs()+update.abs())*(4*torch.finfo(old.dtype).eps);errors.append(((q.double()-expected_q).abs()/tol.clamp_min(1e-30)).max())
    if current_attempt<=250:signed[n].add_(update);absolute[n].add_(update.abs())
   if errors:worst=max(worst,float(torch.stack(errors).max()));assert math.isfinite(worst) and worst<=1
   before.clear();applied+=1
  def forward(model,tokens,step):
   nonlocal current_attempt
   current_attempt=step
   if not parameters:
    parameters.update((n,q) for n,q in model.named_parameters() if n in names);assert set(parameters)==set(names)
   return model(tokens)
  def callback(model,opt,scaler,step):
   if step==251:
    expected.update(model={n:t.detach().cpu().clone() for n,t in model.state_dict().items()},optimizer=copy.deepcopy(opt.state_dict()),scaler=copy.deepcopy(scaler.state_dict()))
    raise Stop251
  hooks=[register_optimizer_step_pre_hook(pre),register_optimizer_step_post_hook(post)]
  output=root/'native';checkpoint=output/'interruption.pt'
  try:N.recovery._run_recovery_condition(**native_args(seed,order,output,checkpoint,state,config,batches,validation,callback,forward))
  except Stop251:pass
  finally:
   for h in hooks:h.remove()
  assert checkpoint.exists() and applied==250
  payload=torch.load(checkpoint,map_location='cpu',weights_only=False);assert payload['step']==250
  scores={};excluded={}
  for n in names:
   den=absolute[n].cpu();num=signed[n].cpu();sign=initial[n].sign();valid=(den>0)&(sign!=0)
   score=torch.full_like(den,float('nan'));score[valid]=sign[valid]*num[valid]/den[valid]
   scores[n]=dict(score=score,absolute_update=den,initial_sign=sign)
   excluded[n]=dict(zero_denominator=int((den==0).sum()),zero_start_sign=int((sign==0).sum()),eligible=int(valid.sum()))
  applied_prefix=int(payload['optimizer_steps_applied']);assert applied_prefix==249
  save_torch(root/'scores.pt',dict(scores=scores,excluded=excluded,applied_updates_1_250=applied_prefix,attempts=250,max_reconstruction_error_fraction=worst))
  save_torch(root/'expected_step251.pt',expected)
  # Restore into a cloned native directory and reproduce update 251.
  verify=root/'restore';shutil.copytree(output,verify);verify_cp=verify/'interruption.pt';observed={}
  def check(model,opt,scaler,step):
   if step==251:
    observed.update(model={n:t.detach().cpu().clone() for n,t in model.state_dict().items()},optimizer=copy.deepcopy(opt.state_dict()),scaler=copy.deepcopy(scaler.state_dict()));raise Stop251
  try:N.recovery._run_recovery_condition(**native_args(seed,order,verify,verify_cp,state,config,batches,validation,check))
  except Stop251:pass
  checks={k:nested_equal(expected[k],observed[k]) for k in expected};assert all(checks.values())
  save_json(done,dict(seed=seed,order=order,protocol_sha256=PROTOCOL_SHA,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),scores_sha256=sha(root/'scores.pt'),
   initial_checkpoint=str(cp),initial_sha256=sha(cp),train_hash_1_251=N.recovery._batch_sequence_hash(batches[:251]),validation_hash=N.recovery._batch_sequence_hash(validation),
   expert_matrices=len(names),excluded_totals={k:sum(x[k] for x in excluded.values()) for k in ('zero_denominator','zero_start_sign','eligible')},
   applied_updates_1_250=applied_prefix,amp_overflow_attempts_1_250=1,max_reconstruction_error_fraction=worst,restore_step251_checks=checks,status='VERIFIED'))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('seed',type=int,choices=SEEDS);a.add_argument('order',choices=ORDERS);z=a.parse_args();run(z.seed,z.order)
