"""Two unchanged native segments with explicit adaptation flags and diagnostics."""
from af_common import *
from af_native import segment
from af_diagnostics import diagnose,model_state,build
from moe_lth.data import build_dataloaders
import math,copy

def batches(v):
 R.seed_everything(v['seed']);c=v['config'];tr,va=build_dataloaders(c['data'],128,v['seed']);train=R._materialize_batches(tr,5000);val=R._materialize_validation_batches(va,32)
 if R._batch_sequence_hash(train[:2500])!=v['canonical_pair']['data_hash'] or R._batch_sequence_hash(train[2500:])!=v['extension_identity']['training_batch_hash'] or R._batch_sequence_hash(val)!=v['canonical_pair']['validation_hash']:raise RuntimeError('Batch identity mismatch')
 return train,val

def full_snapshot(v,step):
 if v['full_reuse']:return Path(v['full_snapshots'][str(step)])
 st=read(OUT/'status'/f"{v['id']}_full.json");return Path(st['attempt'])/f'snapshots/step_{step:04d}.pt'

def snapshots_diagnostics(v,condition,attempt,masks,val):
 for step in STEPS:
  model=build(load(full_snapshot(v,step))['model'],v['config'],condition,torch.device('cuda'))
  previous=full_snapshot(v,max(s for s in STEPS if s<step)) if step else None
  diagnose(model,val[0],masks,condition,v['config'],step,attempt,full_snapshot(v,step),previous,full_snapshot(v,0));del model;torch.cuda.empty_cache()

def reused(job,v,attempt):
 train,val=batches(v);masks=load_masks(v['mask']) if v['mask'] else {};snapshots_diagnostics(v,'full',attempt,masks,val)
 source=Path(v['full_attempt']);curves=[];applied=skips=0
 for offset,name in [(0,'segment_0'),(2500,'segment_2500')]:
  r=read(source/name/'record.json');applied+=r['optimizer_steps_applied'];skips+=r['amp_overflow_step_count']
  for line in (source/name/'metrics.jsonl').read_text().splitlines():
   x=json.loads(line)
   if offset and x['step']==0:continue
   curves.append(dict(step=offset+x['step'],loss=x['loss']))
  for line in (source/name/'gradient_stats/gradient_stats.jsonl').read_text().splitlines():
   x=json.loads(line);norm=x.get('global_grad_norm_pre_clip')
   append(attempt/'clipping.jsonl',dict(step=offset+x['step'],coefficient=min(1.,1./(norm+1e-6)) if norm is not None else None,source='reconstructed from historical robust preclip norm; not exact native FP32 return',gradient_valid=x.get('gradient_valid')))
 save(attempt/'curves.json',curves)
 return dict(final_ce=curves[-1]['loss'],applied_updates=applied,amp_skips=skips,reused=True,source=str(source),native_training_logs=[str(source/n/'gradient_stats/gradient_stats.jsonl') for n in ('segment_0','segment_2500')],initial_ce=curves[0]['loss'],checkpoint=read(HIST/'reruns'/next(c['id'] for c in read(HIST/'cohort.json')['conditions'] if c['seed']==v['seed'] and (c['sparsity'] or 0)==v['sparsity'])/'completed.json')['checkpoint'])

def run(job,v,attempt):
 if job['reuse']:return reused(job,v,attempt)
 c=v['config'];train,val=batches(v);masks=load_masks(v['mask']) if v['mask'] else {};model=load_model_from_checkpoint(c['model'],v['initial'],torch.device('cpu'));apply_masks_(model,masks);initial=model_state(model);del model
 if state_dict_hash(initial)!=v['initial_model_hash']:raise RuntimeError('Initialization changed')
 records=[];router=v['initial'];first_hash=None
 for offset in (0,2500):
  dest=attempt/f'segment_{offset}';current={'step':offset};frozen_parameters={};frozen_checks={'count':0}
  def assembled(m):
   frozen_parameters.clear();frozen_parameters.update({n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad})
  def clip(norm,coefficient):append(dest/'clipping.jsonl',dict(step=current['step'],native_returned_grad_norm=norm,native_clipping_coefficient=coefficient,source='actual unchanged native clip_grad_norm_ return'))
  def forward(m,t,step):current['step']=offset+step;return R._condition_forward(m,t,False,1.)
  def callback(m,opt,scaler,step):
   absolute=offset+step
   for n,p in m.named_parameters():
    if n in frozen_parameters and not torch.equal(p,frozen_parameters[n]):raise RuntimeError('Frozen parameter changed '+n)
   frozen_checks['count']+=1
   if step==0:
    ids={id(p) for g in opt.param_groups for p in g['params']}
    if any((id(p) in ids)!=p.requires_grad for p in m.parameters()) or opt.state:raise RuntimeError('Optimizer membership/reset mismatch')
    reset=dict(step=absolute,model_hash=state_dict_hash(m.state_dict()),optimizer_entries=len(opt.state),scaler=scaler.state_dict(),rng_reset_seed=v['seed'],trainable_groups=sorted(TRAINABLE[job['condition']]))
    save(attempt/f'reset_{offset}.json',reset)
    if offset and reset['model_hash']!=first_hash:raise RuntimeError('Boundary changed weights')
   if absolute in STEPS and not (offset and step==0):
    for n,mask in masks.items():
     if torch.count_nonzero(dict(m.named_parameters())[n][~mask.to(next(m.parameters()).device)]):raise RuntimeError('Mask regrowth')
    path=attempt/f'snapshots/step_{absolute:04d}.pt';path.parent.mkdir(exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(dict(model=model_state(m),config=c,step=absolute),tmp);tmp.replace(path)
    previous=attempt/f'snapshots/step_{max(s for s in STEPS if s<absolute):04d}.pt' if absolute else None
    reference=path if job['condition']=='full' else full_snapshot(v,absolute)
    diagnose(m,val[0],masks,job['condition'],c,absolute,attempt,reference,previous,attempt/'snapshots/step_0000.pt')
  record=segment(c,initial,router,masks,train[offset:offset+2500],val,dest,job['condition'],v['seed'],offset,callback,forward,clip,assembled)
  save(dest/'record.json',record);save(dest/'freeze_checks.json',dict(passed=True,checks=frozen_checks['count'],frozen_parameters=len(frozen_parameters)))
  records.append(record);router=str(dest/'checkpoints/final_recovered.pt');initial=load(router)['model'];first_hash=state_dict_hash(initial)
 curves=[]
 for offset in (0,2500):
  for x in map(json.loads,(attempt/f'segment_{offset}/metrics.jsonl').read_text().splitlines()):
   if offset and x['step']==0:continue
   curves.append(dict(step=offset+x['step'],loss=x['loss']))
 save(attempt/'curves.json',curves)
 if not math.isfinite(curves[-1]['loss']):raise RuntimeError('Nonfinite endpoint')
 return dict(final_ce=curves[-1]['loss'],initial_ce=curves[0]['loss'],applied_updates=sum(r['optimizer_steps_applied'] for r in records),amp_skips=sum(r['amp_overflow_step_count'] for r in records),reused=False,checkpoint=router,native_training_logs=[str(attempt/f'segment_{o}/gradient_stats/gradient_stats.jsonl') for o in (0,2500)],frozen_groups_bit_identical=True,reset_weights_preserved=True)
