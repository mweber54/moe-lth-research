from gcommon import *
from diagnostics import Diagnostics
import worker as W
from moe_lth.experiments import run_router_age_recovery as R
import argparse

def run(cid,dest,short=None,instrument=True,audit=False):
 job=next(r for r in rows(OUT/'run_manifest.csv') if r['condition_id']==cid);job={**job,'source_attempts':2500,'recovery_attempts':2500,'arm':'G'}
 info=read(OUT/'runtime_sources.json')[job['source_id']]
 W.OUT=OUT;W.source_info=lambda seed,budget:info
 if short is not None:
  oldread=W.pd.read_csv
  def smoke_manifest(path,*args,**kwargs):
   df=oldread(path,*args,**kwargs)
   if str(path)==str(OUT/'manifests/run_manifest.csv'):
    df.loc[df.job_type=='dense','status']='REUSED'
    for i,row in df[df.job_type=='dense'].iterrows():df.loc[i,'artifact_path']=str(ROOT/'results/dataset_a_full_lower_sparsity/dense_table_references'/f'dense_e32_seed{int(row.seed)}_R{int(row.router_age)}')
   return df
  W.pd.read_csv=smoke_manifest
 original=R._run_recovery_condition
 def wrapped(**kw):
  if short is not None:kw['recovery_steps']=short;kw['train_batches']=kw['train_batches'][:short];kw['train_batch_hash']=R._batch_sequence_hash(kw['train_batches']);kw['diagnostic_steps']=(0,short)
  if instrument:
   with P.preserve_rng():d=Diagnostics(info,dest,kw['device'])
   d.validation=kw['validation_batches'];d.validation_cache={}
   evaluate_original=R._evaluate_condition_loss
   def cached_evaluate(*args,**kwargs):
    step=kwargs.get('recovery_step',args[6] if len(args)>6 else 0)
    if step not in d.validation_cache:
     started=time.monotonic();d.validation_cache[step]=evaluate_original(*args,**kwargs);d.validation_timings[step]=time.monotonic()-started
    return d.validation_cache[step]
   R._evaluate_condition_loss=cached_evaluate
   prior=kw['state_snapshot_callback']
   def callback(m,o,s,step):
    prior(m,o,s,step)
    if audit and step in STEPS:
     from preservation import capture
     before=capture(m,o,s,kw['train_batches']);d(m,o,s,step);after=capture(m,o,s,kw['train_batches']);assert before==after,'Diagnostic changed training state'
     save(Path(dest)/f'preservation_step{step}.json',dict(passed=True,before=before,after=after))
    else:d(m,o,s,step)
   kw['state_snapshot_callback']=callback;kw['diagnostic_steps']=STEPS if short is None else (0,short)
  result=original(**kw)
  if instrument:save(Path(dest)/'instrumentation_timing.json',dict(training_seconds=d.training_seconds,diagnostic_seconds=d.elapsed,validation_seconds_by_step=d.validation_timings,peak_reserved_bytes=torch.cuda.max_memory_reserved(),host=__import__('socket').getfqdn(),pid=os.getpid(),gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda))
  return result
 R._run_recovery_condition=wrapped
 # Native worker's completion checks assume 2500; short smoke calls stop after the
 # native engine returns, before those assertions, preserving its training engine.
 if short is not None:
  class SmokeDone(Exception):pass
  result={}
  def smoke(**kw):result.update(wrapped(**kw));raise SmokeDone()
  R._run_recovery_condition=smoke
  try:W.recovery_job(job,Path(dest))
  except SmokeDone:return result
 return W.recovery_job(job,Path(dest))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--attempt',required=True);p.add_argument('--short',type=int);p.add_argument('--disabled',action='store_true');p.add_argument('--audit-preservation',action='store_true');a=p.parse_args();torch.set_num_threads(2)
 dest=Path(a.attempt)
 if dest.exists() or OUT.resolve() not in dest.resolve().parents:raise RuntimeError('Fresh attempt under experiment root required')
 result=run(a.condition,dest,a.short,not a.disabled,a.audit_preservation);save(dest/'completed.json',dict(condition_id=a.condition,result=result,unix=time.time()))
