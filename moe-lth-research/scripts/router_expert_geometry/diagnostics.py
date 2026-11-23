from gcommon import *
from metrics import geometry,responses,reference_responses,basis
from moe_lth.models import TinyMoELanguageModel
from moe_lth.experiments import run_router_age_recovery as R
class Diagnostics:
 def __init__(self,info,dest,device):
  self.info=info;self.dest=Path(dest);self.device=device;self.elapsed=0.;self.started=time.monotonic();self.initial=None;self.initial_routes={};self.applied=0;self.skips=0;self.training_seconds=0.;self.last_end=None;self.validation_timings={}
  self.probe=torch.load(OUT/'probes'/f"{info['source_id']}.pt",weights_only=True)
  self.source=TinyMoELanguageModel(info['config']['model']).to(device)
  self.source.load_state_dict(torch.load(info['checkpoints']['100']['path'],map_location=device,weights_only=False)['model'])
  self.source.eval()
  self.source_state={k:v.detach().cpu().clone() for k,v in self.source.state_dict().items()}
  with protected(self.source):
   self.frozen_ref=reference_responses(self.source,self.probe['inputs'])
   source_native=self.source(self.probe['tokens'].to(device));pos=self.probe['positions'].to(device)
   self.native_source_selected=[r.selected_expert_indices.reshape(-1,self.source.config['top_k'])[pos].cpu() for r in source_native.routes]
 def __call__(self,model,optimizer,scaler,step):
  now=time.monotonic()
  if step and self.last_end is not None:self.training_seconds+=now-self.last_end-self.validation_timings.get(step,0.)
  if step:
   with (self.dest/'gradient_stats/gradient_stats.jsonl').open('rb') as f:
    f.seek(0,2);f.seek(max(0,f.tell()-65536));line=f.read().splitlines()[-1]
   flag=json.loads(line)['optimizer_step_applied'];self.applied+=int(flag);self.skips+=int(not flag)
  if step==0:
   save(self.dest/'host_info.json',dict(host=__import__('socket').getfqdn(),pid=os.getpid(),gpu=torch.cuda.get_device_name(self.device) if self.device.type=='cuda' else None,torch_version=torch.__version__,cuda_version=torch.version.cuda,source_id=self.info['source_id'],source_checkpoints=self.info['checkpoints'],probe_sha256=sha(OUT/'probes'/f"{self.info['source_id']}.pt")))
  if step%25==0:
   save(self.dest/'heartbeat.json',dict(step=step,unix=time.time(),host=__import__('socket').getfqdn(),pid=os.getpid()))
  if step not in STEPS:
   self.last_end=time.monotonic();return
  allocated_before=torch.cuda.memory_allocated(self.device) if self.device.type=='cuda' else None
  reserved_before=torch.cuda.memory_reserved(self.device) if self.device.type=='cuda' else None
  started=time.monotonic();dest=self.dest/'geometry'/f'step_{step:04d}';dest.mkdir(parents=True,exist_ok=False)
  with protected(model):
   state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   if self.initial is None:self.initial=state
   table(dest/'weights.csv',geometry(state,self.source_state,self.initial))
   bases={(layer,e):basis(state[f'blocks.{layer}.moe.experts.{e}.fc1.weight'])[1] for layer in range(len(model.blocks)) for e in range(model.config['num_experts'])}
   native=model(self.probe['tokens'].to(self.device));inputs=[x.reshape(-1,x.shape[-1])[self.probe['positions'].to(x.device)] for x in native.pre_router_hidden_states]
   for mode,xs in [('frozen',self.probe['inputs']),('native',inputs)]:
    ref=self.frozen_ref if mode=='frozen' else reference_responses(self.source,xs)
    values,raw=responses(model,xs,ref,self.initial_routes.get(mode),native_traces=native.routes if mode=='native' else None,positions=self.probe['positions'],expert_bases=bases)
    for row in values:
     row['source_same_input_assignment_agreement']=row['source_assignment_agreement']
     if mode=='native':row['source_assignment_agreement']=float((raw[str(row['layer'])]['selected']==self.native_source_selected[row['layer']]).float().mean())
    if step==0:self.initial_routes[mode]=raw
    table(dest/f'{mode}.csv',values);torch.save(raw,dest/f'{mode}_routing.pt')
   torch.save({k:v.cpu() for k,v in state.items()},dest/'diagnostic_model_state.pt')
  with protected(model,strict_fp32=False):loss=R._evaluate_condition_loss(model,self.validation,self.device,False,1.,recovery_step=step)
  duration=time.monotonic()-started;self.elapsed+=duration
  save(dest/'metadata.json',dict(attempted_step=step,applied_updates=self.applied,amp_skips=self.skips,validation_loss=loss,training_seconds=self.training_seconds,native_validation_seconds=sum(v for k,v in self.validation_timings.items() if k!=25),diagnostic_seconds=duration,cumulative_diagnostic_seconds=self.elapsed,training_and_native_framework_seconds=time.monotonic()-self.started-self.elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None,peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device) if self.device.type=='cuda' else None,probe_sha256=sha(OUT/'probes'/f"{self.info['source_id']}.pt"),diagnostic_allocated_before=allocated_before,diagnostic_reserved_before=reserved_before,diagnostic_allocated_after=torch.cuda.memory_allocated(self.device) if self.device.type=='cuda' else None,cpu_peak_rss_kib=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss,checkpoint_kind='diagnostic model state, not resumable'))
  save(self.dest/'heartbeat.json',dict(step=step,unix=time.time(),host=__import__('socket').getfqdn(),pid=os.getpid()))
  print('GEOMETRY',step,'seconds',duration,flush=True)
  self.last_end=time.monotonic()
