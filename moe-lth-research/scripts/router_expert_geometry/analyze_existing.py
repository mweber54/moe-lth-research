from gcommon import *
from metrics import geometry,responses,reference_responses,basis
from moe_lth.models import TinyMoELanguageModel
from moe_lth.utils import configure_device
import argparse,fcntl

def analyze(seed):
 torch.set_num_threads(2);lock=(OUT/f'analysis_seed{seed}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX);device=torch.device('cuda');configure_device(device);sid=f'source_e32_seed{seed}_T2500';info=read(OUT/'runtime_sources.json')[sid];probe=torch.load(OUT/'probes'/f'{sid}.pt',weights_only=True)
 states={a:torch.load(rec['path'],map_location='cpu',weights_only=False)['model'] for a,rec in info['checkpoints'].items()}
 model=TinyMoELanguageModel(info['config']['model']).to(device);source=TinyMoELanguageModel(info['config']['model']).to(device);source.load_state_dict(states['100'])
 with protected(source):
  ref=reference_responses(source,probe['inputs']);source_native=source(probe['tokens'].to(device));pos=probe['positions'].to(device);source_selected=[r.selected_expert_indices.reshape(-1,model.config['top_k'])[pos].cpu() for r in source_native.routes]
 root=OUT/'existing'/sid;root.mkdir(parents=True,exist_ok=True)
 def measure(label,state,initial,origin):
  path=root/label
  if (path/'done.json').exists() and (path/'native.csv').exists():return
  start=time.monotonic();model.load_state_dict(state)
  with protected(model):
   if not (path/'weights.csv').exists():table(path/'weights.csv',geometry(state,states['100'],initial))
   bases={(layer,e):basis(state[f'blocks.{layer}.moe.experts.{e}.fc1.weight'])[1] for layer in range(len(model.blocks)) for e in range(model.config['num_experts'])}
   if not (path/'frozen.csv').exists():
    data,raw=responses(model,probe['inputs'],ref,expert_bases=bases);table(path/'frozen.csv',data);torch.save(raw,path/'frozen_routing.pt')
   native=model(probe['tokens'].to(device));xs=[x.reshape(-1,x.shape[-1])[pos] for x in native.pre_router_hidden_states];native_ref=reference_responses(source,xs)
   data,raw=responses(model,xs,native_ref,native_traces=native.routes,positions=probe['positions'],expert_bases=bases)
   for row in data:
    row['source_same_input_assignment_agreement']=row['source_assignment_agreement'];row['source_assignment_agreement']=float((raw[str(row['layer'])]['selected']==source_selected[row['layer']]).float().mean())
   table(path/'native.csv',data);torch.save(raw,path/'native_routing.pt')
  save(path/'done.json',dict(state=label,origin=origin,host=__import__('socket').getfqdn(),gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,seconds=time.monotonic()-start,probe_sha256=sha(OUT/'probes'/f'{sid}.pt')));print(seed,label,'DONE',flush=True)
 for age in AGES:
  base={k:states[str(age)][k] if '.router.' in k else v for k,v in states['0'].items()}
  measure(f'source_age_R{age}',states[str(age)],base,info['checkpoints'][str(age)])
  measure(f'unmasked_init_R{age}',base,base,info['checkpoints']['0'])
 for sp in SP:
  for kind in ('learned','random'):
   mask=torch.load(info['masks'][str(sp)][kind+'_path'],map_location='cpu',weights_only=True)
   trained={k:v*mask[k] if k in mask else v for k,v in states['100'].items()}
   initial={k:v*mask[k] if k in mask else v for k,v in states['0'].items()}
   measure(f'masked_trained_{kind}_s{sp:.2f}',trained,initial,info['masks'][str(sp)])
   for age in AGES:
    base={k:states[str(age)][k] if '.router.' in k else v for k,v in initial.items()}
    measure(f'masked_init_R{age}_{kind}_s{sp:.2f}',base,base,info['masks'][str(sp)])
 # Historical endpoints are never reused as measured trajectories.
 for package in ('dataset_b_lower_sparsity','dataset_a_full_lower_sparsity'):
  manifest=ROOT/'results'/package/'manifests/run_manifest.csv'
  for job in rows(manifest):
   if job.get('num_experts')!='32' or int(float(job.get('seed',-1)))!=seed or job.get('job_type') not in ('sparse','dense'):continue
   if int(float(job.get('router_age',-1))) not in AGES:continue
   if job['job_type']=='sparse' and float(job.get('sparsity',0)) not in SP:continue
   if job['status'] not in ('COMPLETE','REUSED'):continue
   if (root/('historical_'+job['condition_id'])/'native.csv').exists():continue
   p=Path(job['artifact_path'])/'checkpoints/final_recovered.pt'
   if not p.exists():save(root/f'missing_historical_{job["condition_id"]}.json',dict(path=str(p),package=package));continue
   actual=sha(p)
   if job.get('artifact_sha256') and actual!=job['artifact_sha256']:raise RuntimeError('Historical hash mismatch')
   state=torch.load(p,map_location='cpu',weights_only=False)['model'];age=int(float(job['router_age']))
   if job['job_type']=='dense':
    base={k:states[str(age)][k] if '.router.' in k else v for k,v in states['0'].items()}
    measure(f'historical_dense_e32_seed{seed}_R{age}',state,base,dict(path=str(p),sha256=actual,package=package,condition_id=job['condition_id']));continue
   sp=float(job['sparsity']);kind=job['mask_type'];mask=torch.load(info['masks'][str(sp)][kind+'_path'],map_location='cpu',weights_only=True)
   base={k:states[str(age)][k] if '.router.' in k else v*mask[k] if k in mask else v for k,v in states['0'].items()}
   measure('historical_'+job['condition_id'],state,base,dict(path=str(p),sha256=actual,package=package))
 for job in rows(ROOT/'results/final_dataset_a/inputs/dense_recovery_manifest.csv'):
  if job['num_experts']!='32' or int(job['seed'])!=seed or int(job['router_age']) not in AGES:continue
  p=Path(job['path'])/'checkpoints/final_recovered.pt'
  if not p.exists():save(root/f'missing_historical_{job["condition_id"]}.json',dict(path=str(p),kind='dense'));continue
  assert sha(p)==job['final_checkpoint_sha256'];age=int(job['router_age']);state=torch.load(p,map_location='cpu',weights_only=False)['model'];base={k:states[str(age)][k] if '.router.' in k else v for k,v in states['0'].items()}
  measure('historical_'+job['condition_id'],state,base,dict(path=str(p),sha256=sha(p),kind='dense'))
 save(root/'complete.json',dict(seed=seed,unix=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);analyze(p.parse_args().seed)
