"""Inference-only initial/endpoint comparisons. No optimizer and no training entrypoint.

Two deterministic shards; per-group locks and immutable cache identities. The
historical full validation losses define labels, never the smaller probe losses.
"""
import argparse,copy,csv,fcntl,gzip,hashlib,json,os,socket,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from numerics import spectrum,geometry,cka,associations
OUT=ROOT/'results/ticket_geometry_endpoint_v1'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp')
 q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def frame(p,rows):
 if not rows:return
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
def npy(t):return t.detach().cpu().numpy()

def capture(model,tokens):
 with torch.no_grad():
  y=model(tokens)
  return [x.detach() for x in y.pre_router_hidden_states],[(npy(r.selected_expert_indices.reshape(-1,model.blocks[i].moe.top_k)),npy(r.accepted_mask)) for i,r in enumerate(y.routes)]

def stateselection(model,inputs):
 with torch.no_grad():
  return [(npy((r:=b.moe(x)[2]).selected_expert_indices.reshape(-1,b.moe.top_k)),npy(r.accepted_mask)) for b,x in zip(model.blocks,inputs)]

def load_model(cfg,state,device):
 m=TinyMoELanguageModel(cfg).to(device).float().eval();m.load_state_dict(state,strict=True);return m

def masks_checked(state,mask):
 for n,m in mask.items():
  if n not in state or tuple(m.shape)!=tuple(state[n].shape):raise RuntimeError('Mask shape/key mismatch '+n)
  if not bool(((m==0)|(m==1)).all()):raise RuntimeError('Nonbinary mask '+n)
  if bool(state[n][~m.bool()].any()):raise RuntimeError('Pruned weights nonzero '+n)

def initial_state(base,router,mask):
 return {n:(router[n].clone() if parameter_group(n)=='router' else (t*mask[n] if n in mask else t.clone())) for n,t in base.items()}

def validate_initial(state,meta):
 for group,key in [('expert','expert_state_hash'),('shared','shared_state_hash'),('router','initial_router_state_hash')]:
  if not meta.get(key):raise RuntimeError('Missing initialization provenance '+key)
  h=state_dict_hash({n:v for n,v in state.items() if parameter_group(n)==group})
  if h!=meta[key]:raise RuntimeError('Initialization hash mismatch: '+key)

def metadata(path,expected_loss):
 p=Path(path);m=read(p/'metadata.json');s=read(p/'summary.json')
 if abs(float(s['final_validation_loss'])-expected_loss)>1e-9:raise RuntimeError('Historical endpoint loss mismatch '+str(p))
 if m.get('router_mode')!='trainable' or m.get('optimizer')!='fresh_AdamW' or int(m['recovery_steps'])!=2500:
  raise RuntimeError('Unsupported historical protocol '+str(p))
 if m.get('confidence_control') or float(m.get('temperature',1))!=1:raise RuntimeError('Unsupported confidence treatment')
 return m

def analyze_state(model,initial,source,dense,source_inputs,source_assign,initial_inputs,initial_assign,initial_fixed_assign,tokens,targets,sid,step,rows):
 inputs,assign=capture(model,tokens)
 fixed_assign=stateselection(model,source_inputs)
 with torch.no_grad():probe_loss=float(torch.nn.functional.cross_entropy(model(tokens).logits.flatten(0,1),targets.flatten()))
 layers=[]
 for l,b in enumerate(model.blocks):
  x=inputs[l].reshape(-1,model.config['d_model']);xref=source_inputs[l].reshape_as(x)
  with torch.no_grad():dense_native=dense_inputs_global[l].reshape_as(x)
  layers.append(dict(condition=sid,step=step,layer=l,source_hidden_cka=cka(npy(x),npy(xref)),initial_hidden_cka=cka(npy(x),npy(initial_inputs[l].reshape_as(x))),dense_hidden_cka=cka(npy(x),npy(dense_native)),source_assignment_agreement=float((assign[l][0]==source_assign[l][0]).all(1).mean()),initial_assignment_agreement=float((assign[l][0]==initial_assign[l][0]).all(1).mean()),probe_ce=probe_loss))
  routers=npy(b.moe.router.projection.weight)
  # Geometry references at the same endpoint, original source, and own initialization.
  for e,expert in enumerate(b.moe.experts):
   refs={label:spectrum(npy(m.blocks[l].moe.experts[e].fc1.weight)) for label,m in [('source',source),('initial',initial),('dense',dense)]}
   metric=geometry(npy(expert.fc1.weight),routers,e,refs)
   rows.append(dict(kind='geometry',condition=sid,step=step,layer=l,expert=e,**metric))
  for mode,z,current,frozen in [('fixed_source',xref,fixed_assign[l],initial_fixed_assign[l]),('native',x,assign[l],initial_assign[l])]:
   with torch.no_grad():
    scores=npy(b.moe.router.projection(z));act=[]
    for e,expert in enumerate(b.moe.experts):
     act.append(npy(expert.activation(expert.fc1(z)).mean(-1)))
     # Same input fed to sparse and dense experts: compare full FFN outputs.
     sample=z[::4];v=expert(sample);u=dense.blocks[l].moe.experts[e](sample)
     denom=float(u.square().sum());valid=(v.norm(dim=-1)>0)&(u.norm(dim=-1)>0)
     rows.append(dict(kind='function',condition=sid,step=step,layer=l,expert=e,mode=mode,dense_relative_output_error=float((v-u).square().sum()/denom) if denom else None,dense_output_cosine=float(torch.nn.functional.cosine_similarity(v[valid],u[valid],dim=-1).mean()) if bool(valid.any()) else None,n_output_pairs=len(sample),n_cosine_pairs=int(valid.sum())))
   acts=np.stack(act,axis=1)
   ids,accepted=current;fi,fa=frozen;ss,sa=source_assign[l]
   sets={'all':(None,None),'current_proposed':(ids,np.ones_like(accepted)), 'current_accepted':current,'initial_proposed':(fi,np.ones_like(fa)),'initial_accepted':frozen,'source_proposed':(ss,np.ones_like(sa))}
   for r in associations(scores,acts,sets):rows.append(dict(kind='activation',condition=sid,step=step,layer=l,mode=mode,**r))
 emb=npy(model.token_embedding.weight);layers.append(dict(condition=sid,step=step,layer=-1,source_embedding_cka=cka(emb,npy(source.token_embedding.weight)),dense_embedding_cka=cka(emb,npy(dense.token_embedding.weight)),probe_ce=probe_loss))
 return layers


def run_group(g,args):
 global dense_inputs_global
 dest=OUT/g['id'];dest.mkdir(parents=True,exist_ok=True)
 lock=(dest/'group.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 start=time.monotonic();identities={}
 def verified(path,expected=None):
  p=Path(path)
  if not p.is_file():raise FileNotFoundError(str(p))
  h=sha(p)
  if expected and h!=expected:raise RuntimeError('SHA mismatch '+str(p))
  identities[str(p)]=h;return p
 def state(path,expected):
  p=verified(path,expected)
  d=torch.load(p,map_location='cpu',weights_only=False)
  s=d['model'] if 'model' in d else d
  if not all(torch.isfinite(v).all() for v in s.values()):raise RuntimeError('Nonfinite model '+str(p))
  return s
 info=g['source'];cfg=copy.deepcopy(info['config']);e=g['num_experts'];age=g['age']
 if int(cfg['model']['num_experts'])!=e:raise RuntimeError('Architecture mismatch')
 cps=info['checkpoints'];base=state(cps['0']['path'],cps['0']['sha256']);source_state=state(cps['100']['path'],cps['100']['sha256']);router=state(cps[str(age)]['path'],cps[str(age)]['sha256'])
 dense_state=state(Path(g['dense']['path'])/'checkpoints/final_recovered.pt',g['dense']['sha256'])
 dm=metadata(g['dense']['path'],g['dense']['loss'])
 for n in ['metadata.json','summary.json']:verified(Path(g['dense']['path'])/n)
 dense_initial=initial_state(base,router,{})
 validate_initial(dense_initial,dm)
 for k in ['train_path','validation_path']:
  cfg['data'][k]=str(verified(ROOT/cfg['data'][k]))
 # Stable validation sequence indices; preserve sequence order for capacity effects.
 _,val=build_dataloaders(cfg['data'],batch_size=32,seed=g['seed'],reshuffle_each_epoch=False)
 if len(val.dataset)<32:raise RuntimeError('Need at least 32 validation sequences')
 rng=np.random.default_rng(41807);indices=np.sort(rng.choice(len(val.dataset),32,replace=False))
 tokens=torch.stack([val.dataset[int(i)][0] for i in indices]).to(args.device)
 targets=torch.stack([val.dataset[int(i)][1] for i in indices]).to(args.device)
 source=load_model(cfg['model'],source_state,args.device)
 source_inputs,source_assign=capture(source,tokens)
 controls=[dict(id=g['id'],path=g['dense']['path'],sha256=g['dense']['sha256'],loss=g['dense']['loss'],gap=0,ticket=False,sparsity=0)]+g['sparse']
 summary=[]
 for c in controls:
  cp=Path(c['path']);meta=metadata(cp,c['loss'])
  for field in ['seed','router_age_percent','router_step','training_batch_sequence_hash','validation_batch_sequence_hash','recovery_steps','shared_state_hash','initial_router_state_hash']:
   if meta.get(field) is None or meta.get(field)!=dm.get(field):raise RuntimeError('Dense comparator mismatch '+c['id']+' '+field)
  if int(meta['router_age_percent'])!=age or int(meta['seed'])!=g['seed']:raise RuntimeError('Plan/metadata mismatch')
  for name in ['metadata.json','summary.json']:verified(cp/name)
  mask={}
  if c['sparsity']:
   verified(c['mask_path']);mask=torch.load(c['mask_path'],map_location='cpu',weights_only=True)
   if state_dict_hash(mask)!=c['mask_hash'] or meta['mask_hash']!=c['mask_hash']:raise RuntimeError('Mask hash mismatch')
  init=initial_state(base,router,mask);validate_initial(init,meta)
  final=dense_state if not c['sparsity'] else state(cp/'checkpoints/final_recovered.pt',c['sha256'])
  masks_checked(final,mask)
  key=hashlib.sha256(json.dumps(dict(group=g,condition=c,identities=identities,code=CODE,indices=indices.tolist(),probe='validation32_seed41807',metrics_version=1),sort_keys=True).encode()).hexdigest()
  out=dest/c['id'];out.mkdir(exist_ok=True)
  if (out/'complete.json').is_file():
   done=read(out/'complete.json')
   if done['key']!=key:raise RuntimeError('Cache identity changed; use a new output version')
   if all(sha(out/n)==h for n,h in done['outputs'].items()):
    print('CACHED',c['id'],flush=True);summary.append(done);continue
   raise RuntimeError('Cached output changed')
  initial=load_model(cfg['model'],init,args.device);initial_inputs,initial_assign=capture(initial,tokens);initial_fixed=stateselection(initial,source_inputs)
  raw=[];layer_rows=[]
  for step,st,ds in [(0,init,dense_initial),(2500,final,dense_state)]:
   model=load_model(cfg['model'],st,args.device);dense=load_model(cfg['model'],ds,args.device)
   dense_inputs_global,_=capture(dense,tokens)
   with torch.no_grad():layer_rows+=analyze_state(model,initial,source,dense,source_inputs,source_assign,initial_inputs,initial_assign,initial_fixed,tokens,targets,c['id'],step,raw)
   del model,dense
  del initial
  tmp=out/'experts.jsonl.gz.tmp'
  with gzip.open(tmp,'wt') as f:
   for r in raw:f.write(json.dumps(r,allow_nan=False)+'\n')
  tmp.replace(out/'experts.jsonl.gz');frame(out/'layers.csv',layer_rows)
  done=dict(key=key,condition=c,group=g['id'],num_experts=e,seed=g['seed'],age=age,host=socket.getfqdn(),checkpoint_hashes=identities.copy(),probe_sequence_indices=indices.tolist(),probe_note='validation subset, already used for historical ticket selection; not independent test data',training_launched=False,steps=[0,2500],initialization_verified=True,outputs={n:sha(out/n) for n in ['experts.jsonl.gz','layers.csv']})
  save(out/'complete.json',done);summary.append(done);print('COMPLETE',c['id'],flush=True)
 save(dest/'complete.json',dict(group=g['id'],conditions=len(summary),elapsed_seconds=time.monotonic()-start,host=socket.getfqdn()))
 return time.monotonic()-start

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',default='cuda');parser.add_argument('--shard',type=int,choices=[0,1],default=0);parser.add_argument('--limit-groups',type=int);parser.add_argument('--min-free-mib',type=int,default=12288);args=parser.parse_args()
 import torch
 from moe_lth.models import TinyMoELanguageModel
 from moe_lth.data import build_dataloaders
 from moe_lth.pruning.router_age import parameter_group,state_dict_hash
 torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 CODE={str(p.relative_to(ROOT)):sha(p) for p in list(HERE.glob('*.py'))+list((ROOT/'src/moe_lth/models').glob('*.py'))+[ROOT/'src/moe_lth/data.py',ROOT/'src/moe_lth/pruning/router_age.py']}
 plan=read(HERE/'plan.json')
 canonical=ROOT/'results/dataset_a_full_lower_sparsity/analysis_final/canonical_combined.csv'
 if sha(canonical)!=plan['canonical_sha256']:raise RuntimeError('Canonical table changed; regenerate plan')
 groups=plan['groups'][args.shard::2]
 if args.limit_groups is not None:
  if args.limit_groups<1:parser.error('limit must be positive')
  groups=groups[:args.limit_groups]
 OUT.mkdir(parents=True,exist_ok=True)
 for g in groups:
  if args.device.startswith('cuda'):
   free,total=torch.cuda.mem_get_info(args.device)
   if free<args.min_free_mib*1024**2:raise RuntimeError('Insufficient free GPU memory; rerun later')
  save(OUT/f'shard_{args.shard}_status.json',dict(state='RUNNING',group=g['id'],host=socket.getfqdn(),pid=os.getpid(),time=time.time()))
  try:elapsed=run_group(g,args)
  except Exception as exc:
   save(OUT/f'shard_{args.shard}_status.json',dict(state='FAILED',group=g['id'],error=repr(exc),time=time.time()));raise
  print('GROUP_SECONDS',g['id'],elapsed,flush=True)
 save(OUT/f'shard_{args.shard}_status.json',dict(state='BENCHMARK_COMPLETE' if args.limit_groups else 'COMPLETE',groups=len(groups),time=time.time()))
