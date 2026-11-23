"""Measure verified historical and instrumented snapshots; output hashes gate reuse."""
from common import *
from measure import *
from gradient import diagnostic
from runner import batches
import argparse,fcntl
_ORIGINAL_STATE=state
_CHECKED={}
def state(p):
 p=Path(p);name=str(p)
 if name not in _CHECKED:
  expected={x['path']:x['sha256'] for x in frozen()['files']}.get(name)
  if expected:verify_file(p,expected)
  else:
   meta=p.with_suffix('.json')
   if not meta.exists():raise RuntimeError('Unregistered model snapshot '+name)
   verify_file(p,read(meta)['snapshot_sha256'])
  _CHECKED[name]=True
 return _ORIGINAL_STATE(p)

def valid_cache(dest,key):
 p=dest/'completed.json'
 if not p.exists():return False
 d=read(p)
 return d['key']==key and all((dest/n).exists() and sha(dest/n)==h for n,h in d['outputs'].items())
def _write_probe(job,val):
 p=OUT/'probes'/f'seed_{job["seed"]}.pt';p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():return load(p)
 src=model(state(job['source_reference']),job['config']);tokens,targets=val[0];positions=torch.randperm(tokens.numel(),generator=torch.Generator().manual_seed(72419))[:2048]
 with protected(src):
  o=src(tokens.cuda());full=[x.cpu() for x in o.pre_router_hidden_states]
 probe=dict(tokens=tokens,targets=targets,positions=positions,source_full_inputs=full,validation_hash=R._batch_sequence_hash(val),batch_index=0)
 torch.save(probe,p);save(p.with_suffix('.json'),dict(sha256=sha(p),batch=0,positions=positions.tolist(),validation_hash=probe['validation_hash'],source=verify_file(job['source_reference']),selection='fixed randperm seed 72419, no reordering for capacity',independent_test=False));return probe

def write_probe(job,val):
 directory=OUT/'probes';directory.mkdir(parents=True,exist_ok=True)
 with (directory/f'seed_{job["seed"]}.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);return _write_probe(job,val)

def run_seed(seed,benchmark=False,track='historical'):
 jobs=[j for j in frozen()['conditions'] if j['seed']==seed];j=jobs[0];c=j['config'];_,val=batches(j,total=1);assert R._batch_sequence_hash(val)==j['canonical_pair']['validation_hash'];probe=write_probe(j,val)
 source_state=state(j['source_reference']);src=model(source_state,c)
 with protected(src):_,source_cache=capture(src,probe['tokens'],probe['positions']);source_assign=assignments(src,probe)
 del src
 origins={};initials={};initial_cache={};initial_assign={};mask_sources={}
 for j in jobs:
  masks=load_masks(j['mask']) if j['mask'] else {};init=masked(state(j['initial']),masks);initials[j['id']]=init;im=model(init,c)
  with protected(im):_,initial_cache[j['id']]=capture(im,probe['tokens'],probe['positions']);initial_assign[j['id']]=assignments(im,probe)
  del im
  if j['mask']:
   meta=read(Path(j['mask']).with_suffix('.json'));mask_sources[j['id']]=state(meta['trained_checkpoint'])
  else:mask_sources[j['id']]=source_state
  if track=='historical':origins[j['id']]={0:None,2500:j['boundary'],**{int(k):v for k,v in j['historical_checkpoints'].items()}}
  else:
   base=OUT/'reruns'/j['id'];done=base/'completed.json'
   if not done.exists():continue
   d=read(done);verify_file(d['checkpoint'],d['sha256']);attempt=Path(d['attempt']);origins[j['id']]={step:str(attempt/'snapshots'/f'step_{step:04d}.pt') for step in STEPS}
 if not origins:return
 densejob=next(j for j in jobs if j['sparsity'] is None);denseid=densejob['id'];densefinal=state(densejob['historical_checkpoints']['5000']) if track=='historical' else state(origins[denseid][5000]);raw_benchmark=[]
 for step in sorted(set(s for ss in origins.values() for s in ss)):
  densepath=origins[denseid][step];ds=initials[denseid] if densepath is None else state(densepath);dm=model(ds,c)
  with protected(dm):do,dcache=capture(dm,probe['tokens'],probe['positions'])
  dlog=do.logits.detach().reshape(-1,256)[probe['positions'].cuda()].float();del dm,do
  for j in jobs:
   if j['id'] not in origins or step not in origins[j['id']]:continue
   origin=origins[j['id']][step];s=initials[j['id']] if origin is None else state(origin);masks=load_masks(j['mask']) if j['mask'] else {}
   for n,mask in masks.items():assert not torch.count_nonzero(s[n][~mask])
   dest=OUT/'analysis'/track/j['id']/f'step_{step:04d}';dest.mkdir(parents=True,exist_ok=True)
   key=hashlib.sha256(json.dumps(dict(origin=verify_file(origin) if origin else dict(init=verify_file(j['initial']),mask=verify_file(j['mask']) if j['mask'] else None),probe=sha(OUT/'probes'/f'seed_{seed}.pt'),code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},cohort=sha(OUT/'cohort.json')),sort_keys=True).encode()).hexdigest()
   if valid_cache(dest,key):continue
   started=time.monotonic();m=model(s,c);before=state_dict_hash(m.state_dict())
   refs=dict(mask_selection_source=mask_sources[j['id']],recovery_initial=initials[j['id']],same_step_dense=ds,fixed_dense_final=densefinal)
   with protected(m):
    measures,raw,cache=probe_measure(m,probe,source_cache,initial_cache[j['id']],source_assign,initial_assign[j['id']],dcache,ds)
    wg,qe=weight_geometry(s,refs,torch.device('cuda'));emb,erows=embedding(s,dict(original_initial=state(j['initial']),source=source_state,same_step_dense=ds))
   with protected(m,strict_fp32=False):ce=R._evaluate_condition_loss(m,val,torch.device('cuda'),False,1.)
   raw['positions']=probe['positions'].numpy();raw['tokens']=probe['tokens'].numpy();raw['targets']=probe['targets'].numpy()
   x=torch.tensor(raw['logits'],device='cuda');lp=F.log_softmax(x,dim=-1);dlp=F.log_softmax(dlog,dim=-1);raw['kl_sparse_to_dense']=(lp.exp()*(lp-dlp)).sum(-1).cpu().numpy();raw['dense_probe_ce']=F.cross_entropy(dlog,probe['targets'].reshape(-1)[probe['positions']].cuda(),reduction='none').cpu().numpy()
   raw['per_token_ce_difference']=raw['probe_ce']-raw['dense_probe_ce']
   np.savez_compressed(dest/'raw.npz',**raw)
   for name,rows in measures.items():table(dest/(name+'.csv'),rows)
   table(dest/'weight_geometry.csv',wg);table(dest/'router_energy.csv',qe);table(dest/'embedding.csv',emb);table(dest/'embedding_rows.csv',erows)
   if step in (0,2500,5000):
    with protected(m):gr=diagnostic(m,probe,masks)
    table(dest/'gradient_identity.csv',gr)
   assert state_dict_hash(m.state_dict())==before
   seconds=time.monotonic()-started
   result=dict(key=key,seed=seed,step=step,condition=j['id'],sparsity=j['sparsity'],track=track,validation_ce=ce,seconds=seconds,peak_cuda_bytes=torch.cuda.max_memory_allocated(),model_preserved=True,probe_kl_mean=float(raw['kl_sparse_to_dense'].mean()),probe_ce_mean=float(raw['probe_ce'].mean()),outputs={p.name:sha(p) for p in dest.iterdir() if p.name!='completed.json'})
   save(dest/'completed.json',result);print(j['id'],track,step,round(seconds,2),'s',flush=True);del m,s
   if benchmark:
    raw_benchmark.append(result)
    counts=[r['n'] for r in measures['groups'] if r['selection']=='current_accepted'];save(OUT/'benchmark.json',dict(result=result,count_min=min(counts),count_median=float(np.median(counts)),under8=sum(n<8 for n in counts),groups=len(counts)))
    cfg=OUT/'probe_config.json'
    if not cfg.exists():save(cfg,dict(full_batch_sequences=128,sequence_length=128,selected_positions=2048,counterfactual_positions=128,generator_seed=72419,min_samples=8,variance_floor=1e-12,benchmark=sha(OUT/'benchmark.json'),frozen_unix=time.time()))
    return
 # Coverage is explicit, including missing early historical states.
 table(OUT/'coverage'/f'{track}_seed{seed}.csv',[dict(condition=j['id'],step=s,available=s in origins.get(j['id'],{}),reconstructed=s==0,track=track) for j in jobs for s in STEPS])

# Keep the public probe data name separate from the measurement function.
probe_measure=probe

def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS));p.add_argument('--benchmark',action='store_true');p.add_argument('--track',choices=['historical','rerun'],default='historical');a=p.parse_args();torch.set_num_threads(2);R.configure_device(torch.device('cuda'))
 for seed in a.seeds:
  directory=OUT/'locks';directory.mkdir(parents=True,exist_ok=True)
  with (directory/f'analysis_{a.track}_{seed}.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run_seed(seed,a.benchmark,a.track)
if __name__=='__main__':main()
