"""Run bounded, resumable checkpoint-only measurements. No optimizer exists here."""
from common import *
from measurements import gram,gram_compare,activation_probe
from moe_lth.models import TinyMoELanguageModel
import argparse,time,re,resource

def grams(state,device):
    result={}
    names=[n for n in state if n.endswith('.fc1.weight') and '.moe.experts.' in n]
    for start in range(0,len(names),8):
        chosen=names[start:start+8];w=torch.stack([state[n] for n in chosen]).to(device=device,dtype=torch.float64)
        # Transfer once per chunk. Scalar comparison reductions remain on CPU,
        # avoiding thousands of GPU synchronizations per checkpoint.
        gs=(w.transpose(1,2)@w).cpu()
        result.update({n:gs[i] for i,n in enumerate(chosen)})
    return result

def load_state(p):
    d=torch.load(p,map_location='cpu',weights_only=False)
    return d['model'] if 'model' in d else d

def mask_state(state,mask):return {n:t*mask[n] if n in mask else t for n,t in state.items()}

class Runner:
    def __init__(self,device):
        self.device=torch.device(device);self.sources=read(BASE/'runtime_sources.json')
        self.jobs=pd.read_csv(BASE/'run_manifest.csv');self.code=code_hash();self.identity_cache={};self.mask_cache={}
        self.coverage=[];self.audits=[];self.completed=0
        OUT.mkdir(parents=True,exist_ok=True)
        self.original_manifest_hash=sha(BASE/'run_manifest.csv')
        self.original_analysis_hashes={str(p.relative_to(BASE)):sha(p) for p in (BASE/'analysis').iterdir() if p.is_file()}
        save(OUT/'implementation_config.json',dict(config=CONFIG,code_hash=self.code,original_manifest_sha256=self.original_manifest_hash,
             original_analysis_hashes=self.original_analysis_hashes,torch_version=torch.__version__,device=str(self.device),
             disclosure='Measurements and corrected aggregation selected after inspecting original rank-16 outcomes.'))
    def identity(self,p,expected=None):
        p=Path(p)
        if not p.is_file():raise FileNotFoundError(str(p))
        if str(p) not in self.identity_cache:self.identity_cache[str(p)]=sha(p)
        actual=self.identity_cache[str(p)]
        if expected and actual!=expected:raise RuntimeError('Checkpoint/input SHA mismatch: '+str(p))
        return dict(path=str(p),sha256=actual,size=p.stat().st_size)
    def masks(self,info,sp,kind):
        k=(info['source_id'],str(sp),kind)
        if k not in self.mask_cache:
            m=info['masks'][str(float(sp))];p=m[kind+'_path'];self.identity(p,m[kind+'_file_sha256'])
            mask=torch.load(p,map_location='cpu',weights_only=True)
            from gcommon import P
            assert P.state_dict_hash(mask)==m[kind+'_state_hash']
            self.mask_cache[k]=mask
        return self.mask_cache[k]
    def measure(self,label,state,info,probe,origin,refs,meta,mask=None,expected_init=None):
        dest=OUT/'cache'/info['source_id']/label
        key=hashlib.sha256(json.dumps(dict(origin=origin,refs={k:v[1] for k,v in refs.items()},probe=self.probe_identity,
                    code=self.code,config=CONFIG,meta=meta),sort_keys=True).encode()).hexdigest()
        if (dest/'completed.json').exists() and read(dest/'completed.json')['cache_key']==key:
            self.completed+=1;return 'cached'
        started=time.monotonic();bad=[]
        for n,t in state.items():
            if not bool(torch.isfinite(t).all()):bad.append(n+':nonfinite')
            if mask is not None and n in mask and bool((t[~mask[n].bool()]!=0).any()):bad.append(n+':mask_violation')
            if expected_init is not None and not torch.equal(t,expected_init[n]):bad.append(n+':initialization_mismatch')
        if bad:raise RuntimeError(label+' '+str(bad))
        check_done=time.monotonic();self.model.load_state_dict(state,strict=True)
        before=tensor_hash(self.model.state_dict())
        hashing_done=time.monotonic()
        with protected(self.model):
            gs=grams(state,self.device);records=[]
            for n,g in gs.items():
                layer=int(n.split('.')[1]);expert=int(n.split('.experts.')[1].split('.')[0])
                for ref,(reference,identity) in refs.items():
                    d=gram_compare(g,reference[n]);d.update(layer=layer,expert=expert,reference=ref,
                        current_gram_norm=float(g.norm()),reference_gram_norm=float(reference[n].norm()))
                    records.append(d)
            gram_done=time.monotonic();groups,pools,within,drift,raw=activation_probe(self.model,probe)
            activation_done=time.monotonic()
        assert tensor_hash(self.model.state_dict())==before,'Diagnostics changed model tensors'
        dest.mkdir(parents=True,exist_ok=True)
        frame(dest/'gram_experts.csv',records);frame(dest/'activation_groups.csv',groups);frame(dest/'activation_layer_pools.csv',pools)
        frame(dest/'activation_within_token.csv',within);frame(dest/'input_drift.csv',drift)
        np.savez_compressed(dest/'token_pairs.npz',**raw,positions=probe['positions'].numpy(),tokens=probe['tokens'].numpy())
        result=dict(cache_key=key,meta=meta,origin=origin,reference_identities={k:v[1] for k,v in refs.items()},
             probe=self.probe_identity,model_tensor_sha256=before,finite_tensors=True,mask_applied_verified=mask is not None,
             expected_initialization_verified=expected_init is not None,model_preserved=True,elapsed_seconds=time.monotonic()-started,
             peak_cuda_allocated=torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None,
             peak_cpu_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             timing_seconds=dict(tensor_checks=check_done-started,load_and_hash=hashing_done-check_done,
                                 gram=gram_done-hashing_done,activation=activation_done-gram_done,
                                 hash_and_write=time.monotonic()-activation_done))
        result['output_hashes']={p.name:sha(p) for p in dest.iterdir() if p.name!='completed.json'}
        save(dest/'completed.json',result);self.completed+=1
        print('DONE',label,round(result['elapsed_seconds'],2),'seconds',flush=True);return result
    def seed(self,seed,benchmark=False,recoveries_only=False):
        sid=f'source_e32_seed{seed}_T2500';info=self.sources[sid]
        pp=BASE/'probes'/f'{sid}.pt';self.probe_identity=self.identity(pp,read(pp.with_suffix('.json'))['probe_sha256'])
        probe=torch.load(pp,map_location='cpu',weights_only=True)
        assert probe['tokens'].shape==(8,128) and torch.equal(probe['positions'].sort().values,torch.arange(1024))
        state={};ids={}
        for a in ('0','50','100'):
            rec=info['checkpoints'][a];ids[a]=self.identity(rec['path'],rec['sha256']);state[a]=load_state(rec['path'])
        # Local constructor uses preserved RNG; measurements never advance caller RNG.
        from gcommon import P
        with P.preserve_rng():self.model=TinyMoELanguageModel(info['config']['model']).to(self.device)
        source_g=grams(state['100'],self.device)
        source_ref=(source_g,dict(category='source_final_dense',**ids['100']))
        jobs=self.jobs[self.jobs.seed.eq(seed)]
        for age in (0,50,100):
            agejobs=jobs[jobs.router_age.eq(age)];initial_g={}
            for step in STEPS:
                dense=agejobs[agejobs.mask_type.eq('dense')].iloc[0]
                dp=Path(dense['attempt'])/'geometry'/f'step_{step:04d}'/'diagnostic_model_state.pt'
                if not dp.exists():
                    self.coverage.append(dict(condition_id=dense.condition_id,step=step,status='missing_dense_snapshot',path=str(dp)));continue
                did=self.identity(dp);ds=load_state(dp);dg=grams(ds,self.device)
                dense_ref=(dg,dict(category='matched_new_dense_recovery',condition_id=dense.condition_id,step=step,**did))
                for _,j in agejobs.iterrows():
                    if benchmark and j.mask_type!='dense':continue
                    p=Path(j['attempt'])/'geometry'/f'step_{step:04d}'/'diagnostic_model_state.pt'
                    meta=dict(condition_id=j.condition_id,seed=seed,router_age=age,sparsity=float(j.sparsity),mask_type=j.mask_type,
                              step=step,category='recovery',dense_condition_id=j.dense_condition_id)
                    if not p.exists():self.coverage.append(dict(**meta,status='missing_snapshot',path=str(p)));continue
                    identity=did if j.mask_type=='dense' else self.identity(p)
                    cm=read(p.parent/'metadata.json');assert cm['attempted_step']==step and cm['probe_sha256']==self.probe_identity['sha256']
                    meta['validation_loss']=cm['validation_loss']
                    mask=None if j.mask_type=='dense' else self.masks(info,j.sparsity,j.mask_type)
                    current=ds if j.mask_type=='dense' else load_state(p)
                    initial={n:state[str(age)][n] if '.router.' in n else t for n,t in state['0'].items()}
                    if mask is not None:initial=mask_state(initial,mask)
                    if step==0:initial_g[j.condition_id]=grams(initial,self.device)
                    initidentity=dict(category='recovery_initialization',source_initial=ids['0'],source_router=ids[str(age)],
                                      mask_identity=None if mask is None else info['masks'][str(float(j.sparsity))][j.mask_type+'_state_hash'])
                    refs={'source_final':source_ref,'matched_dense':dense_ref,'recovery_initial':(initial_g[j.condition_id],initidentity)}
                    res=self.measure(f'{j.condition_id}/step_{step:04d}',current,info,probe,identity,refs,meta,mask,initial if step==0 else None)
                    if step==2500:
                        fp=Path(j['attempt'])/'checkpoints/final_recovered.pt';fid=self.identity(fp,j.artifact_sha256)
                        assert tensor_hash(load_state(fp))==tensor_hash(current),'Final snapshot differs from native checkpoint'
                    if benchmark:
                        save(OUT/'benchmark.json',res if isinstance(res,dict) else read(OUT/'cache'/sid/f'{j.condition_id}/step_{step:04d}/completed.json'))
                        return
                del ds,dg
        if recoveries_only:return
        # Audit every stored existing state, with reconstruction from verified original weights.
        existing=BASE/'existing'/sid
        for folder in sorted(existing.iterdir()):
            done=folder/'done.json'
            if not done.exists() or not (folder/'weights.csv').exists():continue
            label=folder.name;rec=read(done);assert rec['probe_sha256']==self.probe_identity['sha256']
            age=None;sp=0.;kind='dense';mask=None;category='existing';origin={}
            if label.startswith('source_age_R'):
                age=int(label.split('_R')[1]);current=state[str(age)];origin=ids[str(age)];category='source_age'
            elif label.startswith('unmasked_init_R'):
                age=int(label.split('_R')[1]);current={n:state[str(age)][n] if '.router.' in n else t for n,t in state['0'].items()}
                origin=dict(source_initial=ids['0'],router=ids[str(age)]);category='unmasked_initialization'
            elif label.startswith('masked_'):
                m=re.search(r'(learned|random)_s([0-9.]+)',label);kind=m[1];sp=float(m[2]);mask=self.masks(info,sp,kind)
                current=mask_state(state['100'] if label.startswith('masked_trained') else state['0'],mask)
                if label.startswith('masked_init'):
                    age=int(re.search(r'_R(\d+)_',label)[1]);current={n:state[str(age)][n] if '.router.' in n else t for n,t in current.items()};category='masked_initialization'
                else:category='masked_trained'
                origin=dict(source=ids['100'] if category=='masked_trained' else ids['0'],router=None if age is None else ids[str(age)],
                            mask=info['masks'][str(sp)][kind+'_state_hash'])
            elif label.startswith('historical_'):
                p=Path(rec['origin']['path'])
                if not p.exists():self.coverage.append(dict(condition_id=label,step=2500,status='missing_historical_weights',path=str(p)));continue
                origin=self.identity(p,rec['origin'].get('sha256'));current=load_state(p);category='historical_endpoint'
                age=int(re.search(r'_R(\d+)',label)[1]);m=re.search(r'(learned|random)_s([0-9.]+)',label)
                if m:kind=m[1];sp=float(m[2]);mask=self.masks(info,sp,kind)
            else:self.coverage.append(dict(condition_id=label,status='unrecognized_existing_state'));continue
            meta=dict(condition_id=label,seed=seed,router_age=age,sparsity=sp,mask_type=kind,step=2500 if category=='historical_endpoint' else None,
                      category=category,dense_condition_id=None,validation_loss=None)
            refs={'source_final':source_ref}
            if category in ('masked_initialization','unmasked_initialization'):
                j=jobs[jobs.router_age.eq(age)&jobs.mask_type.eq('dense')].iloc[0]
                dp=Path(j['attempt'])/'geometry/step_0000/diagnostic_model_state.pt';refs['matched_dense']=(grams(load_state(dp),self.device),dict(category='matched_new_dense_initialization',**self.identity(dp)))
            if category=='historical_endpoint':
                # Historical comparator belongs to the historical family, never substitute a new run.
                candidates=[existing/f'historical_dense_e32_seed{seed}_R{age}',existing/f'historical_dense_e32_seed{seed}_R{age}']
                for candidate in candidates:
                    if (candidate/'done.json').exists():
                        dr=read(candidate/'done.json')['origin'];p=Path(dr['path'])
                        if p.exists():refs['matched_historical_dense']=(grams(load_state(p),self.device),dict(category='historical_matched_dense_endpoint',**self.identity(p,dr.get('sha256'))))
                        break
                # Historical validation losses remain in their original summaries, never a trajectory.
                summary=Path(origin['path']).parent.parent/'summary.json'
                if summary.exists():meta['validation_loss']=read(summary).get('final_validation_loss')
            self.measure(label,current,info,probe,origin,refs,meta,mask)
        save(OUT/'coverage'/f'seed_{seed}.json',dict(seed=seed,missing=self.coverage,completed=self.completed))
        assert sha(BASE/'run_manifest.csv')==self.original_manifest_hash
        assert all(sha(BASE/p)==h for p,h in self.original_analysis_hashes.items()),'Original outputs changed'

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--device',default='cuda');p.add_argument('--benchmark',action='store_true');p.add_argument('--recoveries-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);runner=Runner(a.device)
    for seed in ([a.seed] if a.seed else SEEDS):runner.seed(seed,a.benchmark,a.recoveries_only)
if __name__=='__main__':main()
