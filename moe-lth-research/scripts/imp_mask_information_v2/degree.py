"""Finite binary swap chains with explicit empirical admission diagnostics."""
from v2_common import *
import ctypes
import subprocess
import numpy as np
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.masks import save_masks

_LIB=None


def kernel():
    global _LIB
    if _LIB is None:
        source=Path(__file__).with_name('swaps.cpp')
        target=OUT/'build'/f'swaps_{sha(source)[:16]}.so'
        target.parent.mkdir(exist_ok=True)
        with lock(target.with_suffix('.lock')):
            if not target.exists():
                tmp=target.with_suffix('.building.so')
                subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(tmp)],check=True)
                tmp.replace(target)
        lib=ctypes.CDLL(str(target));lib.binary_swaps.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.c_int64,ctypes.c_int64,ctypes.c_uint64,ctypes.c_uint64];lib.binary_swaps.restype=ctypes.c_uint64
        _LIB=lib
    return _LIB.binary_swaps


def same_degrees(a,b):
    return a.shape==b.shape and a.dtype==b.dtype==torch.bool and torch.equal(a.sum(0),b.sum(0)) and torch.equal(a.sum(1),b.sum(1))


def chain(original,seed):
    if original.ndim!=2 or original.dtype!=torch.bool:raise ValueError('Binary matrix required')
    base=original.cpu().contiguous();a=base.numpy().astype(np.uint8,copy=True)
    count=int(a.sum());n=a.size
    if count in (0,n):raise ValueError('Degenerate mask; near-copy training prohibited')
    fn=kernel();last=0;accepted=0;records=[];ten=None;last_twenty=None
    for length in [2,5,10,20]:
        attempts=(length-last)*count
        # Restarting a PRNG stream between segments uses explicitly separated seeds.
        segment_seed=(int(seed)+length*0x9E3779B97F4A7C15)&((1<<64)-1)
        acc=int(fn(a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),a.shape[0],a.shape[1],segment_seed,attempts));accepted+=acc
        now=torch.from_numpy(a.copy()).bool()
        if not same_degrees(base,now):raise RuntimeError('Swap chain changed degrees')
        overlap=int(torch.logical_and(base,now).sum())/count
        records.append(dict(length=length,attempts=length*count,accepted=accepted,rejection_rate=1-accepted/(length*count),survivor_overlap=overlap))
        if length==10:ten=now
        if length==20:last_twenty=now
        last=length
    p=count/n
    intersection=int(torch.logical_and(ten,last_twenty).sum())/n
    correlation=(intersection-p*p)/(p*(1-p))
    passed=accepted>=2*count and records[-1]['survivor_overlap']<=.5 and abs(records[-1]['survivor_overlap']-records[-2]['survivor_overlap'])<=.02 and abs(correlation)<=.20
    diagnostics=dict(seed=seed,eligible=n,survivors=count,segments=records,accepted_swaps=accepted,attempts=20*count,rejection_rate=1-accepted/(20*count),survivor_overlap=records[-1]['survivor_overlap'],overlap_change_10_to_20=abs(records[-1]['survivor_overlap']-records[-2]['survivor_overlap']),binary_correlation_10_to_20=correlation,degree_counts_exact=True,passed=passed,uniform_sampling_claimed=False,independent_stationary_draws_claimed=False)
    return last_twenty,diagnostics


def matrix_seed(seed,name,replicate,diagnostic=False):
    return int.from_bytes(hashlib.sha256(f'v2-degree:{seed}:{replicate}:{name}:{diagnostic}'.encode()).digest()[:8],'little')


def prepare_degree(manifest):
    phase1=[j for j in manifest['jobs'] if j['phase']==1]
    if not all(job_status(j)['status'] in ('COMPLETE','REUSED') for j in phase1):raise RuntimeError('Phase1 incomplete')
    effects={s:np.mean([result(j)['final_ce'] for j in phase1 if j['seed']==s and j['condition']=='matrix'])-next(result(j)['final_ce'] for j in phase1 if j['seed']==s and j['condition']=='IMP') for s in SEEDS}
    if not matrix_gate(effects):raise RuntimeError('Replication gate not passed')
    save(OUT/'degree_preparation_status.json',dict(status='running',host=socket.getfqdn(),pid=os.getpid(),updated=time.time()))
    save(OUT/'degree_sampler_runtime.json',dict(host=socket.getfqdn(),compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],kernel_source_sha256=sha(Path(__file__).with_name('swaps.cpp')),rng='std::mt19937_64; fixed independent segment seeds',lengths=[2,5,10,20],uniform_sampling_claimed=False,independent_stationary_draws_claimed=False))
    diagnostics=[];production=[];all_pass=True;seen=set()
    for seed in SEEDS:
        c=manifest['cohorts'][str(seed)];original=load_masks(c['mask_source'])
        # Two independent starts per matrix, checked at all registered lengths.
        for name,m in sorted(original.items()):
            checks=[]
            for replicate in range(2):
                _,d=chain(m,matrix_seed(seed,name,replicate,True));d.update(training_seed=seed,matrix=name,diagnostic_chain=replicate);checks.append(d);diagnostics.append(d)
            between=abs(checks[0]['survivor_overlap']-checks[1]['survivor_overlap'])
            for d in checks:d['between_chain_overlap_difference']=between;d['passed']=d['passed'] and between<=.02
            if not all(d['passed'] for d in checks):all_pass=False
        save(OUT/'degree_diagnostics.json',dict(checks=diagnostics,passed=all_pass,complete=False))
        save(OUT/'degree_preparation_status.json',dict(status='diagnostics',last_seed=seed,pid=os.getpid(),host=socket.getfqdn(),updated=time.time()))
        if not all_pass:break
    if not all_pass:
        save(OUT/'degree_diagnostics.json',dict(checks=diagnostics,passed=False,complete=True,limitation='Prespecified mixing/overlap diagnostics failed; no degree-mask recovery admitted.'))
        return False
    # Construct all production masks before admitting even the first recovery.
    for job in [j for j in manifest['jobs'] if j['phase']==2]:
        original=load_masks(manifest['cohorts'][str(job['seed'])]['mask_source']);masks={};ds=[]
        for name,m in sorted(original.items()):
            masks[name],d=chain(m,matrix_seed(job['mask_seed'],name,job['replicate']));d.update(job=job['id'],training_seed=job['seed'],matrix=name);ds.append(d)
        valid=all(d['passed'] for d in ds)
        digest=recovery._mask_hash(masks)
        if digest in seen:valid=False
        seen.add(digest)
        if not valid:
            save(OUT/'degree_diagnostics.json',dict(checks=diagnostics,production=production+ds,passed=False,complete=True,limitation='Production diagnostic or distinct-mask check failed; no degree training.'))
            return False
        save_masks(masks,job['mask_path']);loaded=load_masks(job['mask_path'])
        if not all(torch.equal(masks[n],loaded[n]) and same_degrees(original[n],loaded[n]) for n in original):raise RuntimeError('Degree mask persistence failed')
        save(OUT/'degree_masks'/f"{job['id']}.json",dict(mask_path=job['mask_path'],mask_hash=digest,mask_sha256=sha(job['mask_path']),checks=ds,passed=True))
        production.extend(ds)
        save(OUT/'degree_preparation_status.json',dict(status='production',last_job=job['id'],pid=os.getpid(),host=socket.getfqdn(),updated=time.time()))
    save(OUT/'degree_diagnostics.json',dict(checks=diagnostics,production=production,passed=True,complete=True,compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],kernel_source_sha256=sha(Path(__file__).with_name('swaps.cpp')),uniform_sampling_claimed=False,independent_stationary_draws_claimed=False))
    return True
