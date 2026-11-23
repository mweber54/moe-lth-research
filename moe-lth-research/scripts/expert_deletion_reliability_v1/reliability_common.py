"""Fixed-budget, one-source checkpoint-only deletion assay; historical artifacts read-only."""
from __future__ import annotations
import contextlib,hashlib,importlib.util,json,os,socket,sys,threading,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/expert_local_deletion_dose_v1'))
import dose_common as dc
s=dc.s
D=ROOT/'results/expert_deletion_reliability_v1'
PROBE_SEED=171032
BOOTSTRAP_SEED=1732000
RESAMPLES=2000
MIN_ACCEPTED=32
FLOOR=1e-12
save,table,load,sha,digest=dc.save,dc.table,dc.load,dc.sha,dc.digest


def content_hash(dataset,index):
    x,y=dataset[index]
    return hashlib.sha256(torch.cat((x,y[-1:])).numpy().tobytes()).hexdigest()


def choose_sequences(dataset,old_indices,old_content_hashes,seed=PROBE_SEED,needed=4096):
    """Deterministic availability-only greedy sampler; no model/routing input."""
    order=np.random.default_rng(seed).permutation(len(dataset));used=set(map(int,old_indices));seen=set(old_content_hashes);chosen=[]
    rejections=dict(source_overlap=0,duplicate_context_content=0)
    for raw in order:
        i=int(raw)
        if i in used or i-1 in used or i+1 in used:
            rejections['source_overlap']+=1;continue
        h=content_hash(dataset,i)
        if h in seen:rejections['duplicate_context_content']+=1;continue
        chosen.append((i,h));used.add(i);seen.add(h)
        if len(chosen)==needed:break
    return chosen,rejections


def batch_identity(i):
    reg=s.read(D/'configuration.json');p=reg['probes'][i]
    return dict(batch=i,block=p['block'],partition=p['partition'],inputs_sha256=p['inputs_sha256'],targets_sha256=p['targets_sha256'],sequence_contexts_sha256=p['sequence_contexts_sha256'])


def verify_frozen():
    reg=s.read(D/'configuration.json');seal=s.read(D/'freeze.json')
    for path,key in [('configuration.json','configuration_sha256'),('protocol.md','protocol_sha256'),('probes.pt','probes_sha256')]:assert sha(D/path)==seal[key],path
    for p,h in seal['implementation'].items():assert sha(ROOT/p)==h,p
    for p,h in reg['source_files'].items():assert sha(ROOT/p)==h,p
    return reg


def interventions():return load(dc.D/'intervention_coordinates.pt')
def cache_path(kind,t,b):return D/'cache'/f'{kind}_{t}_{b:02d}.pt'
def cache_identity(kind,t,b):
    reg=s.read(D/'configuration.json');j=s.jobs()[kind]
    return dict(checkpoint=reg['checkpoints'][j['id']][str(t)],batch=batch_identity(b),interventions_sha256=reg['source_files'][str((dc.D/'intervention_coordinates.pt').relative_to(ROOT))],configuration_sha256=sha(D/'configuration.json'),implementation_sha256=s.read(D/'freeze.json')['implementation_sha256'],numerics='FP32_eval_inference_TF32off_native')


def energies(base_blocks,outputs):
    n=sum(len(b['output']) for b in base_blocks);elements=sum(b['output'].numel() for b in base_blocks)
    num=den=gate_num=gate_den=0.
    for b,z in zip(base_blocks,outputs):
        y=b['output'].double();d=z.double()-y;g=b['gates'].double().reshape(-1,1)
        num+=float(d.square().sum());den+=float(y.square().sum());gate_num+=float((g*d).square().sum());gate_den+=float((g*y).square().sum())
    return dict(output_elements=elements,output_difference_energy=num,baseline_output_energy=den,gate_weighted_difference_energy=gate_num,baseline_gate_weighted_energy=gate_den,output_rms_difference=(num/elements)**.5 if elements else None,baseline_output_rms=(den/elements)**.5 if elements else None,relative_output_rms_difference=(num/den)**.5 if elements and (den/elements)**.5>dc.OUTPUT_EPS else None,gate_weighted_rms_difference=(gate_num/elements)**.5 if elements else None,accepted_output_positions=n)


def aggregate_energies(rows):
    fields=('output_elements','output_difference_energy','baseline_output_energy','gate_weighted_difference_energy','baseline_gate_weighted_energy')
    total={k:sum(r[k] for r in rows) for k in fields};n=total['output_elements'];num=total['output_difference_energy'];den=total['baseline_output_energy']
    ratios=[r['relative_output_rms_difference'] for r in rows if r['relative_output_rms_difference'] is not None and np.isfinite(r['relative_output_rms_difference'])]
    total.update(pooled_output_rms_difference=(num/n)**.5 if n else None,pooled_baseline_output_rms=(den/n)**.5 if n else None,pooled_relative_output_rms_difference=(num/den)**.5 if n and (den/n)**.5>dc.OUTPUT_EPS else None,pooled_gate_weighted_rms_difference=(total['gate_weighted_difference_energy']/n)**.5 if n else None,mean_defined_batch_relative_rms=float(np.mean(ratios)) if ratios else None,defined_batch_ratios=len(ratios))
    return total


def subset_statistics(delta,baseline,deleted,accepted):
    result=[];sequence=[]
    for subset,mask in [('whole',torch.ones(len(delta),dtype=torch.bool)),('baseline_accepted',accepted),('complement',~accepted)]:
        count=int(mask.sum());d=float(delta[mask].sum());b=float(baseline[mask].double().sum());z=float(deleted[mask].double().sum())
        result.append(dict(subset=subset,target_tokens=count,baseline_loss_sum=b,deleted_loss_sum=z,damage_sum=d,baseline_ce=b/count if count else None,deleted_ce=z/count if count else None,damage=d/count if count else None))
        sequence.append(dict(subset=subset,counts=mask.reshape(-1,128).sum(1).to(torch.int32),paired_loss_sums=(delta*mask).reshape(-1,128).sum(1)))
    return result,sequence


def coverage_stats(base,e):
    proposed=base['ids']==e;accepted=proposed&base['accepted'];pg=base['gates'][proposed].double();ag=base['gates'][accepted].double();np_=int(proposed.sum());na=int(accepted.sum())
    out=dict(proposed_tokens=np_,accepted_tokens=na,capacity_drops=np_-na,acceptance_rate=na/np_ if np_ else None)
    for label,g in [('proposed',pg),('accepted',ag)]:
        out.update({label+'_gate_sum':float(g.sum()),label+'_gate_mean':float(g.mean()) if len(g) else None,label+'_gate_min':float(g.min()) if len(g) else None,label+'_gate_max':float(g.max()) if len(g) else None})
    return out


def mask_check(m,masks):
    p=dict(m.named_parameters())
    for n,mask in masks.items():assert torch.count_nonzero(p[n][~mask.to(p[n].device).bool()])==0


@contextlib.contextmanager
def worker():
    import fcntl
    D.mkdir(parents=True,exist_ok=True);handles=[];stop=threading.Event()
    try:
        for label in ('study',socket.gethostname()):
            p=D/'locks'/(label+'.lock');p.parent.mkdir(exist_ok=True);f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(f)
        def beat():
            while not stop.is_set():save(D/'heartbeat.json',dict(host=socket.gethostname(),pid=os.getpid(),unix=time.time()));stop.wait(15)
        thread=threading.Thread(target=beat,daemon=True);thread.start()
        try:yield
        finally:stop.set();thread.join()
    finally:
        for f in handles:f.close()
