"""Expert-local checkpoint deletion doses. No training/optimizer entrypoints."""
from __future__ import annotations
import contextlib
import importlib.util
import json
import os
import random
import socket
import threading
import time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
_spec=importlib.util.spec_from_file_location('stage1_readonly',ROOT/'scripts/expert_role_deletion_v1/study.py')
s=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(s)
D=ROOT/'results/expert_local_deletion_dose_v1'
ORDER_SEEDS=(170901,170902,170903,170904)
DOSES=(0,1,2,4,8)
OUTPUT_EPS=1e-12  # absolute baseline RMS floor; undefined ratios, never clipped
MIN_ACCEPTED=32
save,table,load,sha,digest=s.save,s.table,s.load,s.sha,s.digest


def build_panel(groups):
    """Unique expert/coordinate-set interventions and their nested ordering aliases."""
    orders={str(seed):np.random.default_rng(seed).permutation(8).tolist() for seed in ORDER_SEEDS}
    unique={};aliases=[]
    for e in s.EXPERTS:
        for ordering,order in orders.items():
            for dose in DOSES:
                keys=tuple(sorted(f'e{e}_g{i}' for i in order[:dose]))
                identity=(e,keys)
                if identity not in unique:
                    idx=torch.cat([groups[k]['indices'] for k in keys]).sort().values if keys else torch.empty(0,dtype=torch.long)
                    iid=f'e{e}_n{dose}_'+digest(idx)[:16]
                    unique[identity]=dict(intervention=iid,expert=e,parameter=s.name(e),kind='weight_union',dose_groups=dose,groups=list(keys),indices=idx)
                aliases.append(dict(expert=e,ordering_seed=int(ordering),dose_groups=dose,ordered_groups=';'.join(f'e{e}_g{i}' for i in order[:dose]),intervention=unique[identity]['intervention']))
        iid=f'e{e}_output_zero'
        unique[(e,'output_zero')]=dict(intervention=iid,expert=e,parameter=s.name(e),kind='expert_output_zero',dose_groups=None,groups=[],indices=torch.empty(0,dtype=torch.long))
    return list(unique.values()),aliases,orders


def rng_snapshot():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state().clone(),cuda=[r.clone() for r in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])
def equal_tree(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


@contextlib.contextmanager
def numerical_guard(model):
    """Enforce preservation, not merely restore silently, for each evaluation."""
    rng=rng_snapshot();modes=[m.training for m in model.modules()]
    flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    assert all(p.grad is None for p in model.parameters())
    with s.protected(model),torch.inference_mode(),torch.autocast('cuda',enabled=False):
        inner_rng=rng_snapshot()
        yield
        assert equal_tree(inner_rng,rng_snapshot()),'Evaluation consumed RNG'
        assert all(p.grad is None for p in model.parameters())
    assert equal_tree(rng,rng_snapshot()),'RNG restoration failed'
    assert modes==[m.training for m in model.modules()]
    assert flags==(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)


@contextlib.contextmanager
def output_hooks(model,experts,records,zero_expert=None):
    handles=[]
    try:
        for e in experts:
            def hook(module,args,out,e=e):
                effective=torch.zeros_like(out) if zero_expert==e else out
                records[e]=(args[0].detach().clone(),effective.detach().clone())
                return effective if zero_expert==e else None
            handles.append(model.blocks[1].moe.experts[e].register_forward_hook(hook))
        yield
    finally:
        for h in handles:h.remove()


def dispatch_positions(trace,e,capacity):
    ids=trace.selected_expert_indices.reshape(-1,1)
    assignments=torch.nonzero(ids==e,as_tuple=False)
    if len(assignments)>capacity:
        scores=trace.selected_probabilities.reshape(-1,1)[assignments[:,0],assignments[:,1]]
        assignments=assignments[scores.topk(capacity).indices]
    return assignments[:,0]


def evaluate(model,batches,experts,zero_expert=None):
    """Native whole batches; capture expert outputs in actual dispatch order."""
    losses=[];ids=[];accepted=[];gates=[];aux=[];outputs={e:[] for e in experts}
    with numerical_guard(model):
      for x,y in batches:
        records={}
        with output_hooks(model,experts,records,zero_expert):out=model(x.cuda())
        tr=out.routes[1];n=x.numel();moe=model.blocks[1].moe
        capacity=max(1,int(np.ceil(moe.capacity_factor*n*moe.top_k/moe.num_experts)))
        loss=torch.nn.functional.cross_entropy(out.logits.flatten(0,1),y.cuda().flatten(),reduction='none')
        assert torch.isfinite(loss).all();losses.append(loss.cpu());aux.append(float(out.auxiliary_loss))
        ids.append(tr.selected_expert_indices.cpu().flatten());accepted.append(tr.accepted_mask.cpu().flatten());gates.append(tr.selected_probabilities.cpu().flatten())
        for e in experts:
            pos=dispatch_positions(tr,e,capacity)
            mask=(tr.selected_expert_indices.flatten()==e)&tr.accepted_mask.flatten()
            assert torch.equal(pos.sort().values,torch.where(mask)[0])
            if len(pos):
                inp,z=records[e]
                assert torch.equal(inp,out.pre_router_hidden_states[1].reshape(n,-1)[pos])
                assert z.shape[0]==len(pos)
            else:
                inp=out.pre_router_hidden_states[1].reshape(n,-1)[:0];z=inp.clone()
            outputs[e].append(dict(positions=pos.cpu(),inputs=inp.cpu(),output=z.cpu(),gates=tr.selected_probabilities.flatten()[pos].cpu()))
    return dict(loss=torch.cat(losses),ids=torch.cat(ids),accepted=torch.cat(accepted),gates=torch.cat(gates),auxiliary_batch_means=aux,outputs=outputs)


def replay_outputs(model,expert,baseline):
    """Replay expert alone on baseline *native dispatch* rows; no isolated-token routing."""
    result=[]
    with numerical_guard(model):
        for block in baseline['outputs'][expert]:
            x=block['inputs'];out=model.blocks[1].moe.experts[expert](x.cuda()).cpu() if len(x) else block['output'].clone()
            result.append(out)
    return result


def output_metrics(blocks,outputs):
    count=sum(len(b['output']) for b in blocks)
    if not count:
        return dict(accepted_positions=0,output_elements=0,baseline_output_rms=None,output_rms_difference=None,relative_output_rms_difference=None,gate_weighted_rms_difference=None,baseline_gate_weighted_rms=None,output_coverage='unavailable_no_accepted_positions',relative_rms_status='undefined_no_accepted_positions')
    elements=sum(b['output'].numel() for b in blocks);energy=delta=weighted=weighted_base=0.
    for b,z in zip(blocks,outputs):
        y=b['output'].double();difference=z.double()-y;gate=b['gates'].double().reshape(-1,1)
        energy+=float(y.square().sum());delta+=float(difference.square().sum())
        weighted+=float((difference*gate).square().sum());weighted_base+=float((y*gate).square().sum())
    rms=(energy/elements)**.5;diff=(delta/elements)**.5
    return dict(accepted_positions=count,output_elements=elements,baseline_output_rms=rms,output_rms_difference=diff,relative_output_rms_difference=diff/rms if rms>OUTPUT_EPS else None,gate_weighted_rms_difference=(weighted/elements)**.5,baseline_gate_weighted_rms=(weighted_base/elements)**.5,output_coverage='adequate' if count>=MIN_ACCEPTED else 'low_traffic',relative_rms_status='defined' if rms>OUTPUT_EPS else 'undefined_near_zero_baseline')


def sequence_rows(delta,accepted,sequence_ids):
    rows=[];delta=delta.double().reshape(-1,128);accepted=accepted.reshape(-1,128)
    assert len(delta)==len(sequence_ids)
    for i,(z,mask) in enumerate(zip(delta,accepted)):
        rows.append(dict(sequence_ordinal=i,sequence_id=int(sequence_ids[i]),target_tokens=128,accepted_tokens=int(mask.sum()),complement_tokens=int((~mask).sum()),whole_damage=float(z.mean()),accepted_damage=float(z[mask].mean()) if mask.any() else None,complement_damage=float(z[~mask].mean()) if (~mask).any() else None))
    return rows


def measurement_rows(meta,delta,baseline,metrics,individual):
    accepted=(baseline['ids']==meta['expert'])&baseline['accepted'];rows=[]
    for subset,mask in [('whole',torch.ones(len(delta),dtype=torch.bool)),('baseline_accepted',accepted),('complement',~accepted)]:
        n=int(mask.sum());damage=float(delta[mask].double().mean()) if n else None
        component=float(individual[mask].double().mean()) if individual is not None and n else None
        rows.append(dict(**meta,subset=subset,target_tokens=n,coverage='unavailable_empty' if n==0 else ('low_traffic' if subset=='baseline_accepted' and n<MIN_ACCEPTED else 'adequate'),damage=damage,baseline_ce=float(baseline['loss'][mask].double().mean()) if n else None,sum_individual_damage=component,nonadditivity=damage-component if damage is not None and component is not None else None,**metrics))
    return rows


def source_audit():
    # Read-only: never invoke Stage 1 verify.main(), which writes its own audit file.
    s.verify_inputs();seal=s.read(s.D/'output_seal.json')
    checked=[]
    for p,h in seal['files'].items():
        assert sha(s.D/p)==h,p;checked.append(dict(path=str((s.D/p).relative_to(ROOT)),sha256=h))
    assert s.read(s.D/'completion.json')['measurement_complete']
    groups=load(s.D/'group_coordinates.pt');manifest={r['group']:r for r in __import__('csv').DictReader((s.D/'group_manifest.csv').open())}
    masks=s.load_masks(s.jobs()['sparse']['mask'])
    for e in s.EXPERTS:
        gg=[g for g in groups.values() if g['expert']==e];idx=torch.cat([g['indices'] for g in gg])
        assert len(torch.unique(idx))==len(idx) and masks[s.name(e)].flatten()[idx].all()
    for key,g in groups.items():assert digest(g['indices'])==manifest[key]['coordinate_sha256']
    candidates=load(s.D/'probe_candidates.pt');n=s.read(s.D/'probe_config.json')['batches_per_probe']
    assert n==2
    A={i for b in candidates[:n] for i in b['sequence_ids']};B={i for b in candidates[2:2+n] for i in b['sequence_ids']}
    assert not (A&B) and not ({i-1 for i in A}&B) and not ({i+1 for i in A}&B)
    for kind,j in s.jobs().items():
        meta=s.read(s.cp_path(j,2500).with_suffix('.json'));after=s.read(s.cp_path(j,2500).with_name('step_2500_after_reset.json'))
        assert meta['segment_offset']==0 and meta['optimizer_entries']>0 and after['optimizer_entries']==0 and meta['model_hash']==after['model_hash']
        for t in s.STEPS:
            path=s.cache_path(kind,t,'A');cache=load(path);assert cache['verified'] and cache['identity']==s.cache_identity(kind,t,'A')
    return checked


def check_baseline(old,new):
    for k in ('loss','ids','accepted','gates'):assert torch.equal(old[k],new[k]),'Stage 1 baseline mismatch: '+k
    assert old['auxiliary_batch_means']==new['auxiliary_batch_means']


def implementation_files():
    return {str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}
def verify_registration():
    reg=s.read(D/'registration.json');seal=s.read(D/'input_seal.json')
    assert sha(D/'registration.json')==seal['registration_sha256']
    assert sha(D/'intervention_coordinates.pt')==seal['coordinates_sha256']
    assert sha(D/'protocol.md')==seal['protocol_sha256']
    for p,h in seal['implementation'].items():assert sha(ROOT/p)==h,p
    for p,h in reg['stage1_files'].items():assert sha(ROOT/p)==h,p
    return reg


@contextlib.contextmanager
def worker():
    import fcntl
    D.mkdir(parents=True,exist_ok=True);handles=[];stop=threading.Event()
    try:
        for label in ('study',socket.gethostname()):
            p=D/'locks'/(label+'.lock');p.parent.mkdir(exist_ok=True);f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(f)
        def beat():
            while not stop.is_set():
                save(D/'heartbeat.json',dict(host=socket.gethostname(),pid=os.getpid(),unix=time.time()));stop.wait(15)
        thread=threading.Thread(target=beat,daemon=True);thread.start()
        try:yield
        finally:stop.set();thread.join()
    finally:
        for f in handles:f.close()
