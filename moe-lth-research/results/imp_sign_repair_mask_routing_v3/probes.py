"""Replay AdamW displacement probes and optionally change selected gate amplitudes."""
import argparse
import fcntl
import tempfile
from contextlib import contextmanager
import socket

from audit import OUT,p
from replay import N,DEVICE,admission,classify

from moe_lth.models.moe_layer import MoEFeedForward
from moe_lth.models.router import RouteOverride

torch=p.torch
SCALE=.5


@contextmanager
def gate_intervention(counter):
    """Use native local slot IDs and capacity priority, scaling live gate gradients."""
    hooks=[]
    seen=set()

    def bind(model):
        for layer in model.modules():
            if not isinstance(layer,MoEFeedForward) or layer in seen:
                continue
            seen.add(layer)
            expected={}

            def before(module,args,kwargs,expected=expected):
                x=args[0]
                override=args[1] if len(args)>1 else kwargs.get("override_ids")
                assert override is None
                native=module.router(x.reshape(-1,x.shape[-1]))
                ids=native.selected_experts
                selected=native.selected_probability
                expected.update(ids=ids,score=selected)
                return (x,RouteOverride(ids,SCALE*selected,selected.detach())),{
                    key:value for key,value in kwargs.items() if key!="override_ids"}

            def after(module,args,kwargs,output,expected=expected):
                trace=output[2]
                ids=expected["ids"]
                score=expected["score"]
                assert torch.equal(trace.selected_expert_indices.reshape_as(ids),ids)
                assert torch.equal(trace.selected_probabilities.reshape_as(score),SCALE*score)
                if counter["calls"] < 32:
                    import math
                    capacity=max(1,math.ceil(module.capacity_factor*ids.numel()/module.num_experts))
                    mask=torch.zeros_like(ids,dtype=torch.bool)
                    for expert in range(module.num_experts):
                        assignments=torch.nonzero(ids==expert,as_tuple=False)
                        if kwargs.get("capacity_enabled",True) and len(assignments)>capacity:
                            priority=score[assignments[:,0],assignments[:,1]]
                            assignments=assignments[priority.topk(capacity).indices]
                        mask[assignments[:,0],assignments[:,1]]=True
                    assert torch.equal(trace.accepted_mask,mask)
                    counter["verified_capacity"]+=1
                counter["calls"]+=1

            hooks.extend((layer.register_forward_pre_hook(before,with_kwargs=True),
                          layer.register_forward_hook(after,with_kwargs=True)))
    try:
        yield bind
    finally:
        for handle in hooks:
            handle.remove()


def probe(entry,material,condition):
    root=OUT/"probes"/condition
    root.mkdir(parents=True,exist_ok=True)
    seed,order=entry["seed"],entry["order"]
    output=root/f"probe_{seed}_{order}.pt"
    endpoint=root/f"dense_sign_{seed}_{order}.pt"
    audit_path=root/f"audit_{seed}_{order}.json"
    if audit_path.is_file():
        checked=p.read(audit_path)
        assert p.digest(output)==checked["probe_sha256"]
        assert p.digest(endpoint)==checked["dense_sign_sha256"]
        return
    workspace=p.Path(tempfile.mkdtemp(prefix=f"attempt_{seed}_{order}_",dir=root))
    produced=workspace/output.name
    v1=p.OUT
    baseline=N.recovery._run_recovery_condition
    native_segment=N._run_segment
    counts=dict(calls=0,verified_capacity=0)
    state={}
    live_model=None

    with gate_intervention(counts) as bind:
        def intercept_segment(*args,**kwargs):
            native=N.recovery._run_recovery_condition

            def capture(**kw):
                forward=kw["training_forward_callback"]

                def train(model,tokens,step):
                    nonlocal live_model
                    live_model=model
                    if condition=="scaled":
                        bind(model)
                    return forward(model,tokens,step)

                kw["training_forward_callback"]=train
                return native(**kw)

            N.recovery._run_recovery_condition=capture
            try:
                return native_segment(*args,**kwargs)
            finally:
                N.recovery._run_recovery_condition=native

        p.OUT=workspace
        N._run_segment=intercept_segment
        try:
            p.probe(entry,material)
        finally:
            p.OUT=v1
            N.recovery._run_recovery_condition=baseline
            N._run_segment=native_segment

    assert live_model is not None, "Probe did not expose its model"
    observed=torch.load(produced,weights_only=True,mmap=True)
    assert observed["applied_updates"]==100, (
        "Expected exactly 100 optimizer updates",
        observed["applied_updates"]
    )
    state.update({
        name:classify(value.detach()).cpu()
        for name,value in live_model.named_parameters()
        if ".moe.experts." in name and value.ndim==2
    })
    assert state
    archived=torch.load(v1/f"probe_{seed}_{order}.pt",weights_only=True,mmap=True)
    assert observed["train_hash"]==archived["train_hash"]
    assert set(state)==set(archived["scores"])
    if condition=="native":
        assert observed["attempts"]==archived["attempts"]
        for name,(score,activity) in observed["scores"].items():
            for a,b in zip((score,activity),archived["scores"][name]):
                assert torch.allclose(a,b,rtol=0,atol=0,equal_nan=True),name
    elif not counts["verified_capacity"]:
        raise RuntimeError("Scaled probe did not exercise capacity verification.")
    p.save_tensor(endpoint,state)
    produced.replace(output)
    p.write(audit_path,dict(condition=condition,probe_sha256=p.digest(output),
                dense_sign_sha256=p.digest(endpoint),routing_calls=counts,attempts=observed["attempts"],
                applied=observed["applied_updates"],archive_exact=condition=="native"))


def main(condition,shard):
    manifest=p.read(OUT/"audit_manifest.json")
    for entry in manifest["inputs"]:
        if entry["order"]!=("A","B")[shard]:
            continue
        admission(socket.gethostname())
        material=N._load_training_material(entry["seed"])
        probe(entry,material,condition)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition",choices=("native","scaled"))
    parser.add_argument("shard",type=int,choices=(0,1))
    args=parser.parse_args()
    with (OUT/f"probe_{args.condition}_{args.shard}.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main(args.condition,args.shard)
