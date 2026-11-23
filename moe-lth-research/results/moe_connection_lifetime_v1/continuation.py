"""Native two-cut continuation with resumable cut-boundary records."""
import argparse
import fcntl
import json
import time
from pathlib import Path

import torch

from forecast import evaluate_panel
from lifetime_engineering import HERE, branch, masks, restore


def mask_digest(active):
    import hashlib
    digest=hashlib.sha256()
    for n,m in sorted(active.items()):
        digest.update(n.encode()+b"\0")
        digest.update(m.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(seed,order,t,arm,material,folder,groups,sets_hash):
    saved,cfg,batches,panel,model,opt,scaler,dtype=material
    folder.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda")
    with (folder/"lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.monotonic()
        active={} if arm=="dense" else masks(model,groups,{"H":"R","R":"H","0":"M"}[arm])
        final={} if arm=="dense" else masks(model,groups,"M")
        ident=dict(seed=seed,order=order,t=t,arm=arm,sets_sha256=sets_hash,final_mask_sha256=mask_digest(final),
                   prefix_step=t,train_hash=branch.N.recovery._batch_sequence_hash(batches[t:]),
                   validation_hash=branch.N.recovery._batch_sequence_hash(panel))
        if seed in (37,43,53):
            cell=f"seed{seed}_{order}"
            state_meta=HERE/"prefixes"/cell/f"state{t}.json"
            seal=json.loads((HERE/"seals"/f"{cell}_{t}.json").read_text())
            assert branch.sha(state_meta)==seal["state_record_sha256"]
            ident.update(starting_state_sha256=json.loads(state_meta.read_text())["sha256"],
                         forecast_sha256=seal["forecast_sha256"],freeze_sha256=seal["freeze_sha256"])
        ckfile=folder/"checkpoint.pt"
        pre_applied=int(saved["optimizer_steps_applied"])
        total_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
        def count(m):
            return total_parameters-sum(int((~a).sum()) for a in m.values())
        early_live,final_live=count(active),count(final)
        if ckfile.exists():
            ck=torch.load(ckfile,map_location="cpu",weights_only=False)
            assert ck["identity"]==ident and ck["global_step"]==ck["next_batch_index"]
            restore(ck,model,opt,scaler)
            step=ck["global_step"]
            applied=ck["applied_since_branch"]
            losses=ck["losses"]
            seconds=ck["elapsed_seconds"]
            sparse=ck["live_parameter_tokens"]
            skips=ck["overflow_attempts"]
            if step>=t+250:
                active=final
            branch.verify_zeros(model,opt,active)
            del ck
        else:
            restore(saved,model,opt,scaler)
            step=t
            applied=sparse=seconds=0
            skips=[]
            losses={"pre_first":evaluate_panel(model,panel,device,True)}
            branch.enforce(model,opt,active)
            branch.verify_intervention(saved,model,opt,active)
            losses["post_first"]=evaluate_panel(model,panel,device,True)
        while step<2500:
            applied_now=branch.update(model,opt,scaler,dtype,cfg,batches[step],active,device)
            sparse+=batches[step][1].numel()*(early_live if step<t+250 else final_live)
            step+=1
            applied+=int(applied_now)
            if not applied_now:
                skips.append(step)
            if step==t+250:
                losses["pre_second"]=evaluate_panel(model,panel,device,True)
                active=final
                branch.enforce(model,opt,active)
                branch.verify_zeros(model,opt,active)
                losses["post_second"]=evaluate_panel(model,panel,device,True)
            if step in (t+1,t+251,t+500,2500) or (arm=="dense" and step in (500,750,751,1000)):
                losses[str(step)]=evaluate_panel(model,panel,device,True)
            if step%250==0:
                branch.verify_zeros(model,opt,active)
                ck=branch.checkpoint(model,opt,scaler,ident,step,applied,losses)
                ck.update(live_parameter_tokens=sparse,elapsed_seconds=seconds+time.monotonic()-start,overflow_attempts=skips)
                branch.save_pt(ckfile,ck)
                print(step,applied,flush=True)
        assert step==2500 and applied+len(skips)==2500-t
        assert mask_digest(active)==ident["final_mask_sha256"]
        branch.verify_zeros(model,opt,final)
        expected=evaluate_panel(model,panel,device,True)
        assert expected==losses["2500"]
        assert ckfile.exists()
        result=dict(identity=ident,attempts=step,successful_updates=pre_applied+applied,
                    elapsed_seconds=seconds+time.monotonic()-start,live_parameter_tokens=sparse,
                    live_during_window=early_live,live_after=final_live,
                    overflow_attempts=skips,checkpoint_sha256=branch.sha(ckfile),measurements=losses)
        branch.save_json(folder/"verified.json",result)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("arm",choices=("H","R"))
    a=p.parse_args()
    check=json.loads((HERE/"smoke_7_A/verified.json").read_text())
    assert check["resume_exact"] and check["readout_isolated"] and check["commutation"]
    folder=HERE/"control_groups"
    digest=branch.sha(folder/"sets.pt")
    assert digest==json.loads((folder/"audit.json").read_text())["sha256"]
    material=branch.load_state(7,"A",torch.device("cuda"))
    run(7,"A",250,a.arm,material,HERE/"random_control"/a.arm,
        torch.load(folder/"sets.pt",weights_only=True),digest)
