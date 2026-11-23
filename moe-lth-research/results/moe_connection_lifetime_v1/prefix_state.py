"""Generate prefix-only dense states with the archive's applied AdamW score."""
import copy
import json
import math
import time
from pathlib import Path

import torch

from groups import HERE, V2, branch, construct, scope, MatchingGateFailure
from lifetime_engineering import restore
from moe_lth.utils import seed_everything

DEV=(7,17,29)
FRESH=(37,43,53)
ROOT=HERE.parents[1]


def streams(seed,order):
    # Fresh records inherit the native model/data configuration, overriding the
    # seed and output location only. Use the same epoch materializer and order.
    manifest=json.loads((ROOT/"results/imp_recovery_order_stability_v1/canonical_stream_manifest.json").read_text())
    cfg=copy.deepcopy(manifest["seeds"]["7"]["config"])
    cfg.update(seed=seed,output_dir=str(HERE/"prefixes"/f"seed{seed}_{order}"))
    x,y=branch.N._source_stream(seed,cfg)
    return cfg,branch.N._apply_order(x,branch.N._permutations()[order]["segment_0"]),y


def fresh_model(seed,cfg):
    seed_everything(seed)
    model=branch.TinyMoELanguageModel(copy.deepcopy(cfg["model"])).cuda()
    # The native recovery setup constructs a second CPU model when loading the
    # initial router. Its parameters are overwritten by the same seed-0 state.
    branch.TinyMoELanguageModel(copy.deepcopy(cfg["model"]))
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"])
    dtype=branch.resolve_autocast_dtype(cfg["training"]["precision"],torch.device("cuda"))
    scaler=branch.create_grad_scaler(dtype==torch.float16)
    return model,opt,scaler,dtype


def location(seed,order):
    return HERE/"prefixes"/f"seed{seed}_{order}"


def material(seed,order,t):
    if t==250 and seed in DEV:
        return branch.load_state(seed,order,torch.device("cuda"))
    cfg,batches,validation=streams(seed,order)
    path=location(seed,order)/f"state{t}.pt"
    record=json.loads(path.with_suffix(".json").read_text())
    assert branch.sha(path)==record["sha256"]
    saved=torch.load(path,map_location="cpu",weights_only=False)
    model,opt,scaler,dtype=fresh_model(seed,cfg)
    restore(saved,model,opt,scaler)
    assert saved["step"]==t and record["prefix_batch_hash"]==branch.N.recovery._batch_sequence_hash(batches[:t])
    assert record["panel_hash"]==branch.N.recovery._batch_sequence_hash(validation)
    return saved,cfg,batches,validation,model,opt,scaler,dtype


def generate(seed,order,t):
    started=time.monotonic()
    folder=location(seed,order)
    folder.mkdir(parents=True,exist_ok=True)
    cell=f"seed{seed}_{order}"
    if t==500:
        seal=HERE/"seals"/f"{cell}_250.json"
        assert json.loads(seal.read_text())["forecast_sha256"]==branch.sha(HERE/"groups"/cell/"forecast250.json")
        base,cfg,batches,val,model,opt,scaler,dtype=material(seed,order,250)
        start=250
        total_applied=base["optimizer_steps_applied"]
    else:
        assert t==250
        cfg,batches,val=streams(seed,order)
        model,opt,scaler,dtype=fresh_model(seed,cfg)
        start=total_applied=0
    params=dict(model.named_parameters())
    names=scope(model.state_dict())
    if start==0:
        initial={n:params[n].detach().sign() for n in names}
        signed={n:torch.zeros_like(params[n],dtype=torch.float64) for n in names}
        absolute={n:torch.zeros_like(signed[n]) for n in names}
    else:
        if seed in DEV:
            # V1 stored the normalized score and its denominator; restore the
            # signed numerator consistently for the step-500 descriptive score.
            obj=torch.load(branch.V1/"prefixes"/cell/"scores.pt",weights_only=True,mmap=True)["scores"]
            initial={n:obj[n]["initial_sign"].cuda() for n in names}
            absolute={n:obj[n]["absolute_update"].cuda() for n in names}
            signed={n:torch.nan_to_num(obj[n]["score"].cuda())*initial[n]*absolute[n] for n in names}
            del obj
        else:
            state=torch.load(folder/"accum250.pt",weights_only=True)
            initial,signed,absolute=({n:x.cuda() for n,x in state[k].items()} for k in ("initial","signed","absolute"))
    def record(optim,args,kwargs):
        group,=optim.param_groups
        b1,b2=group["betas"]
        for n in names:
            p=params[n]
            if p.grad is None:
                continue
            st=optim.state[p]
            s=int(st["step"])
            den=(st["exp_avg_sq"].sqrt()/math.sqrt(1-b2**s)).add_(group["eps"])
            delta=(st["exp_avg"].double()/den.double()).mul_(-group["lr"]/(1-b1**s))
            signed[n].add_(delta)
            absolute[n].add_(delta.abs())
    hook=opt.register_step_post_hook(record)
    skips=[]
    for idx in range(start,t):
        applied=branch.update(model,opt,scaler,dtype,cfg,batches[idx],{},torch.device("cuda"))
        total_applied+=int(applied)
        if not applied:
            skips.append(idx+1)
    hook.remove()
    if start==0:
        branch.save_pt(folder/"accum250.pt",dict(initial=initial,signed=signed,absolute=absolute))
    scores={n:dict(score=(initial[n]*signed[n]/absolute[n]).masked_fill((initial[n]==0)|(absolute[n]==0),float("nan")).cpu(),
                 absolute_update=absolute[n].cpu(),initial_sign=initial[n].cpu()) for n in names}
    if t==250 and seed in DEV:
        raw=torch.load(branch.V1/"prefixes"/cell/"native/interruption.pt",map_location="cpu",weights_only=False)
        assert branch.nested_equal(model.state_dict(),raw["model"])
        assert branch.nested_equal(opt.state_dict(),raw["optimizer"])
        assert branch.nested_equal(scaler.state_dict(),raw["scaler"])
        ref=torch.load(branch.V1/"prefixes"/cell/"scores.pt",weights_only=True,mmap=True)["scores"]
        for n in names:
            torch.testing.assert_close(scores[n]["score"],ref[n]["score"],rtol=0,atol=0,equal_nan=True)
    path=folder/f"state{t}.pt"
    branch.save_pt(path,dict(step=t,model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),
                             scheduler=None,optimizer_steps_applied=total_applied,**branch.rng_get()))
    branch.save_pt(folder/f"scores{t}.pt",scores)
    branch.save_json(path.with_suffix(".json"),dict(sha256=branch.sha(path),scores_sha256=branch.sha(folder/f"scores{t}.pt"),
                     prefix_batch_hash=branch.N.recovery._batch_sequence_hash(batches[:t]),
                     panel_hash=branch.N.recovery._batch_sequence_hash(val),applied=total_applied,skips_since_start=skips,elapsed_seconds=time.monotonic()-started))


def fresh_groups(seed,order):
    started=time.monotonic()
    assert seed in FRESH
    cell=f"seed{seed}_{order}"
    folder=location(seed,order)
    meta=json.loads((folder/"state250.json").read_text())
    assert branch.sha(folder/"scores250.pt")==meta["scores_sha256"]
    assert branch.sha(folder/"state250.pt")==meta["sha256"]
    state=torch.load(folder/"state250.pt",map_location="cpu",weights_only=False,mmap=True)["model"]
    scores=torch.load(folder/"scores250.pt",weights_only=True,mmap=True)
    destination=HERE/"groups"/cell
    destination.mkdir(parents=True,exist_ok=True)
    violations=[]
    try:
        groups,rows=construct(state,scores,seed,order)
    except MatchingGateFailure as failure:
        groups,rows,violations=failure.groups,failure.rows,failure.violations
    branch.save_pt(destination/"sets.pt",groups)
    audit=dict(matrices=rows,sha256=branch.sha(destination/"sets.pt"),violations=violations,matching_seconds=time.monotonic()-started)
    branch.save_json(destination/"audit.json",audit)
    if violations:
        branch.save_json(destination/"ineligible.json",audit)
