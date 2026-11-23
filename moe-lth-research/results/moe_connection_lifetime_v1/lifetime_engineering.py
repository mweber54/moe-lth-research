"""7/A row isolation, cut commutation and native restart across attempt 500."""
import argparse
import copy
import json
from pathlib import Path

from groups import HERE, branch, torch


def masks(model, groups, key):
    params = dict(model.named_parameters())
    result = {}
    for name, sets in groups.items():
        mask = torch.ones_like(params[name], dtype=torch.bool)
        mask.flatten()[sets[key].to(mask.device)] = False
        result[name] = mask
    return result


@torch.no_grad()
def copy_rows(readout, donor, groups):
    source = dict(donor.named_parameters())
    target = dict(readout.named_parameters())
    for name, sets in groups.items():
        key = name.replace(".fc2.weight", ".fc1.weight")
        rows = sets["S"].to(target[key].device)
        target[key].index_copy_(0, rows, source[key].index_select(0,rows))


def verify_rows(readout, donor, base, groups):
    expected = copy.deepcopy(base.state_dict())
    for name, sets in groups.items():
        key = name.replace(".fc2.weight", ".fc1.weight")
        rows = sets["S"].to(expected[key].device)
        expected[key][rows] = donor.state_dict()[key][rows]
    assert branch.nested_equal(readout.state_dict(),expected)


def restore(payload, model, opt, scaler):
    model.load_state_dict(payload["model"])
    opt.load_state_dict(payload["optimizer"])
    # PyTorch deliberately preserves the identity of a non-capturable Adam
    # step tensor on load. Detach every such scalar from the immutable donor
    # checkpoint before allowing the optimizer to mutate it.
    for live_group,saved_group in zip(opt.param_groups,payload["optimizer"]["param_groups"]):
        for param,state_id in zip(live_group["params"],saved_group["params"]):
            original=payload["optimizer"]["state"].get(state_id,{})
            for key,value in opt.state.get(param,{}).items():
                source=original[key]
                if torch.is_tensor(value) and value.device==source.device and value.data_ptr()==source.data_ptr():
                    opt.state[param][key]=value.clone()
    scaler.load_state_dict(payload["scaler"])
    branch.rng_set(payload)
    model.train()


def smoke(phase):
    root = HERE/"smoke_7_A_isolated"
    root.mkdir(exist_ok=True)
    device = torch.device("cuda:0")
    saved,cfg,batches,panel,model,opt,scaler,dtype = branch.load_state(7,"A",device)
    group_file = HERE/"groups/seed7_A/sets.pt"
    digest = branch.sha(group_file)
    assert digest == json.loads(group_file.with_name("audit.json").read_text())["sha256"]
    groups = torch.load(group_file,weights_only=True)
    hm,rm,mm = (masks(model,groups,k) for k in ("H","R","M"))
    def step(index,active):
        return int(branch.update(model,opt,scaler,dtype,cfg,batches[index],active,device))
    def logits():
        with torch.no_grad():
            model.eval()
            out = model(panel[0][0].to(device)).logits.detach().cpu()
            model.train()
            return out
    if phase == "forward":
        # V2's separate-process engineering checkpoint extends the native
        # model/moment oracle with post-attempt RNG, CE and the batch cursor.
        v2=HERE.with_name("moe_online_reinforcement_pruning_v2")
        reference_file=v2/"engineering/seed7_A/untouched/step251.pt"
        replay=torch.load(reference_file,map_location="cpu",weights_only=False,mmap=True)
        restore(saved,model,opt,scaler)
        pre=branch.evaluate(model,panel,device)
        applied_251=step(250,{})
        post=branch.evaluate(model,panel,device)
        checks=dict(model=branch.nested_equal(model.state_dict(),replay["model"]),
                    optimizer=branch.nested_equal(opt.state_dict(),replay["optimizer"]),
                    scaler=branch.nested_equal(scaler.state_dict(),replay["scaler"]),
                    rng=branch.rng_equal(branch.rng_get(),replay),
                    scheduler=replay["scheduler"] is None,
                    cursor=replay["global_step"]==replay["next_batch_index"]==251,
                    applied=applied_251==replay["applied_since_branch"],
                    ce_pre=pre==replay["losses"]["pre_mask_250"],
                    ce_post=post==replay["losses"]["251"])
        assert all(checks.values()),checks
        branch.save_json(root/"native251.json",dict(checks=checks,pre_ce=pre,post_ce=post,
                         applied=applied_251,reference_sha256=branch.sha(reference_file),
                         update_batch_sha256=branch.N.recovery._batch_sequence_hash([batches[250]])))
        del replay
        from panels import select
        from forecast import evaluate_panel
        u,r=select(cfg,7,"A",250,batches)[0]
        def alternate_readouts(active):
            restore(saved,model,opt,scaler)
            branch.enforce(model,opt,active)
            applied=step(u,active)
            branch.enforce(model,opt,mm)
            whole=evaluate_panel(model,[batches[r]],device)["ce"]
            donor=copy.deepcopy(model)
            restore(saved,model,opt,scaler)
            branch.enforce(model,opt,mm)
            copy_rows(model,donor,groups)
            return applied,evaluate_panel(model,[batches[r]],device)["ce"],whole
        labelled={k:alternate_readouts(m) for k,m in (("H",rm),("R",hm))}
        # Relabel and execute in reverse physical order, including new native
        # optimizer attempts from the unmodified checkpoint for both arms.
        swapped={k:alternate_readouts(m) for k,m in (("H",hm),("R",rm))}
        assert labelled["H"]==swapped["R"] and labelled["R"]==swapped["H"]
        for j in (1,2):
            assert labelled["R"][j]-labelled["H"][j]==-(swapped["R"][j]-swapped["H"][j])
        branch.save_json(root/"swapped_labels.json",dict(forward=labelled,swapped=swapped,
                      update_batch=u,readout_batch=r,duplicate_floor_partner=0.,duplicate_floor_whole=0.))
        # Compare both orders from independent restores, including all Adam slots.
        for first,second in ((hm,rm),(rm,hm)):
            restore(saved,model,opt,scaler)
            branch.enforce(model,opt,first)
            branch.enforce(model,opt,second)
            if first is hm:
                ref = {n:t.cpu().clone() for n,t in model.state_dict().items()}
                reference_opt = copy.deepcopy(opt.state_dict())
                reference_logits = logits()
            else:
                assert branch.nested_equal(ref,model.state_dict())
                assert branch.nested_equal(reference_opt,opt.state_dict())
                assert torch.equal(reference_logits,logits())
        del ref, reference_opt
        restore(saved,model,opt,scaler)
        branch.enforce(model,opt,mm)
        base = copy.deepcopy(model)
        restore(saved,model,opt,scaler)
        branch.enforce(model,opt,rm)
        applied = step(250,rm)
        readout = copy.deepcopy(base)
        copy_rows(readout,model,groups)
        verify_rows(readout,model,base,groups)
        del base,readout
        for index in range(251,500):
            applied += step(index,rm)
        branch.verify_zeros(model,opt,rm)
        branch.save_pt(root/"pre_second_cut.pt",branch.checkpoint(model,opt,scaler,digest,500,applied,{}))
    else:
        ck = torch.load(root/"pre_second_cut.pt",map_location="cpu",weights_only=False)
        assert ck["identity"] == digest and ck["global_step"] == ck["next_batch_index"] == 500
        restore(ck,model,opt,scaler)
        applied = ck["applied_since_branch"]
        del ck
    branch.enforce(model,opt,mm)
    applied += step(500,mm)
    branch.verify_zeros(model,opt,mm)
    current = branch.checkpoint(model,opt,scaler,digest,501,applied,{"first_panel_logits":logits()})
    if phase == "forward":
        branch.save_pt(root/"expected501.pt",current)
    else:
        ref = torch.load(root/"expected501.pt",map_location="cpu",weights_only=False)
        assert all(branch.nested_equal(current[k],ref[k]) for k in current if k not in branch.rng_get())
        assert branch.rng_equal(current,ref)
        branch.save_json(root/"verified.json",dict(step=501,applied_since_250=applied,sets_sha256=digest,
                         commutation=True,readout_isolated=True,resume_exact=True))

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=["forward","resume"])
    smoke(p.parse_args().phase)
