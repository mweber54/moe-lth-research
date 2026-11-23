from copy import deepcopy
import json
import pytest
import torch
from moe_lth.config import DEFAULT_CONFIG
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.imp import SPARSITIES, nested_expert_local_masks
from moe_lth.pruning.masks import expert_weight_parameters, apply_masks_
from moe_lth.pruning.rewind import register_mask_gradient_hooks
from moe_lth.pruning.router_age import component_state_dict, state_dict_hash, selected_experts_per_batch
from moe_lth.training.checkpoint import save_checkpoint
from moe_lth.experiments import run_router_age_recovery as r


def tiny():
    c=deepcopy(DEFAULT_CONFIG)
    c['seed']=7
    c['model'].update(vocab_size=32,max_seq_len=8,num_layers=1,num_heads=2,d_model=8,num_experts=2,expert_hidden_size=16,dropout=.1)
    c['training'].update(precision='fp32',learning_rate=.001,weight_decay=.1,grad_clip=1.)
    return c


def test_nested_exact_ties_and_scope():
    model=TinyMoELanguageModel(tiny()['model'])
    masks={n:torch.ones_like(p,dtype=torch.bool) for n,p in expert_weight_parameters(model).items()}
    with torch.no_grad():
        for p in expert_weight_parameters(model).values(): p.fill_(1.)
    original=deepcopy(masks)
    for s in SPARSITIES[1:]:
        new=nested_expert_local_masks(model,masks,s)
        assert all(not (new[n] & ~masks[n]).any() for n in new)
        for expert in range(2):
            entries=[m for n,m in new.items() if f'.experts.{expert}.' in n]
            assert sum(int(m.sum()) for m in entries)==round(sum(m.numel() for m in entries)*(1-s))
        assert state_dict_hash(new)==state_dict_hash(nested_expert_local_masks(model,masks,s))
        masks=new
    with pytest.raises(ValueError): nested_expert_local_masks(model,masks,.2)
    with pytest.raises(ValueError): nested_expert_local_masks(model,{'router':torch.ones(1,dtype=torch.bool)},.8)
    assert set(masks)==set(original)


def test_decay_gradient_mask_and_router_learning():
    torch.manual_seed(7)
    model=TinyMoELanguageModel(tiny()['model'])
    masks=nested_expert_local_masks(model,{n:torch.ones_like(p,dtype=torch.bool) for n,p in expert_weight_parameters(model).items()},.8)
    apply_masks_(model,masks)
    handles=register_mask_gradient_hooks(model,masks)
    router=component_state_dict(model,'router')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
    for _ in range(3):
        optimizer.zero_grad()
        out=model(torch.randint(0,32,(2,8)))
        (out.logits.square().mean()+out.auxiliary_loss).backward()
        for n,p in expert_weight_parameters(model).items():
            if p.grad is not None: assert not p.grad[~masks[n]].any()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if '.router.' in n)
        optimizer.step();apply_masks_(model,masks)
        assert all(not p.detach()[~masks[n]].any() for n,p in expert_weight_parameters(model).items())
    assert state_dict_hash(router)!=state_dict_hash(component_state_dict(model,'router'))
    for h in handles:h.remove()


def recovery_run(tmp_path, dest, interrupt=False):
    torch.set_num_threads(1)
    c=tiny();torch.manual_seed(7)
    model=TinyMoELanguageModel(c['model'])
    cp=tmp_path/'initial.pt'
    if not cp.exists():save_checkpoint(cp,model,None,0,None,c)
    masks=nested_expert_local_masks(model,{n:torch.ones_like(p,dtype=torch.bool) for n,p in expert_weight_parameters(model).items()},.8)
    apply_masks_(model,masks)
    state=deepcopy(model.state_dict())
    batches=[(torch.randint(0,32,(2,8)),torch.randint(0,32,(2,8))) for _ in range(3)]
    probe=[batches[0][0]]
    reference=selected_experts_per_batch(model,probe,torch.device('cpu'))
    def stop(model,optimizer,scaler,step):
        if step==2:raise RuntimeError('simulated interruption')
    return r._run_recovery_condition(config=c,condition_name='smoke',pruned_base_state=state,
        router_checkpoint=str(cp),router_age_percent=0,router_step=0,masks=masks,
        expert_hash=state_dict_hash(component_state_dict(model,'expert')),
        shared_hash=state_dict_hash(component_state_dict(model,'shared')),mask_hash=r._mask_hash(masks),
        reference_selected=reference,calibration_batches=probe,train_batches=batches,
        validation_batches=batches[:1],train_batch_hash=r._batch_sequence_hash(batches),
        validation_batch_hash=r._batch_sequence_hash(batches[:1]),device=torch.device('cpu'),recovery_steps=3,
        dense_loss=3.,output_dir=dest,confidence_control=False,target_confidence=None,seed=7,sparsity=.8,
        interruption_checkpoint=dest/'interruption.pt',interruption_interval=1,
        state_snapshot_callback=stop if interrupt else None)


def test_smoke_restore_and_exact_interruption_resume(tmp_path):
    full=recovery_run(tmp_path,tmp_path/'full')
    with pytest.raises(RuntimeError,match='simulated'):
        recovery_run(tmp_path,tmp_path/'resumed',True)
    resumed=recovery_run(tmp_path,tmp_path/'resumed')
    assert full['final_validation_loss']==resumed['final_validation_loss']
    assert full['final_router_state_hash']==resumed['final_router_state_hash']
    assert full['optimizer_steps_applied']==resumed['optimizer_steps_applied']==3
    assert full['trainable_router_gradient_observed'] and full['router_parameter_drift_final']>0
    assert full['optimizer_state_entries_at_start']==0
    assert full['initial_router_state_hash']==resumed['initial_router_state_hash']
    a=torch.load(tmp_path/'full/checkpoints/final_recovered.pt',weights_only=False)['model']
    b=torch.load(tmp_path/'resumed/checkpoints/final_recovered.pt',weights_only=False)['model']
    assert state_dict_hash(a)==state_dict_hash(b)


def test_all_discovery_rounds_reset_and_independent_recovery(tmp_path):
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('imp_runner',Path(__file__).parents[1]/'scripts/imp_80_85/run.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    torch.set_num_threads(1);torch.manual_seed(7)
    c=tiny();model=TinyMoELanguageModel(c['model'])
    cp=tmp_path/'step_0.pt';save_checkpoint(cp,model,None,0,None,c)
    masks={n:torch.ones_like(p,dtype=torch.bool) for n,p in expert_weight_parameters(model).items()}
    original=deepcopy(model.state_dict())
    batches=[(torch.randint(0,32,(2,8)),torch.randint(0,32,(2,8))) for _ in range(2)]
    reference=selected_experts_per_batch(model,[batches[0][0]],torch.device('cpu'))
    saved={}
    for k,target in enumerate(SPARSITIES[1:]):
        dest=tmp_path/f'round{k}'
        record=runner.run_stage(c,cp,masks,batches,batches[:1],reference,torch.device('cpu'),dest,3.,{'sparsity':SPARSITIES[k]})
        assert record['initial_router_state_hash']==state_dict_hash({n:t for n,t in original.items() if '.router.' in n})
        assert record['shared_state_hash']==state_dict_hash({n:t for n,t in original.items() if '.router.' not in n and '.experts.' not in n})
        expected={n:t*masks[n] if n in masks else t for n,t in original.items() if '.experts.' in n}
        assert record['expert_state_hash']==state_dict_hash(expected)
        trained=runner.load_model_from_checkpoint(c['model'],str(dest/'checkpoints/final_recovered.pt'),torch.device('cpu'))
        new=nested_expert_local_masks(trained,masks,target)
        assert all(not (new[n]&~masks[n]).any() for n in new)
        masks=new
        if target in (.8,.85):saved[target]=deepcopy(masks)
    for target,mask in saved.items():
        record=runner.run_stage(c,cp,mask,batches,batches[:1],reference,torch.device('cpu'),tmp_path/f'final{target}',3.,{'sparsity':target})
        assert record['initial_router_state_hash']==state_dict_hash({n:t for n,t in original.items() if '.router.' in n})
        assert record['optimizer_state_entries_at_start']==0
        assert record['trainable_router_gradient_observed']
