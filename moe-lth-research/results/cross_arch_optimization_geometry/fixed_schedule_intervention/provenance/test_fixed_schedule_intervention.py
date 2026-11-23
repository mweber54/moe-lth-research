from copy import deepcopy
from pathlib import Path
import torch
from moe_lth.config import DEFAULT_CONFIG
from moe_lth.models import TinyMoELanguageModel
from moe_lth.experiments import run_router_age_recovery as r
from moe_lth.experiments.fixed_schedule_intervention import Schedule
from moe_lth.experiments.optimization_geometry_diagnostics import digest
from moe_lth.pruning.router_age import state_dict_hash,selected_experts_per_batch
from moe_lth.training.checkpoint import save_checkpoint


def test_capture_and_self_replay_preserve_recovery(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(7)
    c=deepcopy(DEFAULT_CONFIG);c['seed']=7;c['device']='cpu'
    c['model'].update(vocab_size=256,max_seq_len=8,num_layers=2,num_heads=2,d_model=16,num_experts=2,expert_hidden_size=32,dropout=0.,top_k=1,capacity_factor=.6)
    c['training'].update(precision='fp32',learning_rate=.0003,weight_decay=.01,grad_clip=1.)
    model=TinyMoELanguageModel(c['model']);state=deepcopy(model.state_dict());cp=tmp_path/'reference.pt';save_checkpoint(cp,model,None,0,None,c)
    batches=[(torch.randint(0,256,(4,8)),torch.randint(0,256,(4,8))) for _ in range(2)]
    args=dict(config=c,condition_name='schedule_unit_test',pruned_base_state=state,router_checkpoint=str(cp),router_age_percent=100,router_step=0,masks={},expert_hash=state_dict_hash({k:v for k,v in state.items() if r.parameter_group(k)=='expert'}),shared_hash=state_dict_hash({k:v for k,v in state.items() if r.parameter_group(k)=='shared'}),mask_hash='dense_no_mask',reference_selected=selected_experts_per_batch(model,[batches[0][0]],torch.device('cpu')),calibration_batches=[batches[0][0]],train_batches=batches,validation_batches=batches[:1],train_batch_hash=r._batch_sequence_hash(batches),validation_batch_hash=r._batch_sequence_hash(batches[:1]),device=torch.device('cpu'),recovery_steps=2,dense_loss=10.,confidence_control=False,target_confidence=None,seed=7,sparsity=0.)
    r._run_recovery_condition(**args,output_dir=tmp_path/'baseline')
    cap=Schedule(tmp_path/'schedule','capture');r._run_recovery_condition(**args,output_dir=tmp_path/'capture',training_forward_callback=cap.train,validation_forward_callback=cap.validation);cap.finish_capture({'test':True})
    replay=Schedule(tmp_path/'schedule','replay');r._run_recovery_condition(**args,output_dir=tmp_path/'replay',training_forward_callback=replay.train,validation_forward_callback=replay.validation);replay.verify_complete_replay()
    states=[torch.load(tmp_path/k/'checkpoints/final_recovered.pt',weights_only=False)['model'] for k in ['baseline','capture','replay']]
    assert digest(states[0])==digest(states[1])==digest(states[2])
    # A different recipient keeps the donor token sets while retaining live gate gradients.
    other=TinyMoELanguageModel(c['model']);reader=Schedule(tmp_path/'schedule','replay')
    out=reader.train(other,batches[0][0],1);out.logits.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in other.named_parameters() if '.router.' in n)
    bad=Schedule(tmp_path/'schedule','replay')
    try:bad.train(other,batches[0][0].roll(1,0),1)
    except RuntimeError as exc:assert 'Input sequence mismatch' in str(exc)
    else:raise AssertionError('Mismatched input was accepted')
