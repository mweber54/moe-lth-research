from copy import deepcopy
import csv
import torch
from moe_lth.config import DEFAULT_CONFIG
from moe_lth.models import TinyMoELanguageModel
from moe_lth.experiments import optimization_geometry_diagnostics as d
from moe_lth.experiments import run_router_age_recovery as r
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.router_age import state_dict_hash,selected_experts_per_batch
from moe_lth.training.checkpoint import save_checkpoint


def test_invalid_and_zero_cosines_are_unavailable():
    assert d.stats(torch.zeros(3),torch.ones(3))['cosine'] is None
    assert d.stats(torch.tensor([float('nan')]),torch.ones(1))['status']=='nonfinite'
    assert d.stats(torch.tensor([1.,2.]),torch.tensor([1.,2.]))['cosine']>1-1e-12


def test_diagnostics_leave_recovery_identical(tmp_path):
    torch.set_num_threads(1)
    c=deepcopy(DEFAULT_CONFIG);c['seed']=7;c['device']='cpu'
    c['model'].update(vocab_size=256,max_seq_len=8,num_layers=2,num_heads=2,d_model=16,num_experts=2,expert_hidden_size=32,dropout=0.,top_k=1,capacity_factor=.75)
    c['training'].update(precision='fp32',learning_rate=.0003,weight_decay=.01,grad_clip=1.)
    torch.manual_seed(7);initial=TinyMoELanguageModel(c['model']);state=deepcopy(initial.state_dict());cp=tmp_path/'reference.pt';save_checkpoint(cp,initial,None,0,None,c)
    masks=expert_local_magnitude_masks(initial,.85)
    ticket={k:v*masks[k] if k in masks else v for k,v in state.items()}
    batches=[(torch.randint(0,256,(4,8)),torch.randint(0,256,(4,8))) for _ in range(2)]
    kwargs=dict(config=c,condition_name='unit_test',pruned_base_state=ticket,router_checkpoint=str(cp),router_age_percent=0,router_step=0,masks=masks,expert_hash=state_dict_hash({k:v for k,v in ticket.items() if r.parameter_group(k)=='expert'}),shared_hash=state_dict_hash({k:v for k,v in ticket.items() if r.parameter_group(k)=='shared'}),mask_hash=r._mask_hash(masks),reference_selected=selected_experts_per_batch(initial,[batches[0][0]],torch.device('cpu')),calibration_batches=[batches[0][0]],train_batches=batches,validation_batches=batches[:1],train_batch_hash=r._batch_sequence_hash(batches),validation_batch_hash=r._batch_sequence_hash(batches[:1]),device=torch.device('cpu'),recovery_steps=2,dense_loss=10.,confidence_control=False,target_confidence=None,seed=7,sparsity=.85)
    r._run_recovery_condition(**kwargs,output_dir=tmp_path/'baseline')
    observed=[]
    def callback(model,opt,scaler,step):
        rng=d.rng_state();before=d.digest(opt.state_dict());scale=d.digest(scaler.state_dict());mode=model.training
        dense=deepcopy(model);dest=tmp_path/f'diag{step}';dest.mkdir()
        d.measure(model,dense,masks,batches[0],.1,dest,dict(experts=2,seed=7,age=0,step=step))
        assert d.digest(opt.state_dict())==before and d.digest(scaler.state_dict())==scale
        assert d.digest(rng)==d.digest(d.rng_state()) and model.training==mode
        rows=list(csv.DictReader((dest/'gradient_alignment_forced_sparse_routes.csv').open()))
        assert all(float(x['cosine'])>1-1e-6 for x in rows if x['cosine'])
        observed.append(step)
    r._run_recovery_condition(**kwargs,output_dir=tmp_path/'instrumented',state_snapshot_callback=callback)
    a=torch.load(tmp_path/'baseline/checkpoints/final_recovered.pt',weights_only=False)['model'];b=torch.load(tmp_path/'instrumented/checkpoints/final_recovered.pt',weights_only=False)['model']
    assert d.digest(a)==d.digest(b)
    assert observed==[0,1,2]
