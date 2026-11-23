import pytest
import torch
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.masks import expert_weight_parameters, apply_masks_
from moe_lth.pruning.mask_information import randomize, assert_control, groups


def fixture():
    model = TinyMoELanguageModel(dict(vocab_size=16,max_seq_len=4,num_layers=2,num_heads=2,d_model=4,num_experts=2,expert_hidden_size=8,dropout=0.,top_k=1,capacity_factor=1.25))
    masks = {}
    for i,(name,p) in enumerate(expert_weight_parameters(model).items()):
        m=torch.zeros(p.numel(),dtype=torch.bool);m[:3+i]=True
        masks[name]=m.reshape(p.shape)
    return model,masks


@pytest.mark.parametrize('condition',['global','expert','matrix'])
def test_counts_repeatability_and_nonprunable_weights(condition):
    model,masks=fixture()
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    a=randomize(model,masks,condition,17); b=randomize(model,masks,condition,17)
    assert_control(model,masks,a,condition)
    assert all(torch.equal(a[n],b[n]) for n in a)
    c=randomize(model,masks,condition,29)
    assert any(not torch.equal(a[n],c[n]) for n in a)
    apply_masks_(model,a)
    assert all(torch.equal(p,before[n]) for n,p in model.named_parameters() if n not in a)
    assert len(groups(masks,'expert')) == 4


@pytest.mark.parametrize('value',[False,True])
def test_forced_equality(value):
    model,masks=fixture();masks={n:torch.full_like(m,value) for n,m in masks.items()}
    for condition in ['global','expert','matrix']:
        a=randomize(model,masks,condition,7)
        assert all(torch.equal(a[n],masks[n]) for n in a)


def test_rejects_bad_masks_and_unchanged_control():
    model,masks=fixture()
    with pytest.raises(ValueError): randomize(model,{},'matrix',7)
    with pytest.raises(ValueError): randomize(model,{n:m.float() for n,m in masks.items()},'matrix',7)
    bad={n:m.clone() for n,m in masks.items()};name=next(iter(bad));bad[name]=bad[name].flatten()
    with pytest.raises(ValueError): randomize(model,bad,'matrix',7)
    with pytest.raises(ValueError): assert_control(model,masks,masks,'matrix')
    with pytest.raises(ValueError): randomize(model,masks,'unknown',7)


def test_mask_enforcement_with_momentum_and_weight_decay():
    from moe_lth.pruning.rewind import register_mask_gradient_hooks
    model,masks=fixture(); optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
    for p in model.parameters(): p.grad=torch.ones_like(p)
    optimizer.step()  # populate momentum before masking
    hooks=register_mask_gradient_hooks(model,masks);apply_masks_(model,masks)
    for _ in range(3):
        optimizer.zero_grad();sum(p.sum() for p in model.parameters()).backward();optimizer.step();apply_masks_(model,masks)
        for n,p in expert_weight_parameters(model).items(): assert torch.count_nonzero(p[~masks[n]]) == 0
    for h in hooks: h.remove()
