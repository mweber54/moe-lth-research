import importlib.util
from pathlib import Path
import torch

P=Path(__file__).resolve().parents[1]/'scripts/imp_initial_value_placement_v1/run.py'
S=importlib.util.spec_from_file_location('placement_test',P)
study=importlib.util.module_from_spec(S);S.loader.exec_module(study)

def test_transformation_multiset_counts_and_rng():
    w=torch.linspace(-2,3,77).reshape(7,11)
    mask=torch.arange(77).reshape(7,11)%4==0
    before=torch.get_rng_state().clone()
    for mask_seed in (None,92001,92002):
        a=study.transform(w,mask,17,'layer.expert.fc1.weight',mask_seed,93001)
        b=study.transform(w,mask,17,'layer.expert.fc1.weight',mask_seed,93001)
        assert all(torch.equal(x,y) for x,y in zip(a,b))
        assert a[1].sum()==mask.sum()
        study.validate_value(w,*a)
    assert torch.equal(before,torch.get_rng_state())

def test_unchanged_original_and_nested_within_mask():
    w=torch.arange(60,dtype=torch.float32).reshape(6,10)-30
    m=torch.arange(60).reshape(6,10)%3==0
    for base in (None,92001,92002):
        original,mask,c,p=study.transform(w,m,7,'fc2.weight',base,None)
        shuffled,mask2,c2,p2=study.transform(w,m,7,'fc2.weight',base,93002)
        assert torch.equal(mask,mask2)
        assert torch.equal(original[mask],w[mask])
        assert torch.equal(mask.sum(0),mask2.sum(0))
        assert torch.equal(mask.sum(1),mask2.sum(1))
        assert torch.equal(torch.sort(original[mask]).values,torch.sort(shuffled[mask]).values)
        if base is None:assert torch.equal(original,w*m)

def test_estimand_and_equal_order_source_aggregation():
    v=dict(imp_original=2.,imp_shuffle1=5.,imp_shuffle2=7.,random1_original=3.,random1_shuffle=4.,random2_original=4.,random2_shuffle=6.)
    c=study.contrast(v)
    assert c['delta_imp']==4 and c['delta_random']==1.5 and c['interaction']==2.5
    assert c['A_original']-c['A_shuffled']==c['interaction']
    # Constant offsets in any source/order do not alter within-mask penalties.
    assert study.contrast({k:x+10 for k,x in v.items()})==c
    order_b=study.contrast({**v,'imp_shuffle1':7.})
    assert (c['interaction']+order_b['interaction'])/2==3.

def test_original_reconstruction_preserves_signed_zero_bits():
    w=torch.tensor([[-2.,3.],[-4.,5.]])
    mask=torch.tensor([[False,True],[False,True]])
    output,*_=study.transform(w,mask,7,'fc1.weight')
    expected=w*mask
    assert torch.equal(output.view(torch.int32),expected.view(torch.int32))
