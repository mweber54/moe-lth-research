import math,json
import pytest
import torch
from scripts.imp_first_moe_amplitude_v1 import engine as e,report as r

def test_four_variants_global_scaling_and_identity():
    c0=torch.tensor([[[2.,0.],[0.,1.]]]);c1=torch.tensor([[[0.,6.],[3.,0.]]]);donors={'start':{'c':c0},'end':{'c':c1}}
    expected=[c0,3*c0,c1/3,c1]
    for name,value in zip(e.VARIANTS,expected):
        result=e.variant_tensor(donors,name,3.);assert torch.equal(result['c'],value)
        native=c0 if name.startswith('START') else c1
        assert torch.equal(result['c']==0,native==0)
    assert torch.equal(e.variant_tensor(donors,e.VARIANTS[1],1.)['c'],c0)
    assert torch.equal(e.variant_tensor(donors,e.VARIANTS[2],1.)['c'],c1)

def test_rms_scale_fp64_zeros_and_unavailable_are_not_zero_effects(tmp_path):
    estimate=e.estimate_scale(8.,32.,8);assert estimate['a0']==1 and estimate['a1']==2 and estimate['scale']==2
    # Actual element weighting, including rejected zeros; not a batch RMS average.
    expected=math.sqrt((32.+4.)/(8.+8.));assert e.estimate_scale(16.,36.,16)['a1']==expected
    for s0,s1 in [(0.,1.),(1.,0.),(float('inf'),1.)]:
        unavailable=e.estimate_scale(s0,s1,8);assert unavailable['scale'] is None and unavailable['status']=='UNAVAILABLE'
        with pytest.raises(RuntimeError):e.variant_tensor({'start':{'c':torch.ones(2,2)},'end':{'c':torch.ones(2,2)}},e.VARIANTS[1],unavailable['scale'])
    row=r.contrasts(dict(zip(e.VARIANTS,[10.,None,None,6.])));assert row['B_original']==4 and row['B_content_end'] is None
    path=tmp_path/'missing.json';e.c._json_write(path,row);assert json.loads(path.read_text())['B_content_end'] is None

def test_contrast_bookkeeping_signs_and_two_identities():
    z=r.contrasts(dict(zip(e.VARIANTS,[10.,7.,8.,6.])));assert z==dict(B_original=4.,B_scale_start=3.,B_content_end=1.,B_content_start=2.,B_scale_end=2.,I=-1.)
    assert z['B_original']==z['B_scale_start']+z['B_content_end']==z['B_content_start']+z['B_scale_end']
    neg=r.contrasts(dict(zip(e.VARIANTS,[3.,4.,5.,2.])));assert neg['B_scale_start']==-1 and neg['B_content_start']==-2 and neg['I']==4

def fixture():
    torch.manual_seed(712)
    cfg=dict(vocab_size=19,max_seq_len=8,num_layers=4,num_heads=2,d_model=8,num_experts=3,expert_hidden_size=12,dropout=0.,top_k=1,capacity_factor=.5)
    models={t:e.c.TinyMoELanguageModel(cfg) for t in e.TIMES};models['end'].load_state_dict(models['start'].state_dict())
    with torch.no_grad():models['end'].token_embedding.weight.add_(torch.randn_like(models['end'].token_embedding.weight)*.2)
    states={t:{n:v.clone() for n,v in m.state_dict().items()} for t,m in models.items()};identity=dict(seed=17,model_type='sparse',checkpoints={t:dict(state_sha256=e.c.state_dict_hash(states[t])) for t in e.TIMES})
    x=torch.randint(0,19,(2,8));y=torch.randint(0,19,(2,8))
    return cfg,models,states,identity,x,y

def test_scaled_boundary_preserves_recipient_and_current_input_replay():
    cfg,models,states,identity,x,y=fixture();bh=e.c.common.batch_hash([(x,y)])
    for m in models.values():m.eval();e.old.replay.install(m)
    with torch.inference_mode(),e.c._precision_guard():
        cache={};native={}
        for t in e.TIMES:native[t],cache[t]=e.old.capture(models[t],x,y,dict(seed=17,model_type='sparse',step=1000 if t=='start' else 1500,batch_hash=bh))
        assert not torch.equal(cache['start']['u'],cache['end']['u'])
        for recipient in e.TIMES:
            for mode in e.MODES:
                for name in e.VARIANTS:
                    variant=e.variant_tensor(cache,name,1.7);out,obs=e.insert(models[recipient],x,y,cache[recipient],variant,mode)
                    assert torch.equal(obs['u'],cache[recipient]['u'])
                    assert torch.equal(obs['h'],cache[recipient]['u']+variant['c'])
                    assert torch.equal(out.pre_router_hidden_states[0],cache[recipient]['z'])
                    if mode==e.MODES[1]:
                        for a,b in zip(out.routes[1:],native[recipient].routes[1:]):assert torch.equal(a.accepted_mask,b.accepted_mask) and torch.equal(a.selected_probabilities,b.selected_probabilities)

def test_calibration_has_no_ce_and_partial_batches_preserve_state(monkeypatch):
    cfg,models,states,identity,x,y=fixture();batches=[(x,y),(x[:1],y[:1])];entry=e.c._rng_snapshot()
    def forbidden(*args,**kwargs):raise AssertionError('Calibration cannot compute a task loss')
    with monkeypatch.context() as patch:
        patch.setattr(e.c,'_ce_vector',forbidden)
        calibration=e.evaluate(states,{t:{} for t in e.TIMES},cfg,batches,identity,torch.device('cpu'),calibration=True)
    assert calibration['rows']==[] and calibration['target_tokens']==24
    assert sum(z['elements'] for z in calibration['calibration_batch_sums'])==24*8
    assert e.c._rng_equal(entry,e.c._rng_snapshot())
    val=e.evaluate(states,{t:{} for t in e.TIMES},cfg,batches,identity,torch.device('cpu'),scale=1.5,pilot=True)
    assert val['checks']['status']=='PASS' and len(val['rows'])==32
    assert e.c._rng_equal(entry,e.c._rng_snapshot())
    for recipient in e.TIMES:
        for mode in e.MODES:
            for name in e.VARIANTS:assert sum(z['target_tokens'] for z in val['rows'] if (z['recipient'],z['mode'],z['variant'])==(recipient,mode,name))==24
