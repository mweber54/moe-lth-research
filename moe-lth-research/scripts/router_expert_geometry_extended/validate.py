from common import *
from measurements import *
from moe_lth.models import TinyMoELanguageModel
import random

def validate():
    torch.set_num_threads(2);g=torch.Generator().manual_seed(1223);w=torch.randn(12,8,generator=g,dtype=torch.float64)
    original=gram(w);ident=gram_compare(original,original);assert ident['shape_distance']==0 and abs(ident['cosine']-1)<1e-12
    scaled=gram_compare(gram(w*3),original);assert scaled['shape_distance']<1e-12 and abs(scaled['trace_ratio']-9)<1e-12
    q=torch.linalg.qr(torch.randn(12,12,generator=g,dtype=torch.float64))[0]
    orthogonal=gram_compare(gram(q@w),original);assert orthogonal['shape_distance']<1e-12
    a=torch.diag(torch.tensor([1.,2.,3.,4.],dtype=torch.float64));b=torch.diag(torch.tensor([4.,3.,2.,1.],dtype=torch.float64))
    assert torch.linalg.matrix_rank(a)==torch.linalg.matrix_rank(b)==4
    assert gram_compare(gram(a),gram(b))['shape_distance']>.5
    zero=gram_compare(torch.zeros_like(original),original);assert zero['shape_distance'] is None and zero['cosine'] is None
    assert association(np.ones(10),np.arange(10))[2]=='zero_variance_score'
    assert association(np.arange(7),np.arange(7))[2]=='insufficient_samples'
    config=dict(vocab_size=32,max_seq_len=8,num_layers=2,num_heads=2,d_model=8,num_experts=4,expert_hidden_size=16,dropout=.1,top_k=1,capacity_factor=1.25)
    model=TinyMoELanguageModel(config);model.train();model.blocks[1].moe.experts[2].eval()
    tokens=torch.randint(0,32,(2,8),generator=g);positions=torch.randperm(16,generator=g)
    with protected(model):native=model(tokens);inputs=[x.reshape(-1,8)[positions].clone() for x in native.pre_router_hidden_states]
    probe=dict(tokens=tokens,positions=positions,inputs=inputs)
    before=dict(state=tensor_hash(model.state_dict()),modes=[m.training for m in model.modules()],cpu_rng=torch.get_rng_state().clone(),
                python=random.getstate(),numpy=np.random.get_state(),gradients={n:p.grad for n,p in model.named_parameters()})
    captured={};expert=model.blocks[0].moe.experts[0]
    with protected(model):
        handle=expert.fc2.register_forward_pre_hook(lambda m,args:captured.update(pre_fc2=args[0].detach().clone()))
        x=inputs[0];out=expert(x);handle.remove()
        act=expert.activation(expert.fc1(x));assert torch.equal(captured['pre_fc2'],act)
        assert torch.allclose(out,expert.fc2(act))
        groups,pools,within,drift,raw=activation_probe(model,probe)
        assert np.allclose(raw['fixed_L0_activations'][:,0],act.mean(-1).numpy())
        assert np.allclose(raw['fixed_L0_scores'],model.blocks[0].moe.router.projection(x).numpy())
        assert np.array_equal(raw['fixed_L0_selected'],raw['native_L0_selected'])
        assert np.array_equal(raw['fixed_L0_accepted'],raw['native_L0_accepted'])
    assert tensor_hash(model.state_dict())==before['state'];assert [m.training for m in model.modules()]==before['modes']
    assert torch.equal(torch.get_rng_state(),before['cpu_rng']);assert random.getstate()==before['python']
    npafter=np.random.get_state();assert npafter[0]==before['numpy'][0] and np.array_equal(npafter[1],before['numpy'][1]) and npafter[2:]==before['numpy'][2:]
    assert all(p.grad is before['gradients'][n] for n,p in model.named_parameters())
    # Exact cyclic null vectorization must match literal z-score pooling.
    rng=np.random.default_rng(123);s=rng.normal(size=(100,4));a=s*.2+rng.normal(size=(100,4));sel=np.arange(100)[:,None]%4;acc=np.ones_like(sel,dtype=bool)
    gr,po,wi=summarize_pairs(s,a,sel,acc,'test',0)
    rr=next(r for r in po if r['selection']=='all_expert')
    x=np.concatenate([(s[:,i]-s[:,i].mean())/s[:,i].std() for i in range(4)]);y=np.concatenate([(a[:,i]-a[:,i].mean())/a[:,i].std() for i in range(4)])
    assert abs(rr['pearson']-association(x,y)[0])<1e-12
    if torch.cuda.is_available():
        model=model.cuda();before_model=tensor_hash(model.state_dict());before_modes=[m.training for m in model.modules()]
        cpu_rng=torch.get_rng_state().clone();cuda_rng=[t.clone() for t in torch.cuda.get_rng_state_all()]
        flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        with protected(model):
            _,_,_,_,raw_gpu=activation_probe(model,probe)
            actual=model.blocks[0].moe.experts[0].activation(model.blocks[0].moe.experts[0].fc1(inputs[0].cuda())).mean(-1).cpu().numpy()
            assert np.allclose(raw_gpu['fixed_L0_activations'][:,0],actual,atol=1e-6,rtol=1e-5)
        assert tensor_hash(model.state_dict())==before_model and [m.training for m in model.modules()]==before_modes
        assert torch.equal(torch.get_rng_state(),cpu_rng)
        assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
        assert flags==(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    save(OUT/'validation.json',dict(passed=True,tests=['identical Gram','scale invariance and energy','same full row space/different Gram shape',
         'left orthogonal Gram invariance','zero norm undefined','zero variance and sample floor','native expert activation before route weight',
         'native/fixed capacity original batch order','model tensors/module modes/gradients/CPU Python NumPy RNG unchanged','vectorized cyclic null equals literal z-score pooling',
         'GPU batched activation equals native GELU; CPU/CUDA RNG, modes, weights and TF32 flags unchanged']))
    print('VALIDATION PASS',flush=True)
if __name__=='__main__':validate()
