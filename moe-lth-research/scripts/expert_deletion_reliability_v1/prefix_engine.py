"""Reuse checkpoint/batch prefix strictly upstream of the fixed layer1 intervention.

Ephemeral GPU tensors only. Complete-batch native layer1 and all downstream layers
are rerun. Prefix reuse is valid only because every intervention is layer1 fc1 or
layer1 expert-output zero, and original weights are restored between interventions.
"""
from reliability_common import *
import accelerated
import weakref
from moe_lth.models.transformer import TinyMoELanguageModel,LanguageModelOutput
_ORIGINAL=TinyMoELanguageModel.forward
_MODELS=weakref.WeakSet()


def forward(self,token_ids,route_overrides=None):
    if route_overrides is not None:return _ORIGINAL(self,token_ids,route_overrides)
    cached=getattr(self,'_diagnostic_fixed_prefix',None)
    if cached is None or not torch.equal(cached['tokens'],token_ids):
        data={};handles=[]
        def first(module,args,out):data.update(aux0=out[1],route0=out[2],pre0=out[3])
        def residual(module,args):data['residual1']=args[0].detach().clone()
        def pre(module,args):data['pre1']=args[0].detach().clone()
        try:
            handles=[self.blocks[0].register_forward_hook(first),self.blocks[1].moe_norm.register_forward_pre_hook(residual),self.blocks[1].moe.register_forward_pre_hook(pre)]
            out=_ORIGINAL(self,token_ids,route_overrides)
        finally:
            for h in handles:h.remove()
        data['tokens']=token_ids.detach().clone();self._diagnostic_fixed_prefix=data;_MODELS.add(self)
        return out
    expert,aux,trace=self.blocks[1].moe(cached['pre1'])
    hidden=cached['residual1']+expert;routes=[cached['route0'],trace];pre=[cached['pre0'].detach(),cached['pre1'].detach()];auxes=[cached['aux0'],aux]
    seq=token_ids.shape[1];mask=torch.full((seq,seq),float('-inf'),device=token_ids.device).triu(1)
    for block in self.blocks[2:]:
        hidden,a,r,p=block(hidden,mask,None);routes.append(r);pre.append(p.detach());auxes.append(a)
    logits=self.lm_head(self.final_norm(hidden))
    return LanguageModelOutput(logits=logits,auxiliary_loss=torch.stack(auxes).mean(),routes=routes,pre_router_hidden_states=pre)


@contextlib.contextmanager
def enabled():
    old=TinyMoELanguageModel.forward;TinyMoELanguageModel.forward=forward
    try:
        with accelerated.enabled():yield
    finally:
        TinyMoELanguageModel.forward=old
        for m in list(_MODELS):
            if hasattr(m,'_diagnostic_fixed_prefix'):del m._diagnostic_fixed_prefix
        _MODELS.clear()


def identity(kind,t,b):
    base=cache_identity(kind,t,b);path=cache_path(kind,t,b)
    if path.exists():
        actual=load(path)['identity'];copy=dict(actual);route=copy.pop('routing_engine_sha256',None);prefix=copy.pop('prefix_engine_sha256',None)
        assert copy==base
        if route is not None:assert route==sha(Path(accelerated.__file__)) and s.read(D/'acceleration_validation.json')['passed']
        if prefix is not None:assert prefix==sha(Path(__file__)) and s.read(D/'prefix_validation.json')['passed']
        return actual
    validation=s.read(D/'prefix_validation.json');assert validation['passed'] and validation['source_sha256']==sha(Path(__file__))
    return dict(**base,routing_engine_sha256=sha(Path(accelerated.__file__)),prefix_engine_sha256=sha(Path(__file__)))


def validate():
    import measure
    reg=verify_frozen();torch.set_num_threads(2);m=s.model_for(s.jobs()['sparse'],1000);batch=load(D/'probes.pt')[0];reference=load(cache_path('sparse',1000,0));side=D/'host_compatibility'/'prefix_registered_profile.pt';measure.cache_path=lambda kind,t,b:side
    measure.cache_identity=lambda kind,t,b:dict(**cache_identity(kind,t,b),routing_engine_sha256=sha(Path(accelerated.__file__)),prefix_engine_sha256=sha(Path(__file__)))
    stats=dict(native_complete_batch_forwards=0,native_evaluation_wall_seconds=0.,cached_batches_reused=0);started=time.perf_counter()
    with enabled():comparison=measure.measure_batch(m,'sparse',1000,0,batch,reg,stats)
    assert dc.equal_tree(reference['baseline'],comparison['baseline'])
    assert dc.equal_tree(reference['paired_sequence_statistics'],comparison['paired_sequence_statistics'])
    assert dc.equal_tree(reference['rows'],comparison['rows'])
    save(D/'prefix_validation.json',dict(passed=True,source_sha256=sha(Path(__file__)),routing_sha256=sha(Path(accelerated.__file__)),all_56_interventions_bitwise_equal=True,all_CE_output_energy_sequence_statistics_equal=True,registered_batch=0,extra_unique_contexts=0,no_hidden_tensors_saved=True,wall_seconds=time.perf_counter()-started,stats=stats))
    print('Prefix cache exact-profile validation passed',time.perf_counter()-started,'seconds',flush=True)

if __name__=='__main__':
    import argparse,measure
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['validate','measure']);a=ap.parse_args()
    with worker(),s.protected():
        if a.stage=='validate':validate()
        else:
            measure.cache_identity=identity
            with enabled():measure.run('measure')
