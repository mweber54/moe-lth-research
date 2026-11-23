"""Bitwise-equivalent native top1 dispatch with fewer CPU/GPU synchronizations.

Only integer assignment-index construction changes. Router, scores/topk, capacity,
expert calls, gates, expert order, index_add and trace definitions remain native.
Scientific definitions and original frozen source files are unchanged.
"""
from reliability_common import *
import math
from moe_lth.models.moe_layer import MoEFeedForward,RouteTrace
from moe_lth.models.router import RouteOverride
_ORIGINAL=MoEFeedForward.forward


def forward(self,inputs,override_ids=None):
    if self.top_k!=1 or override_ids is not None:return _ORIGINAL(self,inputs,override_ids)
    batch,seq,width=inputs.shape;flat=inputs.reshape(-1,width)
    router=self.router(flat,None,None);selected=router.selected_experts;prob=router.selected_probability
    # Exact stable token order within expert, equal to native row-major nonzero.
    ids=selected.detach().flatten().cpu().numpy();counts=np.bincount(ids,minlength=self.num_experts)
    order=torch.from_numpy(np.argsort(ids,kind='stable')).to(selected.device)
    output=torch.zeros_like(flat);accepted=torch.zeros_like(selected,dtype=torch.bool)
    capacity=max(1,math.ceil(self.capacity_factor*len(flat)*self.top_k/self.num_experts));offset=0
    for e,expert in enumerate(self.experts):
        n=int(counts[e]);positions=order[offset:offset+n];offset+=n
        if not n:continue
        if n>capacity:positions=positions[prob[positions,0].topk(capacity).indices]
        slots=torch.zeros_like(positions)
        accepted[positions,slots]=True
        result=expert(flat[positions]);weights=prob[positions,slots,None]
        output=output.index_add(0,positions,result*weights)
    usage=torch.bincount(selected.flatten(),minlength=self.num_experts).float();usage=usage/max(1,selected.numel())
    meanprob=router.probabilities.mean(dim=0);aux=self.num_experts*torch.sum(usage.detach()*meanprob)
    trace=RouteTrace(selected_experts=selected[:,0].reshape(batch,seq),selected_probability=prob[:,0].reshape(batch,seq),probabilities=router.probabilities.reshape(batch,seq,self.num_experts),selected_expert_indices=selected.reshape(batch,seq,self.top_k),selected_probabilities=prob.reshape(batch,seq,self.top_k),entropy=router.entropy.reshape(batch,seq),margin=router.margin.reshape(batch,seq),usage=usage,dropped_fraction=1.0-accepted.float().mean(),accepted_mask=accepted)
    return output.reshape(batch,seq,width),aux,trace


@contextlib.contextmanager
def enabled():
    old=MoEFeedForward.forward;MoEFeedForward.forward=forward
    try:yield
    finally:MoEFeedForward.forward=old


def identity(kind,t,b):
    base=cache_identity(kind,t,b);path=cache_path(kind,t,b)
    if path.exists():
        actual=load(path)['identity'];candidate=dict(actual);engine=candidate.pop('routing_engine_sha256',None)
        assert candidate==base
        if engine is not None:assert engine==sha(Path(__file__)) and s.read(D/'acceleration_validation.json')['passed']
        return actual
    assert s.read(D/'acceleration_validation.json')['passed']
    return dict(**base,routing_engine_sha256=sha(Path(__file__)))


def validate():
    import measure
    reg=verify_frozen();torch.set_num_threads(2);m=s.model_for(s.jobs()['sparse'],1000);batch=load(D/'probes.pt')[0]
    reference=load(cache_path('sparse',1000,0));side=D/'host_compatibility'/'accelerated_registered_profile.pt';measure.cache_path=lambda kind,t,b:side
    measure.cache_identity=lambda kind,t,b:dict(**cache_identity(kind,t,b),routing_engine_sha256=sha(Path(__file__)))
    stats=dict(native_complete_batch_forwards=0,native_evaluation_wall_seconds=0.,cached_batches_reused=0)
    started=time.perf_counter()
    with enabled():comparison=measure.measure_batch(m,'sparse',1000,0,batch,reg,stats)
    assert dc.equal_tree(reference['baseline'],comparison['baseline'])
    assert dc.equal_tree(reference['paired_sequence_statistics'],comparison['paired_sequence_statistics'])
    assert dc.equal_tree(reference['rows'],comparison['rows'])
    result=dict(passed=True,source_sha256=sha(Path(__file__)),frozen_sources_changed=False,registered_batch=0,all_56_interventions_bitwise_equal=True,baseline_and_paired_sequence_statistics_bitwise_equal=True,all_CE_and_output_energies_bitwise_equal=True,extra_unique_contexts=0,wall_seconds=time.perf_counter()-started,stats=stats,engineering_change='replace128 dynamic GPU nonzero synchronizations per forward with stable integer grouping; native floating-point computation unchanged')
    save(D/'acceleration_validation.json',result);print('Acceleration exact-profile validation passed',result['wall_seconds'],'seconds',flush=True)


def main(stage):
    import measure
    if stage=='validate':validate();return
    assert sha(Path(__file__))==s.read(D/'acceleration_validation.json')['source_sha256']
    measure.cache_identity=identity
    with enabled():measure.run('measure')

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['validate','measure']);a=ap.parse_args()
    with worker(),s.protected():main(a.stage)
