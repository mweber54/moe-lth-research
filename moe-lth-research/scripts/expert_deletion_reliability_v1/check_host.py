"""Cross-host engineering check on already registered batch0; no new contexts."""
from reliability_common import *
import measure as measurement


def main():
    start=time.perf_counter();reg=verify_frozen();torch.set_num_threads(2);kind='sparse';t=1000;b=0
    reference=load(cache_path(kind,t,b));m=s.model_for(s.jobs()[kind],t);probes=load(D/'probes.pt')
    side=D/'host_compatibility'/'adder_registered_profile.pt'
    measurement.cache_path=lambda kind,t,b:side
    stats=dict(native_complete_batch_forwards=0,native_evaluation_wall_seconds=0.,cached_batches_reused=0)
    comparison=measurement.measure_batch(m,kind,t,b,probes[b],reg,stats)
    assert torch.equal(reference['baseline']['loss'],comparison['baseline']['loss'])
    for key in ('ids','accepted','gates'):assert torch.equal(reference['baseline'][key],comparison['baseline'][key])
    assert dc.equal_tree(reference['paired_sequence_statistics'],comparison['paired_sequence_statistics'])
    primary=('baseline_loss_sum','deleted_loss_sum','damage_sum','baseline_ce','deleted_ce','damage')
    maxenergy=0.
    for a,z in zip(reference['rows'],comparison['rows']):
        assert a['intervention']==z['intervention'] and a['subset']==z['subset']
        for k in primary:assert a[k]==z[k],(k,a[k],z[k])
        for k in ('output_difference_energy','baseline_output_energy','gate_weighted_difference_energy','baseline_gate_weighted_energy'):
            assert np.isclose(a[k],z[k],rtol=1e-12,atol=1e-12),(k,a[k],z[k]);maxenergy=max(maxenergy,abs(a[k]-z[k]))
    save(D/'host_compatibility.json',dict(passed=True,host=socket.gethostname(),gpu=torch.cuda.get_device_name(),registered_batch=0,checkpoint='sparse1000',baseline_losses_bitwise_equal=True,all_union_and_control_CE_statistics_equal=True,paired_sequence_sums_counts_equal=True,baseline_routing_gates_equal=True,energy_tolerance=dict(rtol=1e-12,atol=1e-12),maximum_absolute_energy_roundoff=maxenergy,new_contexts=0,wall_seconds=time.perf_counter()-start,stats=stats))
    print('Registered profile cross-host compatibility passed',flush=True)
if __name__=='__main__':
    with worker(),s.protected():main()
