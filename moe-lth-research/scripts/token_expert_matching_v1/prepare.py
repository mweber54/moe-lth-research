from run import *
import shutil


def prepare():
    if (DEST/'registration.json').exists():raise RuntimeError('Frozen registration already exists')
    manifest=frozen();jobs=[]
    for variant in manifest['variants']:
        for kind,draw in [('donor',None),('identity',None)]+[('shuffle',x) for x in PERMUTATIONS]:
            job=dict(id=variant['id']+'_'+kind+(f'_{draw}' if draw is not None else ''),variant_id=variant['id'],seed=variant['seed'],sparsity=variant['sparsity'],kind=kind,draw=draw)
            job['pilot']=variant['seed']==17 and variant['sparsity'] in (0.,.8) and draw in (None,101)
            jobs.append(job)
    save(DEST/'registration.json',dict(seeds=SEEDS,permutation_seeds=PERMUTATIONS,variants=manifest['variants'],jobs=jobs,source_manifest_sha256=sha(OUT/'manifest.json'),source_seal_sha256=sha(OUT/'seal.json'),pilot_criteria=dict(identity_native_improvement_retained_min=.9,identity_final_ce_native_ratio_max=1.5,max_amp_skips_per5000=500,finite_endpoints=True,penalty_sign_used=False),independent_unit='source seed',endpoint=5000))
    save(DEST/'donor_inventory.json',[dict(variant=v['id'],initial=v['initial'],mask=v['mask'],initial_model_hash=v['initial_model_hash'],mask_hash=v['mask_hash'],historical_native_attempt=v['full_attempt'],status='MISSING_COMPLETE_TRAINING_TRACE') for v in manifest['variants']])
    estimate=read(OUT/'resource_estimate.json');hours=2*estimate['median_measured_segment_seconds']/3600
    # Tensor payload 4 layers*16384 tokens*(ids8+gate4+accept1+position8+accepted order<=8).
    trace_bytes=15*5000*4*16384*29
    save(DEST/'resource_estimate.json',dict(total_runs=75,donors=15,recipients=60,pilot_runs=6,estimated_gpu_hours=75*hours*1.35,pilot_gpu_hours=6*hours*1.35,trace_bytes_upper_estimate=trace_bytes,checkpoint_bytes_estimate=75*4_000_000_000,suggested_free_storage_gib=550,free_bytes=shutil.disk_usage(DEST).free,basis='Historical native segment median;35% provisional tracing/checksum/diagnostic overhead. Expansion re-estimates from measured pilot wall time and actual bytes.'))
    save(DEST/'validation.json',dict(production_ready=False,reason='Awaiting both-host canonical numerical checks'))
    save(DEST/'seal.json',dict(protocol_sha256=sha(DEST/'protocol.md'),registration_sha256=sha(DEST/'registration.json'),implementation={str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')},source_input_inventory_sha256=sha(OUT/'input_hashes.json')))


if __name__=='__main__':prepare()
