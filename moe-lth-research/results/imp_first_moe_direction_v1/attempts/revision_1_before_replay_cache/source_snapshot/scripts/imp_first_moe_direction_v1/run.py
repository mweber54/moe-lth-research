"""Two-stage eligibility sealing and bounded 160-cell direction evaluation."""
import argparse,hashlib,json,math,os,socket,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.imp_first_moe_direction_v1 import engine as e,report as r
from scripts.imp_first_moe_amplitude_v1 import run as amplitude
c=e.c;OUT=r.OUT;ORDER=(17,7,29,43,61)

def canonical(value):return json.loads(json.dumps(value,sort_keys=True))
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def source_files():return sorted(list((ROOT/'scripts/imp_first_moe_direction_v1').glob('*.py'))+[ROOT/'tests/test_imp_first_moe_direction_v1.py',Path(amplitude.__file__),Path(e.amp.__file__),Path(e.old.__file__),Path(e.old.replay.__file__),Path(c.__file__)]+list((ROOT/'src/moe_lth/models').glob('*.py')))
def seal():
    audit=c._json_read(OUT/'audit_completion.json');c._need(audit['status']=='PASS','Audit not complete')
    for name,key in [('checkpoint_and_probe_hashes.json','provenance_sha256'),('artifact_inventory.csv','inventory_sha256')]:c._need(c._sha256(OUT/name)==audit[key],'Audited source identity changed')
    prov=c._json_read(OUT/'checkpoint_and_probe_hashes.json');counts={(p['validation_batches'],p['validation_target_tokens']) for p in prov['probes']};c._need(len(counts)==1,'Source validation counts inconsistent');nb,nt=next(iter(counts))
    config=dict(id=OUT.name,seeds=list(c.SEEDS),model_types=list(c.MODEL_TYPES),selected_layer=0,selection='post hoc follow-up to completed interface/amplitude studies',condition='router_experts_frozen',start_step=1000,end_step=1500,recipients=list(e.TIMES),modes=list(e.MODES),variants=list(e.VARIANTS),logical_cells=160,pilot_seed=17,primary_mode=e.MODES[0],eligibility_candidate='exact FP64 nonzero in both native donor vectors; no epsilon or floor; abort on unusable FP32 construction before CE',eligibility_evidence='existing sealed training-only four-complete-batch probes and dtype-aware synthetic tests; no new validation losses before rule seal',construction_dtype='FP64 then cast to recipient FP32',norm_rtol=e.NORM_RTOL,direction_l2_atol=e.DIRECTION_L2_ATOL,fp64_unit_norm_atol=e.UNIT_ATOL,outside_S='same recipient-native contribution in all four variants',validation=dict(batches=nb,target_tokens=nt,rebatching=False,precision='FP32',tf32=False,independent_test=False),ce_atol=1e-6,logit_atol=1e-6,logit_rtol=1e-5,contrasts=list(r.METRICS),loss_scope='whole validation population; batch/sequence differences nested in seed',full_donor_context='reuse verified prior alpha1 cells only; no causal full-minus-common support effect',limits=r.LIMITS)
    sealed=dict(configuration=config,source_hashes={str(p.relative_to(ROOT)):c._sha256(p) for p in source_files()},audit=audit);path=OUT/'specification_seal.json'
    if path.exists():c._need(c._json_read(path)==sealed,'Source/specification changed: preserve attempts and explicitly revise/invalidate affected caches')
    else:c._json_write(OUT/'analysis_config.json',config);(OUT/'protocol.md').write_text(r.PROTOCOL);c._json_write(path,sealed)
    return prov,c._sha256(path)

def cache_identity(pair,spec,probe=None,eligibility_sha=None):
    value=dict(seed=pair['seed'],model_type=pair['model_type'],checkpoints=pair['checkpoints'],specification_sha256=spec)
    if probe is not None:value.update(probe_sha256=probe['materialized_probe_sha256'],probe_batch_hashes=probe['complete_batch_hashes'])
    if eligibility_sha is not None:value.update(eligibility_sha256=eligibility_sha,validation_batch_hashes=pair['prior_cache']['identity']['validation_batch_hashes'],prior_interface_cache_sha256=pair['prior_cache']['sha256'],grid=dict(recipients=e.TIMES,modes=e.MODES,variants=e.VARIANTS))
    return canonical(value)
def read_cache(path,identity):
    if not path.exists():return None
    result=c._json_read(path);c._need(result['identity']==identity and result['checks']['status']=='PASS','Invalid cache identity/checks');return result

def validate(result,probe):
    rows=result['losses'];expected={(rr,p,v,i) for rr in e.TIMES for p in e.MODES for v in e.VARIANTS for i in range(probe['validation_batches'])};c._need(len(rows)==len(expected) and {(z['recipient'],z['mode'],z['variant'],z['batch_index']) for z in rows}==expected,'Incomplete/duplicate readout grid')
    for rr in e.TIMES:
        for p in e.MODES:
            for v in e.VARIANTS:c._need(sum(z['target_tokens'] for z in rows if (z['recipient'],z['mode'],z['variant'])==(rr,p,v))==probe['validation_target_tokens'],'Wrong target coverage')
    c._need(len(result['coverage'])==probe['validation_batches'] and len(result['norm_checks'])==4*probe['validation_batches'],'Coverage or norm-check record missing')
    for z in rows:c._need(len(z['sequence_nll'])==z['sequence_count'] and z['sequence_count']*z['sequence_length']==z['target_tokens'] and abs(math.fsum(z['sequence_nll'])-z['nll_sum'])/z['target_tokens']<=1e-12,'Sequence/batch NLL aggregation mismatch')

def full_context(pair,previous):
    rows=[];nb=len(pair['prior_cache']['identity']['validation_batch_hashes'])
    for rr in e.TIMES:
        for donor in e.TIMES:
            for p in e.MODES:
                previous_mode=e.old.MODES[0] if p==e.MODES[0] else e.old.MODES[1];selected=[previous[rr,donor,previous_mode,i] for i in range(nb)];n=sum(int(z['target_tokens']) for z in selected)
                rows.append(dict(seed=pair['seed'],model_type=pair['model_type'],recipient=rr,donor=donor,mode=p,task_ce=math.fsum(float(z['nll_sum']) for z in selected)/n,target_tokens=n,batches=nb,scope='verified_full_donor_substitution_not_common_support',source='completed_interface_cache',prior_cache_sha256=pair['prior_cache']['sha256']))
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evaluate',action='store_true');parser.add_argument('--seal-only',action='store_true');args=parser.parse_args();OUT.mkdir(exist_ok=True);prov,spec=seal()
    if not args.evaluate:
        if not (OUT/'completion.json').exists():c._json_write(OUT/'completion.json',dict(status='SPECIFICATION_SEALED',scientifically_complete=False))
        return
    attempt=OUT/'attempts'/f'attempt_{time.time_ns()}';attempt.mkdir(parents=True);c._json_write(attempt/'invocation.json',dict(host=socket.getfqdn(),pid=os.getpid(),command=sys.argv,specification_sha256=spec,torch_version=e.torch.__version__,cuda_version=e.torch.version.cuda))
    command=[sys.executable,'-m','pytest','tests/test_imp_first_moe_direction_v1.py','tests/test_imp_first_moe_amplitude_v1.py','tests/test_imp_first_moe_interface_v1.py','tests/test_imp_expert_contribution_ablation_v1.py','tests/test_model.py','-q'];tests=subprocess.run(command,cwd=ROOT,capture_output=True,text=True);testrecord=dict(command=command,returncode=tests.returncode,output=tests.stdout+tests.stderr);c._json_write(attempt/'synthetic_checks.json',testrecord);c._need(tests.returncode==0,'Numerical tests failed before eligibility/CE')
    try:admission=c.gpu_admission_check()
    except RuntimeError as exc:c._json_write(OUT/'completion.json',dict(status='RESOURCE_BLOCKED',scientifically_complete=False,reason=str(exc),attempt=str(attempt)));return
    admission['host']=socket.getfqdn();c._json_write(OUT/'gpu_admission.json',admission)
    pairs={(p['seed'],p['model_type']):p for p in prov['pairs']};probes={p['seed']:p for p in prov['probes']};variants=c.adaptation_variants();calibrations=[];readouts=[];context=[];new_cal=0;new_eval=0;profile_forwards=0;cache_reuse=0;started=time.perf_counter()
    with c.exclusive_claim(OUT/'evaluation.lock'):
        c.gpu_admission_check();device=e.torch.device('cuda');e.torch.cuda.reset_peak_memory_stats()
        for seed in ORDER:
            probe=probes[seed];path=Path(probe['materialized_probe_path']);c._need(c._sha256(path)==probe['materialized_probe_sha256'],'Training probe changed');batches=e.torch.load(path,map_location='cpu',weights_only=True);c._need([c.common.batch_hash([b]) for b in batches]==probe['complete_batch_hashes'],'Probe sequence/target identities differ')
            for m in c.MODEL_TYPES:
                pair=pairs[seed,m];identity=cache_identity(pair,spec,probe=probe);path=OUT/'eligibility_cache'/f'{digest(identity)}.json';result=read_cache(path,identity)
                if result is None:
                    states,masks=amplitude.load_states(pair,variants[seed]);result=e.evaluate(states,masks,variants[seed]['config']['model'],batches,identity,device,calibration=True);new_cal+=result['forwards'];del states,masks
                    c._need(result['losses']==[] and sum(z['positions'] for z in result['coverage'])==probe['probe_target_tokens'],'Eligibility calibration computed losses or wrong coverage');c._json_write(path,result)
                calibrations.append(result);print('ELIGIBILITY_PROBE_PASS',seed,m,flush=True)
        evidence=[]
        for z in calibrations:
            errors=[v['maximum_relative_norm_error'] for v in z['norm_checks'] if v['maximum_relative_norm_error'] is not None];directions=[v['maximum_direction_l2_error'] for v in z['norm_checks'] if v['maximum_direction_l2_error'] is not None];cov=r.pooled_coverage(z)
            evidence.append(dict(seed=z['identity']['seed'],model_type=z['identity']['model_type'],positions=cov['positions'],eligible_positions=cov['eligible_positions'],minimum_positive_start_norm=cov['minimum_positive_start_norm'],minimum_positive_end_norm=cov['minimum_positive_end_norm'],max_relative_norm_error=max(errors) if errors else None,max_direction_l2_error=max(directions) if directions else None,numerical_floor_exclusions=0,checks=z['checks']))
        eligibility=dict(status='PASS',rule='S = positions with both native FP64 L2 norms strictly greater than zero',floor=None,epsilon=None,construction='FP64 unit vectors and hybrids; cast to recipient FP32',norm_rtol=e.NORM_RTOL,direction_l2_atol=e.DIRECTION_L2_ATOL,fp64_unit_norm_atol=e.UNIT_ATOL,numerically_unusable_policy='abort before hybrid loss; no invented directions or silent exclusions and no outcome-based threshold changes',dtype_tests='stable near-zero FP32 vectors pass; a deliberately unrepresentable smallest-subnormal hybrid is refused before any forward',training_only_evidence=sorted(evidence,key=lambda z:(z['seed'],z['model_type'])),source_specification_sha256=spec,cache_hashes={p.name:c._sha256(p) for p in sorted((OUT/'eligibility_cache').glob('*.json'))},new_validation_losses_before_this_seal=False)
        path=OUT/'direction_eligibility.json'
        if path.exists():c._need(c._json_read(path)==eligibility,'Eligibility rule/evidence changed; refuse CE-based revision')
        else:c._json_write(path,eligibility)
        eligibility_sha=c._sha256(path);c._json_write(OUT/'progress.json',dict(status='ELIGIBILITY_SEALED',validated_pairs=0,logical_cells=0,eligibility_sha256=eligibility_sha))
        for seed in ORDER:
            batches,vhash=c._validation_batches(seed,variants[seed]);probe=probes[seed];c._need(vhash==probe['validation_sha256'] and [c.common.batch_hash([b]) for b in batches]==probe['validation_batch_hashes'],'Validation boundaries/targets changed')
            for m in c.MODEL_TYPES:
                c._need(c._sha256(OUT/'direction_eligibility.json')==eligibility_sha,'Eligibility changed after seal');pair=pairs[seed,m];identity=cache_identity(pair,spec,eligibility_sha=eligibility_sha);path=OUT/'cache'/f'{digest(identity)}.json';result=read_cache(path,identity);previous=amplitude.prior_cells(pair);context+=full_context(pair,previous)
                if result is None:
                    states,masks=amplitude.load_states(pair,variants[seed])
                    if not (OUT/'evaluation_profile.json').exists():
                        tick=time.perf_counter();pilot=e.evaluate(states,masks,variants[seed]['config']['model'],batches[:1],identity,device,prior_rows=previous);elapsed=time.perf_counter()-tick;profile_forwards+=pilot['forwards'];c._json_write(OUT/'evaluation_profile.json',dict(status='PASS',engineering_pilot_seed=17,one_complete_batch_seconds=elapsed,projected_remaining_seconds=elapsed*len(batches)*10,projected_cache_bytes=len(json.dumps(pilot).encode())*len(batches)*10,peak_memory_bytes=e.torch.cuda.max_memory_allocated(),profile_forwards=pilot['forwards'],eligibility_sha256=eligibility_sha));del pilot
                    result=e.evaluate(states,masks,variants[seed]['config']['model'],batches,identity,device,prior_rows=previous);new_eval+=result['forwards'];del states,masks;validate(result,probe);c._json_write(path,result)
                else:validate(result,probe);cache_reuse+=1
                readouts.append(result);c._json_write(OUT/'progress.json',dict(status='EVALUATING',validated_pairs=len(readouts),logical_cells=len(readouts)*16,expected_pairs=10));print('VALIDATED_PAIR',seed,m,flush=True)
            if seed==17:c._json_write(OUT/'pilot_completion.json',dict(status='PASS',seed=17,pairs=2,retained_in_cohort=True,criteria='numerical/provenance and resource feasibility only; no effect sign/magnitude filter'))
        key=lambda z:(c.SEEDS.index(z['identity']['seed']),c.MODEL_TYPES.index(z['identity']['model_type']));calibrations.sort(key=key);readouts.sort(key=key)
        verified=r.write_outputs(calibrations,readouts,context)
        norms=[v for z in readouts for v in z['norm_checks']];maxnorm=max(v['maximum_relative_norm_error'] for v in norms if v['maximum_relative_norm_error'] is not None);maxdir=max(v['maximum_direction_l2_error'] for v in norms if v['maximum_direction_l2_error'] is not None)
        c._json_write(OUT/'numerical_checks.json',dict(status='PASS',synthetic_checks=testrecord,eligibility_sha256=eligibility_sha,maximum_validation_relative_norm_error=maxnorm,maximum_validation_direction_l2_error=maxdir,calibration_checks=[dict(seed=z['identity']['seed'],model_type=z['identity']['model_type'],checks=z['checks']) for z in calibrations],validation_checks=[dict(seed=z['identity']['seed'],model_type=z['identity']['model_type'],checks=z['checks']) for z in readouts],independent_export_checks=verified,pilot=c._json_read(OUT/'pilot_completion.json')))
        c._json_write(OUT/'compute_accounting.json',dict(host=socket.getfqdn(),wall_seconds=time.perf_counter()-started,eligibility_new_forwards=new_cal,validation_new_forwards=new_eval,profile_forwards=profile_forwards,reused_pair_caches=cache_reuse,logical_cells=160,prior_full_donor_context_cells=80,profile=c._json_read(OUT/'evaluation_profile.json'),peak_memory_bytes=e.torch.cuda.max_memory_allocated(),cache_bytes=sum(p.stat().st_size for p in (OUT/'cache').glob('*.json')),training_updates=0,backward_passes=0,optimizer_updates=0,new_masks=0,permanent_parameter_changes=0,attempt=str(attempt)))
        c._json_write(OUT/'completion.json',dict(status='COMPLETE',scientifically_complete=True,logical_cells=160,validated_pairs=10,seeds=list(c.SEEDS),specification_sha256=spec,eligibility_sha256=eligibility_sha,independent_checks=verified));c._json_write(OUT/'progress.json',dict(status='COMPLETE',validated_pairs=10,logical_cells=160));r.bundle();c._json_write(attempt/'completion.json',dict(status='COMPLETE',new_forwards=new_cal+new_eval+profile_forwards))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        failure=dict(status='FAILED',scientifically_complete=False,error=repr(exc),traceback=traceback.format_exc());c._json_write(OUT/'attempts'/f'failure_{time.time_ns()}.json',failure);c._json_write(OUT/'completion.json',failure);raise
