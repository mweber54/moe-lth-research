"""Sealed training-only calibration followed by bounded checkpoint-only validation."""
import argparse,hashlib,json,math,os,socket,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.imp_first_moe_amplitude_v1 import engine as e,report as r
from scripts.imp_first_moe_interface_v1 import run as prior
c=e.c;OUT=r.OUT
ORDER=(17,7,29,43,61)

def normalized(value):return json.loads(json.dumps(value,sort_keys=True))
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def files():
    return sorted(list((ROOT/'scripts/imp_first_moe_amplitude_v1').glob('*.py'))+[ROOT/'tests/test_imp_first_moe_amplitude_v1.py',Path(prior.__file__),Path(e.old.__file__),Path(e.old.replay.__file__),Path(c.__file__)]+list((ROOT/'src/moe_lth/models').glob('*.py')))
def seal():
    a=c._json_read(OUT/'audit_completion.json');need=c._need;need(a['status']=='PASS','Audit not passed')
    need(c._sha256(OUT/'checkpoint_and_probe_hashes.json')==a['provenance_sha256'],'Provenance changed');need(c._sha256(OUT/'artifact_inventory.csv')==a['inventory_sha256'],'Inventory changed')
    prov=c._json_read(OUT/'checkpoint_and_probe_hashes.json');counts={(p['validation_batches'],p['validation_target_tokens']) for p in prov['probes']};need(len(counts)==1,'Inconsistent source validation counts');nb,nt=next(iter(counts))
    config=dict(id=OUT.name,seeds=list(c.SEEDS),model_types=list(c.MODEL_TYPES),layer=0,condition='router_experts_frozen',start_step=1000,end_step=1500,recipients=list(e.TIMES),modes=list(e.MODES),variants=list(e.VARIANTS),cells=160,pilot_seed=17,primary_mode=e.MODES[0],calibration='existing sealed four-complete-batch training probe; one FP64 RMS ratio per seed/model; no CE',validation=dict(batches=nb,target_tokens=nt,rebatching=False,independent_test=False,precision='FP32',tf32=False),tolerances=dict(ce_atol=1e-6,logit_atol=1e-6,logit_rtol=1e-5,relative_rms_linearity=1e-6),unavailable='explicit unavailable scaled cells and null contrasts; no epsilon',previous_unscaled_reuse='exact verified prior pair caches; all pilot A/D re-evaluated plus first complete batch in every other pair',insertion='previously verified complete first-MoE contribution only; one global multiply or divide, recipient residual unchanged',metrics=list(r.METRICS),no_validation_refit=True,limits=r.LIMITS)
    sealed=dict(configuration=config,source_hashes={str(p.relative_to(ROOT)):c._sha256(p) for p in files()},audit=a)
    path=OUT/'specification_seal.json'
    if path.exists():need(c._json_read(path)==sealed,'Specification/source changed: preserve failed attempt and explicitly revise before resuming')
    else:c._json_write(OUT/'analysis_config.json',config);(OUT/'protocol.md').write_text(r.PROTOCOL);c._json_write(path,sealed)
    return prov,c._sha256(path)

def load_states(pair,variant):
    states,masks,checkpoints=prior.load_pair(pair['seed'],pair['model_type'],variant);c._need(checkpoints==pair['checkpoints'],'State differs from audited prior interface state');return states,masks

def identity(pair,spec,probe=None,calibration_seal=None):
    result=dict(seed=pair['seed'],model_type=pair['model_type'],checkpoints=pair['checkpoints'],source_specification_sha256=spec)
    if probe is not None:result.update(probe_sha256=probe['materialized_probe_sha256'],probe_batch_hashes=probe['complete_batch_hashes'])
    if calibration_seal is not None:result.update(calibration_seal_sha256=calibration_seal,prior_cache_sha256=pair['prior_cache']['sha256'],validation_batch_hashes=pair['prior_cache']['identity']['validation_batch_hashes'],logical_grid=dict(recipients=e.TIMES,modes=e.MODES,variants=e.VARIANTS))
    return normalized(result)

def cached(path,ident):
    if not path.exists():return None
    result=c._json_read(path);c._need(result['identity']==ident and result['checks']['status']=='PASS','Invalid cache identity/checks');return result

def prior_cells(pair):
    source=pair['prior_cache'];c._need(c._sha256(Path(source['path']))==source['sha256'],'Previous interface cache changed')
    result=c._json_read(Path(source['path']));c._need(result['identity']==source['identity'],'Prior cache identity changed')
    rows={(z['recipient'],z['donor'],z['mode'],int(z['batch_index'])):z for z in result['batches'] if z['alpha']==1.}
    c._need(len(rows)==8*len(source['identity']['validation_batch_hashes']),'Unscaled cache coverage wrong');return rows

def validate_readout(result,expected_batches,expected_tokens):
    rows=result['rows'];expected={(rr,p,v,i) for rr in e.TIMES for p in e.MODES for v in e.VARIANTS for i in range(expected_batches)}
    c._need(len(rows)==len(expected) and {(z['recipient'],z['mode'],z['variant'],z['batch_index']) for z in rows}==expected,'Readout logical grid incomplete/duplicated')
    for rr in e.TIMES:
        for p in e.MODES:
            for v in e.VARIANTS:c._need(sum(z['target_tokens'] for z in rows if (z['recipient'],z['mode'],z['variant'])==(rr,p,v))==expected_tokens,'Target coverage wrong')
    c._need(result['checks']['status']=='PASS','Readout not validated')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evaluate',action='store_true');parser.add_argument('--seal-only',action='store_true');args=parser.parse_args();OUT.mkdir(exist_ok=True)
    prov,spec=seal()
    if not args.evaluate:
        if not (OUT/'completion.json').exists():c._json_write(OUT/'completion.json',dict(status='SPECIFICATION_SEALED',scientifically_complete=False))
        return
    attempt=OUT/'attempts'/f'attempt_{time.time_ns()}';attempt.mkdir(parents=True)
    c._json_write(attempt/'invocation.json',dict(host=socket.getfqdn(),pid=os.getpid(),spec_sha256=spec,command=sys.argv,torch_version=e.torch.__version__,cuda_version=e.torch.version.cuda))
    tests=subprocess.run([sys.executable,'-m','pytest','tests/test_imp_first_moe_amplitude_v1.py','tests/test_imp_first_moe_interface_v1.py','tests/test_imp_expert_contribution_ablation_v1.py','tests/test_model.py','-q'],cwd=ROOT,capture_output=True,text=True)
    testrecord=dict(command=tests.args,returncode=tests.returncode,output=tests.stdout+tests.stderr);c._json_write(attempt/'synthetic_checks.json',testrecord);c._need(tests.returncode==0,'Numerical tests failed before calibration/cohort')
    try:admission=c.gpu_admission_check()
    except RuntimeError as exc:
        c._json_write(OUT/'completion.json',dict(status='RESOURCE_BLOCKED',scientifically_complete=False,reason=str(exc),attempt=str(attempt)));return
    admission['host']=socket.getfqdn();c._json_write(OUT/'gpu_admission.json',admission)
    pairs={(p['seed'],p['model_type']):p for p in prov['pairs']};probes={p['seed']:p for p in prov['probes']};variants=c.adaptation_variants();calibrations=[];scales=[];readouts=[];fresh_calibration_forwards=0;fresh_validation_forwards=0;profile_forwards=0;reused=0;started=time.perf_counter()
    with c.exclusive_claim(OUT/'evaluation.lock'):
        c.gpu_admission_check();device=e.torch.device('cuda');e.torch.cuda.reset_peak_memory_stats()
        for seed in ORDER:
            probe=probes[seed];path=Path(probe['materialized_probe_path']);c._need(c._sha256(path)==probe['materialized_probe_sha256'],'Probe bytes changed')
            batches=e.torch.load(path,map_location='cpu',weights_only=True);c._need([c.common.batch_hash([b]) for b in batches]==probe['complete_batch_hashes'],'Probe complete-batch identity changed')
            for m in c.MODEL_TYPES:
                pair=pairs[seed,m];ident=identity(pair,spec,probe=probe);cache_path=OUT/'calibration_cache'/f'{digest(ident)}.json';result=cached(cache_path,ident)
                if result is None:
                    states,masks=load_states(pair,variants[seed]);result=e.evaluate(states,masks,variants[seed]['config']['model'],batches,ident,device,calibration=True);fresh_calibration_forwards+=result['forwards'];del states,masks
                    c._need(result['target_tokens']==probe['probe_target_tokens'] and len(result['calibration_batch_sums'])==probe['registered']['batch_count'],'Calibration coverage differs')
                    sums=result['calibration_batch_sums'];result['scale_estimate']=e.estimate_scale(math.fsum(z['start_sumsq'] for z in sums),math.fsum(z['end_sumsq'] for z in sums),sum(z['elements'] for z in sums));c._json_write(cache_path,result)
                estimate=result['scale_estimate'];scales.append(dict(seed=seed,model_type=m,**estimate,probe_batches=len(batches),probe_target_tokens=result['target_tokens'],probe_identity_sha256=probe['registered']['probe_identity_sha256'],scale_uses_validation_losses=False));calibrations.append(result)
                print('CALIBRATION_COMPLETE',seed,m,estimate['status'],flush=True)
        scales.sort(key=lambda z:(c.SEEDS.index(z['seed']),c.MODEL_TYPES.index(z['model_type'])))
        scale_path=OUT/'calibration_scales.csv'
        if scale_path.exists():
            old=c._read_csv(scale_path)
            c._need(len(old)==10 and all(float(o['scale'])==n['scale'] if o['scale'] else n['scale'] is None for o,n in zip(old,scales)),'Calibration scale cache changed')
        else:c._csv_write(scale_path,scales)
        scale_sealed=dict(specification_sha256=spec,calibration_scales_sha256=c._sha256(scale_path),calibration_cache_hashes={p.name:c._sha256(p) for p in sorted((OUT/'calibration_cache').glob('*.json'))},probe_identity_hashes={str(s):probes[s]['registered']['probe_identity_sha256'] for s in c.SEEDS},no_new_validation_interventions_before_this_seal=True)
        seal_path=OUT/'calibration_seal.json'
        if seal_path.exists():c._need(c._json_read(seal_path)==scale_sealed,'Calibration seal changed; no validation refit allowed')
        else:c._json_write(seal_path,scale_sealed)
        scale_digest=c._sha256(seal_path);scale_map={(z['seed'],z['model_type']):z['scale'] for z in scales}
        c._json_write(OUT/'progress.json',dict(status='CALIBRATION_SEALED',calibration_pairs=10,validated_pairs=0,logical_cells=0,calibration_seal_sha256=scale_digest))
        for seed in ORDER:
            batches,vhash=c._validation_batches(seed,variants[seed]);probe=probes[seed];c._need(c.common.batch_hash(batches)==probe['validation_sha256'],'Validation changed since calibration seal')
            for m in c.MODEL_TYPES:
                c._need(c._sha256(seal_path)==scale_digest and c._sha256(scale_path)==scale_sealed['calibration_scales_sha256'],'Calibration scales modified after seal')
                pair=pairs[seed,m];ident=identity(pair,spec,calibration_seal=scale_digest);cache_path=OUT/'cache'/f'{digest(ident)}.json';result=cached(cache_path,ident)
                if result is None:
                    states,masks=load_states(pair,variants[seed]);previous=prior_cells(pair);scale=scale_map[seed,m]
                    if not (OUT/'evaluation_profile.json').exists():
                        tick=time.perf_counter();profile=e.evaluate(states,masks,variants[seed]['config']['model'],batches[:1],ident,device,scale,previous,pilot=True);elapsed=time.perf_counter()-tick;profile_forwards+=profile['forwards']
                        c._json_write(OUT/'evaluation_profile.json',dict(status='PASS',engineering_pilot_seed=17,one_complete_batch_seconds=elapsed,projected_remaining_seconds=elapsed*len(batches)*10,projected_cache_bytes=len(json.dumps(profile).encode())*len(batches)*10,peak_memory_bytes=e.torch.cuda.max_memory_allocated(),calibration_seal_sha256=scale_digest,profile_forwards=profile['forwards']));del profile
                    result=e.evaluate(states,masks,variants[seed]['config']['model'],batches,ident,device,scale,previous,pilot=seed==17);fresh_validation_forwards+=result['forwards'];del states,masks
                    validate_readout(result,probe['validation_batches'],probe['validation_target_tokens']);c._json_write(cache_path,result)
                else:validate_readout(result,probe['validation_batches'],probe['validation_target_tokens']);reused+=1
                readouts.append(result);c._json_write(OUT/'progress.json',dict(status='EVALUATING',validated_pairs=len(readouts),expected_pairs=10,logical_cells=len(readouts)*16,calibration_seal_sha256=scale_digest));print('VALIDATED_PAIR',seed,m,flush=True)
            if seed==17:c._json_write(OUT/'pilot_completion.json',dict(status='PASS',seed=17,retained_in_cohort=True,criteria='numerical/provenance checks and resource feasibility only; no CE-contrast sign or magnitude selection',pairs=2,calibration_seal_sha256=scale_digest))
        readouts.sort(key=lambda z:(c.SEEDS.index(z['identity']['seed']),c.MODEL_TYPES.index(z['identity']['model_type'])));calibrations.sort(key=lambda z:(c.SEEDS.index(z['identity']['seed']),c.MODEL_TYPES.index(z['identity']['model_type'])))
        r.write_outputs(calibrations,readouts,scales);verified=r.verify_exports()
        c._json_write(OUT/'numerical_checks.json',dict(status='PASS',synthetic_checks=testrecord,calibration_checks=[dict(seed=z['identity']['seed'],model_type=z['identity']['model_type'],checks=z['checks']) for z in calibrations],validation_checks=[dict(seed=z['identity']['seed'],model_type=z['identity']['model_type'],checks=z['checks']) for z in readouts],calibration_seal_sha256=scale_digest,independent_export_checks=verified,pilot=c._json_read(OUT/'pilot_completion.json'),scales_not_refit=True))
        unavailable=sum(z['status']!='AVAILABLE' for z in scales)
        c._json_write(OUT/'compute_accounting.json',dict(host=socket.getfqdn(),wall_seconds=time.perf_counter()-started,calibration_new_forwards=fresh_calibration_forwards,validation_new_forwards=fresh_validation_forwards,profile_forwards=profile_forwards,cache_pairs_reused=reused,prior_unscaled_logical_cells_reused=80,new_scaled_logical_cells=80-unavailable*8,core_logical_cells=160,profile=c._json_read(OUT/'evaluation_profile.json'),peak_memory_bytes=e.torch.cuda.max_memory_allocated(),cache_bytes=sum(p.stat().st_size for p in (OUT/'cache').glob('*.json')),training_updates=0,optimizer_updates=0,new_masks=0,permanent_parameter_changes=0,attempt=str(attempt)))
        c._json_write(OUT/'completion.json',dict(status='COMPLETE' if not unavailable else 'COMPLETE_WITH_UNAVAILABLE_COMPARISONS',scientifically_complete=not unavailable,logical_cells=160,unavailable_scaled_pairs=unavailable,validated_pairs=10,seeds=list(c.SEEDS),specification_seal_sha256=spec,calibration_seal_sha256=scale_digest,independent_checks=verified))
        r.bundle();c._json_write(attempt/'completion.json',dict(status='COMPLETE',new_forwards=fresh_calibration_forwards+fresh_validation_forwards+profile_forwards))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        failure=dict(status='FAILED',scientifically_complete=False,error=repr(exc),traceback=traceback.format_exc());c._json_write(OUT/'attempts'/f'failure_{time.time_ns()}.json',failure);c._json_write(OUT/'completion.json',failure);raise
