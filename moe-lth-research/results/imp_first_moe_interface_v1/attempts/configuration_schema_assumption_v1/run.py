"""Bounded checkpoint evaluation. Never trains, discovers masks or saves model weights."""
from __future__ import annotations
import argparse,hashlib,json,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.imp_first_moe_interface_v1 import engine as e,analysis as a
c=e.c;OUT=a.OUT;OLD=ROOT/'results/imp_expert_contribution_ablation_v1'

def load_pair(seed,m,variant):
    states={};masks={};checkpoints={}
    for t,offset in zip(e.TIMES,(0,500)):
        state,mask,cid,digest=c._state_for(seed,m,offset,variant)
        source_path=c.ADAPTATION/'prefixes'/f'seed{seed}'/'reconstruction_interruption.pt' if offset==0 else Path(c._branch(seed,m)[0]['final_checkpoint'])
        payload=c._load_payload(source_path)
        c._need(payload.get('config',{}).get('model')==variant['config']['model'],f'Saved checkpoint model configuration mismatch: {source_path}')
        c._need(c._sha256(source_path)==digest,'Checkpoint source file changed during configuration verification')
        config_hash=hashlib.sha256(json.dumps(variant['config']['model'],sort_keys=True).encode()).hexdigest()
        states[t]=state;masks[t]=mask;checkpoints[t]=dict(checkpoint_id=cid,source_path=str(source_path),checkpoint_file_sha256=digest,state_sha256=c.state_dict_hash(state),mask_sha256=c._mask_hash(mask),model_configuration_sha256=config_hash,saved_configuration_verified=True)
    return states,masks,checkpoints

def seal():
    OUT.mkdir(parents=True,exist_ok=True)
    audit=c._json_read(OUT/'audit_completion.json')
    c._need(audit['status']=='PASS','Source audit not passed')
    c._need(c._sha256(OUT/'checkpoint_and_validation_provenance.json')==audit['provenance_sha256'],'Provenance changed')
    c._need(c._sha256(OUT/'artifact_inventory.csv')==audit['inventory_sha256'],'Inventory changed')
    config=dict(analysis_id=OUT.name,post_hoc_layer_selection=True,selected_layer=0,seeds=list(c.SEEDS),model_types=list(c.MODEL_TYPES),condition='router_experts_frozen',steps=dict(start=1000,end=1500),recipients=list(e.TIMES),donors=list(e.TIMES),alphas=list(e.ALPHAS),modes=list(e.MODES),logical_cells=400,primary_mode=e.MODES[0],modest_attenuation_alpha=.9,replication_unit='source seed; n=5',validation=dict(batches=31,target_tokens=492416,rebatching=False,precision='FP32',tf32=False,independent_test=False),tolerances=dict(ce_absolute=1e-6,logit_atol=1e-6,logit_rtol=1e-5),deduplication='within recipient/mode/batch alpha-zero donor identity; own native baseline output reused after first-batch hook equality; explicit logical coverage',cache_unit='complete seed/model pair; both checkpoint identities, every complete batch hash, donor, recipient, alpha, mode, spec and implementation',diagnostic_expert_chunk_size=1024,actual_model_chunking=False,limits=a.LIMITS,prior_absolute_loss_sha256=c._sha256(OLD/'absolute_losses.csv'),prior_batch_loss_sha256=c._sha256(OLD/'per_batch_losses.csv'),prior_temporal_contrast_sha256=c._sha256(OLD/'seed_level_temporal_contrasts.csv'))
    files=list((ROOT/'scripts/imp_first_moe_interface_v1').glob('*.py'))+[ROOT/'tests/test_imp_first_moe_interface_v1.py',Path(c.__file__),Path(e.replay.__file__)]+list((ROOT/'src/moe_lth/models').glob('*.py'))
    sealed=dict(configuration=config,source_sha256={str(p.relative_to(ROOT)):c._sha256(p) for p in sorted(files)},audit=audit)
    path=OUT/'seal.json'
    if path.exists():c._need(c._json_read(path)==sealed,'Seal mismatch: preserve old attempt; do not overwrite')
    else:
        c._json_write(OUT/'analysis_config.json',config);(OUT/'interface_definition.md').write_text(a.DEFINITION);c._json_write(path,sealed)
    return c._sha256(path)

def previous_checks(result):
    old=c._read_csv(OLD/'per_batch_losses.csv')
    ident=result['identity'];errors=[]
    for row in result['batches']:
        if row['alpha']==1. and row['recipient']!=row['donor']:continue
        if row['alpha'] not in (0.,1.):continue
        policy='BASELINE' if row['alpha']==1. else 'ZERO_L0';offset=0 if row['recipient']=='start' else 500;mode='BASELINE_REPLAY' if row['mode']==e.MODES[1] else 'NATIVE_RESPONSE'
        ref=next(z for z in old if (int(z['seed']),z['model_type'],int(z['offset']),z['mode'],z['intervention'],int(z['batch_index']))==(ident['seed'],ident['model_type'],offset,mode,policy,row['batch_index']))
        c._need(ref['batch_sha256']==row['batch_hash'] and int(ref['target_tokens'])==row['target_tokens'],'Previous batch/target identity differs')
        error=abs(float(ref['nll_sum'])-row['nll_sum'])/row['target_tokens'];c._need(error<=1e-6,'Previous native/ZERO_L0 per-batch CE mismatch');errors.append(error)
    return dict(status='PASS',maximum_per_batch_ce_error=max(errors),reference='previous ablation baseline and ZERO_L0, exact checkpoint/batch/mode')

def validate_cache(result,identity):
    c._need(result['identity']==identity and result['checks']['status']=='PASS','Cache identity or checks invalid')
    rs=result['batches'];expected={(r,d,alpha,p,bi) for r in e.TIMES for d in e.TIMES for alpha in e.ALPHAS for p in e.MODES for bi in range(31)}
    c._need(len(rs)==1240 and {(r['recipient'],r['donor'],r['alpha'],r['mode'],r['batch_index']) for r in rs}==expected,'Cache not scientifically complete')
    for r,d,alpha,p,_ in expected:
        rows=[z for z in rs if (z['recipient'],z['donor'],z['alpha'],z['mode'])==(r,d,alpha,p)]
        c._need(sum(z['target_tokens'] for z in rows)==492416,'Cache target coverage wrong')
    # Same donor/alpha is exactly the same contribution perturbation in each recipient.
    norms=[r for r in result['interface'] if r['kind']=='substitution']
    lookup={(r['recipient'],r['donor'],r['alpha'],r['mode'],r['batch_index']):r for r in norms}
    c._need(len(lookup)==1240,'Missing substitution norm coverage')
    for r,d,alpha,p,bi in expected:
        s=lookup['start',d,alpha,p,bi];end=lookup['end',d,alpha,p,bi]
        c._need(s['attenuation_displacement_sumsq']==end['attenuation_displacement_sumsq'] and s['donor_contribution_sumsq']==end['donor_contribution_sumsq'],'Fixed-donor perturbation differs by recipient')
    previous_checks(result)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evaluate',action='store_true');parser.add_argument('--seal-only',action='store_true');args=parser.parse_args()
    seal_hash=seal()
    if not args.evaluate:
        c._json_write(OUT/'completion.json',dict(status='SEALED',scientifically_complete=False));return
    tests=subprocess.run([sys.executable,'-m','pytest','tests/test_imp_first_moe_interface_v1.py','tests/test_imp_expert_contribution_ablation_v1.py','tests/test_model.py','-q'],cwd=ROOT,capture_output=True,text=True)
    c._json_write(OUT/'numerical_checks.json',dict(status='PRECOHORT_PASS' if tests.returncode==0 else 'FAIL',synthetic_checks=dict(command=tests.args,returncode=tests.returncode,output=tests.stdout+tests.stderr),cohort_evaluated=False))
    c._need(tests.returncode==0,'Numerical tests failed before cohort dispatch')
    try:admission=c.gpu_admission_check()
    except RuntimeError as exc:
        c._json_write(OUT/'completion.json',dict(status='RESOURCE_BLOCKED',scientifically_complete=False,reason=str(exc)));print(exc,flush=True);return
    admission['host']=socket.getfqdn();c._json_write(OUT/'gpu_admission.json',admission)
    variants=c.adaptation_variants();provenance=c._json_read(OUT/'checkpoint_and_validation_provenance.json');readouts=[];started=time.perf_counter();new_forwards=0;cache_reused=0;preflight_forwards=0;profile_forwards=0
    with c.exclusive_claim(OUT/'evaluation.lock'):
        c.gpu_admission_check();device=e.torch.device('cuda');e.torch.cuda.reset_peak_memory_stats()
        for seed in c.SEEDS:
            batches,vhash=c._validation_batches(seed,variants[seed]);batch_hashes=[c.common.batch_hash([b]) for b in batches]
            for m in c.MODEL_TYPES:
                states,masks,checkpoints=load_pair(seed,m,variants[seed])
                pair=next(r for r in provenance['first_interface_pair_verification'] if (r['seed'],r['model_type'])==(seed,m))
                for t,offset in zip(e.TIMES,(0,500)):c._need(pair['state_hashes'][str(offset)]==checkpoints[t]['state_sha256'],'Checkpoint differs from audited pair')
                identity=dict(seed=seed,model_type=m,checkpoints=checkpoints,validation_sha256=vhash,validation_batch_hashes=batch_hashes,seal_sha256=seal_hash,logical_grid=dict(recipients=e.TIMES,donors=e.TIMES,alphas=e.ALPHAS,modes=e.MODES))
                # Normalize tuples to JSON lists so cache identities compare exactly on resume.
                identity=json.loads(json.dumps(identity,sort_keys=True));key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();path=OUT/'cache'/f'{key}.json'
                if path.exists():
                    result=c._json_read(path);validate_cache(result,identity);cache_reused+=1
                else:
                    if not (OUT/'precohort_checks.json').exists():
                        pre=e.evaluate_pair(states,masks,variants[seed]['config']['model'],batches,identity,device,preflight=True);prior=previous_checks(pre);preflight_forwards+=pre['unique_forwards'];c._json_write(OUT/'precohort_checks.json',dict(identity=identity,checks=pre['checks'],prior=prior,full_batches=31,target_tokens=492416,unique_forwards=pre['unique_forwards']))
                        del pre
                    if not (OUT/'evaluation_profile.json').exists():
                        tick=time.perf_counter();pilot=e.evaluate_pair(states,masks,variants[seed]['config']['model'],batches[:1],identity,device);seconds=time.perf_counter()-tick;profile_forwards+=pilot['unique_forwards'];size=len(json.dumps(pilot).encode());c._json_write(OUT/'evaluation_profile.json',dict(status='PASS',one_complete_batch_seconds=seconds,projected_evaluation_seconds=seconds*31*10,peak_memory_bytes=e.torch.cuda.max_memory_allocated(),projected_cache_storage_bytes=size*31*10,profile_unique_forwards=pilot['unique_forwards'],logical_cells=400));del pilot
                    result=e.evaluate_pair(states,masks,variants[seed]['config']['model'],batches,identity,device);result['checks']['previous_readout_reproduction']=previous_checks(result);validate_cache(result,identity);new_forwards+=result['unique_forwards'];c._json_write(path,result)
                readouts.append(result);c._json_write(OUT/'progress.json',dict(validated_pairs=len(readouts),expected_pairs=10,logical_cells=len(readouts)*40));print('VALIDATED_PAIR',seed,m,flush=True)
        verified=a.write_outputs(readouts)
        c._json_write(OUT/'numerical_checks.json',dict(status='PASS',synthetic_checks=dict(command=tests.args,returncode=tests.returncode,output=tests.stdout+tests.stderr),precohort=c._json_read(OUT/'precohort_checks.json'),cohort_checks=[dict(seed=r['identity']['seed'],model_type=r['identity']['model_type'],checks=r['checks']) for r in readouts],independent_exported_validation=verified,FP32=True,TF32=False))
        c._json_write(OUT/'compute_accounting.json',dict(host=socket.getfqdn(),evaluation_wall_seconds=time.perf_counter()-started,profile=c._json_read(OUT/'evaluation_profile.json'),new_cohort_forwards=new_forwards,preflight_forwards=preflight_forwards,profile_forwards=profile_forwards,cache_pairs_reused=cache_reused,unique_validated_pairs=len(readouts),logical_cells=400,alpha_zero_logical_aliases=40,model_checkpoint_count=20,training_updates=0,optimizer_updates=0,new_masks=0,permanent_parameter_changes=0,peak_memory_bytes=e.torch.cuda.max_memory_allocated(),cache_bytes=sum(p.stat().st_size for p in (OUT/'cache').glob('*.json'))))
        c._json_write(OUT/'completion.json',dict(status='COMPLETE',scientifically_complete=True,seeds=list(c.SEEDS),validated_pairs=10,logical_cells=400,per_batch_logical_records=12400,independent_identity_checks=verified,seal_sha256=seal_hash))
        a.bundle()
if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback
        failure=dict(status='FAILED',scientifically_complete=False,error=repr(exc),traceback=traceback.format_exc());c._json_write(OUT/'attempts'/f'failure_{time.time_ns()}.json',failure);c._json_write(OUT/'completion.json',failure);raise
