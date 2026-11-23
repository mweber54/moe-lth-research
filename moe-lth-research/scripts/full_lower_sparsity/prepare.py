"""Prepare the exact 5 x 5 x 11 x 4 learned-mask extension; never retrain sources."""
import sys, math, fcntl
from support import *
sys.path.insert(0,str(ROOT/'scripts/nominal_exposure_dataset'))
import pilot_common as P
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
ES=(8,12,16,24,32); SEEDS=(7,17,29,43,61); AGES=tuple(range(0,101,10)); SP=(.5,.55,.6,.65)
def cid(e,s,a,sp): return f'A_e{e}_seed{s}_R{a}_sparse_learned_s{sp:.2f}'
def grid():
    return [dict(condition_id=cid(e,s,a,sp),num_experts=e,seed=s,router_age=a,
        router_checkpoint_step=25*a,source_attempts=2500,recovery_attempts=2500,
        arm='A',job_type='sparse',mask_type='learned',sparsity=sp,
        source_id=f'source_e{e}_seed{s}_T2500',dense_condition_id=f'dense_e{e}_seed{s}_R{a}',
        stage=1,status='PENDING',artifact_path='',artifact_sha256='',block_reason='',execution_host='')
        for e in ES for s in SEEDS for a in AGES for sp in SP]
def prepare():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'prepare.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'extension_seal.json').exists(): validate_seal(); print('EXISTING PREPARATION VERIFIED'); return
    if (OUT/'manifests/run_manifest.csv').exists(): raise RuntimeError('Unsealed manifest: inspect interrupted preparation')
    P.torch.set_num_threads(2)
    ad=ROOT/'results/final_dataset_a'
    cpfile=ad/'inputs/source_checkpoint_manifest.csv'; dfile=ad/'canonical/dataset_a_dense.csv'
    cps=rows(cpfile); dense=rows(dfile)
    cpmap={(int(r['num_experts']),int(r['seed']),int(r['router_age'])):r for r in cps}
    dm={(int(r['architecture']),int(r['seed']),int(r['router_age'])):r for r in dense}
    keys={(e,s,a) for e in ES for s in SEEDS for a in AGES}
    assert len(cps)==len(cpmap)==275 and set(cpmap)==keys
    assert len(dense)==len(dm)==275 and set(dm)==keys
    inputs={str(cpfile):sha(cpfile),str(dfile):sha(dfile)}
    missing=[]
    for key,r in cpmap.items():
        p=Path(r['path'])
        if not p.is_file(): missing.append(str(p)); continue
        if sha(p)!=r['sha256']: raise RuntimeError('Checkpoint hash mismatch: '+str(p))
        inputs[str(p)]=r['sha256']
    if missing:
        save(OUT/'missing_checkpoints.json',missing)
        raise RuntimeError(f'{len(missing)} required checkpoints missing; see missing_checkpoints.json. No training launched.')
    old=ROOT/'results/dataset_b_lower_sparsity'
    oldmanifest=old/'manifests/run_manifest.csv'; oldsources=old/'runtime_sources.json'
    prior={r['condition_id']:r for r in rows(oldmanifest) if r['job_type']=='sparse' and r['mask_type']=='learned'}
    assert len(prior)==60 and all(r['status']=='COMPLETE' for r in prior.values())
    oldsrc=read(oldsources)
    verify_map(ROOT,read(old/'extension_seal.json')['code'])
    inputs[str(oldmanifest)]=sha(oldmanifest); inputs[str(oldsources)]=sha(oldsources)
    sources={}
    for e in ES:
      for seed in SEEDS:
        sid=f'source_e{e}_seed{seed}_T2500'
        initial=P.A.load(cpmap[e,seed,0]['path']); cfg=initial['config']
        assert not P.A.protocol_errors(cfg,e,seed)
        assert int(cfg['model']['num_experts'])==e
        cp={str(age):{'path':cpmap[e,seed,age]['path'],'sha256':cpmap[e,seed,age]['sha256']} for age in AGES}
        with P.preserve_rng():
            model=TinyMoELanguageModel(cfg['model'])
            final=P.A.load(cpmap[e,seed,100]['path']); assert final['step']==2500
            model.load_state_dict(final['model']); del final,initial
            masks={}
            for sp in SP:
                m=expert_local_magnitude_masks(model,sp)
                h=P.state_dict_hash(m)
                if e==32:
                    for age in (0,50,100):
                        assert oldsrc[sid]['checkpoints'][str(age)]['sha256']==cp[str(age)]['sha256'], '32E source identity differs'
                    record=oldsrc[sid]['masks'][str(sp)]
                    assert h==record['learned_state_hash'], 'Existing 32E mask differs'
                    p=Path(record['learned_path']); assert sha(p)==record['learned_file_sha256']
                else:
                    p=OUT/'masks'/f'{sid}_s{sp:.2f}.pt'; p.parent.mkdir(exist_ok=True)
                    if p.exists():
                        assert P.state_dict_hash(P.torch.load(p,map_location='cpu',weights_only=True))==h
                    else:
                        tmp=p.with_suffix('.tmp'); P.torch.save(m,tmp); tmp.replace(p)
                masks[str(sp)]={'learned_path':str(p),'learned_state_hash':h,'learned_file_sha256':sha(p)}
                inputs[str(p)]=sha(p)
            del model
        for k in ('train_path','validation_path'):
            path=Path(cfg['data'][k]); path=path if path.is_absolute() else ROOT/path
            inputs[str(path)]=sha(path)
        sources[sid]={'config':cfg,'checkpoints':cp,'masks':masks}
        print('SOURCE AND MASKS VERIFIED',sid,flush=True)
    records=grid(); references=[]
    for key,r in dm.items():
        e,seed,age=key; loss=float(r['final']); assert math.isfinite(loss)
        # Exported, finalized Dataset A endpoint: not a fabricated checkpoint.
        dest=OUT/'dense_table_references'/r['condition_id']
        save(dest/'summary.json',{'final_validation_loss':loss})
        save(dest/'provenance.json',{'reference_type':'finalized_dataset_a_table_endpoint',
            'table_sha256':sha(dfile),'original_record':r,
            'note':'Original dense checkpoint need not be present; no new dense training.'})
        inputs[str(dest/'summary.json')]=sha(dest/'summary.json')
        inputs[str(dest/'provenance.json')]=sha(dest/'provenance.json')
        references.append(dict(condition_id=r['condition_id'],num_experts=e,seed=seed,router_age=age,
            job_type='dense',status='REUSED',artifact_path=str(dest),artifact_sha256='',stage=0,
            block_reason='',execution_host=''))
    for r in records:
        if r['condition_id'] not in prior: continue
        oldrow=prior[r['condition_id']]; dest=Path(oldrow['artifact_path'])
        done=read(dest/'pilot_completed.json'); meta=read(dest/'metadata.json'); result=read(dest/'validated_result.json')
        assert done['condition_id']==r['condition_id']
        assert sha(dest/'checkpoints/final_recovered.pt')==done['final_checkpoint_sha256']==oldrow['artifact_sha256']
        expected=sources[r['source_id']]['masks'][str(r['sparsity'])]['learned_state_hash']
        assert result['mask_hash']==expected
        assert meta['seed']==r['seed'] and meta['router_step']==r['router_checkpoint_step']
        assert meta['recovery_steps']==2500 and meta['router_mode']=='trainable'
        assert abs(result['dense_reference_loss']-float(dm[r['num_experts'],r['seed'],r['router_age']]['final']))<1e-10
        assert math.isfinite(result['final_validation_loss'])
        for name in ('pilot_completed.json','metadata.json','validated_result.json','checkpoints/final_recovered.pt'):
            inputs[str(dest/name)]=sha(dest/name)
        r.update(status='REUSED',artifact_path=str(dest),artifact_sha256=done['final_checkpoint_sha256'])
    assert sum(r['status']=='REUSED' for r in records)==60
    assert sum(r['status']=='PENDING' for r in records)==1040
    inputs[str(P.A.OUT/'canonical_data_hashes.json')]=sha(P.A.OUT/'canonical_data_hashes.json')
    for name in ('manifests','logs','recoveries'): (OUT/name).mkdir(exist_ok=True)
    save(OUT/'runtime_sources.json',sources)
    P.csv(OUT/'manifests/runtime_initial.csv',records+references)
    P.csv(OUT/'manifests/run_manifest.csv',records+references)
    code={str(p.relative_to(ROOT)):sha(p) for folder in (Path(__file__).parent,ROOT/'scripts/nominal_exposure_dataset',ROOT/'src/moe_lth') for p in folder.rglob('*.py')}
    code['scripts/fixed_budget_dataset/common.py']=sha(ROOT/'scripts/fixed_budget_dataset/common.py')
    save(OUT/'extension_seal.json',{'code':code,'inputs':inputs,'immutable':{
        'runtime_sources.json':sha(OUT/'runtime_sources.json'),
        'manifests/runtime_initial.csv':sha(OUT/'manifests/runtime_initial.csv')}})
    print('READY: 1100 learned conditions; 60 reused; 1040 pending; 275 dense table references',flush=True)
if __name__=='__main__': prepare()
