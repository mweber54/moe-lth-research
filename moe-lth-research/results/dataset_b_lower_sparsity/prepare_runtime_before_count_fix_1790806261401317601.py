"""Called under the parent's exclusive lock, after all 225 parent jobs finish."""
import sys, math
from pathlib import Path
from support import *
sys.path.insert(0,str(ROOT/'scripts/nominal_exposure_dataset'))
import pilot_common as P
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks

def prepare():
    P.torch.set_num_threads(2)
    parent=rows(PARENT/'manifests/run_manifest.csv')
    if len(parent)!=225 or any(r['status'] in ('RUNNING','FAILED_UNCERTAIN') for r in parent):
        raise RuntimeError('Parent has unresolved workers or an invalid manifest')
    if (OUT/'extension_seal.json').exists():
        validate_seal(); return
    if (OUT/'manifests/run_manifest.csv').exists():
        raise RuntimeError('Unsealed runtime manifest exists; inspect interrupted preparation')
    verify_map(OUT,read(OUT/'preparation_hashes.json'))
    checks=read(PARENT/'audits/preflight.json')
    verify_map(ROOT,checks['code_hashes'])
    plan=rows(OUT/'manifests/job_plan.csv')
    if len(plan)!=320 or {r['condition_id'] for r in plan}!=expected_ids():
        raise RuntimeError('Unexpected extension grid')
    plan=[r for r in plan if r['arm']=='A']
    assert len(plan)==80
    assert sum(r['mask_type']=='learned' for r in plan)==60
    assert sum(r['mask_type']=='random' for r in plan)==20
    assert all(int(r['source_attempts'])==2500 and int(r['recovery_attempts'])==2500 for r in plan)
    lookup={r['condition_id']:r for r in parent}
    assert all(lookup[r['dense_condition_id']]['status'] in ('COMPLETE','REUSED') for r in plan)

    sources=read(OUT/'manifests/source_inputs.json')
    cache=OUT/'runtime_sources'; cache.mkdir(exist_ok=True)
    for seed in ():  # Arm A uses existing short sources only
        sid=f'source_e32_seed{seed}_T10000'
        record=lookup[sid]; dest=Path(record['artifact_path'])
        done=read(dest/'pilot_completed.json')
        if done['condition_id']!=sid or sha(dest/'pilot_source_identity.json')!=done['source_identity_sha256']:
            raise RuntimeError('Invalid source completion: '+sid)
        info=read(dest/'pilot_source_identity.json')
        if info['checkpoints']['100']['sha256']!=record['artifact_sha256']:
            raise RuntimeError('Source manifest digest mismatch')
        for cp in info['checkpoints'].values():
            if sha(Path(cp['path']))!=cp['sha256']: raise RuntimeError('Source checkpoint changed')
        cached=cache/(sid+'.json')
        if cached.exists():
            entry=read(cached)
            if entry['source_final_sha256']!=info['checkpoints']['100']['sha256']:
                raise RuntimeError('Cached mask source changed')
        elif sid in sources:
            entry=sources[sid]
        else:
            with P.preserve_rng():
                model=TinyMoELanguageModel(info['config']['model'])
                payload=P.A.load(info['checkpoints']['100']['path'])
                assert payload['step']==10000
                model.load_state_dict(payload['model']); del payload
                masks={}
                for sp in (.50,.55,.60,.65):
                    learned=expert_local_magnitude_masks(model,sp)
                    random=P.random_mask(learned,sid,sp)
                    assert P.check_masks(learned,random)
                    item={'random_seed':P.mask_seed(sid,sp)}
                    for kind,mask in [('learned',learned),('random',random)]:
                        path=OUT/'masks'/f'{sid}_s{sp:.2f}_{kind}.pt'
                        if path.exists():
                            existing=P.torch.load(path,map_location='cpu',weights_only=True)
                            assert P.state_dict_hash(existing)==P.state_dict_hash(mask)
                        else:
                            temp=path.with_suffix('.tmp'); P.torch.save(mask,temp); temp.replace(path)
                        item.update({kind+'_path':str(path),kind+'_state_hash':P.state_dict_hash(mask),kind+'_file_sha256':sha(path)})
                    masks[str(sp)]=item
                del model
            entry={'config':info['config'],'checkpoints':info['checkpoints'],'masks':masks,
                   'source_final_sha256':info['checkpoints']['100']['sha256']}
            save(cached,entry)
        sources[sid]=entry
        print('LONG SOURCE MASKS READY',sid,flush=True)
    sources={sid:info for sid,info in sources.items() if sid.endswith('_T2500')}
    assert len(sources)==5
    inputs={}
    for sid,info in sources.items():
        for cp in info['checkpoints'].values(): inputs[cp['path']]=cp['sha256']
        for item in info['masks'].values():
            learned=P.torch.load(item['learned_path'],map_location='cpu',weights_only=True)
            random=P.torch.load(item['random_path'],map_location='cpu',weights_only=True)
            assert P.check_masks(learned,random)
            for kind,mask in [('learned',learned),('random',random)]:
                assert P.state_dict_hash(mask)==item[kind+'_state_hash']
                inputs[item[kind+'_path']]=item[kind+'_file_sha256']
        for key in ('train_path','validation_path'):
            path=Path(info['config']['data'][key]); path=path if path.is_absolute() else ROOT/path
            inputs[str(path)]=sha(path)
    canonical=P.A.OUT/'canonical_data_hashes.json'; inputs[str(canonical)]=sha(canonical)
    dense=[]
    for cid in sorted({r['dense_condition_id'] for r in plan}):
        r=lookup[cid].copy(); path=Path(r['artifact_path'])
        inputs[str(path/'checkpoints/final_recovered.pt')]=r['artifact_sha256']
        for name in ('summary.json','metadata.json'):
            inputs[str(path/name)]=sha(path/name)
        summary=read(path/'summary.json'); meta=read(path/'metadata.json')
        assert math.isfinite(summary['final_validation_loss'])
        assert int(meta['recovery_steps'])==int(float(r['recovery_attempts']))
        assert int(meta['seed'])==int(float(r['seed']))
        assert int(meta['router_step'])==int(float(r['router_checkpoint_step']))
        r.update(status='REUSED',job_type='dense',stage=0,block_reason='')
        dense.append(r)
    assert len(dense)==15
    for r in plan:
        r.update(status='PENDING',stage='1',artifact_path='',artifact_sha256='',block_reason='',execution_host='')
    verify_map(Path('/'),inputs)
    save(OUT/'runtime_sources.json',sources)
    for name in ('logs','recoveries'): (OUT/name).mkdir(exist_ok=True)
    P.csv(OUT/'manifests/runtime_initial.csv',plan+dense)
    P.csv(OUT/'manifests/run_manifest.csv',plan+dense)
    code={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}
    code.update(checks['code_hashes'])
    immutable={'runtime_sources.json':sha(OUT/'runtime_sources.json'),
               'manifests/runtime_initial.csv':sha(OUT/'manifests/runtime_initial.csv')}
    immutable.update(read(OUT/'preparation_hashes.json'))
    save(OUT/'extension_seal.json',{'code':code,'immutable':immutable,'inputs':inputs})
    print('RUNTIME READY: 80 Arm A sparse jobs; 15 reused dense references',flush=True)
if __name__=='__main__':
    raise SystemExit('Use launch.py so the parent lock is held.')
