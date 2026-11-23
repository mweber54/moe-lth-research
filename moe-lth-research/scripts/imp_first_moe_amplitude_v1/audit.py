"""Audit exact interface states and the already sealed training-only probe; no forwards."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.imp_first_moe_interface_v1 import run as prior
c=prior.c;OUT=ROOT/'results/imp_first_moe_amplitude_v1';OLD=ROOT/'results/imp_first_moe_interface_v1'
from diagnostics import probe_identity,probe_indices

def check_seal(path,key):
    sealed=c._json_read(path)
    for name,digest in sealed[key].items():c._need(c._sha256(ROOT/name)==digest,f'Sealed source changed: {name}')
    return sealed

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    seal=check_seal(OLD/'seal.json','source_sha256');p=c._json_read(OLD/'checkpoint_and_validation_provenance.json')
    c._need(c._sha256(OLD/'checkpoint_and_validation_provenance.json')==seal['audit']['provenance_sha256'],'Interface provenance mismatch')
    c._need(c._json_read(OLD/'completion.json')['scientifically_complete'],'Interface incomplete')
    c._need(c._json_read(OLD/'numerical_checks.json')['status']=='PASS','Interface checks failed')
    check_seal(ROOT/'results/imp_expert_contribution_ablation_v1/source_configuration_seal.json','source_hashes')
    amendment=c._json_read(OLD/'aggregate_metadata_amendment_seal.json');c._need(c._sha256(ROOT/amendment['source'])==amendment['sha256'],'Prior reporting amendment changed')
    inventory=[]
    for path in [OLD/n for n in ('seal.json','completion.json','numerical_checks.json','absolute_losses.csv','per_batch_losses.csv','interface_statistics.csv','checkpoint_and_validation_provenance.json','aggregate_metadata_amendment_seal.json')]+[c.ADAPTATION/n for n in ('registration.json','registration_revision_6.json','input_code_configuration_hashes.json','diagnostic_probe_manifest.json')]:inventory.append(c._source_row(path,'verified_reference'))
    manifest=c._json_read(c.ADAPTATION/'diagnostic_probe_manifest.json');variants=c.adaptation_variants();caches={}
    for path in sorted((OLD/'cache').glob('*.json')):
        result=c._json_read(path);prior.validate_cache(result,result['identity']);ident=result['identity'];key=(ident['seed'],ident['model_type']);c._need(key not in caches,'Duplicate prior pair cache')
        caches[key]=dict(path=str(path),sha256=c._sha256(path),identity=ident)
    c._need(set(caches)=={(s,m) for s in c.SEEDS for m in c.MODEL_TYPES},'Prior cache cohort missing')
    pairs=[];probes=[]
    for seed in c.SEEDS:
        variant=variants[seed]
        for m in c.MODEL_TYPES:
            states,masks,checkpoints=prior.load_pair(seed,m,variant)
            c._need(checkpoints==caches[seed,m]['identity']['checkpoints'],'Current checkpoints differ from interface cache')
            groups={}
            for group in ('router','experts','shared'):
                names=[n for n in states['start'] if c.common.group_for_parameter(n)==group]
                changed=[n for n in names if not c.torch.equal(states['start'][n],states['end'][n])]
                c._need(bool(changed) if group=='shared' else not changed,f'Unexpected frozen/shared change: {seed}/{m}/{group}')
                groups[group]=dict(start_sha256=c._group_hash(states['start'],group),end_sha256=c._group_hash(states['end'],group),bias_tensors=sum(n.endswith('bias') for n in names),changed_names=changed)
            for t in ('start','end'):
                inventory.append(c._source_row(Path(checkpoints[t]['source_path']),'verified_checkpoint',checkpoints[t]['checkpoint_file_sha256']))
            inventory.append(c._source_row(Path(variant['mask']),'verified_registered_mask'))
            pairs.append(dict(seed=seed,model_type=m,checkpoints=checkpoints,groups=groups,prior_cache=caches[seed,m]))
            del states,masks
            print(seed,m,'exact interface pair verified',flush=True)
        rng=c._rng_snapshot()
        try:train,validation=c.common.materialize_data(variant,2500)
        finally:c._restore_rng(rng)
        registered=next(r for r in manifest['seeds'] if int(r['seed'])==seed)
        indices=probe_indices(seed,2500);c._need(indices==registered['batch_indices'],'Probe selection differs')
        selected=[train[i] for i in indices];c._need(probe_identity(seed,indices,selected)==registered['probe_identity_sha256'],'Probe identity differs')
        for row,(x,y) in zip(registered['batch_hashes'],selected):
            digest=hashlib.sha256()
            for tensor in (x,y):
                v=tensor.detach().cpu().contiguous();digest.update(str(tuple(v.shape)).encode());digest.update(str(v.dtype).encode());digest.update(v.numpy().tobytes())
            c._need(digest.hexdigest()==row['sha256'],'Training-only probe batch hash differs')
            c._need(x.shape==train[0][0].shape,'Calibration probe is not a complete original batch')
        path=OUT/'probe_data'/f'seed{seed}.pt';path.parent.mkdir(exist_ok=True);tmp=path.with_suffix('.tmp');c.torch.save(selected,tmp);tmp.replace(path)
        vhash=c.common.batch_hash(validation);c._need(vhash==caches[seed,'sparse']['identity']['validation_sha256'],'Validation population differs')
        c._need([c.common.batch_hash([b]) for b in validation]==caches[seed,'sparse']['identity']['validation_batch_hashes'],'Validation boundaries/order differ')
        probes.append(dict(seed=seed,registered=registered,materialized_probe_path=str(path),materialized_probe_sha256=c._sha256(path),complete_batch_hashes=[c.common.batch_hash([b]) for b in selected],probe_target_tokens=sum(y.numel() for _,y in selected),validation_batches=len(validation),validation_target_tokens=sum(y.numel() for _,y in validation),validation_batch_hashes=[c.common.batch_hash([b]) for b in validation],validation_sha256=vhash,previously_used_in_training_and_diagnostics=True,independent_test_set=False))
        inventory.append(c._source_row(path,'verified_training_probe_copy'))
        del train,validation,selected
        print(seed,'sealed training probe and validation verified',flush=True)
    result=dict(status='PASS',prior_interface_seal_sha256=c._sha256(OLD/'seal.json'),prior_provenance_sha256=c._sha256(OLD/'checkpoint_and_validation_provenance.json'),inherited_common_parent_provenance=p['seeds'],pairs=pairs,probes=probes,scope='Freshly verify all used checkpoints, masks, configs, frozen groups, prior caches and probe/data identities; retain inherited verified transition provenance without rereading unrelated historical trajectories.')
    c._json_write(OUT/'checkpoint_and_probe_hashes.json',result);c._csv_write(OUT/'artifact_inventory.csv',inventory)
    c._json_write(OUT/'audit_completion.json',dict(status='PASS',provenance_sha256=c._sha256(OUT/'checkpoint_and_probe_hashes.json'),inventory_sha256=c._sha256(OUT/'artifact_inventory.csv'),source_sha256=c._sha256(Path(__file__))))
if __name__=='__main__':main()
