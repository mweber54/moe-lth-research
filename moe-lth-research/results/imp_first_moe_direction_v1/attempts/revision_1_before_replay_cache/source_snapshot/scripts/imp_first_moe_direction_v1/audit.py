"""Read-only verification of the exact interface/amplitude states and sealed probes."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.imp_first_moe_amplitude_v1 import audit as amplitude_audit
from scripts.imp_first_moe_interface_v1 import run as prior
c=prior.c;AMP=ROOT/'results/imp_first_moe_amplitude_v1';OLD=ROOT/'results/imp_first_moe_interface_v1';OUT=ROOT/'results/imp_first_moe_direction_v1'
def main():
    OUT.mkdir(exist_ok=True)
    spec=amplitude_audit.check_seal(AMP/'specification_seal.json','source_hashes');source=c._json_read(AMP/'checkpoint_and_probe_hashes.json')
    c._need(c._sha256(AMP/'checkpoint_and_probe_hashes.json')==spec['audit']['provenance_sha256'],'Amplitude provenance changed')
    c._need(c._json_read(AMP/'completion.json')['scientifically_complete'],'Amplitude cohort incomplete')
    c._need(c._json_read(AMP/'numerical_checks.json')['status']=='PASS','Amplitude numerical checks failed')
    amplitude_audit.check_seal(OLD/'seal.json','source_sha256');amplitude_audit.check_seal(ROOT/'results/imp_expert_contribution_ablation_v1/source_configuration_seal.json','source_hashes')
    oldseal=c._json_read(OLD/'seal.json');c._need(c._sha256(OLD/'checkpoint_and_validation_provenance.json')==oldseal['audit']['provenance_sha256'],'Interface provenance changed')
    inventory=[]
    for path in [AMP/n for n in ('specification_seal.json','calibration_seal.json','completion.json','numerical_checks.json','checkpoint_and_probe_hashes.json','calibration_scales.csv')]+[OLD/n for n in ('seal.json','completion.json','numerical_checks.json','absolute_losses.csv','per_batch_losses.csv')]+[c.ADAPTATION/n for n in ('registration.json','registration_revision_6.json','diagnostic_probe_manifest.json')]:inventory.append(c._source_row(path,'verified_reference'))
    variants=c.adaptation_variants();pairs=[];probes=[]
    for seed in c.SEEDS:
        for m in c.MODEL_TYPES:
            original=next(p for p in source['pairs'] if (p['seed'],p['model_type'])==(seed,m));states,masks,checkpoints=prior.load_pair(seed,m,variants[seed]);c._need(checkpoints==original['checkpoints'],'Checkpoint differs from prior studies')
            groups={}
            for group in ('router','experts','shared'):
                names=[n for n in states['start'] if c.common.group_for_parameter(n)==group];changed=[n for n in names if not c.torch.equal(states['start'][n],states['end'][n])]
                c._need(bool(changed) if group=='shared' else not changed,f'Unexpected shared/frozen group changes: {seed}/{m}/{group}')
                groups[group]=dict(start_sha256=c._group_hash(states['start'],group),end_sha256=c._group_hash(states['end'],group),bias_count=sum(n.endswith('bias') for n in names),changed_names=changed)
            for t,offset in zip(('start','end'),(0,500)):
                path=Path(checkpoints[t]['source_path']);payload=c._load_payload(path);cfg=payload['identity']['config'] if offset==0 else payload['config']
                c._need(cfg['data']==variants[seed]['config']['data'],'Saved data config mismatch')
                inventory.append(c._source_row(path,'verified_checkpoint',checkpoints[t]['checkpoint_file_sha256']))
            mask_path=Path(variants[seed]['mask']);oldmask=next(z for z in c._read_csv(AMP/'artifact_inventory.csv') if z['path']==str(mask_path) and z['role']=='verified_registered_mask');inventory.append(c._source_row(mask_path,'verified_mask',oldmask['sha256']))
            cache=original['prior_cache'];c._need(c._sha256(Path(cache['path']))==cache['sha256'],'Prior interface cache changed');result=c._json_read(Path(cache['path']));prior.validate_cache(result,cache['identity'])
            pairs.append(dict(original,groups=groups));del states,masks,payload,result
            print(seed,m,'exact shared-only pair verified',flush=True)
        probe=next(p for p in source['probes'] if p['seed']==seed);path=Path(probe['materialized_probe_path']);c._need(c._sha256(path)==probe['materialized_probe_sha256'],'Training probe changed')
        batches=c.torch.load(path,map_location='cpu',weights_only=True);c._need([c.common.batch_hash([b]) for b in batches]==probe['complete_batch_hashes'],'Probe coordinates differ')
        validation,vhash=c._validation_batches(seed,variants[seed]);c._need(vhash==probe['validation_sha256'] and [c.common.batch_hash([b]) for b in validation]==probe['validation_batch_hashes'],'Validation identities differ')
        c._need(len(validation)==probe['validation_batches'] and sum(y.numel() for _,y in validation)==probe['validation_target_tokens'],'Validation counts differ')
        probes.append(probe);inventory.append(c._source_row(path,'verified_training_probe',probe['materialized_probe_sha256']));del batches,validation
    c._json_write(OUT/'checkpoint_and_probe_hashes.json',dict(status='PASS',pairs=pairs,probes=probes,inherited_common_parent_provenance=source['inherited_common_parent_provenance'],prior_amplitude_spec_sha256=c._sha256(AMP/'specification_seal.json'),prior_interface_spec_sha256=c._sha256(OLD/'seal.json'),data_and_model_configuration_verified=True,source_scope='Exact already verified post-pruning step1000 and step1500 shared-only states; no trajectory substitution or reconstruction'))
    c._csv_write(OUT/'artifact_inventory.csv',inventory)
    c._json_write(OUT/'audit_completion.json',dict(status='PASS',source_sha256=c._sha256(Path(__file__)),provenance_sha256=c._sha256(OUT/'checkpoint_and_probe_hashes.json'),inventory_sha256=c._sha256(OUT/'artifact_inventory.csv')))
if __name__=='__main__':main()
