"""Read-only source audit. Does not call a model forward or any training API."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.imp_capacity_diagnostic_v1 import run as c
OUT=ROOT/'results/imp_first_moe_interface_v1'
PREVIOUS=ROOT/'results/imp_expert_contribution_ablation_v1'
def main():
    OUT.mkdir(parents=True,exist_ok=True);c.OUT=OUT
    provenance,inventory,variants=c.audit_sources()
    prior=c._json_read(PREVIOUS/'source_configuration_seal.json')
    for name,digest in prior['source_hashes'].items():c._need(c._sha256(ROOT/name)==digest,f'Prior sealed source changed: {name}')
    c._need(c._sha256(PREVIOUS/'checkpoint_and_validation_provenance.json')==prior['provenance_sha256'],'Prior provenance seal changed')
    for d in ('imp_capacity_diagnostic_v1','imp_expert_contribution_ablation_v1'):
        c._need(c._json_read(ROOT/'results'/d/'completion.json')['scientifically_complete'],f'Prior incomplete: {d}')
    inventory.extend(c._source_row(PREVIOUS/n,'previous_ablation_'+n) for n in ['source_configuration_seal.json','analysis_config.json','completion.json','numerical_checks.json','absolute_losses.csv','per_batch_losses.csv','seed_level_temporal_contrasts.csv','shared_parameter_verification.json'])
    pairs=[]
    for seed in c.SEEDS:
        for m in c.MODEL_TYPES:
            states={t:c._state_for(seed,m,t,variants[seed])[0] for t in (0,500)}
            groups={}
            for group in ('router','experts','shared'):
                names=[n for n in states[0] if c.common.group_for_parameter(n)==group]
                changed=[n for n in names if not c.torch.equal(states[0][n],states[500][n])]
                c._need(bool(changed) if group=='shared' else not changed,f'Unexpected pair change: {seed}/{m}/{group}')
                groups[group]=dict(start_hash=c._group_hash(states[0],group),end_hash=c._group_hash(states[500],group),changed_names=changed,bias_count=sum(n.endswith('bias') for n in names))
            pairs.append(dict(seed=seed,model_type=m,groups=groups,state_hashes={str(t):c.state_dict_hash(states[t]) for t in (0,500)}))
            print(seed,m,'checkpoint pair verified',flush=True)
    provenance['first_interface_pair_verification']=pairs
    provenance['previous_ablation_reference']=str(PREVIOUS)
    c._json_write(OUT/'checkpoint_and_validation_provenance.json',provenance)
    c._csv_write(OUT/'artifact_inventory.csv',inventory)
    c._json_write(OUT/'audit_completion.json',dict(status='PASS',provenance_sha256=c._sha256(OUT/'checkpoint_and_validation_provenance.json'),inventory_sha256=c._sha256(OUT/'artifact_inventory.csv'),audit_source_sha256=c._sha256(Path(__file__))))
if __name__=='__main__':main()
