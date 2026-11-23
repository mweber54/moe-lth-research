"""Read-only consistency verification of completed Stage 1 outputs."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch
spec=importlib.util.spec_from_file_location('role_study',Path(__file__).with_name('study.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
D=s.D


def main():
    seal=s.read(D/'output_seal.json')
    for p,h in seal['files'].items():assert s.sha(D/p)==h,p
    s.verify_inputs()
    damage=pd.read_csv(D/'deletion_damage.csv');roles=pd.read_csv(D/'routing_role_metrics.csv');forecasts=pd.read_csv(D/'prospective_forecasts.csv');unions=pd.read_csv(D/'combined_deletion_checks.csv');manifest=pd.read_csv(D/'group_manifest.csv');groups=s.load(D/'group_coordinates.pt')
    assert len(damage)==2*4*2*32*3 and len(roles)==2*4*2*4*2
    assert len(manifest)==32 and len(groups)==32
    masks=s.load_masks(s.jobs()['sparse']['mask'])
    for e in s.EXPERTS:
        gg=[g for g in groups.values() if g['expert']==e];idx=torch.cat([g['indices'] for g in gg])
        assert len(torch.unique(idx))==len(idx)
        assert masks[s.name(e)].flatten()[idx].all()
        for g in gg:assert len(g['indices'])==int(masks[s.name(e)].sum())//100
    for key,g in groups.items():assert s.digest(g['indices'])==manifest.set_index('group').loc[key,'coordinate_sha256']
    candidates=s.load(D/'probe_candidates.pt');n=s.read(D/'probe_config.json')['batches_per_probe']
    a={i for b in candidates[:n] for i in b['sequence_ids']};b={i for batch in candidates[2:2+n] for i in batch['sequence_ids']}
    assert not (a&b) and not {i+1 for i in a}&b and not {i-1 for i in a}&b
    hashes=s.read(D/'checkpoint_and_probe_hashes.json')
    for batch,meta in zip(candidates,hashes['probe_candidates']):
        assert s.digest(batch['batch'][0])==meta['input_sha256'] and s.digest(batch['batch'][1])==meta['target_sha256']
    lookup=damage[damage.subset=='whole'].set_index(['kind','step','probe','group']).damage
    for r in forecasts.itertuples():
        current=lookup.loc[(r.kind,r.decision_step,'A',r.group)];selected=current<=r.tolerance
        if r.rule=='persistent':
            prev=s.STEPS[s.STEPS.index(r.decision_step)-1];selected=selected and lookup.loc[(r.kind,prev,'A',r.group)]<=r.tolerance
        assert bool(r.selected)==bool(selected)
        assert r.future_step==s.STEPS[s.STEPS.index(r.decision_step)+1]
        assert np.isclose(r.future_B_damage,lookup.loc[(r.kind,r.future_step,'B',r.group)],rtol=0,atol=1e-15)
    for key,z in damage.groupby(['kind','step','probe','group']):
        whole=z[z.subset=='whole'].iloc[0];parts=z[z.subset!='whole'];parts=parts[parts.target_tokens>0]
        assert parts.target_tokens.sum()==whole.target_tokens
        assert np.isclose((parts.damage*parts.target_tokens).sum()/whole.target_tokens,whole.damage,atol=1e-14)
    assert (unions.coordinates<=unions.cap).all()
    for r in unions[unions.status=='measured'].itertuples():
        selected=r.groups.split(';');assert sum(len(groups[k]['indices']) for k in selected)==r.coordinates
        for key in selected:
            assert lookup.loc[(r.kind,r.decision_step,'A',key)]<=r.tolerance
            prev=s.STEPS[s.STEPS.index(r.decision_step)-1]
            assert lookup.loc[(r.kind,prev,'A',key)]<=r.tolerance
        summed=sum(lookup.loc[(r.kind,r.step,r.probe,k)] for k in selected)
        assert np.isclose(summed,r.sum_individual_damage,atol=1e-14)
        assert np.isclose(r.union_damage-r.sum_individual_damage,r.nonadditivity,atol=1e-14)
    checks=s.read(D/'numerical_checks.json');assert checks['training_updates']==0 and len(checks['production_checks'])==16
    assert all(all(r[k] for k in ['state_restored','repeat_equal','zero_dose_equal','gradients_preserved','rng_preserved','mask_integrity']) for r in checks['production_checks'])
    s.save(D/'final_audit.json',dict(passed=True,damage_rows=len(damage),role_rows=len(roles),forecast_rows=len(forecasts),combined_rows=len(unions),output_hashes_verified=len(seal['files']),group_membership_disjoint=True,mask_survivors_only=True,probe_windows_disjoint=True,prospective_no_future_selection=True,conditional_weighting_reconciles=True,union_cap_and_additivity_accounting=True))
    print('Stage 1 output verification passed',flush=True)
if __name__=='__main__':main()
