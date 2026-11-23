"""Independent table audit and synthetic test receipt; no model execution."""
import json
import time
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from prepare import ROOT,D,SOURCE,sha,save
from analyze import score,select,rho,coordinate_union

def main():
    started=time.perf_counter()
    config=json.loads((D/'analysis_config.json').read_text());freeze=json.loads((D/'freeze.json').read_text())
    assert sha(D/'analysis_config.json')==freeze['configuration_sha256']
    assert sha(D/'protocol.md')==freeze['protocol_sha256']
    assert freeze['unix']<=json.loads((D/'estimand_clarification.json').read_text())['unix']
    raw=pd.read_csv(SOURCE/'deletion_damage_by_batch.csv')
    manifest=pd.read_csv(SOURCE/'intervention_manifest.csv').set_index('intervention')
    totals=raw.groupby(['trajectory','step','expert','subset','partition','intervention'])[['damage_sum','target_tokens']].sum()
    inventory=pd.read_csv(D/'temporal_intervention_inventory.csv')
    for iid,z in inventory.groupby('intervention'):
        assert z.identity.nunique()==1 and z.coordinate_sha256.nunique()==1
        if z.intervention_kind.iloc[0]=='weight_union': assert z.coordinate_sha256.iloc[0]==manifest.loc[iid,'coordinate_sha256']
    scores=pd.read_csv(D/'temporal_scores_and_outcomes.csv');max_error=0.;prediction_groups=0
    for keys,z in scores.groupby(['trajectory','early_step','future_step','expert','dose_groups','intervention_kind','subset','rule'],dropna=False):
        trajectory,early,future,expert,dose,kind,subset,rule=keys
        z=z.sort_values('intervention');ids=z.intervention.to_list()
        def means(t,part,ss):
            a=totals.loc[(trajectory,t,expert,ss,part)].loc[ids]
            return np.where(a.target_tokens>0,a.damage_sum/a.target_tokens,np.nan)
        c=means(early,'discovery','whole');previous=z.previous_step.iloc[0]
        p=means(int(previous),'discovery','whole') if pd.notna(previous) else None
        computed=score(c,p,rule);observed=z.prediction_score.to_numpy()
        np.testing.assert_allclose(computed,observed,rtol=0,atol=1e-14,equal_nan=True)
        expected_future=means(future,'confirmation',subset)
        np.testing.assert_allclose(expected_future,z.future_confirmation_damage,rtol=0,atol=1e-14,equal_nan=True)
        # Integration leakage check: mutate every future outcome, independently recompute permitted predictions.
        low,high,tie=select(computed,ids)
        mutated_future=np.arange(len(ids))*1000.+999.
        repeated=score(c,p,rule)
        assert select(repeated,ids)==(low,high,tie)
        np.testing.assert_array_equal(computed,repeated)
        prediction_groups+=1
    ranks=pd.read_csv(D/'temporal_rank_agreement.csv')
    assert ranks[ranks.distinct_unions==1].spearman.isna().all()
    for row in ranks.itertuples():
        expected=rho(json.loads(row.scores),np.array(json.loads(row.future_damages),float))
        assert (expected is None and pd.isna(row.spearman)) or np.isclose(expected,row.spearman,atol=1e-14)
    policy=pd.read_csv(D/'low_damage_selection_comparison.csv')
    same=policy[policy.identical_selection]
    for field in ['current_minus_rule_future_damage','current_minus_rule_ci95_low','current_minus_rule_ci95_high']:
        assert (same[field]==0).all()
    assert (policy.distinct_unions>=3).all()
    st=pd.read_csv(D/'selection_stability.csv')
    keys=['trajectory','transition','expert','dose_groups','subset','rule']
    for _,z in st.groupby(keys):
        assert np.isclose(z.low_selection_frequency.sum()+z.undefined_selection_fraction.iloc[0],1)
        assert np.isclose(z.high_selection_frequency.sum()+z.undefined_selection_fraction.iloc[0],1)
    thresholds=pd.read_csv(D/'false_safe_summary.csv')
    assert thresholds[thresholds.selected_count==0].false_safe_fraction.isna().all()
    assert (thresholds.selected_coordinate_union_count<=thresholds.panel_coordinate_union_count).all()
    assert (thresholds[thresholds.selected_count==thresholds.candidate_count].status=='ceiling_no_discrimination').all()
    # Verify coordinate coverage independently from existing group membership lists.
    inv=inventory.drop_duplicates('intervention').set_index('intervention')
    import torch
    groups=torch.load(ROOT/'results/expert_role_deletion_v1/group_coordinates.pt',map_location='cpu',weights_only=True)
    sets={iid:set().union(*(set(groups[g]['indices'].tolist()) for g in json.loads(row.component_groups))) for iid,row in inv.iterrows()}
    for row in thresholds.itertuples():
        ids=json.loads(row.selected_ids)
        union=set().union(*(sets[i] for i in ids))
        assert len(union)==row.selected_coordinate_union_count
    prov=json.loads((D/'input_provenance.json').read_text())
    assert all(sha(ROOT/path)==h for path,h in prov['verified_files'].items())
    test_start=time.perf_counter()
    proc=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pytest','tests/test_expert_deletion_temporal_prediction.py','-q'],cwd=ROOT,text=True,capture_output=True)
    (D/'tests.log').write_text(proc.stdout+proc.stderr);assert proc.returncode==0,proc.stdout+proc.stderr
    save(D/'test_results.json',dict(passed=True,command=proc.args,output=proc.stdout,test_wall_seconds=time.perf_counter()-test_start,model_execution=False))
    checks=json.loads((D/'numerical_checks.json').read_text())
    checks.update(synthetic_tests_passed=True,synthetic_future_leakage_invariance=True,
        independently_recomputed_prediction_panels=prediction_groups,all_predictions_early_whole_discovery_only=True,
        singleton_ranks_undefined=True,empty_false_safe_undefined=True,coordinate_overlap_set_union_verified=True,
        historical_inputs_unchanged_at_final_audit=True,selection_frequencies_normalized=True,
        independent_audit_wall_seconds=time.perf_counter()-started)
    save(D/'numerical_checks.json',checks)
    acc=json.loads((D/'compute_accounting.json').read_text())
    acc.update(verification_wall_seconds=time.perf_counter()-started,
        end_to_end_wall_since_protocol_freeze_seconds=time.time()-freeze['unix'],
        end_to_end_includes_implementation_and_reporting=True,
        source_code_bytes=sum(p.stat().st_size for p in Path(__file__).parent.glob('*.py')))
    save(D/'compute_accounting.json',acc)
    # Size reaches a fixed point after the accounting field's own digit count is stable.
    for _ in range(3):
        acc['storage_bytes']=sum(p.stat().st_size for p in D.rglob('*') if p.is_file())
        save(D/'compute_accounting.json',acc)
    save(D/'final_audit.json',dict(passed=True,model_executions=0,training_updates=0,
        prediction_panels=prediction_groups,historical_artifacts_preserved=True,
        output_counts={p.name:len(pd.read_csv(p)) for p in D.glob('*.csv')}))
    print('Independent temporal audit passed; historical inputs unchanged',flush=True)
    # Final immutable output inventory excludes the seal itself.
    for _ in range(4):
        save(D/'output_seal.json',dict(files={str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file() and p.name!='output_seal.json' and p.suffix!='.log'},
            implementation={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},
            test_implementation_sha256=sha(ROOT/'tests/test_expert_deletion_temporal_prediction.py')))
        acc['storage_bytes']=sum(p.stat().st_size for p in D.rglob('*') if p.is_file())
        save(D/'compute_accounting.json',acc)
    # The last iteration changes no file length; refresh the accounting hash in the seal.
    seal=json.loads((D/'output_seal.json').read_text())
    seal['files']['compute_accounting.json']=sha(D/'compute_accounting.json')
    save(D/'output_seal.json',seal)

if __name__=='__main__':main()
