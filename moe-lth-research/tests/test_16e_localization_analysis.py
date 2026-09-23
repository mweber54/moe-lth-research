"""Synthetic artifact test for reporting; no models trained or scientific outputs retained."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_16e_router_age_localization as a

def test_report_pipeline():
    original_out=a.OUT;original_developmental=a.developmental
    with tempfile.TemporaryDirectory() as tmp:
        a.OUT=Path(tmp);ages=[0,20,40,50,60,70,80,100];rows=[];runs=[]
        for seed in (7,17,29):
            for age in ages:
                gap=.1+abs(age-60)*.001+seed*.00001
                rows.append(dict(experts=16,seed=seed,router_age=age,dense_initial_loss=5.,sparse_initial_loss=5.1,immediate_pruning_shock=.1,dense_early_auc=3000.,sparse_early_auc=3100.+gap,dense_final_loss=1.8,sparse_final_loss=1.8+gap,ticket_gap=gap))
                for kind in ('dense','sparse'):
                    dest=Path(tmp)/f'recovery/{seed}/{age}/{kind}';(dest/'checkpoints').mkdir(parents=True)
                    (dest/'checkpoints/final_recovered.pt').write_bytes(b'synthetic fixture')
                    (dest/'metadata.json').write_text(json.dumps(dict(integrity_checks_passed=True,optimizer_state_entries_at_start=0,recovery_steps=2500,scheduler='none',precision='fp16',router_mode='trainable',confidence_control=False)))
                    pd.DataFrame([dict(step=s,loss=2.) for s in range(0,2501,50)]).to_json(dest/'metrics.jsonl',orient='records',lines=True)
                    runs.append(dict(condition_dir=str(dest),final_checkpoint_sha256=a.sha256_file(dest/'checkpoints/final_recovered.pt')))
        outcomes=pd.DataFrame(rows);outcomes.to_csv(Path(tmp)/'outcomes.csv',index=False);pd.DataFrame(runs).to_csv(Path(tmp)/'run_manifest.csv',index=False)
        (Path(tmp)/'localization_decision.json').write_text(json.dumps(dict(stage_a_ages=[0,20,40,60,80,100],stage_b_ages=[50,70],stage_c_ages=[])))
        def fake_developmental(outcomes):
            old=pd.read_csv(a.ROOT/'results/r20_8_vs_r60_12/developmental_vectors.csv');metrics=[]
            for row in outcomes.to_dict('records'):
                x=old[(old.expert_count==12)&(old.seed==row['seed'])&(old.router_age==60)].iloc[0].to_dict();x.update(row)
                x.update(entropy=x['normalized_entropy']*2.7,token_specialization=x['specialization']*2.7,assignment_agreement_to_r100=x['assignment_agreement_to_R100'],js_distance_to_r100=x['functional_js_to_R100'],cumulative_expert_exposure=x['cumulative_exposure_per_expert'])
                metrics.append(x)
            return pd.DataFrame(metrics)
        a.developmental=fake_developmental
        with contextlib.redirect_stdout(io.StringIO()):a.analyze()
        assert (Path(tmp)/'final_report.md').exists()
        assert len(list((Path(tmp)/'figures').glob('*.png')))==10
        assert json.loads((Path(tmp)/'localization_decision.json').read_text())['candidate_optimum']==60
        assert len(pd.read_csv(Path(tmp)/'cross_architecture_optima.csv'))==3
    a.OUT=original_out;a.developmental=original_developmental

if __name__=='__main__':test_report_pipeline();print('Synthetic localization reporting checks passed')
