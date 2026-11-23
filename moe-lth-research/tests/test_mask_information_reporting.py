"""Exercise reporting with synthetic data confined to pytest's temp directory."""
import importlib.util
import json
from pathlib import Path
import sys


def test_partial_pilot_reports_nesting_and_five_figures(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts/imp_mask_information'))
    spec=importlib.util.spec_from_file_location('mask_info_analysis_test',root/'scripts/imp_mask_information/analyze.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'OUT',tmp_path)
    records=[]
    for condition,loss in [('IMP',1.5),('matrix',1.53),('matrix',1.49)]:
        curve=tmp_path/f'{len(records)}.jsonl'
        curve.write_text(json.dumps(dict(step=0,loss=3.0))+'\n'+json.dumps(dict(step=2500,loss=loss))+'\n')
        records.append(dict(condition=condition,final_ce=loss,curves_path=str(curve),sparsity=.8,num_experts=16))
    module.write_csv(tmp_path/'pilot_results.csv',records)
    module.analyze()
    text=(tmp_path/'summary.md').read_text()
    assert '3/16' in text
    assert 'No seed-level confidence interval' in text
    summary=module.rows(tmp_path/'pilot_analysis.csv')[0]
    assert summary['n_masks']=='2'
    assert summary['independent_training_seeds']=='1'
    assert abs(float(summary['delta_random_minus_imp'])-.01)<1e-10
    assert len(list((tmp_path/'figures').glob('*.png')))==5


def test_no_data_does_not_invent_figures(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts/imp_mask_information'))
    spec=importlib.util.spec_from_file_location('mask_info_empty_test',root/'scripts/imp_mask_information/analyze.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'OUT',tmp_path)
    module.analyze()
    assert 'No mask advantage can be estimated' in (tmp_path/'summary.md').read_text()
    assert not (tmp_path/'figures').exists()
