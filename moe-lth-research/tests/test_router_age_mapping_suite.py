"""Disposable synthetic reporting checks. No experimental recoveries."""
import contextlib,io,json,sys,tempfile
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_router_age_cross_architecture_map as mapping

def test_mapping_report():
    original_root=mapping.ROOT;original_out=mapping.OUT;original_loader=mapping.load_architecture
    frames={e:original_loader(e) for e in (8,12,16)}
    for e in (24,32):
        frames[e]=frames[16].copy();frames[e]['experts']=e;frames[e]['historical']=False;frames[e]['new']=True
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'results').mkdir()
        for name in ['expert_count_replication','router_development_12expert_retry4_20260902','r20_8_vs_r60_12']:(root/'results'/name).symlink_to(original_root/'results'/name,target_is_directory=True)
        repeat=root/'results/16e_seed7_reproducibility';repeat.mkdir()
        (repeat/'reproducibility_summary.json').write_text(json.dumps(dict(status='complete',verdict='16E SEED7 EXACTLY REPRODUCED',original_winner=100,rerun_winner=100)))
        pd.DataFrame({'router_age':[50,60,70,80,90,100],'original_ticket_gap':[.3,.2,.21,.22,.23,.18],'rerun_ticket_gap':[.3,.2,.21,.22,.23,.18]}).to_csv(repeat/'original_vs_rerun.csv',index=False)
        mapping.ROOT=root;mapping.OUT=root/'results/router_age_cross_architecture_map';mapping.load_architecture=lambda e:frames[e]
        with contextlib.redirect_stdout(io.StringIO()):mapping.main()
        assert len(list((mapping.OUT/'figures').glob('*.png')))==16
        assert len(pd.read_csv(mapping.OUT/'architecture_summary.csv'))==5
        report=(mapping.OUT/'final_report.md').read_text();assert '31. **Was a follow-up launched?**' in report
        assert not pd.read_csv(mapping.OUT/'unified_outcomes.csv').reproducibility_rerun.any()
    mapping.ROOT=original_root;mapping.OUT=original_out;mapping.load_architecture=original_loader

if __name__=='__main__':
    test_mapping_report()
    # Exercise both generated localization analyses using the prior disposable fixture.
    source=(Path(__file__).parent/'test_16e_localization_analysis.py').read_text()
    for e in (24,32):
        code=source.replace('import analyze_16e_router_age_localization as a',f'import analyze_{e}e_router_age_localization as a')
        with contextlib.redirect_stdout(io.StringIO()):exec(compile(code,str(Path(__file__).parent/'test_16e_localization_analysis.py'),'exec'),{'__name__':'__main__','__file__':str(Path(__file__).parent/'test_16e_localization_analysis.py')})
    print('Mapping and 24E/32E synthetic reporting checks passed')
