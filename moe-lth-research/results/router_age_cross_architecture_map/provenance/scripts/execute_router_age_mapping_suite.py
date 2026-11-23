#!/usr/bin/env python3
"""Strictly ordered mapping suite; resumes completed phases, never overwrites a recovery."""
import argparse,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from run_16e_router_age_localization import ROOT,write_json,sha256_file
OUT=ROOT/'results/router_age_cross_architecture_map'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--existing-repeat-pid',type=int);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    manifest_path=OUT/'workflow_manifest.json'
    if manifest_path.exists():raise FileExistsError('Mapping workflow already has a manifest; inspect it before any restart')
    scripts=['run_16e_seed7_reproducibility.py','run_24e_router_age_localization.py','analyze_24e_router_age_localization.py','run_32e_router_age_localization.py','analyze_32e_router_age_localization.py','analyze_router_age_cross_architecture_map.py']
    historical=[ROOT/'results/16e_router_age_localization/outcomes.csv',ROOT/'results/16e_router_age_localization/localization_decision.json',ROOT/'results/expert_count_replication/sparse_results.csv',ROOT/'results/expert_count_replication/dense_baselines.csv',ROOT/'results/router_development_12expert_retry4_20260902/ticket_gap_seed_level.csv',ROOT/'results/router_development_12expert_retry4_20260902/dense_baselines.csv']
    state={'status':'waiting_for_seed7_repeat','existing_repeat_pid':args.existing_repeat_pid,'independent_seeds':[7,17,29],'repeat_is_independent_seed':False,'phases':['16E_seed7_repeat','24E_localization','32E_localization','unified_descriptive_map'],'code_hashes':{s:sha256_file(ROOT/'scripts'/s) for s in scripts},'historical_hashes':{str(p):sha256_file(p) for p in historical},'mechanistic_interventions':0}
    write_json(manifest_path,state)
    (OUT/'suite_protocol.md').write_text('# Router-age mapping suite\n\nStrict order: finish and interpret one seed-7 repeat (R50/60/70/80/90/100, 12 recoveries), then 24E, then 32E, then unified analysis. Each new architecture uses seeds 7/17/29, 85% expert-local pruning, E_T masks, E_0 rewind, native trainable age router, same historical 2500-step recovery. Coarse ages 0/20/40/60/80/100 precede adaptive missing 10%-age refinement. The prior paired-seed-SD ambiguity rule is retained, preregistered before new outcomes. Optional Stage C omitted; residual ambiguity reported as a basin. No mechanism interventions.\n\nThe original 16E seed-7 observation remains primary, n=3; its repeat is never pooled. Same historical metric definitions and caveats apply. The unified primary verdict conservatively prioritizes a cross-architecture descriptive intermediate pattern if >=3 mean winners are interior, marking seed-conditioning whenever any such system lacks local/endpoint consistency; otherwise reports recovery-emergence only with >=2 interior systems and <=1/5 pruning-shock and early-AUC agreements, or no single phenomenon. This is a report-classification convention, not a statistical test. No universal, threshold, scaling-law or causal conclusion is selected automatically from five systems. Feature concentration is compared relative to all sampled same-seed cross-architecture age pairs, preserving historical feature-range scaling; no p-values.\n')
    try:
        if args.existing_repeat_pid:
            while Path(f'/proc/{args.existing_repeat_pid}').exists():
                stat=Path(f'/proc/{args.existing_repeat_pid}/stat').read_text().split()
                if stat[2]=='Z':break
                time.sleep(30)
        else:
            subprocess.run([sys.executable,'-u','scripts/run_16e_seed7_reproducibility.py'],cwd=ROOT,check=True)
        repeat_root=ROOT/'results/16e_seed7_reproducibility'
        if Path('/tmp/16e_seed7_reproducibility_execution.log').exists():shutil.copy2('/tmp/16e_seed7_reproducibility_execution.log',repeat_root/'execution.log')
        repeat=json.loads((repeat_root/'reproducibility_summary.json').read_text());assert repeat['status']=='complete'
        assert (repeat_root/'artifact_integrity.json').exists() and (repeat_root/'final_report.md').exists()
        state['repeat_verdict']=repeat['verdict'];write_json(manifest_path,state)
        print('[mapping] Phase 1:',repeat['verdict'],flush=True)
        for e in (24,32):
            for name,digest in state['code_hashes'].items():assert sha256_file(ROOT/'scripts'/name)==digest,f'Code changed: {name}'
            if shutil.disk_usage(OUT).free<40*1024**3:raise RuntimeError(f'Insufficient 40 GiB reserve before {e}E; no artifacts deleted')
            state['status']=f'{e}e_localization_running';write_json(manifest_path,state)
            # Architecture preflight occurs inside this runner, after the previous phase report.
            root=ROOT/f'results/{e}e_router_age_localization';root.mkdir(parents=True,exist_ok=True)
            with (root/'execution.log').open('a') as log:
                subprocess.run([sys.executable,'-u',f'scripts/run_{e}e_router_age_localization.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            decision=json.loads((root/'localization_decision.json').read_text());assert decision['status']=='complete'
            assert (root/'final_report.md').exists() and (root/'artifact_integrity.json').exists()
            state[f'{e}e_verdict']=decision['verdict'];write_json(manifest_path,state)
            print(f'[mapping] {e}E:',decision['verdict'],flush=True)
        for name,digest in state['historical_hashes'].items():assert sha256_file(Path(name))==digest,f'Historical primary observation changed: {name}'
        state['status']='unified_analysis_running';write_json(manifest_path,state)
        subprocess.run([sys.executable,'-u','scripts/analyze_router_age_cross_architecture_map.py'],cwd=ROOT,check=True)
        state['status']='complete';write_json(manifest_path,state)
    except BaseException as error:
        state.update(status='failed',error=repr(error));write_json(manifest_path,state)
        (OUT/'workflow_failure.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
