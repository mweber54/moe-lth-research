"""Synthetic loss/ledger files only; zero actual training updates."""
from pathlib import Path
import tempfile,importlib.util,json,csv
import pandas as pd
ROOT=Path.cwd();OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('synthetic_report',OUT/'report.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
legacy=pd.read_csv(ROOT/'results/imp_initial_value_placement_v1/trajectory_losses.csv')
with tempfile.TemporaryDirectory(prefix='matched_recovery_synthetic_no_training_') as td:
 target=Path(td);r.OUT=target;(target/'status').mkdir();(target/'workers').mkdir();(target/'jobs').mkdir();manifest=[]
 for seed in (7,17,29):
  for order in ('A','B'):
   for draw in (1,2,3):
    for mask in ('imp','random1','random2'):
     for arm in ('M','Bm'):
      job=f'synthetic_{seed}_{order}_{draw}_{mask}_{arm}';path=target/'jobs'/job;path.mkdir()
      base=legacy[(legacy.seed==seed)&(legacy.order==order)&(legacy.variant==mask+'_original')].set_index('step')
      with (path/'attempt_ledger.jsonl').open('w') as f:
       for n in range(1,2501):f.write(json.dumps({'attempt':n})+'\n')
      with (path/'common_evaluations.jsonl').open('w') as f:
       for step in (0,250,500,1000,1500,2000,2500):
        # Mask and order terms must cancel correctly, preserving all arm pairing.
        delta=.001*draw if mask=='random1' else .002*draw if mask=='random2' else .02+(.10 if arm=='Bm' else .01)*(step>250)+.001*draw+(.01 if order=='B' else 0)
        f.write(json.dumps(dict(step=step,cross_entropy=float(base.loc[step,'cross_entropy'])+delta))+'\n')
      (target/'status'/f'{job}.json').write_text(json.dumps(dict(status='COMPLETE',attempted_updates=2500,applied_updates=2499,amp_skipped_updates=1,host='synthetic_no_training',slot_seconds=600,native_segment_seconds=590,attempt=str(path))))
      manifest.append(dict(job_id=job,seed=seed,order=order,draw=draw,mask_type=mask,arm=arm,initial_state_hash=f'synthetic_{seed}_{mask}_{draw}_{arm}',target_distance=.35,realized_distance=.35,frac_sign_flipped=.03 if arm=='Bm' else 0))
 with (target/'job_manifest.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
 pd.DataFrame([dict(arm='Bm',target_distance=.34,realized_distance=.3401),dict(arm='Bm',target_distance=.36,realized_distance=.3601)]).to_csv(target/'matrix_invariants.csv',index=False)
 (target/'registration.json').write_text(json.dumps(dict(preparation_elapsed_seconds=0)))
 (target/'workers/fixture_accounting.json').write_text(json.dumps(dict(worker_slot_seconds=64800)))
 r.main()
 draws=pd.read_csv(target/'individual_paired_draw_values.csv');assert len(draws)==9 and (abs(draws.C-.09)<1e-12).all() and (abs(draws.R_M-.01)<1e-12).all() and (abs(draws.R_Bm-.10)<1e-12).all()
 summary=pd.read_csv(target/'per_seed_results.csv');assert len(summary)==3 and summary.direct_pass.all() and summary.M_near_zero.all() and summary.Bm_reopens.all()
 accounting=pd.read_csv(target/'actual_resources.csv');assert accounting.attempted_updates.iloc[0]==270000
 with (OUT/'report_check.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['check','passed','actual_training_updates']);w.writeheader();w.writerow(dict(check='synthetic full 108-run report, within-run differences and paired arm contrasts',passed=True,actual_training_updates=0))
 print('synthetic report passed; zero actual training updates')
