from pathlib import Path
import sys, importlib.util, csv, json, time
ROOT=Path.cwd(); sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('placement',ROOT/'scripts/imp_initial_value_placement_v1/run.py'); p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
import torch
out=ROOT/'results/imp_sign_magnitude_step0_v1'
torch.set_num_threads(4)
rows=p.rows(p.OUT/'variant_manifest.csv'); logs=p.rows(p.OUT/'trajectory_losses.csv')
results=[]
for seed in (7,17,29,43,61):
    entry=p.native._seed_entry(seed); config=entry['config']
    p.native.recovery.seed_everything(seed)
    _,loader=p.native.build_dataloaders(config['data'],int(config['training']['batch_size']),seed)
    val=p.native.recovery._materialize_validation_batches(loader,int(config['data']['validation_blocks']))
    h=p.native.recovery._batch_sequence_hash(val)
    assert h==entry['validation_batch_sequence_hash']
    for variant in ('imp_original','random1_original','random2_original'):
        row=next(r for r in rows if int(r['seed'])==seed and r['order']=='A' and r['variant']==variant)
        state,masks=p.variant_state(row)
        payload=p.native._read_model_payload(Path(row['initial_checkpoint']))
        (out/f'config_seed{seed}.json').write_text(json.dumps(payload['config'],indent=2))
        model=p.native.TinyMoELanguageModel(config['model']).cuda(); model.load_state_dict(state)
        # Use the original evaluator itself; a new model-only file stays in this new directory.
        temp=out/'baseline_model.pt';torch.save({'model':state,'step':0},temp)
        measured=p.native._eval_one_snapshot(model,temp,val,torch.device('cuda'),seed,'A','imp80',row['job_id'],h)
        logged=float(next(r for r in logs if r['job_id']==row['job_id'] and r['step']=='0')['cross_entropy'])
        result={'seed':seed,'mask_type':variant,'logged_CE':logged,'CE':measured['cross_entropy'],'difference':measured['cross_entropy']-logged,'validation_hash':h,'target_tokens':measured['target_tokens']}
        results.append(result)
        p.table(out/'baseline_reproduction.csv',results)
        print(json.dumps(result),flush=True)
        del model,state;torch.cuda.empty_cache()
        temp.unlink()
p.save(out/'baseline_gate.json',{'passed':all(abs(r['difference'])<=1e-3 for r in results),'results':results})
