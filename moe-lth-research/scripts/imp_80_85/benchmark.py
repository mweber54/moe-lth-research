"""Short canonical-shape benchmark; never counted as an experimental result."""
import argparse
import shutil
import time
from run import *


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--experts',type=int,default=8,choices=common.ES)
    p.add_argument('--seed',type=int,default=7,choices=common.SEEDS)
    p.add_argument('--steps',type=int,default=20)
    args=p.parse_args()
    if not 2 <= args.steps <= 100: p.error('Benchmark steps must be 2–100')
    pairs=canonical_pairs();pair=pairs[args.experts,args.seed,.8]
    inventory=preflight(pairs,[(args.experts,args.seed)])
    if not all(v['verified'] for v in inventory[0]['checkpoints'].values()):
        raise RuntimeError('Missing benchmark prerequisites')
    init=Path(pair['source_checkpoint_path']);config=common.load(init)['config']
    device=torch.device('cuda');configure_device(device);torch.set_num_threads(4)
    seed_everything(args.seed)
    tr,va=build_dataloaders(config['data'],128,args.seed)
    train=recovery._materialize_batches(tr,args.steps)
    val=recovery._materialize_validation_batches(va,32)
    model=load_model_from_checkpoint(config['model'],str(init),device)
    masks={n:torch.ones_like(v,dtype=torch.bool,device='cpu') for n,v in expert_weight_parameters(model).items()}
    reference=selected_experts_per_batch(model,recovery._calibration_batches(val),device)
    del model
    dest=OUT/'benchmarks'/f'e{args.experts}_seed{args.seed}_{time.time_ns()}'
    record=run_stage(config,init,masks,train,val,reference,device,dest,float(pair['dense_final_validation_loss']),
                     {'num_experts':args.experts,'seed':args.seed,'steps':args.steps,'sparsity':0.,'development_benchmark':True})
    bytes_per_checkpoint=(dest/'checkpoints/final_recovered.pt').stat().st_size
    report={'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__, 'steps':args.steps,
        'duration_seconds':record['duration_seconds'], 'seconds_per_attempt_including_diagnostics':record['duration_seconds']/args.steps,
        'representative_round_hours_linear':record['duration_seconds']/args.steps*2500/3600,
        'full_275_stages_hours_same_architecture_linear':record['duration_seconds']/args.steps*2500*275/3600,
        'checkpoint_bytes':bytes_per_checkpoint,'full_275_checkpoints_gib_same_architecture':bytes_per_checkpoint*275/1024**3,
        'free_disk_gib':shutil.disk_usage(OUT).free/1024**3,
        'limitations':'Short shared-GPU e8 measurement; fixed diagnostics overestimate per-step cost. Other expert counts and contention vary. Includes 225 discovery stages plus 50 recoveries; temporary AdamW resume snapshots add disk usage.'}
    write_json(dest/'benchmark.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
