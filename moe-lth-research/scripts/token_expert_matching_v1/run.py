"""Native two-segment donor/replay runner; fresh attempts only, immutable inputs."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'scripts/imp_adaptation_factorial_v1'))
from af_common import *
from af_native import segment
from af_engine import batches
from af_diagnostics import model_state, displacement
import replay
import argparse
import socket
import traceback

DEST=ROOT/'results/token_expert_matching_v1'
PERMUTATIONS=(101,202,303)


def atomic_tensor(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)


def named_displacement(model,previous,initial,start=0,step=0):
    current=dict(model.named_parameters())
    missing={label:sorted(set(current)-set(state)) for label,state in [('previous',previous),('initial',initial)]}
    if any(missing.values()):raise RuntimeError('Missing named parameters '+str(missing))
    return dict(missing=missing,frozen=[n for n,p in current.items() if not p.requires_grad],
                components=displacement(model,model_state(model),previous,initial,start,step))


def gate_summary(payload):
    rows=[]
    for layer in payload['layers']:
        expert_rows=[]
        for expert in range(len(layer['order'])):
            proposed=layer['ids']==expert
            accepted=proposed & layer['accepted']
            values=layer['gates'][proposed].double()
            expert_rows.append(dict(expert=expert,proposed=int(proposed.sum()),accepted=int(accepted.sum()),gate_min=float(values.min()) if values.numel() else None,gate_max=float(values.max()) if values.numel() else None,gate_mean=float(values.mean()) if values.numel() else None,gate_std=float(values.std(unbiased=False)) if values.numel() else None))
        rows.append(dict(layer=layer['layer'],capacity=layer['capacity'],experts=expert_rows))
    return rows


def run(v,kind,draw=None):
    if kind not in ('donor','identity','shuffle') or (kind=='shuffle' and draw not in PERMUTATIONS):
        raise ValueError('Unregistered condition')
    name=f'{v["id"]}_{kind}'+(f'_{draw}' if draw is not None else '')
    attempt=DEST/'attempts'/name
    donor=DEST/'attempts'/f'{v["id"]}_donor'
    if kind!='donor' and not (donor/'result.json').exists():raise RuntimeError('Complete validated donor required')
    attempt.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    save(attempt/'status.json',dict(status='RUNNING',host=socket.getfqdn(),pid=os.getpid()))
    try:
        train,val=batches(v)
        masks=load_masks(v['mask']) if v['mask'] else {}
        source=load_model_from_checkpoint(v['config']['model'],v['initial'],torch.device('cpu'))
        apply_masks_(source,masks);initial=model_state(source);del source
        if state_dict_hash(initial)!=v['initial_model_hash']:raise RuntimeError('Initial state changed')
        origin=initial
        router=v['initial'];records=[];trace_rows=[];previous=initial;previous_step=0
        for offset in (0,2500):
            dest=attempt/f'segment_{offset}'
            frozen_parameters={}
            def assembled(model):
                if kind!='donor':replay.install(model)
                frozen_parameters.update({n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad})
                save(dest/'parameters.json',[dict(name=n,trainable=p.requires_grad,group=group(n)) for n,p in model.named_parameters()])
            def forward(model,tokens,step):
                absolute=offset+step
                batch_hash=R._batch_sequence_hash([train[absolute-1]])
                path=(donor if kind!='donor' else attempt)/'traces'/f'{absolute:04d}.pt'
                if kind=='donor':
                    output=R._condition_forward(model,tokens,False,1.)
                    with protected(model):
                        payload=replay.capture(model,output,tokens,absolute,batch_hash)
                        atomic_tensor(path,payload)
                        append(attempt/'gate_distributions.jsonl',dict(step=absolute,layers=gate_summary(payload)))
                    trace_rows.append(dict(step=absolute,path=str(path),sha256=sha(path),batch_hash=batch_hash))
                    return output
                expected=donor_inventory[absolute]
                if sha(path)!=expected['sha256']:raise RuntimeError('Trace checksum mismatch')
                payload=load(path)
                output=replay.forward(model,tokens,payload,absolute,batch_hash,draw if kind=='shuffle' else None,v['seed'])
                if absolute in STEPS:
                    with protected(model):
                        params={n:p for n,p in model.named_parameters() if p.requires_grad}
                        task=torch.nn.functional.cross_entropy(output.logits.flatten(0,1),train[absolute-1][1].to(tokens.device).flatten())
                        gradients=dict(zip(params,torch.autograd.grad(task,tuple(params.values()),retain_graph=True,allow_unused=True),strict=True))
                        rows=[]
                        for n,p in model.named_parameters():
                            gradient=gradients.get(n)
                            if gradient is not None and n in masks:gradient=gradient[masks[n].to(gradient.device)]
                            rows.append(dict(parameter=n,group=group(n),trainable=p.requires_grad,missing_gradient=gradient is None,task_l2=float(gradient.detach().double().norm()) if gradient is not None else None))
                        save(attempt/f'task_gradients/{absolute:04d}.json',dict(step=absolute,batch_hash=batch_hash,parameters=rows,precision='production autocast; task-only before loss scaling'))
                append(attempt/'conservation.jsonl',dict(step=absolute,passed=True,permutations=[replay.permute(x,draw,v['seed'],absolute)['permutation_hash'] for x in payload['layers']] if draw is not None else None))
                return output
            def callback(model,opt,scaler,step):
                nonlocal previous,previous_step
                absolute=offset+step
                for n,p in model.named_parameters():
                    if n in frozen_parameters and not torch.equal(p,frozen_parameters[n]):raise RuntimeError('Frozen parameter changed '+n)
                    if n in masks and torch.count_nonzero(p[~masks[n].to(p.device)]):raise RuntimeError('Mask regrowth '+n)
                if step==0:
                    if opt.state:raise RuntimeError('Fresh boundary optimizer required')
                    ids={id(p) for g in opt.param_groups for p in g['params']}
                    if any((id(p) in ids)!=p.requires_grad for p in model.parameters()):raise RuntimeError('Optimizer membership')
                    if state_dict_hash(model.state_dict())!=state_dict_hash(initial):raise RuntimeError('Reset changed state')
                    save(dest/'reset.json',dict(step=absolute,hash=state_dict_hash(model.state_dict()),optimizer_entries=0,scaler=scaler.state_dict(),seed=v['seed']))
                if absolute in STEPS and not(offset and step==0):
                    with protected(model):
                        snapshot=model_state(model)
                        atomic_tensor(attempt/f'snapshots/{absolute:04d}.pt',dict(model=snapshot))
                        diagnostic=named_displacement(model,previous,origin,previous_step,absolute)
                        diagnostic['step']=absolute
                        save(attempt/f'diagnostics/{absolute:04d}.json',diagnostic)
                        previous=snapshot;previous_step=absolute
            donor_inventory={x['step']:x for x in read(donor/'trace_inventory.json')} if kind!='donor' else {}
            record=segment(v['config'],initial,router,masks,train[offset:offset+2500],val,dest,'full' if kind=='donor' else 'expert_shared',v['seed'],offset,callback,forward,assembled=assembled)
            save(dest/'record.json',record);records.append(record)
            router=str(dest/'checkpoints/final_recovered.pt');initial=load(router)['model']
        curves=[]
        for offset in (0,2500):
            for row in map(json.loads,(attempt/f'segment_{offset}/metrics.jsonl').read_text().splitlines()):
                if offset and row['step']==0:continue
                curves.append(dict(step=offset+row['step'],ce=row['loss']))
        save(attempt/'curves.json',curves)
        training=[]
        for offset in (0,2500):
            for row in map(json.loads,(attempt/f'segment_{offset}/gradient_stats/gradient_stats.jsonl').read_text().splitlines()):
                training.append(dict(step=offset+row['step'],**{k:v for k,v in row.items() if k!='step'}))
        save(attempt/'training_curve.json',training)
        if kind=='donor':
            if len(trace_rows)!=5000:raise RuntimeError('Incomplete donor trace')
            save(attempt/'trace_inventory.json',trace_rows)
            canonical=[]
            for index,record in enumerate(records):
                historical=read(Path(v['full_attempt'])/f'segment_{index*2500}/record.json')
                for key in ('optimizer_steps_applied','amp_overflow_step_count'):
                    if record[key]!=historical[key]:raise RuntimeError('Donor AMP accounting changed '+key)
            for offset in (0,2500):
                for row in map(json.loads,(Path(v['full_attempt'])/f'segment_{offset}/metrics.jsonl').read_text().splitlines()):
                    if offset and row['step']==0:continue
                    canonical.append(dict(step=offset+row['step'],ce=row['loss']))
            if curves!=canonical:raise RuntimeError('Passive tracing changed donor CE trajectory')
            if state_dict_hash(load(router)['model'])!=state_dict_hash(load(Path(v['full_attempt'])/'segment_2500/checkpoints/final_recovered.pt')['model']):raise RuntimeError('Passive tracing changed donor final state')
        result=dict(variant=v['id'],seed=v['seed'],sparsity=v['sparsity'],kind=kind,draw=draw,final_ce=curves[-1]['ce'],applied_updates=sum(r['optimizer_steps_applied'] for r in records),amp_skips=sum(r['amp_overflow_step_count'] for r in records),wall_seconds=time.monotonic()-start,checkpoint=router)
        if result['applied_updates']+result['amp_skips']!=5000:raise RuntimeError('Attempt accounting failed')
        save(attempt/'provenance.json',dict(variant=v,initial_hash=v['initial_model_hash'],mask_hash=v['mask_hash'],registration_sha256=sha(DEST/'registration.json'),donor=str(donor),trace_inventory_sha256=sha((attempt if kind=='donor' else donor)/'trace_inventory.json'),gate_distributions=str((attempt if kind=='donor' else donor)/'gate_distributions.jsonl'),gate_distribution_semantics='Same per-expert multisets for identity and shuffled replay; actual counts also logged by native gradient_stats'))
        save(attempt/'result.json',result)
        save(attempt/'status.json',dict(status='COMPLETE',**result))
    except BaseException as error:
        save(attempt/'status.json',dict(status='FAILED',error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--variant',required=True);parser.add_argument('--kind',choices=['donor','identity','shuffle'],required=True);parser.add_argument('--draw',type=int)
    args=parser.parse_args()
    torch.set_num_threads(4);R.configure_device(torch.device('cuda'))
    if socket.getfqdn() not in HOSTS or not torch.__version__.startswith('2.7.0') or not torch.cuda.is_available():raise RuntimeError('Canonical host/runtime required')
    if torch.cuda.mem_get_info()[0]<24*1024**3:raise RuntimeError('GPU admission failed; other jobs preserved')
    manifest=frozen()
    v=next(x for x in manifest['variants'] if x['id']==args.variant)
    for path in (v['initial'],v['mask']):
        if path and sha(path)!=read(OUT/'input_hashes.json')[path]:raise RuntimeError('Changed input '+path)
    if not read(DEST/'validation.json').get('production_ready',False):raise RuntimeError('Production validation incomplete')
    with lock(DEST/'workers'/f'{socket.getfqdn()}.lock',False):
        run(v,args.kind,args.draw)
