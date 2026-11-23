"""Checkpoint-only perturbations. No optimizer, backward, or training calls."""
from pathlib import Path
import sys, importlib.util, csv, json, hashlib, math, time
ROOT=Path.cwd(); OUT=ROOT/'results/imp_sign_magnitude_step0_v1';sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('placement',ROOT/'scripts/imp_initial_value_placement_v1/run.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
import os
import torch
import torch.nn.functional as F

def rng(seed,arm,condition,dose,draw,name):
    key=f'sign-magnitude-step0-v1|{seed}|{arm}|{condition}|{dose}|{draw}|{name}'
    s=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')%(2**63)
    return torch.Generator(device='cuda').manual_seed(s),s

def transform(v,kind,dose,g):
    z=v.clone();n=len(v)
    if kind=='A':
        for sign in (1,-1):
            ids=torch.where(v*sign>0)[0]; k=round(dose*len(ids)); ids=ids[torch.randperm(len(ids),device='cuda',generator=g)[:k]]
            z[ids]=v[ids[torch.randperm(k,device='cuda',generator=g)]]
        assert torch.equal(v.sign(),z.sign())
        for sign in (1,-1):
            assert torch.equal(v[v*sign>0].abs().sort().values,z[z*sign>0].abs().sort().values)
    elif kind=='B':
        eligible=torch.where(v!=0)[0]; ids=eligible[torch.randperm(len(eligible),device='cuda',generator=g)[:round(dose*len(eligible))]]
        z[ids]=v[ids].abs()*v[ids[torch.randperm(len(ids),device='cuda',generator=g)]].sign()
        assert torch.equal(v.abs(),z.abs())
        assert torch.equal(v.sign().sort().values,z.sign().sort().values)
    elif kind=='C': z=v.sign()*v.abs().mean()
    elif kind=='D':
        z=v[torch.randperm(n,device='cuda',generator=g)]
        assert torch.equal(v.sort().values,z.sort().values)
    elif kind=='F': pass
    else: raise ValueError(kind)
    if kind in ('A','B','D'):
        assert torch.allclose(v.double().square().sum(),z.double().square().sum(),rtol=1e-12,atol=1e-12)
    return z

@torch.inference_mode()
def evaluate(model,val,reference=None):
    total=0.;count=0;routes=[];changed=0;ntokens=0
    for b,(x,y) in enumerate(val):
        output=model(x.cuda());y=y.cuda(); logits=output.logits.float()
        total+=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='sum').double().cpu());count+=y.numel()
        ids=torch.stack([r.selected_experts.sort(-1).values for r in output.routes],dim=0)
        if reference is None: routes.append(ids.cpu())
        else:
            # A token changes if its expert set differs in any layer.
            changed+=int((ids!=reference[b].cuda()).any(-1).any(0).sum());ntokens+=ids.shape[1]
    return total/count,(routes if reference is None else changed/ntokens)

def main():
    assert json.loads((OUT/'baseline_gate.json').read_text())['passed'],'Baseline gate failed: STOP'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=p.rows(p.OUT/'variant_manifest.csv');baseline=p.rows(OUT/'baseline_reproduction.csv')
    fields=['seed','mask_type','condition','p','draw','layer','projection','realized_distance','frac_sign_flipped','CE','delta_CE','routing_change','rng_key','matched_condition','invariants_passed']
    suffix=os.environ.get('ANALYSIS_SEED',''); path=OUT/(f'results_seed{suffix}.csv' if suffix else 'results.csv');done=set()
    if path.exists():
        for r in p.rows(path):done.add(tuple(r[k] for k in fields[:7]))
    f=path.open('a',buffering=1);writer=csv.DictWriter(f,fieldnames=fields)
    if not done:writer.writeheader()
    seedslog=(OUT/(f'rng_seeds_seed{suffix}.jsonl' if suffix else 'rng_seeds.jsonl')).open('a',buffering=1)
    for seed in ([int(suffix)] if suffix else (7,17,29,43,61)):
        entry=p.native._seed_entry(seed);config=entry['config'];p.native.recovery.seed_everything(seed)
        _,loader=p.native.build_dataloaders(config['data'],int(config['training']['batch_size']),seed)
        val=p.native.recovery._materialize_validation_batches(loader,int(config['data']['validation_blocks']))
        assert p.native.recovery._batch_sequence_hash(val)==entry['validation_batch_sequence_hash']
        for variant in ('imp_original','random1_original','random2_original'):
            row=next(r for r in manifest if int(r['seed'])==seed and r['order']=='A' and r['variant']==variant)
            state,masks=p.variant_state(row);arm=variant.replace('_original','')
            model=p.native.TinyMoELanguageModel(config['model']).cuda().eval();model.load_state_dict(state)
            original={n:t.cuda() for n,t in state.items()};masks={n:m.cuda() for n,m in masks.items()}
            mask_reference={n:m.clone() for n,m in masks.items()}; coords={n:m.flatten().nonzero().flatten() for n,m in masks.items()}
            values={n:original[n].flatten()[coords[n]] for n in masks}
            p.save(OUT/f'zero_survivors_seed{seed}_{arm}.json',{n:int((v==0).sum()) for n,v in values.items()}); total_energy=sum(float(v.double().square().sum()) for v in values.values())
            base,routes=evaluate(model,val)
            expected=float(next(r for r in baseline if int(r['seed'])==seed and r['mask_type']==variant)['CE']);assert abs(base-expected)<=1e-3
            cases=[(c,d,'all','both') for c in ('A','B') for d in (.1,.25,.5,1.)]+[(c,1.,'all','both') for c in ('C','D','F')]
            cases += [(c,1.,str(l),'both') for c in ('A','B','D') for l in range(config['model']['num_layers'])]
            cases += [(c,1.,'all',proj) for c in ('A','B','D') for proj in ('fc1','fc2')]
            for condition,dose,layer,projection in cases:
                for draw in range(1,6):
                    key=(str(seed),arm,condition,str(dose),str(draw),layer,projection)
                    kinds=[condition]+(['E_'+condition] if condition in ('A','B') and layer=='all' and projection=='both' else [])
                    if all(tuple([str(seed),arm,k,str(dose),str(draw),layer,projection]) in done for k in kinds):continue
                    model.load_state_dict(original);modified={};dist=0.;flips=0;survivors=sum(v.numel() for v in values.values());rngmap={}
                    for n,v in values.items():
                        if layer!='all' and n.split('.')[1]!=layer:continue
                        if projection!='both' and f'.{projection}.' not in n:continue
                        g,s=rng(seed,arm,condition,dose,draw,n);rngmap[n]=s
                        z=transform(v,condition,dose,g);modified[n]=z
                        dist+=float((z.double()-v.double()).square().sum());flips+=int((z.sign()!=v.sign()).sum())
                        target=model.state_dict()[n];target.flatten()[coords[n]]=z
                        assert torch.count_nonzero(target[~masks[n]])==0
                        assert torch.equal(masks[n],mask_reference[n]); assert torch.equal(masks[n].sum(0),mask_reference[n].sum(0))
                        assert torch.equal(masks[n].sum(1),mask_reference[n].sum(1))
                    seedslog.write(json.dumps({'key':key,'tensor_seeds':rngmap})+'\n')
                    ce,route=evaluate(model,val,routes)
                    record=dict(zip(fields[:7],key));record.update(realized_distance=math.sqrt(dist/total_energy),frac_sign_flipped=flips/survivors,CE=ce,delta_CE=ce-base,routing_change=route,rng_key='|'.join(key),matched_condition='',invariants_passed=True)
                    if condition=='F':assert ce==base and route==0
                    if key not in done:writer.writerow(record);done.add(key)
                    if len(kinds)>1:
                        model.load_state_dict(original);noise_dist=0.;noise_flips=0;noise_rng={}
                        for n,z in modified.items():
                            v=values[n];g,s=rng(seed,arm,'E_'+condition,dose,draw,n);noise_rng[n]=s
                            noise=torch.randn(v.shape,device='cuda',generator=g)
                            target_dist=(z.double()-v.double()).norm();noise=noise*(target_dist/noise.double().norm()).float();nz=v+noise
                            noise_dist+=float((nz.double()-v.double()).square().sum());noise_flips+=int((nz.sign()!=v.sign()).sum())
                            assert torch.allclose((nz.double()-v.double()).norm(),target_dist,rtol=1e-6,atol=1e-7)
                            model.state_dict()[n].flatten()[coords[n]]=nz
                            assert torch.count_nonzero(model.state_dict()[n][~masks[n]])==0
                        ce,route=evaluate(model,val,routes)
                        nk=(str(seed),arm,'E_'+condition,str(dose),str(draw),layer,projection)
                        seedslog.write(json.dumps({'key':nk,'tensor_seeds':noise_rng})+'\n')
                        if nk not in done:
                            record.update(condition='E_'+condition,realized_distance=math.sqrt(noise_dist/total_energy),frac_sign_flipped=noise_flips/survivors,CE=ce,delta_CE=ce-base,routing_change=route,rng_key='|'.join(nk),matched_condition=condition)
                            writer.writerow(record);done.add(nk)
                    print(seed,arm,condition,dose,draw,layer,projection,flush=True)
            del model,original,state,masks,values;torch.cuda.empty_cache()
    f.close();seedslog.close();p.save(OUT/(f'completion_seed{suffix}.json' if suffix else 'completion.json'),{'complete':True,'rows':len(done)})
if __name__=='__main__':main()
