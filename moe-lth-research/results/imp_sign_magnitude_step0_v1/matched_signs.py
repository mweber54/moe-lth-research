"""Additional B doses selected solely by weight distance, never by CE."""
import importlib.util,os,math,json,csv
from pathlib import Path
spec=importlib.util.spec_from_file_location('analysis',Path(__file__).with_name('analyze.py'));a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
torch=a.torch;p=a.p;OUT=a.OUT

def main():
    seed=int(os.environ['ANALYSIS_SEED']); assert (OUT/f'completion_seed{seed}.json').exists()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    entry=p.native._seed_entry(seed);config=entry['config'];p.native.recovery.seed_everything(seed)
    _,loader=p.native.build_dataloaders(config['data'],int(config['training']['batch_size']),seed)
    val=p.native.recovery._materialize_validation_batches(loader,int(config['data']['validation_blocks']))
    assert p.native.recovery._batch_sequence_hash(val)==entry['validation_batch_sequence_hash']
    manifest=p.rows(p.OUT/'variant_manifest.csv'); source=p.rows(OUT/f'results_seed{seed}.csv')
    path=OUT/f'matched_seed{seed}.csv';existing=p.rows(path) if path.exists() else [];done={(r['mask_type'],r['target_p'],r['draw']) for r in existing}
    log=(OUT/f'matched_rng_seed{seed}.jsonl').open('a',buffering=1)
    f=path.open('a',buffering=1); fields=list(source[0])+['target_p','target_distance','distance_mismatch','selected_fraction'];w=csv.DictWriter(f,fieldnames=fields)
    if not existing:w.writeheader()
    for variant in ('imp_original','random1_original','random2_original'):
        arm=variant.replace('_original','');row=next(r for r in manifest if int(r['seed'])==seed and r['order']=='A' and r['variant']==variant)
        state,masks=p.variant_state(row);model=p.native.TinyMoELanguageModel(config['model']).cuda().eval();model.load_state_dict(state)
        original={n:t.cuda() for n,t in state.items()};masks={n:m.cuda() for n,m in masks.items()};mask_reference={n:m.clone() for n,m in masks.items()};coords={n:m.flatten().nonzero().flatten() for n,m in masks.items()};values={n:original[n].flatten()[coords[n]] for n in masks}
        energy=sum(float(v.double().square().sum()) for v in values.values());num=sum(v.numel() for v in values.values());base,routes=a.evaluate(model,val)
        for dose in (.1,.25,.5,1.):
            for draw in range(1,6):
                if (arm,str(dose),str(draw)) in done:continue
                model.load_state_dict(original);distance=0.;flips=0;seedmap={}
                for n,v in values.items():
                    g,_=a.rng(seed,arm,'A',dose,draw,n);az=a.transform(v,'A',dose,g);target=(az.double()-v.double()).square().sum()
                    g,s=a.rng(seed,arm,'B_matched',dose,draw,n);seedmap[n]=s
                    pos=torch.where(v>0)[0];neg=torch.where(v<0)[0]
                    pos=pos[torch.randperm(len(pos),device='cuda',generator=g)];neg=neg[torch.randperm(len(neg),device='cuda',generator=g)];num_pairs=min(len(pos),len(neg));pos=pos[:num_pairs];neg=neg[:num_pairs]
                    costs=4*(v[pos].double().square()+v[neg].double().square());cumulative=torch.cat((costs.new_zeros(1),costs.cumsum(0)))
                    k=int((cumulative-target).abs().argmin());z=v.clone();z[pos[:k]]*=-1;z[neg[:k]]*=-1
                    assert torch.equal(z.abs(),v.abs());assert torch.equal(z.sign().sort().values,v.sign().sort().values)
                    assert torch.allclose(z.double().square().sum(),v.double().square().sum(),rtol=1e-12,atol=1e-12)
                    distance+=float((z.double()-v.double()).square().sum());flips+=2*k
                    t=model.state_dict()[n];t.flatten()[coords[n]]=z
                    assert torch.count_nonzero(t[~masks[n]])==0
                    assert torch.equal(masks[n],mask_reference[n]);assert torch.equal(masks[n].sum(0),mask_reference[n].sum(0));assert torch.equal(masks[n].sum(1),mask_reference[n].sum(1))
                ce,route=a.evaluate(model,val,routes);targetrow=next(r for r in source if r['mask_type']==arm and r['condition']=='A' and r['p']==str(dose) and r['draw']==str(draw) and r['layer']=='all' and r['projection']=='both')
                dist=math.sqrt(distance/energy);td=float(targetrow['realized_distance']);key=f'{seed}|{arm}|B_matched|{dose}|{draw}'
                w.writerow(dict(seed=seed,mask_type=arm,condition='B_matched',p=flips/num,draw=draw,layer='all',projection='both',realized_distance=dist,frac_sign_flipped=flips/num,CE=ce,delta_CE=ce-base,routing_change=route,rng_key=key,matched_condition='A',invariants_passed=True,target_p=dose,target_distance=td,distance_mismatch=dist-td,selected_fraction=flips/num))
                log.write(json.dumps({'key':key,'tensor_seeds':seedmap})+'\n'); print(seed,arm,dose,draw,dist-td,flush=True)
        del model,original,state,masks,values;torch.cuda.empty_cache()
    f.close();log.close();p.save(OUT/f'matched_completion_seed{seed}.json',{'complete':True})
if __name__=='__main__':main()
