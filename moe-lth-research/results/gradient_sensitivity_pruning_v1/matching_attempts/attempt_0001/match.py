"""Registered row pairs and directed-column-cycle matching; never calls model forward."""
from common import *
import csv,concurrent.futures
from scipy.stats import ks_2samp


def cycles_from_pairs(pairs,columns,target):
    # Deterministic DFS; cycle deletion cannot create a cycle in an exhausted vertex.
    adjacency=[[] for _ in range(columns)]
    for i,p in enumerate(pairs):adjacency[p['hc']].append(i)
    for edges in adjacency:edges.sort(key=lambda i:(pairs[i]['cost'],pairs[i]['h'],pairs[i]['l']))
    used=set();black=set();chosen=[];cycle_lengths=[]
    sys.setrecursionlimit(10000)
    while True:
        color={v:2 for v in black};stack=[];tree=[];position={};found=None
        def dfs(v):
            nonlocal found
            color[v]=1;position[v]=len(stack);stack.append(v)
            for ei in adjacency[v]:
                if ei in used:continue
                to=pairs[ei]['lc'];c=color.get(to,0)
                if c==1:
                    found=tree[position[to]:]+[ei];return True
                if c==0:
                    tree.append(ei)
                    if dfs(to):return True
                    tree.pop()
            stack.pop();position.pop(v);color[v]=2;black.add(v)
            return False
        for v in range(columns):
            if color.get(v,0)==0 and dfs(v):break
        if found is None:break
        used.update(found);cycle_lengths.append(len(found))
        if len(chosen)+len(found)<=target:chosen.extend(found)
        if len(chosen)==target:break
    return [pairs[i] for i in chosen],cycle_lengths


def balance(a,b):
    sd=float(np.sqrt((np.var(a)+np.var(b))/2));diff=float(np.mean(a)-np.mean(b))
    return {'H_mean':float(np.mean(a)),'L_mean':float(np.mean(b)),
            'SMD':diff/sd if sd else (0. if diff==0 else None),'KS':float(ks_2samp(a,b).statistic)}


def tensor_match(task):
    seed,name,values,A=task;reg=read(D/'registration.json')['matching'];meta=metadata(seed,name)
    shape=values.shape[1:];n=int(np.prod(shape));v=values.reshape(3,-1).astype(np.float64)
    mag=np.abs(v);rms=float(np.sqrt(np.mean(mag[2]**2)));assert rms>0
    t=np.array([250.,500.,1000.]);t-=t.mean()
    slope=(t[:,None]*mag).sum(0)/np.dot(t,t)*750/rms
    variation=np.abs(np.diff(mag,axis=0)).sum(0)/rms
    features=np.stack([mag[2]/rms,slope,variation],axis=1)
    calipers=np.array([.005,.1,.1]);small=np.argsort(mag[2],kind='stable')[:n//2]
    strata=np.empty(n,dtype=np.int16);strata.fill(-1)
    for si,idx in enumerate(np.array_split(small,50)):strata[idx]=si
    sign=np.sign(v[2]).astype(np.int8);row=small//shape[1]
    # Contiguous blocks ordered by row, magnitude stratum, sign; high score ordering is local.
    order=np.lexsort((small,sign[small],strata[small],row));idx=small[order]
    key=(idx//shape[1])*150+strata[idx]*3+(sign[idx]+1)
    splits=np.flatnonzero(np.diff(key))+1;pairs=[]
    score=np.asarray(A).reshape(-1)
    for group in np.split(idx,splits):
        if len(group)<2:continue
        group=group[np.lexsort((group,-score[group]))];available=np.ones(len(group),dtype=bool)
        for j,h in enumerate(group):
            if not available[j] or score[h]<=0:continue
            candidates=np.flatnonzero(available & (score[group]<score[h]) & (score[h]>=2*score[group]))
            if not len(candidates):continue
            diff=np.abs(features[group[candidates]]-features[h]);valid=np.all(diff<=calipers,axis=1)
            candidates=candidates[valid];diff=diff[valid]
            if not len(candidates):continue
            costs=np.sum((diff/calipers)**2,axis=1);best=np.lexsort((group[candidates],costs))[0]
            q=int(candidates[best]);l=int(group[q]);available[j]=available[q]=False
            pairs.append({'h':int(h),'l':l,'hc':int(h%shape[1]),'lc':int(l%shape[1]),'cost':float(costs[best])})
    selected,lengths=cycles_from_pairs(pairs,shape[1],2621)
    h=np.array([p['h'] for p in selected],dtype=np.int64);l=np.array([p['l'] for p in selected],dtype=np.int64);k=len(h)
    assert len(np.unique(h))==k and len(np.unique(l))==k and not np.intersect1d(h,l).size
    assert np.all(strata[h]>=0) and np.all(strata[h]==strata[l]) and np.all(sign[h]==sign[l])
    rh=np.bincount(h//shape[1],minlength=shape[0]);rl=np.bincount(l//shape[1],minlength=shape[0]);ch=np.bincount(h%shape[1],minlength=shape[1]);cl=np.bincount(l%shape[1],minlength=shape[1])
    assert np.array_equal(rh,rl) and np.array_equal(ch,cl)
    b={};ok=k>=656
    if k:
        for j,feature in enumerate(('magnitude','slope','variation')):
            b[feature]=balance(features[h,j],features[l,j]);d=b[feature]
            ok=ok and d['SMD'] is not None and abs(d['SMD'])<=.05 and d['KS']<=.05
        eh=float(np.dot(v[2,h],v[2,h]));el=float(np.dot(v[2,l],v[2,l]));energy=abs(eh-el)/max(eh,el,1e-300)
        ah=float(score[h].mean());al=float(score[l].mean());separation=ah>0 and ah>=2*al
        ok=ok and energy<=.005 and separation
    else:eh=el=ah=al=energy=None;separation=False
    summary=dict(**meta,candidate_pairs=len(pairs),cycle_pairs=k,cycles_examined=len(lengths),
      dose_fraction=k/n,included=bool(ok),excluded_reason='' if ok else ('insufficient_cycles' if k<656 else 'registered_balance_failure'),
      mean_A_H=ah,mean_A_L=al,A_separation_pass=separation,removed_squared_H=eh,removed_squared_L=el,relative_energy_difference=energy)
    b_rows=[dict(**meta,feature=feature,**d) for feature,d in b.items()]
    structural=dict(**meta,pairs=k,included=bool(ok),max_row_count_difference=int(np.abs(rh-rl).max()),max_column_count_difference=int(np.abs(ch-cl).max()),
      row_L1=0,column_L1=0,row_occupancy=float(np.mean(rh>0)),column_occupancy=float(np.mean(ch>0)),
      row_HHI=float(np.sum((rh/k)**2)) if k else None,column_HHI=float(np.sum((ch/k)**2)) if k else None,
      max_row_share=float(rh.max()/k) if k else None,max_column_share=float(ch.max()/k) if k else None,
      exact_sign_counts=True,connectivity_fully_controlled=False)
    return name,h,l,summary,b_rows,structural,{'rows_H':torch.from_numpy(rh),'rows_L':torch.from_numpy(rl),'columns_H':torch.from_numpy(ch),'columns_L':torch.from_numpy(cl)}


def main():
    frozen();gate=read(D/'reliability_review.json');assert gate['status']=='PASS'
    assert not (D/'matching_review.json').exists()
    save(D/'matching_implementation_seal.json',{'registration_sha256':sha(D/'registration.json'),'implementation_sha256':sha(Path(__file__)),'stage':'BEFORE_MATCHING_AND_INTERVENTION_LOSSES','energy_relative_difference':'abs(E_H-E_L)/max(E_H,E_L,1e-300)','SMD':'(mean_H-mean_L)/sqrt((population_variance_H+population_variance_L)/2); identical constants=0, unequal constants undefined/fail'})
    (D/'matching_implementation_seal.json').chmod(0o444)
    summaries=[];balances=[];structures=[];cover=[];candidate_files={}
    provenance=read(D/'checkpoint_and_calibration_provenance.json')
    for seed in SEEDS:
        states=[]
        for step in (250,500,1000):
            c=provenance['seeds'][str(seed)]['checkpoints'][str(step)];assert sha(ROOT/c['path'])==c['sha256'];states.append(load(ROOT/c['path'])['model'])
        st=read(D/'status'/f'measure_seed{seed}.json');p=D/st['attempt'];checks=read(p/'measurement_checks.json')
        with (p/'tensor_reliability_summary.csv').open() as f:eligible={r['parameter'] for r in csv.DictReader(f) if r['eligible']=='True'}
        def tasks():
            for n in names(states[0]):
                if n not in eligible:continue
                path=D/'scores'/f'seed{seed}'/(n+'.pt');assert sha(path)==checks['score_files'][str(path.relative_to(D))]
                yield seed,n,np.stack([state[n].numpy() for state in states]),load(path)['A'].numpy()
        masks={'H':{},'L':{}};all_pairs={'H':{},'L':{}};degrees={};local=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
            # Bounded outstanding tensor tasks: avoid loading the entire score archive into RAM.
            iterator=iter(tasks());pending=[]
            for _ in range(8):
                try:pending.append(pool.submit(tensor_match,next(iterator)))
                except StopIteration:break
            done=0
            while pending:
                future=pending.pop(0);n,h,l,summary,bs,struct,degree=future.result();done+=1
                summaries.append(summary);balances+=bs;structures.append(struct);local.append(summary);degrees[n]=degree
                all_pairs['H'][n]=torch.from_numpy(h);all_pairs['L'][n]=torch.from_numpy(l)
                if summary['included']:masks['H'][n]=torch.from_numpy(h);masks['L'][n]=torch.from_numpy(l)
                if done%32==0:print('Matched',seed,done,'/256',flush=True)
                save(D/'matching_progress.json',{'seed':seed,'tensors':done,'time':time.time()})
                try:pending.append(pool.submit(tensor_match,next(iterator)))
                except StopIteration:pass
        count=sum(x.numel() for x in masks['H'].values());included=len(masks['H'])
        strata=[dict(layer=layer,projection=proj,included=sum(x['included'] for x in local if x['layer']==layer and x['projection']==proj)) for layer in range(4) for proj in ('fc1','fc2')]
        passed=included>=math.ceil(.8*256) and count/67108864>=.005 and all(x['included']>=26 for x in strata)
        cover.append(dict(seed=seed,included_tensors=included,total_tensors=256,removed_per_branch=count,overall_dose=count/67108864,passed=bool(passed),layer_projection=strata))
        p=D/'candidate_masks'/f'seed{seed}.pt';atomic_tensor(p,masks);p.chmod(0o444);candidate_files[str(p.relative_to(D))]=sha(p)
        atomic_tensor(D/'degree_distributions'/f'seed{seed}.pt',degrees)
        q=D/'candidate_coordinate_lists'/f'seed{seed}.pt';atomic_tensor(q,all_pairs);q.chmod(0o444);candidate_files[str(q.relative_to(D))]=sha(q)
        table(D/'matched_group_summary.csv',summaries);table(D/'magnitude_growth_balance.csv',balances);table(D/'structural_balance.csv',structures)
        print('Matching coverage',cover[-1],flush=True)
    passed=all(x['passed'] for x in cover)
    if passed:
        # Only a complete five-source accepted design becomes intervention masks.
        import shutil
        for seed in SEEDS:
            p=D/'masks'/f'seed{seed}.pt';p.parent.mkdir(exist_ok=True);shutil.copyfile(D/'candidate_masks'/p.name,p);p.chmod(0o444)
        save(D/'mask_seal.json',{'registration_sha256':sha(D/'registration.json'),'masks':{str(seed):{'path':f'masks/seed{seed}.pt','sha256':sha(D/'masks'/f'seed{seed}.pt')} for seed in SEEDS},'before_any_intervention_losses':True})
        (D/'mask_seal.json').chmod(0o444)
    save(D/'matching_review.json',{'status':'PASS' if passed else 'STOP_MATCHING','coverage':cover,'candidate_coordinate_file_sha256':candidate_files,'intervention_losses_measured':0})
    print('PASS' if passed else 'STOP_MATCHING',flush=True)
if __name__=='__main__':main()
