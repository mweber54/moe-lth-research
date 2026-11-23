"""Bounded native FP32 calibration, streaming statistics and split-half gate."""
from common import *
import argparse,socket,threading,traceback,shutil
from scipy.stats import rankdata


def correlation(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])


def group_metrics(a,b,idx):
    k=int(math.ceil(len(idx)*.25))
    hi0=np.lexsort((idx,-a))[:k];lo0=np.lexsort((idx,a))[:k]
    hi1=np.lexsort((idx,-b))[:k];lo1=np.lexsort((idx,b))[:k]
    h01=float(b[hi0].mean());l01=float(b[lo0].mean())
    h10=float(a[hi1].mean());l10=float(a[lo1].mean())
    rho=correlation(a,b)
    return dict(rho=rho,defined=rho is not None,
       high_overlap=len(np.intersect1d(hi0,hi1))/k,low_overlap=len(np.intersect1d(lo0,lo1))/k,
       selected_half0_H_scored_half1=h01,selected_half0_L_scored_half1=l01,
       selected_half1_H_scored_half0=h10,selected_half1_L_scored_half0=l10,
       cross_separated=bool(h01>0 and h10>0 and h01>=2*l01 and h10>=2*l10))


def reliability(seed,name,w,half0,half1,exposure,rule):
    meta=metadata(seed,name);order=np.argsort(np.abs(w),kind='stable')[:w.size//2]
    rows=[]
    for scope,indices in [('tensor',order)]+[(str(i),idx) for i,idx in enumerate(np.array_split(order,rule['strata']))]:
        rows.append(dict(**meta,scope=scope,coordinates=len(indices),**group_metrics(half0[indices],half1[indices],indices)))
    sub=rows[1:];valid=[r for r in sub if r['defined']]
    def median(key):return float(np.median([r[key] for r in valid])) if valid else None
    traffic=[r for r in exposure if r['layer']==meta['layer'] and r['expert']==meta['expert']]
    totals=[sum(r['accepted_tokens'] for r in traffic if r['half']==h) for h in (0,1)]
    nonempty=[sum(r['accepted_tokens']>0 for r in traffic if r['half']==h) for h in (0,1)]
    enough=sum(totals)>=rule['minimum_accepted_total'] and min(totals)>=rule['minimum_accepted_per_half'] and min(nonempty)>=rule['minimum_nonempty_batches_per_half']
    valid_fraction=len(valid)/len(sub)
    sep_fraction=sum(r['cross_separated'] for r in valid)/len(valid) if valid else 0
    checks={'traffic':enough,'valid_strata':valid_fraction>=rule['minimum_valid_strata_fraction'],
      'rank':median('rho') is not None and median('rho')>=rule['minimum_median_stratum_rho'],
      'high_overlap':median('high_overlap') is not None and median('high_overlap')>=rule['minimum_median_high_overlap'],
      'low_overlap':median('low_overlap') is not None and median('low_overlap')>=rule['minimum_median_low_overlap'],
      'cross_separation':sep_fraction>=rule['minimum_cross_separated_strata_fraction']}
    summary=dict(**meta,accepted_total=sum(totals),accepted_half0=totals[0],accepted_half1=totals[1],
      nonempty_half0=nonempty[0],nonempty_half1=nonempty[1],valid_strata_fraction=valid_fraction,
      median_stratum_rho=median('rho'),median_high_overlap=median('high_overlap'),median_low_overlap=median('low_overlap'),
      cross_separated_strata_fraction=sep_fraction,eligible=all(checks.values()),
      failed_criteria=';'.join(k for k,v in checks.items() if not v),exposure_status='adequate' if enough else 'INSUFFICIENT_NOT_UNIMPORTANT')
    return rows,summary


def recursive_equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(recursive_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(recursive_equal(x,y) for x,y in zip(a,b))
    return a==b


def synthetic_checks():
    # Includes cancellation, an exactly zero coordinate, and nonstationary gradients.
    g=np.array([[1.,0.,-2.],[ -1.,0.,4.],[3.,0.,1.],[-3.,0.,-3.]],dtype=np.float64)
    w=np.array([2.,0.,-1.]);mean=np.zeros(3);m2=np.zeros(3);ab=np.zeros(3)
    for n,x in enumerate(g,1):
        delta=x-mean;mean+=delta/n;m2+=delta*(x-mean);ab+=np.abs(w*x)
    assert np.allclose(mean,g.mean(0),atol=1e-14)
    assert np.allclose(m2/3,g.var(0,ddof=1),atol=1e-14)
    assert np.allclose(m2/4+mean**2,(g*g).mean(0),atol=1e-14)
    assert np.allclose(ab/4,np.abs(g*w).mean(0),atol=1e-14)
    assert ab[0]/4==4 and abs(w[0]*mean[0])==0
    assert correlation(np.zeros(8),np.zeros(8)) is None
    assert correlation(np.arange(8),np.arange(8))==1
    # Explicit tie averaging: constant score strata must not become reliable by coordinate tie breaks.
    m=group_metrics(np.zeros(8),np.zeros(8),np.arange(8));assert not m['defined'] and not m['cross_separated']
    # Full-gradient archive is permitted only on this tiny native-model fixture.
    cfg=dict(vocab_size=8,max_seq_len=4,d_model=8,num_heads=2,num_layers=1,num_experts=2,
             expert_hidden_size=12,top_k=1,capacity_factor=1.25,dropout=0.)
    with protected():
        torch.manual_seed(1961)
        model=TinyMoELanguageModel(cfg).eval()
        ps=[p for n,p in model.named_parameters() if n in names(model.state_dict())]
        full=[];stream=[torch.zeros_like(p,dtype=torch.float64) for p in ps];absstream=[torch.zeros_like(p,dtype=torch.float64) for p in ps]
        for bi in range(4):
            x=torch.randint(0,8,(2,4));y=torch.randint(0,8,(2,4))
            out=model(x);ce=torch.nn.functional.cross_entropy(out.logits.flatten(0,1),y.flatten())
            gs=torch.autograd.grad(ce,ps,allow_unused=True)
            gs=[torch.zeros_like(p,dtype=torch.float64) if g is None else g.double() for p,g in zip(ps,gs)];full.append(gs)
            for j,(p,z) in enumerate(zip(ps,gs)):
                stream[j]+=(z-stream[j])/(bi+1);absstream[j]+=(p.detach()*z).abs()/4
        for j,p in enumerate(ps):
            archived=torch.stack([v[j] for v in full])
            assert torch.allclose(stream[j],archived.mean(0),rtol=1e-12,atol=1e-14)
            assert torch.allclose(absstream[j],(archived*p.detach()).abs().mean(0),rtol=1e-12,atol=1e-14)
    return {'tiny_native_model_full_vs_streamed_gradients':True,'full_vs_streamed_float64':True,'mean_absolute_not_absolute_mean':True,
      'sample_batch_variance_and_second_moment':True,'undefined_constant_ranks':True,
      'zero_sign_consistency_is_undefined':True,'tolerance':1e-14}


def run(seed,attempt):
    reg=frozen();prov=read(D/'checkpoint_and_calibration_provenance.json')['seeds'][str(seed)]
    cp_path=ROOT/prov['checkpoints']['1000']['path'];assert sha(cp_path)==prov['checkpoints']['1000']['sha256']
    cp=load(cp_path);cfg=ref.variant(seed)['config']
    dev=torch.device('cuda');assert torch.cuda.is_available()
    assert torch.cuda.mem_get_info()[0]>=24*1024**3
    assert shutil.disk_usage(ROOT).free>=80*1024**3
    model=TinyMoELanguageModel(cfg['model']).to(dev);model.load_state_dict(cp['model'],strict=True)
    assert state_dict_hash(model.state_dict())==prov['checkpoints']['1000']['model_sha256']
    ns=names(cp['model']);params=dict(model.named_parameters());ps=[params[n] for n in ns]
    assert sum(p.numel() for p in ps)==67108864 and all(p.dtype==torch.float32 for p in ps)
    del cp
    path=ROOT/prov['calibration_path'];assert sha(path)==prov['calibration_sha256'];cal=load(path)
    assert len(cal['batches'])==32
    for b,p in zip(cal['batches'],prov['calibration']):
        assert batch_hash(b)==p['batch_sha256'] and tensor_hash(b[0])==p['tokens_sha256'] and tensor_hash(b[1])==p['targets_sha256']
        assert 1<=p['absolute_step']<=1000 and tuple(b[0].shape)==(128,128)
    # Existing non-None gradients and heterogeneous modes exercise the preservation contract.
    ps[0].grad=torch.full_like(ps[0],.125)
    model.train();model.blocks[0].moe.eval()
    before_hash=state_dict_hash(model.state_dict());before_rng=ref.rng();before_modes=[m.training for m in model.modules()]
    accum={n:dict(mean=torch.zeros_like(p,dtype=torch.float64),m2=torch.zeros_like(p,dtype=torch.float64),
                  abs0=torch.zeros_like(p,dtype=torch.float64),abs1=torch.zeros_like(p,dtype=torch.float64),
                  positive=torch.zeros_like(p,dtype=torch.int16),nonzero=torch.zeros_like(p,dtype=torch.int16)) for n,p in zip(ns,ps)}
    exposure=[];ces=[];route_hashes=[];unused={n:0 for n in ns}
    tf=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    with protected(model):
      try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;model.eval()
        for bi,(x,y) in enumerate(cal['batches']):
            out=model(x.to(dev));loss=torch.nn.functional.cross_entropy(out.logits.flatten(0,1),y.to(dev).flatten())
            assert torch.isfinite(loss);ces.append(float(loss.detach()))
            grads=torch.autograd.grad(loss,ps,allow_unused=True)
            routes=[]
            for layer,tr in enumerate(out.routes):
                ids=tr.selected_expert_indices.detach().flatten();gates=tr.selected_probabilities.detach().flatten();accepted=tr.accepted_mask.detach().flatten()
                routes.append({'ids':tensor_hash(ids),'accepted':tensor_hash(accepted)})
                for expert in range(32):
                    proposed=ids==expert;accept=proposed&accepted;pg=gates[proposed];ag=gates[accept]
                    exposure.append(dict(seed=seed,batch_index=bi,absolute_step=cal['absolute_steps'][bi],half=bi//16,layer=layer,expert=expert,
                      proposed_tokens=int(proposed.sum()),accepted_tokens=int(accept.sum()),dropped_tokens=int((proposed&~accepted).sum()),
                      proposed_gate_sum=float(pg.sum()),proposed_gate_mean=float(pg.mean()) if pg.numel() else None,
                      proposed_gate_min=float(pg.min()) if pg.numel() else None,proposed_gate_max=float(pg.max()) if pg.numel() else None,
                      accepted_gate_sum=float(ag.sum()),accepted_gate_mean=float(ag.mean()) if ag.numel() else None,
                      accepted_gate_min=float(ag.min()) if ag.numel() else None,accepted_gate_max=float(ag.max()) if ag.numel() else None))
            route_hashes.append(routes)
            for n,p,g in zip(ns,ps,grads):
                a=accum[n]
                if g is None:unused[n]+=1;g=torch.zeros_like(p)
                assert torch.isfinite(g).all()
                z=g.double();delta=z-a['mean'];a['mean'].add_(delta/(bi+1));a['m2'].add_(delta*(z-a['mean']))
                wg=p.detach().double()*z;a['abs'+str(bi//16)].add_(wg.abs())
                a['positive'].add_((wg>0).to(torch.int16));a['nonzero'].add_((wg!=0).to(torch.int16))
            print('Calibration',seed,bi+1,'/32',flush=True)
            save(attempt/'progress.json',{'seed':seed,'batch':bi+1,'time':time.time(),'host':socket.gethostname()})
            del grads,out,loss,g,z,wg,delta
      finally:
        torch.backends.cuda.matmul.allow_tf32=tf[0];torch.backends.cudnn.allow_tf32=tf[1]
    assert state_dict_hash(model.state_dict())==before_hash
    assert recursive_equal(before_rng,ref.rng()) and before_modes==[m.training for m in model.modules()]
    assert torch.all(ps[0].grad==.125)
    assert (torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)==tf
    rel=[];summaries=[];score_files={};zero_counts={}
    for i,(n,p) in enumerate(zip(ns,ps)):
        a=accum.pop(n);w=p.detach().double();a0=a.pop('abs0')/16;a1=a.pop('abs1')/16
        total=(a0+a1)/2;wgmean=w*a['mean'];nz=a.pop('nonzero');pos=a.pop('positive')
        consistency=torch.where(nz>0,torch.maximum(pos,nz-pos).double()/nz.double(),torch.nan)
        stats={'A':total.cpu(),'A_half0':a0.cpu(),'A_half1':a1.cpu(),'S':(-wgmean).cpu(),
          'mean_gradient':a['mean'].cpu(),'gradient_second_moment':(a['m2']/32+a['mean']**2).cpu(),
          'gradient_sample_variance':(a['m2']/31).cpu(),'absolute_mean_wg':wgmean.abs().cpu(),
          'sign_consistency':consistency.cpu(),'nonzero_batches':nz.cpu(),'unused_expert_batches':unused[n]}
        assert all(torch.isfinite(stats[k]).all() for k in stats if torch.is_tensor(stats[k]) and k!='sign_consistency')
        assert torch.all(total*(1+1e-12)+1e-25>=wgmean.abs())  # independent FP64 sums: registered engineering roundoff correction
        rs,summary=reliability(seed,n,p.detach().cpu().numpy().reshape(-1),stats['A_half0'].numpy().reshape(-1),stats['A_half1'].numpy().reshape(-1),exposure,reg['reliability'])
        rel.extend(rs);summaries.append(summary)
        path=D/'scores'/f'seed{seed}'/(n+'.pt');atomic_tensor(path,stats);path.chmod(0o444)
        score_files[str(path.relative_to(D))]=sha(path)
        zero_counts[n]={'A_zero_coordinates':int((total==0).sum()),'sign_undefined_coordinates':int((nz==0).sum()),'unused_expert_batches':unused[n]}
        if (i+1)%32==0:print('Reliability tensors',seed,i+1,'/256',flush=True)
        del a,a0,a1,total,wgmean,w,nz,pos,consistency,stats
    table(attempt/'score_reliability.csv',rel);table(attempt/'tensor_reliability_summary.csv',summaries);table(attempt/'exposure_summary.csv',exposure)
    save(attempt/'measurement_checks.json',{'weight_hash_preserved':True,'RNG_preserved':True,'module_modes_preserved':True,'existing_gradients_preserved':True,'TF32_flags_preserved':True,
      'calibration_batch_token_target_hashes_verified':True,'coordinate_schema_verified':True,'score_files':score_files,
      'zero_and_missing_exposure':zero_counts,'baseline_calibration_mean_CE':float(np.mean(ces)),
      'baseline_calibration_batch_CE':ces,'baseline_route_hashes':route_hashes,'synthetic_checks':synthetic_checks(),
      'no_intervention_loss_or_validation_or_training':True})
    return {'seed':seed,'eligible_tensors':sum(r['eligible'] for r in summaries),'total_tensors':len(summaries),'score_files':len(score_files)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True,choices=SEEDS);args=ap.parse_args()
    frozen();host=socket.gethostname();worker_lock=lock(D/'locks'/f'measure_{host}.lock',blocking=False);seed_lock=lock(D/'locks'/f'seed{args.seed}.lock',blocking=False)
    status=D/'status'/f'measure_seed{args.seed}.json'
    assert not status.exists() or read(status)['status']!='COMPLETE','Measurement already complete'
    base=D/'attempts'/f'measure_seed{args.seed}';base.mkdir(parents=True,exist_ok=True)
    i=1
    while (base/f'attempt_{i:04d}').exists():i+=1
    attempt=base/f'attempt_{i:04d}';attempt.mkdir()
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(15):save(attempt/'heartbeat.json',{'host':host,'pid':os.getpid(),'time':time.time()})
    threading.Thread(target=heartbeat,daemon=True).start()
    save(status,{'status':'RUNNING','attempt':str(attempt.relative_to(D)),'host':host,'pid':os.getpid(),'started':time.time()})
    try:
        result=run(args.seed,attempt)
        save(attempt/'completion.json',dict(status='COMPLETE',**result))
        save(status,dict(status='COMPLETE',attempt=str(attempt.relative_to(D)),host=host,**result))
    except BaseException:
        error=traceback.format_exc();(attempt/'failure.txt').write_text(error)
        save(status,{'status':'FAILED','attempt':str(attempt.relative_to(D)),'host':host,'error':error});raise
    finally:stop.set()
if __name__=='__main__':main()
