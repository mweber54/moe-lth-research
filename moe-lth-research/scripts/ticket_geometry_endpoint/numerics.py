"""Endpoint metrics, all input axes in the model's native coordinate system."""
import numpy as np
from scipy.stats import rankdata


def corr(x,y):
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    if len(x)<8 or not np.isfinite(x).all() or not np.isfinite(y).all():return None,None
    if x.std()<=1e-12 or y.std()<=1e-12:return None,None
    p=float(np.corrcoef(x,y)[0,1]);rx,ry=rankdata(x),rankdata(y)
    return p,float(np.corrcoef(rx,ry)[0,1])


def cka(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    x=x-x.mean(0);y=y-y.mean(0)
    den=np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y)
    return float(np.linalg.norm(x.T@y)**2/den) if den else None


def spectrum(w):
    w=np.asarray(w,dtype=float);g=w.T@w;v,u=np.linalg.eigh(g)
    return g,np.maximum(v[::-1],0),u[:,::-1]


def compare(g,h):
    n,m=np.linalg.norm(g),np.linalg.norm(h)
    return dict(gram_distance=float(np.linalg.norm(g/n-h/m)) if n and m else None,
                trace_ratio=float(np.trace(g)/np.trace(h)) if np.trace(h)>0 else None)


def geometry(w,routers,e,refs):
    g,values,basis=spectrum(w);energy=float(values.sum());r=routers[e]-routers.mean(0)
    norm=float(r@r)
    def q(v):
        den=float(v@v)*energy
        return float(v@g@v/den) if den else None
    out=dict(router_energy=q(r),isotropic_energy=1/w.shape[1],weight_energy=energy)
    other=[q(v) for i,v in enumerate(routers-routers.mean(0)) if i!=e];other=[v for v in other if v is not None]
    out['router_energy_above_shuffle']=out['router_energy']-float(np.mean(other)) if other and out['router_energy'] is not None else None
    for label,(h,ev,b) in refs.items():
        out.update({label+'_'+k:v for k,v in compare(g,h).items()})
    for k in (8,16,32):
        if k>=len(values):continue
        # Match historical singular-value cutoff flags, in double precision.
        s=np.sqrt(values);valid=energy>0 and s[k-1]>s[0]*1e-6 and (s[k-1]-s[k])/max(s[k-1],1e-30)>=1e-3
        out[f'valid_rank{k}']=bool(valid)
        out[f'router_alignment{k}']=float(np.square(basis[:,:k].T@r).sum()/norm) if valid and norm else None
        for label,(h,ev,b) in refs.items():
            rs=np.sqrt(ev);rv=ev.sum()>0 and rs[k-1]>rs[0]*1e-6 and (rs[k-1]-rs[k])/max(rs[k-1],1e-30)>=1e-3
            out[f'{label}_overlap{k}']=float(np.square(basis[:,:k].T@b[:,:k]).sum()/k) if valid and rv else None
    return out


def associations(scores,acts,assignments):
    result=[];n,e=scores.shape
    for label,(ids,accepted) in assignments.items():
        for j in range(e):
            mask=np.ones(n,dtype=bool) if ids is None else ((ids==j)&accepted).any(1)
            x,y=scores[mask,j],acts[mask,j];p,s=corr(x,y)
            null=[corr(scores[mask,k],y)[0] for k in range(e) if k!=j]
            # Above-null only when all null pairings and identity are defined.
            above=p-float(np.mean(null)) if p is not None and all(v is not None for v in null) else None
            result.append(dict(selection=label,expert=j,n=int(mask.sum()),pearson=p,spearman=s,above_shuffle=above))
    return result
