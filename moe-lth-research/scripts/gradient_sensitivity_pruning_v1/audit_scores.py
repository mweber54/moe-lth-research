"""Independent algebra/roundoff audit of retained immutable score arrays."""
from common import *


def main():
    frozen();prov=read(D/'checkpoint_and_calibration_provenance.json');results=[]
    for seed in SEEDS:
        cp=prov['seeds'][str(seed)]['checkpoints']['1000'];assert sha(ROOT/cp['path'])==cp['sha256'];state=load(ROOT/cp['path'])['model']
        maximum_relative_jensen=0.;maximum_absolute_jensen=0.;coordinates=0;undefined=0
        for n in names(state):
            path=D/'scores'/f'seed{seed}'/(n+'.pt');s=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            w=state[n].double();assert tuple(w.shape)==tuple(s['A'].shape)
            assert torch.equal(s['A'],(s['A_half0']+s['A_half1'])/2)
            assert torch.equal(s['S'].abs(),s['absolute_mean_wg'])
            assert torch.equal(s['S'],-w*s['mean_gradient'])
            assert torch.allclose(s['gradient_second_moment'],s['gradient_sample_variance']*(31/32)+s['mean_gradient']**2,rtol=1e-12,atol=1e-25)
            assert torch.all(s['gradient_sample_variance']>=0)
            a=s['A'];am=s['absolute_mean_wg'];violation=torch.clamp(am-a,min=0)
            maximum_absolute_jensen=max(maximum_absolute_jensen,float(violation.max()))
            maximum_relative_jensen=max(maximum_relative_jensen,float((violation/torch.clamp(a,min=1e-300)).max()))
            assert torch.all(am<=a*(1+1e-12)+1e-25)
            nz=s['nonzero_batches'];sc=s['sign_consistency'];assert torch.all((nz>=0)&(nz<=32))
            assert torch.all(torch.isnan(sc[nz==0]));assert torch.all((sc[nz>0]>=.5)&(sc[nz>0]<=1))
            coordinates+=a.numel();undefined+=int((nz==0).sum())
        results.append(dict(seed=seed,coordinates=coordinates,undefined_sign_coordinates=undefined,
          max_Jensen_absolute_roundoff=maximum_absolute_jensen,max_Jensen_relative_roundoff=maximum_relative_jensen,
          half_averages_exact=True,signed_score_identity_exact=True,absolute_signed_mean_identity_exact=True,
          second_moment_sample_variance_identity_passed=True,sign_consistency_definedness_passed=True))
        print(results[-1],flush=True)
    save(D/'score_algebra_audit.json',{'passed':True,'seeds':results,'intervention_losses':0,'tolerances':{'relative':1e-12,'absolute':1e-25},'implementation_sha256':sha(Path(__file__))})
if __name__=='__main__':main()
