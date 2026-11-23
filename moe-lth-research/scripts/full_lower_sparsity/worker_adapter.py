import sys,socket
from support import *
sys.path.insert(0,str(ROOT/'scripts/nominal_exposure_dataset'))
import worker as W
if __name__=='__main__':
    if '--validate' in sys.argv:
        validate_seal(); print('HOST VERIFIED',socket.getfqdn(),flush=True)
    else:
        cid=sys.argv[sys.argv.index('--condition')+1]
        job=next(r for r in rows(OUT/'manifests/run_manifest.csv') if r['condition_id']==cid)
        assert job['job_type']=='sparse' and job['mask_type']=='learned'
        assert int(float(job['source_attempts']))==int(float(job['recovery_attempts']))==2500
        W.OUT=OUT
        W.source_info=lambda seed,budget: read(OUT/'runtime_sources.json')[job['source_id']]
        try: W.main()
        finally:
            dest=Path(sys.argv[sys.argv.index('--attempt')+1])
            if dest.is_dir() and OUT.resolve() in dest.resolve().parents and W.torch.cuda.is_initialized():
                save(dest/'gpu_memory_peak.json',{'host':socket.getfqdn(),
                    'peak_allocated_mib':W.torch.cuda.max_memory_allocated()/1024**2,
                    'peak_reserved_mib':W.torch.cuda.max_memory_reserved()/1024**2})
