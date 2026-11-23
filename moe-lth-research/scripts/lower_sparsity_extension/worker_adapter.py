"""Reuse the parent recovery function without changing its source file."""
import sys, socket
from support import *
sys.path.insert(0,str(ROOT/'scripts/nominal_exposure_dataset'))
import worker as W

def configure():
    W.OUT=OUT
    W.source_info=lambda seed,budget: read(OUT/'runtime_sources.json')[f'source_e32_seed{seed}_T{budget}']

if __name__=='__main__':
    if '--validate' in sys.argv:
        validate_seal(); print('HOST INPUTS VERIFIED',socket.getfqdn(),flush=True)
    else:
        configure()
        try: W.main()
        finally:
            if W.torch.cuda.is_initialized() and '--attempt' in sys.argv:
                dest=Path(sys.argv[sys.argv.index('--attempt')+1])
                if dest.is_dir() and OUT.resolve() in dest.resolve().parents:
                    save(dest/'gpu_memory_peak.json',{'host':socket.getfqdn(),
                        'peak_allocated_mib':W.torch.cuda.max_memory_allocated()/1024**2,
                        'peak_reserved_mib':W.torch.cuda.max_memory_reserved()/1024**2})
