"""Record execution environment without gradients or training."""
from common import *
import platform,socket,subprocess,scipy
save(D/'environment'/f'{socket.gethostname()}.json',{'host':socket.gethostname(),'recorded':time.time(),'python':platform.python_version(),
  'torch':str(torch.__version__),'torch_CUDA':torch.version.cuda,'numpy':np.__version__,'scipy':scipy.__version__,
  'GPU_and_driver':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
  'scoring_mode':'FP32 eval, TF32 disabled; see preserved checks and sealed measure.py',
  'measurement_updates':0,'production_updates':0})
print(socket.gethostname(),torch.__version__,torch.version.cuda,flush=True)
