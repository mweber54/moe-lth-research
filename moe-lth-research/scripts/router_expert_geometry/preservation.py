"""Whole-callback audit, used only in validation, never in production."""
from gcommon import *
import numpy as np

def digest(value):
 h=hashlib.sha256()
 def visit(x):
  if isinstance(x,torch.Tensor):
   t=x.detach().cpu().contiguous();h.update(str((t.dtype,tuple(t.shape))).encode());h.update(t.numpy().tobytes())
  elif isinstance(x,np.ndarray):
   h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
  elif isinstance(x,dict):
   for k in sorted(x,key=repr):h.update(repr(k).encode());visit(x[k])
  elif isinstance(x,(tuple,list)):
   for v in x:visit(v)
  else:h.update(repr(x).encode())
 visit(value);return h.hexdigest()
def capture(model,optimizer,scaler,batches):
 return dict(tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,cuda_autocast=torch.is_autocast_enabled('cuda'),cpu_autocast=torch.is_autocast_enabled('cpu'),parameters_buffers=digest(model.state_dict()),gradients=digest({n:p.grad for n,p in model.named_parameters()}),optimizer=digest(optimizer.state_dict()),amp=digest(scaler.state_dict()),module_modes=[m.training for m in model.modules()],cpu_rng=digest(torch.get_rng_state()),cuda_rng=digest(torch.cuda.get_rng_state_all()),python_rng=digest(random.getstate()),numpy_rng=digest(np.random.get_state()),training_batches=digest(batches))
