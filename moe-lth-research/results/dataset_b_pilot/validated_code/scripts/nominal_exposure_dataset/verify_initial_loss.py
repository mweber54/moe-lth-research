"""CPU reconstruction checks: preserve Dataset A and do not use the occupied GPU."""
from pilot_common import *
from moe_lth.experiments import run_router_age_recovery as R
from moe_lth.data import build_dataloaders
from moe_lth.pruning.router_age import assemble_router_age_model
import torch.nn.functional as F,time

def main():
 torch.set_num_threads(4);seed=17;info=read(OUT/'manifests/short_sources.json')[str(seed)];cfg=info['config'];device=torch.device('cpu');initial=A.load(info['checkpoints']['0']['path']);mask=torch.load(info['masks']['0.95']['learned_path'],map_location='cpu',weights_only=True);base={n:v*mask[n] if n in mask else v for n,v in initial['model'].items()};model=assemble_router_age_model(cfg['model'],base,info['checkpoints']['100']['path'],mask,device);model.eval();tr,va=build_dataloaders(cfg['data'],128,seed);val=R._materialize_validation_batches(va,32);vhash=R._batch_sequence_hash(val);assert vhash==read(A.OUT/'canonical_data_hashes.json')[str(seed)]['validation']
 x,y=val[0]
 with torch.no_grad():
  z=model(x).logits;manual=-F.log_softmax(z,dim=-1).gather(-1,y.unsqueeze(-1)).squeeze(-1);ce=F.cross_entropy(z.reshape(-1,z.shape[-1]),y.reshape(-1),reduction='sum')/y.numel();difference=abs(float(manual.mean())-float(ce));assert difference<1e-4
  diag={'logit_min':float(z.min()),'logit_max':float(z.max()),'mean_current_token_logit':float(z.gather(-1,x.unsqueeze(-1)).mean()),'mean_next_token_logit':float(z.gather(-1,y.unsqueeze(-1)).mean()),'tied_embedding_output_head':model.lm_head.weight.data_ptr()==model.token_embedding.weight.data_ptr(),'manual_token_mean_nll':float(manual.mean()),'independent_ce_token_mean':float(ce),'reduction_difference':difference}
 observed=R._evaluate_condition_loss(model,val,device,False,1.)
 f=pd.read_csv(AD/'canonical/dataset_a_canonical.csv');r=f[(f.num_experts==32)&(f.seed==seed)&(f.router_age==100)&(f.sparsity==.95)].iloc[0];expected=float(r.sparse_postprune_validation_loss);error=abs(observed-expected)
 # CPU vs historical TF32/CUDA is a numerical diagnostic, not dense matching.
 tolerance=.02
 result={'seed':seed,'experts':32,'router_age':100,'sparsity':.95,'cpu_reconstructed_full_validation_loss':observed,'historical_initial_loss':expected,'absolute_difference':error,'predeclared_cpu_vs_tf32_diagnostic_tolerance':tolerance,'passed':difference<1e-4 and error<tolerance,'interpretation':'Large initial token CE is valid under the tied, unscaled random embedding/output head and R0 shared/expert rewind. No loss rescaling or initialization change. Backend tolerance is not a ticket/dense-match criterion.','validation_batch_hash':vhash,**diag}
 jsave(OUT/'audits/initial_loss_verification.json',result);print(json.dumps(result,indent=2),flush=True);assert result['passed']
if __name__=='__main__':main()
