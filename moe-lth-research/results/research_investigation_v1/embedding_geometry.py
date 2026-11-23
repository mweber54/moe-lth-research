"""CPU-only independent geometry audit of all registered rerun E tensors."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,torch
from scipy.linalg import svd
O=Path(__file__).resolve().parent;ROOT=O.parents[1];B=ROOT/'results/high_sparsity_coupling_v1';O.mkdir(exist_ok=True);(O/'E_cache').mkdir(exist_ok=True)
SPECTRAL_K=[8,16,32]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eh(a):return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def spec(a):
 u,s,vh=svd(a,full_matrices=False,check_finite=True,lapack_driver='gesdd');p=s*s/(s@s)
 return (u,s,vh),dict(effective_rank_squared_energy=np.exp(-np.sum(p*np.log(p))),stable_rank=(s@s)/s[0]**2,**{f'top{k}_energy':p[:k].sum() for k in SPECTRAL_K},numerical_rank_1e6=int((s>s[0]*1e-6).sum()))
def normrows(a):return a/np.linalg.norm(a,axis=1,keepdims=True)
def cka(a,b):
 a=a-a.mean(0);b=b-b.mean(0)
 return np.linalg.norm(a.T@b)**2/(np.linalg.norm(a.T@a)*np.linalg.norm(b.T@b))
def comparison(a,b,sa,sb):
 x=normrows(a);y=normrows(b);cos=np.clip((x*y).sum(1),-1,1);angles=np.degrees(np.arccos(cos));g=x@x.T;h=y@y.T;off=~np.eye(len(a),dtype=bool)
 nn=[]
 for mat in [g,h]:
  mat=mat.copy();np.fill_diagonal(mat,-np.inf);nn.append(np.argsort(-mat,axis=1,kind='stable'))
 neighbors=np.array([len(set(na[:5])&set(nb[:5]))/5 for na,nb in zip(*nn)])
 alpha=np.sum(a*b)/np.sum(b*b);scale_res=np.linalg.norm(a-alpha*b)/np.linalg.norm(a)
 aa=a-a.mean(0);bb=b-b.mean(0);cross=svd(bb.T@aa,compute_uv=False,check_finite=False);align=float(cross.sum()/(np.linalg.norm(aa)*np.linalg.norm(bb)))
 z=dict(relative_l2=np.linalg.norm(a-b)/np.linalg.norm(b),rms_ratio=np.linalg.norm(a)/np.linalg.norm(b),cosine_mean=cos.mean(),angle_mean_deg=angles.mean(),angle_median_deg=np.median(angles),angle_p95_deg=np.percentile(angles,95),angle_max_deg=angles.max(),offdiag_cosine_RMSE=np.sqrt(np.mean((g[off]-h[off])**2)),offdiag_Gram_relative_l2=np.linalg.norm(g[off]-h[off])/np.linalg.norm(h[off]),neighbor1_preserved=np.mean(nn[0][:,0]==nn[1][:,0]),neighbor5_fraction_preserved=neighbors.mean(),centered_CKA=cka(a,b),optimal_uniform_scale=alpha,scale_only_relative_residual=scale_res,centered_unit_Procrustes_distance=np.sqrt(max(0,2-2*align)),normalized_spectrum_l2=np.linalg.norm(sa[1]/np.linalg.norm(sa[1])-sb[1]/np.linalg.norm(sb[1])))
 for k in SPECTRAL_K:
  for label,va,vb in [('left_token',sa[0],sb[0]),('right_feature',sa[2].T,sb[2].T)]:
   c=np.clip(svd(va[:,:k].T@vb[:,:k],compute_uv=False,check_finite=False),0,1);theta=np.degrees(np.arccos(c))
   z[f'{label}_k{k}_overlap']=np.mean(c*c);z[f'{label}_k{k}_mean_angle']=theta.mean();z[f'{label}_k{k}_max_angle']=theta.max()
  z[f'cutoff_relative_gap_k{k}']=(sa[1][k-1]-sa[1][k])/sa[1][k-1]
 rows=[dict(token_id=i,angle_deg=angles[i],cosine=cos[i],neighbor1_preserved=nn[0][i,0]==nn[1][i,0],neighbor5_fraction_preserved=neighbors[i],current_neighbor1=int(nn[0][i,0]),reference_neighbor1=int(nn[1][i,0])) for i in range(len(a))]
 return z,rows

def main():
 torch.set_num_threads(1)
 p=pd.read_csv(B/'embedding_weight_sparsity/checkpoint_provenance.csv');p=p[(p.track=='rerun')&(p.role=='registered')].copy();assert len(p)==240
 cfg={'all_registered_rerun_steps':sorted(p.step.unique().tolist()),'n_seeds':5,'dtype':'CPU float64 derived from stored float32','no_training':True,'hash_verification':'Fresh E hash verified against prior audited provenance; full checkpoint file hash is reused from that audit, not freshly streamed here','spectral_rank_definition':'exp(entropy(p)); p=singular_value_squared/sum_squares; uncentered E','stable_rank':'Frobenius norm squared / spectral norm squared','principal_subspaces':SPECTRAL_K,'close_gap_flag':'relative cutoff gap <.01; no single-vector claims','global_rotation_invariant':'token cosine Gram; centered CKA; optimally centered/scaled orthogonal Procrustes residual; left singular subspaces','coordinate_dependent':'raw row angles, right singular subspaces','nulls':'64 iid Gaussian 256x256 matrices, diagnostic random-matrix spectral baseline, not experimental evidence','inputs_sha256':{str(B/'embedding_weight_sparsity/checkpoint_provenance.csv'):sha(B/'embedding_weight_sparsity/checkpoint_provenance.csv')}}
 (O/'geometry_config.json').write_text(json.dumps(cfg,indent=2))
 arrays={};spectra={};records=[]
 for i,r in enumerate(p.itertuples()):
  f=O/'E_cache'/f'{r.E_sha256}.npy'
  if f.exists():a=np.load(f);assert eh(a)==r.E_sha256
  else:
   d=torch.load(r.checkpoint_path,map_location='cpu',weights_only=False,mmap=True);sd=d['model'];assert torch.equal(sd['token_embedding.weight'],sd['lm_head.weight']);a=sd['token_embedding.weight'].numpy().copy();assert eh(a)==r.E_sha256;np.save(f,a)
  if r.E_sha256 not in arrays:
   arrays[r.E_sha256]=a.astype(np.float64);sp,met=spec(arrays[r.E_sha256]);spectra[r.E_sha256]=sp;records.append(dict(E_sha256=r.E_sha256,**met))
  if i%30==0:print(f'geometry E loaded {i+1}/240',flush=True)
 pd.DataFrame(records).to_csv(O/'unique_embedding_spectra.csv',index=False);p.to_csv(O/'geometry_checkpoint_provenance.csv',index=False)
 lookup={(r.seed,r.sparsity,r.step):r for r in p.itertuples()};out=[];rowout=[];svout=[]
 for r in p.itertuples():
  a=arrays[r.E_sha256];sa=spectra[r.E_sha256];meta={k:getattr(r,k) for k in ['seed','sparsity','condition','step','E_sha256']}
  for kind,key in [('initial',(r.seed,0.,0)),('same_step_dense',(r.seed,0.,r.step))]:
   ref=lookup[key];b=arrays[ref.E_sha256];sb=spectra[ref.E_sha256];z,rows=comparison(a,b,sa,sb)
   out.append(dict(**meta,reference=kind,reference_E_sha256=ref.E_sha256,**z));rowout.extend(dict(**meta,reference=kind,**x) for x in rows)
  svout.extend(dict(**meta,index=i,singular_value=s,squared_energy=s*s/np.sum(sa[1]**2)) for i,s in enumerate(sa[1]))
 pd.DataFrame(out).to_csv(O/'embedding_geometry.csv',index=False);pd.DataFrame(rowout).to_csv(O/'token_rotation_neighbors.csv',index=False);pd.DataFrame(svout).to_csv(O/'singular_values.csv',index=False)
 rng=np.random.default_rng(72166);null=[]
 for i in range(64):
  _,m=spec(rng.normal(size=(256,256)));null.append(dict(draw=i,**m))
 pd.DataFrame(null).to_csv(O/'random_matrix_spectral_null.csv',index=False)
 # Known algebraic controls, deliberately distinguished from experiment observations.
 a=arrays[next(iter(arrays))];q=np.linalg.qr(np.random.default_rng(135).normal(size=(256,256)))[0];s0=spec(a)[0];s1=spec(a@q)[0];z,_=comparison(a@q,a,s1,s0)
 assert z['offdiag_cosine_RMSE']<1e-12 and z['centered_CKA']>1-1e-12 and z['angle_mean_deg']>20
 z2,_=comparison(.8*a,a,s0,s0);assert z2['angle_mean_deg']<1e-5 and z2['scale_only_relative_residual']<1e-12
 assert len(arrays)==230
 (O/'geometry_validation.json').write_text(json.dumps({'all_240_E_hashes_verified':True,'unique_E':len(arrays),'global_rotation_preserves_Gram_and_CKA':True,'scale_contraction_does_not_rotate_rows':True,'all_initial_E_equal_within_seed':all(lookup[(s,sp,0)].E_sha256==lookup[(s,0.,0)].E_sha256 for s in [7,17,29,43,61] for sp in [.8,.85])},indent=2))
 print('geometry complete',flush=True)
if __name__=='__main__':main()
