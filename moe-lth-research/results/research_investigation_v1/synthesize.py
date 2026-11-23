"""Reproduce seed summaries, falsification figures, and final research recommendation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;ROOT=O.parents[1];B=ROOT/'results/high_sparsity_coupling_v1';F=O/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
colors={0.:'black',.8:'#0072B2',.85:'#D55E00'};seeds=[7,17,29,43,61];seedcolors=dict(zip(seeds,plt.cm.tab10.colors))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(F/f'{name}.{ext}',dpi=220,bbox_inches='tight')
 plt.close(fig)
def write(name,s):(O/name).write_text(s)
G=pd.read_csv(O/'embedding_geometry.csv');P=pd.read_csv(O/'geometry_checkpoint_provenance.csv');S=P.merge(pd.read_csv(O/'unique_embedding_spectra.csv'),on='E_sha256');N=pd.read_csv(O/'random_matrix_spectral_null.csv')
A=pd.read_csv(O/'gate_contrasts_reproduced.csv');C=pd.read_csv(O/'coupling_from_raw.csv')
END=G[G.step==5000];D=G[(G.reference=='initial')&(G.sparsity==0)]
paired=G[(G.reference=='initial')&(G.sparsity>0)].merge(D,on=['seed','step'],suffixes=('_sparse','_dense'),validate='many_to_one')
metrics=['angle_mean_deg','angle_median_deg','offdiag_cosine_RMSE','neighbor1_preserved','neighbor5_fraction_preserved','centered_CKA','normalized_spectrum_l2','centered_unit_Procrustes_distance']
for m in metrics:paired[m+'_difference']=paired[m+'_sparse']-paired[m+'_dense']
paired.to_csv(O/'embedding_paired_changes.csv',index=False)
rows=[]
for sp in [.8,.85]:
 for m in metrics:
  v=paired[(paired.sparsity_sparse==sp)&(paired.step==5000)][m+'_difference'];rows.append(dict(sparsity=sp,metric=m,n=5,mean=v.mean(),seed_sd=v.std(ddof=1),minimum=v.min(),maximum=v.max(),positive_seeds=int((v>0).sum()),negative_seeds=int((v<0).sum())))
pd.DataFrame(rows).to_csv(O/'embedding_endpoint_seed_summary.csv',index=False)
# Performance tests are all exported, including undefined/unstable cases, not searched for a preferred result.
tickets=pd.read_csv(B/'report/rerun_tickets.csv');perf=[]
for sp in [.8,.85]:
 for step in sorted(set(G.step)-{0}):
  v=paired[(paired.sparsity_sparse==sp)&(paired.step==step)].merge(tickets[['seed','sparsity','gap']],left_on=['seed','sparsity_sparse'],right_on=['seed','sparsity'])
  for m in metrics:
   x=v[m+'_difference'].to_numpy();y=v.gap.to_numpy();rho=float(spearmanr(x,y).statistic) if np.std(x)>0 else np.nan
   loo=[float(spearmanr(np.delete(x,i),np.delete(y,i)).statistic) for i in range(5) if np.std(np.delete(x,i))>0]
   perf.append(dict(sparsity=sp,step=step,metric=m,n_seeds=5,spearman_to_final_gap=rho,leave_one_seed_out_min=min(loo) if loo else np.nan,leave_one_seed_out_max=max(loo) if loo else np.nan,exploratory_only=True))
pd.DataFrame(perf).to_csv(O/'geometry_performance_all_tests.csv',index=False)
rows=pd.read_csv(O/'token_rotation_neighbors.csv');out=rows[(rows.step==5000)&(rows.reference=='initial')].sort_values('angle_deg',ascending=False).groupby(['seed','sparsity']).head(5);out.to_csv(O/'largest_token_rotations.csv',index=False)
# Saved per-expert metrics support the spectral counterexample; this is table-backed, not a fresh expert-weight SVD.
src=ROOT/'results/router_expert_geometry_v1/analysis_extended/v1/svd_endpoint_changes.csv';z=pd.read_csv(src);z=z[(z.layer==-1)&(z.membership=='endpoint_matched')];z.to_csv(O/'source_subspace_counterexample.csv',index=False)
# Recheck unmatched-pair controls directly from the 128 actual fixed-source counterfactual positions.
pro=pd.read_csv(B/'report/measurement_provenance.csv');pro=pro[(pro.track=='rerun')&pro.step.isin([0,5000])];controls=[]
for r in pro.itertuples():
 d=np.load(r.raw_path);vals=[]
 for layer in range(4):
  sel=d[f'fixed_source_L{layer}_selection_all_expert'][:,0];x=d[f'fixed_source_L{layer}_scores'][sel].astype(float);y=d[f'fixed_source_L{layer}_activations'][sel].astype(float);x-=x.mean(0);y-=y.mean(0)
  cm=x.T@y/np.sqrt((x*x).sum(0)[:,None]*(y*y).sum(0)[None,:]);identity=np.diag(cm).mean();unmatched=(cm.sum()-np.trace(cm))/(32*31);vals.append((identity,unmatched))
 identity,unmatched=np.mean(vals,0);controls.append(dict(seed=r.seed,sparsity=r.sparsity if pd.notna(r.sparsity) else 0.,step=r.step,identity=identity,unmatched=unmatched,identity_minus_unmatched=identity-unmatched))
CTRL=pd.DataFrame(controls);CTRL.to_csv(O/'coupling_pairing_controls_from_raw.csv',index=False)
# Primary figure retains all four target/control contrasts, including unfavorable baselines.
fig,axs=plt.subplots(2,2,figsize=(9,7),layout='constrained')
for ax,((e,control),v) in zip(axs.flat,A.groupby(['experts','control_age'])):
 for r in v.itertuples():ax.plot([0,1],[r.assignment_only_contrast,r.assignment_gate_contrast],'-o',color=seedcolors[r.seed],label=f'seed {r.seed}')
 ax.axhline(0,color='gray',ls=':',lw=.8);ax.set_xticks([0,1],['Assignments clamped\nnative gate derivatives','Assignments + coefficients clamped\ngate derivatives removed']);ax.set_ylabel('Target−control sparse−dense CE gap (nats)');ax.set_title(f'{e}E: R{int(v.target_age.iloc[0])} vs R{control}; n=3\nmean magnitude attenuation {v.attenuation.mean():.1%}');ax.legend(fontsize=7)
fig.suptitle('Gate replay collapses router-age contrasts; forward and backward paths both change',fontsize=12);save(fig,'01_gate_intervention_all_contrasts')
fig,axs=plt.subplots(2,2,figsize=(9,7),layout='constrained')
for sp in [.8,.85]:
 for seed in seeds:
  v=paired[(paired.sparsity_sparse==sp)&(paired.seed==seed)].sort_values('step');axs[0,0].plot(v.step,v.angle_mean_deg_difference,color=colors[sp],lw=.8,alpha=.6,ls='-' if sp==.8 else '--')
  v=G[(G.sparsity==sp)&(G.seed==seed)&(G.reference=='same_step_dense')].sort_values('step');axs[1,0].plot(v.step,v.offdiag_cosine_RMSE,color=colors[sp],lw=.8,alpha=.6,ls='-' if sp==.8 else '--')
 for ax in [axs[0,0],axs[1,0]]:ax.plot([],[],color=colors[sp],ls='-' if sp==.8 else '--',label=f'{sp:.0%}, five seed trajectories')
for ax in [axs[0,0],axs[1,0]]:ax.axvline(2500,color='gray',ls=':',lw=.8);ax.set_xlabel('Global recovery step');ax.legend(fontsize=7)
axs[0,0].axhline(0,color='gray',ls=':',lw=.8);axs[0,0].set_ylabel('Excess mean token rotation vs dense (degrees)');axs[1,0].set_ylabel('Row-normalized off-diagonal Gram RMSE')
for sp in [0.,.8,.85]:
 for seed in seeds:
  v=S[(S.sparsity==sp)&(S.seed==seed)].sort_values('step');axs[0,1].plot(v.step,v.effective_rank_squared_energy,color=colors[sp],alpha=.6,lw=.8)
 axs[0,1].plot([],[],color=colors[sp],label='dense' if sp==0 else f'{sp:.0%}')
lo,hi=N.effective_rank_squared_energy.quantile([.025,.975]);axs[0,1].axhspan(lo,hi,color='gray',alpha=.15,label='Gaussian null central 95% (64 draws)');axs[0,1].axvline(2500,color='gray',ls=':');axs[0,1].set_xlabel('Global recovery step');axs[0,1].set_ylabel('Squared-energy entropy effective rank');axs[0,1].legend(fontsize=6)
for i,sp in enumerate([0.,.8,.85]):
 v=END[(END.sparsity==sp)&(END.reference=='initial')];axs[1,1].scatter(np.full(5,i),v.neighbor1_preserved*100,color=colors[sp],s=25)
axs[1,1].set_xticks([0,1,2],['dense','80%','85%']);axs[1,1].set_ylabel('Initial nearest neighbor preserved at 5000 (%)');axs[1,1].set_title('Each point is one seed; no token-level CI')
fig.suptitle('Embedding hypothesis: modest directional/relational drift, random-like spectral concentration\nDotted boundary: optimizer/scaler reset only; synthetic null is a diagnostic control',fontsize=11);save(fig,'02_embedding_hypothesis_challenge')
fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
for ax,sp in zip(axs[:2],[.8,.85]):
 v=C[C.sparsity==sp]
 for r in v.itertuples():ax.plot([0,1],[r.dense_change,r.sparse_change],'-o',color=seedcolors[r.seed],label=f'seed {r.seed}')
 ax.axhline(0,color='gray',ls=':');ax.set_xticks([0,1],['dense','sparse']);ax.set_title(f'{sp:.0%}: paired difference in changes\n{v.difference_in_changes.mean():+.4f}, seed range {v.difference_in_changes.min():+.3f}–{v.difference_in_changes.max():+.3f}');ax.set_ylabel('Change in mean per-expert Pearson');ax.legend(fontsize=6)
for i,sp in enumerate([0.,.8,.85]):
 v=CTRL[(CTRL.sparsity==sp)&(CTRL.step==5000)];axs[2].scatter(np.full(5,i),v.identity_minus_unmatched,color=colors[sp]);axs[2].plot([i-.15,i+.15],[v.identity_minus_unmatched.mean()]*2,color=colors[sp])
axs[2].set_xticks([0,1,2],['dense','80%','85%']);axs[2].axhline(0,color='gray',ls=':');axs[2].set_ylabel('Final identity−unmatched correlation');axs[2].set_title('Unmatched-expert control, same inputs\nFive seeds; table/raw verification')
fig.suptitle('Strong observational alternative: signed coupling differs under matched fixed probes',fontsize=11);save(fig,'03_fixed_probe_coupling')
fig,axs=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
for i,mask in enumerate(['dense','learned','random']):
 v=z[(z['rank']==16)&(z.mask_type==mask)]
 for seed in seeds:
  w=v[v.seed==seed];axs[0].scatter(np.full(len(w),i)+(w.router_age-50)/350,w.source_subspace_overlap_change,s=16,color=seedcolors[seed],alpha=.7)
axs[0].set_xticks([0,1,2],['dense','learned mask','random mask']);axs[0].axhline(0,color='gray',ls=':');axs[0].set_ylabel('Source rank-16 overlap change');axs[0].set_title('28/30 learned decrease; 0/30 random decrease\nDots include dependent ages/sparsities, n=5 seeds')
r=pd.read_csv(O/'alignment_winner_falsification.csv');axs[1].barh(r.metric,r.winner_hits,color='#777777');axs[1].set_xlim(0,6);axs[1].set_xlabel('Correct winner among 6 architecture/seed groups');axs[1].set_title('Alignment often fails to identify best ticket gap\nThree reused seeds; groups are not six independent seeds')
fig.suptitle('Alternative findings and negative controls (audited table-backed reaggregation)',fontsize=11);save(fig,'04_alternatives_and_falsification')
# Dynamic embedding findings make the verdict auditable rather than claiming a large unseen effect.
lines=['# Independent embedding geometry findings','', 'All 240 registered rerun checkpoints were read; all fresh original-dtype E hashes match the previous audit. There are 230 unique 256×256 float32 E tensors, converted to CPU float64. Historical duplicates are not additional replication. No forward/backward or optimizer step was performed.','', '| Reference | Condition | Mean angle (deg) | Off-diagonal cosine RMSE | Top-1 neighbor retained | CKA | Procrustes distance |','|---|---|---:|---:|---:|---:|---:|']
for (ref,sp),v in END.groupby(['reference','sparsity']):
 lines.append(f'| {ref} | {"dense" if sp==0 else f"{sp:.0%}"} | {v.angle_mean_deg.mean():.6f} | {v.offdiag_cosine_RMSE.mean():.6f} | {v.neighbor1_preserved.mean():.4%} | {v.centered_CKA.mean():.9f} | {v.centered_unit_Procrustes_distance.mean():.6f} |')
lines+=['','Means give every seed equal weight. The tiny nonzero self-angle/self-Procrustes distances for dense versus itself are floating-point SVD/acos roundoff, not drift. Row angles depend on the fixed feature coordinate system. Token-cosine Gram comparisons, centered CKA, left/token singular subspaces and optimal Procrustes comparisons are invariant to a global right orthogonal transformation; the right/feature subspaces are not.','', '## Paired endpoint effects and spectral check','']
for sp in [.8,.85]:
 v=paired[(paired.sparsity_sparse==sp)&(paired.step==5000)];x=v.angle_mean_deg_difference;lines.append(f'- {sp:.0%}: mean excess initial-relative rotation {x.mean():+.6f} degrees, seed range {x.min():+.6f} to {x.max():+.6f}, positive in {int((x>0).sum())}/5. At 80%, two seeds have very small negative effects; only 85% has a consistent excess. No multiplicity-adjusted discovery is claimed.')
for sp in [0.,.8,.85]:
 v=S[(S.sparsity==sp)&(S.step==5000)];lines.append(f'- {"Dense" if sp==0 else f"{sp:.0%}"}: squared-energy entropy rank {v.effective_rank_squared_energy.mean():.6f}; top-eight energy {v.top8_energy.mean():.9f}; stable rank {v.stable_rank.mean():.6f}; numerical rank {v.numerical_rank_1e6.min()}–{v.numerical_rank_1e6.max()} at relative threshold 1e-6.')
lines.append(f'- Initialization mean entropy rank {S[(S.sparsity==0)&(S.step==0)].effective_rank_squared_energy.mean():.6f}. The 64 independent Gaussian **null** matrices have entropy rank {N.effective_rank_squared_energy.mean():.6f} ± {N.effective_rank_squared_energy.std():.6f} sample SD and top-eight energy {N.top8_energy.mean():.9f} ± {N.top8_energy.std():.9f}. The null is not trained-model evidence. The observed ~155/256 entropy rank is compatible with a random broad spectrum, not learned rank collapse.')
lines+=['','Normalized spectrum changes are small; more concentrated leading-subspace angles must be interpreted with cutoff gaps. All cutoff gaps and k=8/16/32 principal-angle metrics are exported, using subspaces rather than signed singular-vector dot products. Centering and squared-energy definitions are explicit; centered legacy ranks are not mixed into uncentered values.','', '## Token IDs, neighborhoods, and function','', 'The endpoint top-five rotation IDs for every seed/condition are in largest_token_rotations.csv. No semantic byte classes were fitted or invented. Full IDs, angles, nearest-neighbor IDs and top-five preservation are in token_rotation_neighbors.csv. Small changes can switch nearly tied nearest neighbors, so neighbor rearrangement is not automatically meaningful token-group reorganization. No reproducible semantic group mechanism is established.','', 'There is real relational change beyond mere uniform contraction: row-normalized Gram differences and nonzero optimally aligned residuals persist. Nevertheless effects are modest, dense models drift similarly, and sparse–dense CKA exceeds .999 on average. Stable spectrum alone cannot rule out change, but neither change nor high CKA proves a functional mechanism. Tied output prediction adds a second role for each row.','', 'All predefined geometry/performance correlations at every noninitial step are exported in geometry_performance_all_tests.csv, separately by sparsity with n=5 and leave-one-seed-out ranges. They use matched-seed, fixed-step changes from identical initial E; angle/Gram metrics remove row-scale effects and Procrustes additionally removes a global rotation. No outcome-selected metric or best correlation is promoted to a predictor. There is no independent held-out seed set, and five seeds cannot support a reliable multi-covariate predictive fit.','', '## Falsification of the proposed lead','', 'The strong claim that expert pruning induces a large, general embedding-geometry reorganization is not supported: 80% rotation effects have exceptions, the 85% excess is only about .17 degrees, most neighbors are retained, spectra are random-like, and direct E-gradient histories or causal E interventions are absent. The narrower stable-spectrum-with-modest-drift observation is confirmed but is not by itself a compelling novel paper. A global rotation preserves spectrum/Gram by linear algebra; our synthetic orthogonal control explicitly verifies this distinction.']
write('embedding_geometry_findings.md','\n'.join(lines)+'\n')
# Rank all promising alternatives without silently promoting a requested hypothesis.
write('candidate_findings.md',f'''# Ranked candidate findings

Scores are reviewer judgments from 1 (weak) to 5 (strong), not p-values or calibrated probabilities. Repeated ages, sparsities and checkpoints do not count as independent replications. Novelty scores are provisional after the targeted literature screen.

| Candidate finding | Evidence strength | Effect size | Reproducibility | Novelty potential | Scientific significance | Feasibility |
|---|---:|---:|---:|---:|---:|---:|
| **1. Router-age dependence collapses under combined gate-value/derivative replay, beyond fixing assignments** | 4 | 5 | 3 | 3 | 4 | 5 |
| **2. Coordinate pruning dissociates signed fixed-probe coupling from dense recovery and unsigned geometry** | 4 | 4 | 4 | 3 | 3 | 4 |
| **3. Learned masks recover while moving away from source leading subspaces; random masks move toward them** | 3 | 3 | 4 | 2 | 3 | 3 |

## 1. Selected: a gate/backward-path bottleneck in router-age effects

**Precise observed claim.** In these 8E/12E, 85%-coordinate-pruned expert recovery protocols, all 12 target/control age contrasts in sparse-minus-dense validation CE have much smaller magnitude after replacing native selected gate coefficients with one stored R100 donor schedule **and detaching gate task gradients**, even when expert assignments and capacity acceptances were already clamped. Group-mean attenuation is 94.69%, 98.67%, 99.81%, 99.54%; the maximum residual absolute contrast is {A.assignment_gate_contrast.abs().max():.9f} nats/token. This is a claim about this combined intervention and common readout, not about unique forward-value mediation or all natural MoEs.

Supporting source: `results/gate_amplitude_replay_intervention/outcomes.csv`, recipient `recovery/e*/seed*/R*/{{dense,sparse}}/summary.json`, replay/gradient semantics audits, and `src/moe_lth/experiments/gate_amplitude_replay.py:21`. `gate_contrasts_reproduced.csv` recomputes all 12 effects from the actual condition losses; all 36 Level-2 endpoint summaries match the common-gate outcomes to 2.3e-16. Figure 01 retains all four comparisons and every seed. Three seeds per architecture, reused across ages, are the independent units.

Why the scores: matched training interventions and exact assignment audits make evidence stronger than observational drift (4); the magnitude is large in all comparisons (5); only three reused seeds and one donor cap replication (3); generic gating is established and backward-path isolation is missing (novelty 3); distinguishing discrete assignment from continuous coefficients/derivatives could matter broadly (significance 4); two missing factorial cells can reuse existing artifacts (feasibility 5).

**Attempted falsification / counterevidence:** four Level-1 comparisons already favor the control, so this is not removal of a universal favorable intermediate-age advantage. Assignment-only substitution already attenuates several contrasts substantially. Readout choice can reverse signs, all full-gate sparse endpoints remain worse than their common-readout dense controls, and detached coefficients remove router/upstream task gradients. Once all router-specific forward outputs are fixed, losing age dependence is partly expected from disconnecting the distinguishing pathway. Auxiliary gradients remain a possible residual path. Pattern decomposition does not supply a matched full detached canonical-source control and has severe, heterogeneous secondary losses. These facts rule out a finished forward-gating-mechanism claim.

**Publishable if:** isolated value or Jacobian interventions show a substantial, reproducible sparsity/age interaction on new seeds and a second donor/readout, with preserved controls and a clear mechanistic explanation beyond standard scaling. **Unpublishable if:** ordinary detachment/readout mismatch fully explains it, effects are tiny after isolation, random masks behave identically when learned-mask specificity is claimed, or transfer fails. Strongest candidate for a bounded next test, not a paper-ready discovery.

## 2. Signed coupling is not a universal ticket/geometry readout

From actual fixed-source/all-expert cached arrays, sparse changes are +.139590/.171162 versus dense −.365186. Paired differences-in-changes are **+.504776** at 80% (seed range .405341–.611653) and **+.536348** at 85% (.423316–.662443), all five seeds positive. New raw-array arithmetic agrees with the audited report to 2.3e-16. Unmatched-expert identity controls are independently recomputed in coupling_pairing_controls_from_raw.csv. This effect appears in winners and nonwinners and is not an effect of E acquiring zero coordinates.

Evidence/reproducibility scores reflect raw arrays, matched dense controls, five source seeds and sensitivity tables. Effect size 4 reflects a large bounded correlation contrast. Novelty/significance are reduced because correlation does not identify expert usefulness and the compared GELU/full-softmax implementation differs from the nearest paper. Accepted-pair effects reverse under restrictive all-time coverage: only 11–14/128 units remain. Thus the stronger-sounding universal anti-coupling claim fails. Ahrac et al.'s model/readout/selection differs; this is a boundary-condition observation, not a direct refutation. A no-auxiliary or altered gating comparison would be needed for mechanism.

## 3. Recovery need not restore source leading eigenspaces

Reaggregation of `router_expert_geometry_v1/analysis_extended/v1/svd_endpoint_changes.csv` on endpoint-matched expert populations yields learned-mask decreases in **29/30, 28/30, 29/30** conditions at ranks 8/16/32. Random masks and dense controls increase in every condition (0 decreases out of 30 and 15 respectively). Mean rank-16 changes are learned −.069168, random +.128705, dense +.264856. Counts include three ages and two sparsities within each of five reused seeds; they are not 30 independent trials.

This was checked from audited per-condition measurements, not by recomputing every expert-weight SVD here (evidence 3). Reference dependence, small singular gaps, different starting overlap and magnitude-mask selection are simpler competing explanations. Full-Gram shape has mixed signs and does not mirror leading-subspace trends. Recovery away from a source optimum is not itself surprising, nor does this identify ticket usefulness. The original 30/30 rank-16 claim was invalid because of changing valid populations; retain both counterexamples. Its novelty is lower than a clean path intervention.

## Investigated but not promoted

**Embedding orientation with stable spectra:** all 240 rerun E checkpoints measured independently; see embedding_geometry_findings.md. Dense mean initial-relative rotation is 3.0243°, 80% 3.0854°, 85% 3.1909°. The 85% excess is consistent but only .1665°; 80% has two exceptions. Gaussian null spectra match ~155 entropy rank and all E matrices have numerical rank 256. Stable spectra with modest drift are real but unsurprising and do not establish functional relevance or E-mediated recovery. **NO-GO as a standalone paper claim.**

**Generic gradient alignment predicts ticket winners:** falsified in the existing exactly reproduced 8E/12E diagnostic cohort: native/forced gradient, function and update winners match actual best-gap routers in only 2/6 architecture/seed groups, trajectory distance 1/6. Three reused seeds, not six independent replications. This is useful negative evidence, not proof geometry is irrelevant.

**Architecture-dependent optimum / strict lottery tickets:** lower-sparsity Dataset A has 212/1100 strict individual matches but zero all-five settings among 220; no adjusted age-pair contrast establishes a universal optimum. Learned-mask advantages over random controls are real within their specific grid but are broadly anticipated by pruning literature. Older root reports cannot override finalized audits or missing raw artifacts. No architecture scaling law or novel LTH theorem is established.

## Why not GO or NO-GO overall?

GO would require a new mechanistic claim already isolated and replicated, which is absent. A blanket NO-GO would ignore a large, fully reproducible controlled effect with a small discriminating follow-up. **CONDITIONAL GO**, restricted to resolving that confound; no broad additional compute is justified yet.
''')
write('research_verdict.md',f'''# Research verdict: CONDITIONAL GO

**Strongest finding:** In the existing 85%-sparse 8E/12E recoveries, router-age dependence in matched sparse-minus-dense CE nearly disappears when selected gate coefficients and their task-gradient path are clamped, beyond clamping assignments and capacity. This is a precise, reproducible **combined-intervention** finding; forward values and backward derivatives are not yet separated.

**Key quantitative evidence:** All 12 age contrasts were independently reproduced from condition losses. Mean attenuation across the three seeds in each architecture/control group is 94.7–99.8%; maximum residual contrast is {A.assignment_gate_contrast.abs().max():.7f} nats/token. All 36 full-gate recipient endpoint summaries match their common-readout outcomes. The observational alternative also reproduces: fixed-source coupling differences-in-changes are +.5048/.5363 across all five 32E seeds. The embedding hypothesis is much weaker: only ~.17° excess rotation at 85%, near-identical spectra and random-like entropy rank.

**Most important counterevidence:** Detachment removes an entire router/upstream task-gradient path; coefficient clamping can almost disconnect router-age identity by construction. Auxiliary gradients remain. Four baseline contrasts are already target-unfavorable. One donor/readout, only three reused seeds per architecture, and large native-readout shifts restrict generality. No saved E-gradient trajectory or causal E-mediation evidence exists. These limitations preclude a finished mechanism or novel-paper claim.

**Novelty assessment:** Router-weighted pruning, gating/gradient scaling, representation drift and CKA invariance are established. A factorial separation of gate values, gate Jacobians and sparse/dense compatibility at fixed accepted assignments is plausibly more specific, but priority is unverified beyond the targeted primary-paper screen. See novelty_review.md for links, dates and comparison limits.

**Decisive next experiment:** Add the two missing factorial arms: donor gate values with the native gate Jacobian, and native gate values with its task Jacobian detached. Use existing 8E R20/R100 seeds 7/17/29 and dense/sparse controls: 24 new recoveries, reuse audited A/B cells, common primary readout. Stop after this screen under the predeclared rule. If it passes, confirm on five new seeds and then donor/random-mask controls. Nothing was launched here.

**Estimated research value:** Moderate potential and high immediate information value, low present paper readiness. Reserve roughly 5–25 GPU-hours for the discriminating screen, calibrated by an authorized timing pilot; a fresh-seed confirmation is a separate 20–70+ GPU-hour decision. A clear pathway result with portability could justify a paper; a generic detachment or evaluation artifact should stop the project direction.

**Recommendation: CONDITIONAL GO.** Fund one bounded mechanism-discriminating screen, not more descriptive embedding plots or a large model sweep. NO-GO for a standalone claim that sparse experts produce novel low-rank or strongly rotated embeddings in this cohort.

All outputs and scripts are confined to `results/research_investigation_v1/`. No training code, checkpoint weights or previous experiment outputs were changed.
''')
# Preserve exact code references and consumed input hashes, and check previous outputs remain unchanged.
source_names=['src/moe_lth/models/transformer.py','src/moe_lth/models/router.py','src/moe_lth/models/moe_layer.py','src/moe_lth/models/experts.py','src/moe_lth/pruning/imp.py','src/moe_lth/pruning/masks.py','src/moe_lth/pruning/rewind.py','src/moe_lth/experiments/run_router_age_recovery.py','src/moe_lth/experiments/gate_amplitude_replay.py','scripts/high_sparsity_coupling/runner.py','scripts/high_sparsity_coupling/gradient.py','scripts/high_sparsity_coupling/measure.py']
ledger=json.load(open(O/'consumed_input_sha256.json'));ledger[str(src.relative_to(ROOT))]=sha(src)
for name in source_names:ledger[name]=sha(ROOT/name)
assert all(sha(ROOT/p)==h for p,h in ledger.items());json.dump(ledger,open(O/'consumed_input_sha256.json','w'),indent=2)
prior=json.load(open(B/'analysis_audit_v2/final_artifact_manifest.json'));assert all(sha(B/p)==h for p,h in prior['outputs'].items())
oldemb=json.load(open(B/'embedding_weight_sparsity/artifact_manifest.json'))['sha256'];assert all(sha(B/'embedding_weight_sparsity'/p)==h for p,h in oldemb.items())
figs=list(F.glob('*.png'));assert len(figs)==4
validation={'candidate_reproduction':json.load(open(O/'candidate_reproduction_validation.json')),'geometry':json.load(open(O/'geometry_validation.json')),'previous_report_files_unchanged':len(prior['outputs']),'previous_embedding_artifacts_unchanged':len(oldemb),'all_consumed_inputs_unchanged':True,'figures_png_pdf_svg':4,'training_launched':False,'production_code_modified':False,'decision':'CONDITIONAL GO'}
json.dump(validation,open(O/'completion.json','w'),indent=2)
manifest={str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file() and p.name not in ['artifact_manifest.json'] and p.suffix not in ['.log'] and '__pycache__' not in p.parts};json.dump(manifest,open(O/'artifact_manifest.json','w'),indent=2)
print('Research synthesis complete; protected inputs/artifacts unchanged.',flush=True)
