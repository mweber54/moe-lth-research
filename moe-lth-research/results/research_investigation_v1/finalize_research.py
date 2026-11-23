"""Polish final figures and add strictly observed timing/readout falsification."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;ROOT=O.parents[1];F=O/'figures';B=ROOT/'results/high_sparsity_coupling_v1'
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
colors={7:'#0072B2',17:'#D55E00',29:'#009E73'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(F/f'{name}.{ext}',dpi=220,bbox_inches='tight')
 plt.close(fig)
A=pd.read_csv(O/'gate_contrasts_reproduced.csv')
fig,axs=plt.subplots(2,2,figsize=(9,7),layout='constrained')
for ax,((e,control),v) in zip(axs.flat,A.groupby(['experts','control_age'])):
 for r in v.itertuples():ax.plot([0,1],[r.assignment_only_contrast,r.assignment_gate_contrast],'-o',color=colors[r.seed],label=f'seed {r.seed}')
 ax.axhline(0,color='gray',ls=':',lw=.8);ax.set_xticks([0,1],['Assignments only','+ detached gates']);ax.set_ylabel('Age contrast of sparse−dense CE gap (nats)');ax.set_title(f'{e}E: R{int(v.target_age.iloc[0])} vs R{control}; n=3\nmean magnitude attenuation {v.attenuation.mean():.1%}',fontsize=10);ax.legend(fontsize=7)
fig.suptitle('Gate replay collapses age contrasts; coefficients and task derivatives both change',fontsize=11);save(fig,'01_gate_intervention_all_contrasts')
# Common-readout timing is reconstructed from actual saved per-state losses.
path=ROOT/'results/gate_amplitude_replay_intervention/common_readout_snapshot_losses.csv';loss=pd.read_csv(path)
existingpath=ROOT/'results/gate_amplitude_replay_intervention/common_readout_contrast_trajectories.csv';existing=pd.read_csv(existingpath);rec=[]
for r in existing.itertuples():
 if r.step not in set(loss.step):continue
 target=20 if r.experts==8 else 60
 vals={}
 for age in [target,r.control_age]:
  z=loss[(loss.experts==r.experts)&(loss.seed==r.seed)&(loss.level==r.level)&(loss.age==age)&(loss.step==r.step)]
  assert len(z)==2 and set(z.kind)=={'dense','sparse'}
  vals[age]=float(z[z.kind=='sparse'].loss.iloc[0]-z[z.kind=='dense'].loss.iloc[0])
 value=vals[target]-vals[r.control_age];assert abs(value-r.contrast)<1e-10
 rec.append(dict(experts=r.experts,seed=r.seed,level=r.level,step=r.step,control_age=r.control_age,contrast=value))
T=pd.DataFrame(rec);T.to_csv(O/'gate_observed_trajectories_reproduced.csv',index=False)
# All-arm intersections avoid treating the gate-only 51-point curves as matched observations.
wide=T.pivot(index=['experts','seed','control_age','step'],columns='level',values='contrast').dropna(subset=['native','assign','gate']).reset_index();wide.to_csv(O/'gate_all_arm_common_steps.csv',index=False)
fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for ax,((e,control),v) in zip(axs.flat,wide.groupby(['experts','control_age'])):
 for seed in [7,17,29]:
  z=v[v.seed==seed].sort_values('step')
  ax.plot(z.step,z['assign'],color=colors[seed],ls='-',marker='o',ms=2,lw=1,label=f'{seed}: assignments only')
  ax.plot(z.step,z['gate'],color=colors[seed],ls='--',marker='s',ms=2,lw=1,label=f'{seed}: + detached gates')
 ax.axhline(0,color='gray',ls=':',lw=.8);ax.set_xlabel('Recovery step (observed common checkpoints)');ax.set_ylabel('Age contrast of sparse−dense CE gap (nats)');ax.set_title(f'{e}E, R{20 if e==8 else 60} vs R{control}');ax.legend(fontsize=6,ncol=2)
fig.suptitle('Training evolution under a common donor readout\nLines connect saved observations; no claims about unobserved onset or causal precedence',fontsize=11);save(fig,'05_gate_training_evolution')
# Explicit seed variability for every original comparison.
grouped=[]
for (e,control),v in A.groupby(['experts','control_age']):
 grouped.append(dict(experts=e,control_age=control,n_seeds=3,assignment_mean=v.assignment_only_contrast.mean(),gate_mean=v.assignment_gate_contrast.mean(),increment_mean=v.gate_increment.mean(),increment_seed_sd=v.gate_increment.std(ddof=1),mean_attenuation=v.attenuation.mean(),min_attenuation=v.attenuation.min(),max_attenuation=v.attenuation.max()))
GROUP=pd.DataFrame(grouped);GROUP.to_csv(O/'gate_grouped_seed_effects.csv',index=False)
candidate=(O/'candidate_findings.md').read_text().split('\n## Seed-level variability of the selected effect')[0]
candidate+='\n## Seed-level variability of the selected effect\n\nAll contrasts below use the same common donor readout. Increment is full-gate minus assignment-only age contrast; a negative increment shifts toward the target. SD is descriptive spread over three seeds, not an expert/token-level uncertainty estimate.\n\n| Experts | Control age | Assignment contrast mean | Full-gate contrast mean | Increment mean ± seed SD | Mean attenuation |\n|---|---:|---:|---:|---:|---:|\n'
for r in GROUP.itertuples():candidate+=f'| {r.experts} | R{r.control_age} | {r.assignment_mean:+.6f} | {r.gate_mean:+.6f} | {r.increment_mean:+.6f} ± {r.increment_seed_sd:.6f} | {r.mean_attenuation:.2%} |\n'
(O/'candidate_findings.md').write_text(candidate)

# Align E geometry with recorded full-validation timing; no forward is rerun.
pro=pd.read_csv(B/'report/measurement_provenance.csv');pro=pro[pro.track=='rerun'];s=pro[pro.sparsity.notna()];d=pro[pro.sparsity.isna()];v=s.merge(d[['seed','step','recorded_measurement_ce']],on=['seed','step'],suffixes=('','_dense'));v['validation_gap']=v.recorded_measurement_ce-v.recorded_measurement_ce_dense;v=v[['seed','sparsity','condition','step','validation_gap']];v.to_csv(O/'validation_gap_timecourse.csv',index=False)
g=pd.read_csv(O/'embedding_paired_changes.csv');j=g.merge(v,on=['seed','step'],suffixes=('','_validation'));j=j[np.isclose(j.sparsity_sparse,j.sparsity)];j.to_csv(O/'geometry_validation_gap_timecourse.csv',index=False)
initial=v[v.step==0].validation_gap
text=(O/'embedding_geometry_findings.md').read_text().replace('All predefined geometry/performance correlations','All fixed-list exploratory geometry/performance correlations')
text=text.split('\n## Temporal counterexample')[0]
text+=f'\n## Temporal counterexample and uncertainty\n\nAt initialization all matched E tensors are identical, while sparse-minus-dense validation gaps already range from {initial.min():+.6f} to {initial.max():+.6f} nats/token. Expert masks and other downstream computations differ before E evolves. Thus E differences are not necessary for an initial performance gap; this does not rule out a later contribution. The first postinitial geometry observation is step 25, so simultaneous early observations do not establish which process leads. See geometry_validation_gap_timecourse.csv for every observed paired time point. All performance correlations are post hoc exploratory, with 240 metric/step/sparsity tests exported and no claim that a maximum correlation predicts unseen seeds.\n'
(O/'embedding_geometry_findings.md').write_text(text)
audit=(O/'research_audit.md').read_text();audit=audit.split('\n## Additional timing verification')[0];audit+='\n## Additional timing verification\n\nThe common-readout gate trajectories were independently reconstructed from saved sparse/dense checkpoint losses, including every level and all available points. The comparison figure uses only the intersection of observed native/assignment/gate checkpoints, not the denser gate-only curve. This verifies saved-loss arithmetic; it is not a fresh model evaluation and cannot turn diagnostic onset into mediation evidence. The 32E full-validation/geometry timeline additionally preserves the existing measured CE precision setting.\n';(O/'research_audit.md').write_text(audit)
notes=(F/'figure_notes.md').read_text();notes=notes.split('\n5. **05_')[0];notes+='\n5. **05_gate_training_evolution:** contrasts recomputed from common_readout_snapshot_losses.csv, intersected across all three training levels. Observed steps are marked. Solid lines denote assignment replay; dashed lines denote added detached coefficients; color identifies seed. The native level defines coverage but is omitted from the plotted comparison to retain a legible incremental intervention. Curves are observational summaries of the recorded interventions, not precise causal onset estimates.\n';(F/'figure_notes.md').write_text(notes)
# The final stage makes the earlier table-backed alternative and all figures reproducible.
readme=(O/'README.md').read_text();cmd='OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-highres/bin/python results/research_investigation_v1/finalize_research.py'
if cmd not in readme:readme=readme.replace('synthesize.py\n```','synthesize.py\n'+cmd+'\n```')
(O/'README.md').write_text(readme)
ledger=json.load(open(O/'consumed_input_sha256.json'))
for p in [path,existingpath]:ledger[str(p.relative_to(ROOT))]=sha(p)
assert all(sha(ROOT/p)==h for p,h in ledger.items());json.dump(ledger,open(O/'consumed_input_sha256.json','w'),indent=2)
c=json.load(open(O/'completion.json'));c.update(figures_png_pdf_svg=5,common_gate_trajectory_rows_reproduced=len(T),matched_all_arm_timepoints=len(wide),all_inputs_unchanged_after_finalization=True,analysis_complete=True,all_unique_E_numerical_rank=256)
json.dump(c,open(O/'completion.json','w'),indent=2)
required=['research_audit.md','candidate_findings.md','novelty_review.md','decisive_experiment.md','research_verdict.md']
assert all((O/x).exists() for x in required)
assert all(len(list(F.glob('*.'+ext)))==5 for ext in ['png','pdf','svg'])
manifest={str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file() and p.name!='artifact_manifest.json' and p.suffix!='.log' and '__pycache__' not in p.parts};json.dump(manifest,open(O/'artifact_manifest.json','w'),indent=2)
print('Final research checks passed; five observed-data figure families saved.',flush=True)
