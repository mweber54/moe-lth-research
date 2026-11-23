import sys
sys.path.insert(0,'results/router_gate_factorial_v1')
from support import *
import pandas as pd,numpy as np,re

def md(df):
 d=df.copy()
 for c in d:
  if pd.api.types.is_float_dtype(d[c]):d[c]=d[c].map(lambda x:f'{x:.6f}' if pd.notna(x) else '')
 return '| '+' | '.join(map(str,d.columns))+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in d.itertuples(index=False,name=None))+'\n'
def pivot(data,key='age_contrast'):
 p=data.pivot(index='arm',columns='seed',values=key).reindex(list(ARMS));p['mean']=p[[7,17,29]].mean(axis=1);p['sample_SD']=p[[7,17,29]].std(axis=1);return p.reset_index()
def main():
 e=pd.read_csv(OUT/'full_factorial_results.csv');c=pd.read_csv(OUT/'router_age_contrasts.csv');f=pd.read_csv(OUT/'factorial_effects_per_seed.csv');a=pd.read_csv(OUT/'per_seed_attenuation.csv');b=pd.read_csv(OUT/'all_factorial_budgets.csv');g=pd.read_csv(OUT/'gradient_diagnostics.csv');gates=pd.read_csv(OUT/'forward_gate_diagnostics.csv');j=pd.read_csv(OUT/'fixed_probe_gate_jacobians.csv');u=pd.read_csv(OUT/'adamw_update_reconstruction.csv');rule=read(OUT/'stopping_rule_evaluation.json');post=read(OUT/'postrun_validation.json');pres=read(OUT/'preservation_audit.json')
 full=e.sort_values(['arm','seed','age','kind']);(OUT/'full_factorial_results.md').write_text('# Full factorial endpoints\n\nNats/token; each row is one trained model evaluated using three readouts. A/B are audited reuse; C/D are new. All use E8, the same donor assignments/capacity archive, ages20/100, seeds7/17/29, and matched dense/85% sparse recoveries.\n\n'+md(full.drop(columns='source_file')))
 ep=f[f.readout=='primary'].pivot(index='effect',columns='seed',values='nats');ep['mean']=ep[[7,17,29]].mean(axis=1);ep['sample_SD']=ep[[7,17,29]].std(axis=1)
 means=e.groupby(['arm','age','kind'])[['primary_loss','native_loss','assignment_loss']].mean().reset_index()
 budgets=b.groupby('arm').agg(models=('kind','count'),attempts_min=('attempted_updates','min'),attempts_max=('attempted_updates','max'),applied_min=('applied_updates','min'),applied_max=('applied_updates','max'),total_AMP_skips=('AMP_skips','sum')).reset_index()
 grad=g.groupby('arm').agg(router_task_min=('router_task_norm','min'),router_task_max=('router_task_norm','max'),router_aux_min=('router_aux_norm','min'),expert_task_min=('expert_task_norm','min'),shared_task_min=('shared_task_norm','min'),max_tied_decomposition_error=('tied_decomposition_relative_error','max')).reset_index()
 assessment=read(OUT/'verdict_decision.json')
 text=f'''# Router gate factorial mechanism experiment

**{assessment['verdict']}** — {assessment['reason']}

Exactly24 new C/D recoveries completed. E8 only; router ages20/100; seeds7/17/29; existing85% learned masks; matched dense recoveries. The audited24 A/B models are reused. No fresh-seed confirmation or expanded sweep was launched. All numerical/protocol inputs, source weights, masks, configurations and schedules remain unchanged. One concurrent trailing-newline-only change in the earlier narrative verdict is documented and was left untouched.

## What the factorial changes

A: native gate values/native gate Jacobian. B: donor values/detached gate Jacobian. C: donor values/native gate Jacobian. D: native values/detached gate Jacobian. A denotes the audited **assignment-replay** control, not the original freely routed native training arm. Donor IDs and capacity priorities are identical across all four cells. C supplies an explicit surrogate backward path, not a finite-difference derivative of its donor-constant forward function. Native means the recipient's current selected probability/Jacobian on its current state, not a saved A trajectory.

The native auxiliary-probability loss and coefficient.1 remain active in every cell. Gate detach also cuts indirect paths from later auxiliary losses through earlier gates, exactly as in the existing B intervention. This is therefore not a CE-exclusive graph surgery.

## Validation and preservation

[Pre-training validation](validation_report.md) passed before any recovery: all12243 donor occurrences passed hashes; twelve real initial-state/data/mask identities passed; exact gate forward values, independent native/detached derivative references, FP32 and AMP legacy equivalence, active expert/upstream gradients, and tied-embedding lookup/head decomposition passed. The actual E8 router uses full softmax before top1; it is not the structurally constant post-selection-normalized top1 case. Exact selected probability1 alone is not proof that every softmax Jacobian coordinate is numerically zero, so actual local Jacobians were measured separately.

All three C/R100/dense runs reproduce all51 native donor validation values and every final model weight bit-for-bit, including workers paused for GPU scheduling. C/R100/dense is an identity cell by construction on the donor's own trajectory: its value-only C-A effect is zero there, while the native gate Jacobian remains active. It is a compatibility control, not an independent mechanistic replicate. All24 final checkpoints match their step2500 snapshots, preserve tied embedding values and sparse mask zeros, and contain finite weights. All216 snapshot diagnostics preserve parameters, optimizer, scaler, gradient buffers and RNG. [Post-run checks](postrun_validation.json) reconstruct192 subsequent embedding AdamW updates from clipped gradients, moments and weight decay; maximum error is {u.max_absolute_error.max():.3g}, within explicit FP32 rounding bounds. These are measured optimizer displacements, not inferred gradient directions.

Retained B endpoints were re-evaluated under all three readouts and checked against their audited values in [the endpoint recheck](combined_endpoint_readout_recheck.csv). Their new fixed-probe gradient measurements are explicitly diagnostic derivatives, not historical training gradients. A checkpoints and A/B intermediate snapshots are unavailable; their audited scalar readouts and execution provenance are reused without claiming new evaluations of missing states. [Reuse hashes](audited_reuse_hash_verification.json) and [scientific-input preservation audit](preservation_audit.json) passed. The strict byte check detected one removed final newline in the earlier narrative verdict; its original bytes are proven by SHA256 after adding the newline in memory. That report was left untouched. See [the formatting exception](preservation_exception.md). The frozen analyzer remains intact; a separately saved analyzer changes only this preservation audit and metadata, with an exact retained patch and unchanged estimands/thresholds.

## Full outcomes and router-age contrasts

The complete48-row table with primary, native and assignment-only losses is [Markdown](full_factorial_results.md) / [CSV](full_factorial_results.csv). Absolute condition losses, averaged over the three seeds, are:

{md(means)}

Define G=CE_sparse-CE_dense and T=G_R20-G_R100. Negative T means a smaller sparse-minus-dense penalty at R20. Primary evaluation forces the identical donor assignments **and coefficients**, with FP32 token-weighted CE on the identical complete validation batches. Primary T, by seed:

{md(pivot(c[c.readout=='primary']))}

## Attenuation and factorial effects

Per-seed absolute attenuation is1-|T_arm|/(|T_A|+1e-12); it is secondary to the signed nats contrasts. Near-zero baseline flags use the predeclared1e-5 diagnostic. Primary attenuation:

{md(pivot(a[a.readout=='primary'],'absolute_attenuation'))}

The isolated native-reference effects are C-A (value substitution) and D-A (Jacobian removal). Combined is B-A. Interaction is B-C-D+A. The balanced factorial main effects average both conditional effects: marginal value=((C-A)+(B-D))/2; marginal detach=((D-A)+(B-C))/2. These are effects on T, not effects on absolute CE, and the two notions of main effect should not be conflated when interaction is large.

{md(ep.reset_index())}

[Sparse/dense age components](age_effect_components.csv), [absolute-CE factorial effects](factorial_effects_on_condition_CE.csv), [ticket-gap factorial effects](factorial_effects_on_ticket_gap.csv), [all per-seed effects](factorial_effects_per_seed.csv), [aggregate effects](factorial_effects_aggregate.csv), [attenuation statistics](attenuation_aggregate.csv), and [attenuation-of-means versus mean-of-attenuations](aggregate_attenuation_estimands.csv) retain the distinct estimands. Seeds are the independent units; n=3 is exploratory. No token/expert/time-point pseudo-replication or confirmatory p-values are used.

## Readout sensitivity

Native-readout T:

{md(pivot(c[c.readout=='native']))}

Assignment-only-readout T:

{md(pivot(c[c.readout=='assignment']))}

{assessment['readout_interpretation']}

Readouts measure the same models, not additional training arms. Native evaluation releases assignments and coefficients; assignment-only retains donor assignment/capacity but uses native coefficients. Primary uses donor assignment/capacity and donor coefficients. These are distinct evaluation interventions, so cross-readout reversals cannot be averaged away.

## Mechanistic assessment and stopping rule

{assessment['mechanism']}

The predeclared screen stops after24. Promotion requires at least one isolated effect exceeding.02 nats in the same direction in all three seeds, passed intervention controls, and an effect not solely explained by evaluation-policy reversal. The frozen numerical rule checks are in [stopping_rule_evaluation.json](stopping_rule_evaluation.json); native-readout seed effects are retained alongside primary effects. {assessment['stopping_interpretation']}

The controlled gate interventions identify behavior under the imposed donor schedule and readout. Attenuation alone does not establish natural causal mediation of router age, learned-mask specificity, or general recovery improvement. {assessment['next_control']}

## Gradients, saturation, updates and budgets

Fixed-step0-probe FP32 CE/auxiliary gradient norms across all new snapshot states:

{md(grad)}

[Forward diagnostics](forward_gate_diagnostics.csv) show zero forward-source error for every logged layer/step. [Actual local FP16 Jacobians](fixed_probe_gate_jacobians.csv) match the requested policy in all864 measurements; mean native zero-Jacobian fractions by arm are {j.groupby('arm').native_local_J_exact_zero_fraction.mean().to_dict()}. The native Jacobian can change along the optimization trajectory, so selected-gate0/1 fractions are numerical diagnostics, not automatic mediation evidence. [Embedding gradient/update diagnostics](optimizer_update_diagnostics.csv) identify the fixed probe's source step, observed displacement and moment norms. Lookup/head contributions sum to tied CE gradients; auxiliary gradients remain separately logged. No layer VJP is called an additive causal decomposition.

All arm budgets, including reused controls:

{md(budgets)}

Every model used2500 attempts, fresh AdamW, LR.0003, decay.01, clipping1, FP16 AMP and the same native auxiliary objective. Applied-update counts can differ because AMP skips remain uncompensated, matching the audited protocol. Preclip norms and every attempted update are in legacy gradient logs; postclip norms and actual E updates are in measurements. Raw wall seconds for the two initially paused cells include waiting; [runtime accounting](runtime_accounting.csv) subtracts the logged pause intervals and does not call paused time GPU compute.

Snapshots are0,25,100,250,500,1000,1500,2000,2500. There is no donor validation occurrence at25: its native readout and fixed-step0 derivative diagnostic are labeled, and it is excluded from primary timing comparisons. Primary evaluation remains every50 updates; the full-range trajectory figure uses only jointly observed common steps, with no interpolation represented as evidence. The limited-axis figure is a zoom; [full-range primary trajectories](figures/primary_contrasts_full_range.png), [readout sensitivity](figures/readout_sensitivity.png), and [gradient policies](figures/gradient_policies.png) also have PDF exports.

Only previously used validation data were evaluated. Original A states are missing, all seeds are outcome-selected exploratory units, the Jacobian factor also affects indirect auxiliary paths, and donor schedule portability is not tested here. These limitations constrain the verdict and are not resolved by small scalar attenuation.
'''
 text=re.sub(r'(?<=[a-z])(?=[0-9])',' ',text)
 text=re.sub(r'(?<=[A-Za-z])(?=\.[0-9])',' ',text)
 (OUT/'report.md').write_text(text)
 (OUT/'research_verdict.md').write_text(f"# Research verdict\n\n**{assessment['verdict']}**\n\n{assessment['reason']}\n\n{assessment['mechanism']}\n\n{assessment['stopping_interpretation']}\n\n{assessment['next_control']}\n\nExactly 24 new recoveries completed; numerical/protocol inputs preserved (one prior narrative newline edit documented); no fresh seeds launched. See [full report](report.md).\n")
 print('Report and research verdict written.')
if __name__=='__main__':main()
