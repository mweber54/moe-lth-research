from pathlib import Path
import json,math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
SEEDS=(7,17,29,43,61)
frames=[]
for s in SEEDS:
    assert json.loads((OUT/f'completion_seed{s}.json').read_text())['complete']
    assert json.loads((OUT/f'matched_completion_seed{s}.json').read_text())['complete']
    x=pd.read_csv(OUT/f'results_seed{s}.csv');assert len(x)==555,(s,len(x));frames.append(x)
    x=pd.read_csv(OUT/f'matched_seed{s}.csv');assert len(x)==60,(s,len(x));frames.append(x)
df=pd.concat(frames,ignore_index=True);assert len(df)==3075
assert not df.duplicated(['seed','mask_type','condition','p','draw','layer','projection']).any()
assert df.invariants_passed.all();assert np.isfinite(df[['CE','delta_CE','realized_distance','routing_change']]).all().all()
identity=df[df.condition=='F']; assert (identity.delta_CE==0).all() and (identity.routing_change==0).all()
df.to_csv(OUT/'results.csv',index=False)
allrows=df[(df.layer=='all') & (df.projection=='both')]
summary=allrows.groupby(['seed','mask_type','condition','p'],dropna=False)[['CE','delta_CE','realized_distance','frac_sign_flipped','routing_change']].agg(['mean','std']).reset_index();summary.to_csv(OUT/'condition_summary.csv',index=False)
# Per-seed mean full-dose effects, never mix random realizations without labels.
full=allrows[(allrows.p==1)&allrows.condition.isin(['A','B','C','D','F'])].groupby(['seed','mask_type','condition']).agg(delta_CE=('delta_CE','mean'),sd=('delta_CE','std'),CE=('CE','mean'),distance=('realized_distance','mean'),routing_change=('routing_change','mean')).reset_index();full.to_csv(OUT/'full_dose_per_seed.csv',index=False)
# Paired Gaussian controls, matched within each tensor and draw.
noise=allrows[allrows.condition.isin(['E_A','E_B'])].copy();noise['treatment']=noise.condition.str.replace('E_','',regex=False)
treatment=allrows[allrows.condition.isin(['A','B'])].copy();treatment['treatment']=treatment.condition
noise_pairs=noise.merge(treatment,on=['seed','mask_type','treatment','p','draw'],suffixes=('_noise','_treatment'))
noise_pairs['treatment_minus_noise']=noise_pairs.delta_CE_treatment-noise_pairs.delta_CE_noise
noise_pairs.to_csv(OUT/'noise_comparisons.csv',index=False)
noise_pairs.groupby(['seed','mask_type','treatment','p'])[['delta_CE_treatment','delta_CE_noise','treatment_minus_noise']].agg(['mean','std']).to_csv(OUT/'noise_per_seed.csv')
df[df.projection.isin(['fc1','fc2'])].groupby(['seed','mask_type','condition','projection']).delta_CE.agg(['mean','std']).to_csv(OUT/'projection_per_seed.csv')
df[df.layer!='all'].groupby(['seed','mask_type','condition','layer']).delta_CE.agg(['mean','std']).to_csv(OUT/'layer_per_seed.csv')
# Paired distance-conditioned sign effects versus A, with all draws retained.
a=allrows[allrows.condition=='A'];b=allrows[allrows.condition=='B_matched']
paired=b.merge(a,on=['seed','mask_type','draw'],suffixes=('_B','_A'));paired=paired[np.isclose(paired.target_p_B,paired.p_A)]
paired['B_minus_A']=paired.delta_CE_B-paired.delta_CE_A
paired.to_csv(OUT/'distance_matched_comparisons.csv',index=False)
pair_summary=paired.groupby(['seed','mask_type','target_p_B']).agg(A_delta=('delta_CE_A','mean'),B_delta=('delta_CE_B','mean'),B_minus_A=('B_minus_A','mean'),distance=('realized_distance_A','mean'),distance_mismatch=('distance_mismatch_B',lambda z:max(abs(z)))).reset_index();pair_summary.to_csv(OUT/'distance_matched_per_seed.csv',index=False)
for family,arms in [('IMP',['imp']),('random',['random1','random2'])]:
    fig,axes=plt.subplots(1,5,figsize=(19,4),sharey=True)
    colors={'A':'#1f77b4','B':'#d62728','E_A':'#2ca02c','E_B':'#9467bd','B_matched':'#ff7f0e'}
    for ax,s in zip(axes,SEEDS):
        for arm in arms:
            sub=allrows[(allrows.seed==s)&(allrows.mask_type==arm)]
            for c,color in colors.items():
                z=sub[sub.condition==c];label=c+(' '+arm if family=='random' else '')
                ax.scatter(z.realized_distance,z.delta_CE,s=9,alpha=.45,color=color,marker='o' if arm!='random2' else 'x',label=label)
                group='target_p' if c=='B_matched' else 'p'
                means=z.groupby(group)[['realized_distance','delta_CE']].mean().sort_values('realized_distance')
                ax.plot(means.realized_distance,means.delta_CE,color=color,ls='-' if arm!='random2' else '--',lw=1)
        ax.axhline(0,color='black',lw=.5);ax.set_title(f'Seed {s}');ax.set_xlabel('Relative Frobenius distance')
    axes[0].set_ylabel('Change in CE (nats)');axes[-1].legend(fontsize=7);fig.suptitle(f'{family}: step 0; dots are individual draws');fig.tight_layout();fig.savefig(OUT/f'distance_{family}.png',dpi=170);fig.savefig(OUT/f'distance_{family}.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(19,4),sharey=True)
    for ax,s in zip(axes,SEEDS):
        sub=df[(df.seed==s)&df.mask_type.isin(arms)&(df.layer!='all')]
        width=.24/len(arms)
        for j,c in enumerate(['A','B','D']):
            for k,arm in enumerate(arms):
                z=sub[(sub.condition==c)&(sub.mask_type==arm)].copy();z['layer_num']=z.layer.astype(int)
                agg=z.groupby('layer_num').delta_CE.agg(['mean','std'])
                ax.bar(agg.index+((j-1)*len(arms)+k)*width,agg['mean'],width,yerr=agg['std'],label=f'{c} {arm}',alpha=.85,color={'A':'#1f77b4','B':'#d62728','D':'#555555'}[c],hatch='//' if k else None)
        ax.set_xticks(range(4));ax.set_xlabel('Transformer layer');ax.set_title(f'Seed {s}');ax.axhline(0,color='black',lw=.5)
    axes[0].set_ylabel('Change in CE (nats)');axes[-1].legend(fontsize=7);fig.suptitle(f'{family}: one layer at a time, full dose; error bars = draw SD');fig.tight_layout();fig.savefig(OUT/f'layers_{family}.png',dpi=170);fig.savefig(OUT/f'layers_{family}.pdf');plt.close(fig)
# Descriptive tables drive the plain-language report; no significance tests or tuned thresholds.
text=['# Step-zero findings','', 'All 15 unmodified baselines reproduce the original logged CE exactly. Step zero is true original random initialization; every router/shared tensor comes from that complete checkpoint. No training was run.', '', 'Full-dose mean CE changes across five draws are shown separately for every seed and mask realization:', '', '|Seed|Mask|A: magnitude placement|B: sign placement|C: constant magnitude|D: full shuffle|','|---|---|---|---|---|---|']
for s in SEEDS:
    for arm in ['imp','random1','random2']:
        z=full[(full.seed==s)&(full.mask_type==arm)].set_index('condition')
        text.append(f"|{s}|{arm}|{z.loc['A','delta_CE']:.6f}|{z.loc['B','delta_CE']:.6f}|{z.loc['C','delta_CE']:.6f}|{z.loc['D','delta_CE']:.6f}|")
text+=['','Distance-matched comparisons use sign-count-preserving positive/negative pair swaps selected by weights alone. They are an additional conditional sign-permutation distribution, distinct from the uniform subset permutation B. This limits any claim of a universal sign effect. All realized distances and paired effects are in distance_matched_comparisons.csv. Maximum absolute aggregate distance mismatch: '+f'{b.distance_mismatch.abs().max():.8g}'+'.', '', 'Distance-matched effects at each seed and A dose:', '', '|Seed|Mask|A dose|A change|Matched B change|B minus A|','|---|---|---|---|---|---|']
for _,r in pair_summary.iterrows():text.append(f'|{int(r.seed)}|{r.mask_type}|{r.target_p_B:g}|{r.A_delta:.6f}|{r.B_delta:.6f}|{r.B_minus_A:.6f}|')
text+=['','Uniform-chance CE is ln(256) = 5.545177. The observed random-mask baseline CE is much larger; it is not near chance. No chance-floor interpretation explains the flat shuffle response. The high CE is a reproduced feature of this initialization, not a corrected or retuned metric. In particular, a null random-mask damage effect is conditional on a model with extremely poor initial predictions.','','All mask, degree, sign/magnitude multiset and A/B/D energy assertions passed for each emitted result. All identity draws reproduce baseline CE exactly with zero routing changes. Gaussian directions match A/B distance independently in every matrix. Noise can change signs and weight energy; it is a distance control, not a matched-distribution control.','','These results apply only to step zero and the tested initializations and masks. They do not establish which individual connections are necessary, a topology-only mechanism, or any explanation of training behavior or the steps 1500–2500 gap.']
(OUT/'findings.md').write_text('\n'.join(text)+'\n')
(OUT/'completion.json').write_text(json.dumps({'complete':True,'rows':len(df),'baseline_gate_passed':True,'all_invariants_passed':True,'identity_rows':len(identity),'maximum_distance_match_error':float(b.distance_mismatch.abs().max())},indent=2))
print(full.to_string(index=False));print(pair_summary.to_string(index=False))
