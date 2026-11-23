import os
"""Numerical scientific interpretation of exported tables; no input mutation."""
from pathlib import Path
import json,math
import numpy as np,pandas as pd
from scipy import stats
O=Path(os.environ.get('DATASET_A_ANALYSIS_OUT',str(Path(__file__).resolve().parents[1])))
f=pd.read_csv(O/'canonical_lower_sparsity.csv');ss=pd.read_csv(O/'setting_summary.csv');c=pd.read_csv(O/'canonical_combined.csv');cs=pd.read_csv(O/'combined_setting_summary.csv');ac=pd.read_csv(O/'paired_age_contrasts.csv');sc=pd.read_csv(O/'paired_sparsity_contrasts.csv');bc=pd.read_csv(O/'exploratory_age_selection.csv');rc=pd.read_csv(O/'replicated_candidate_counts.csv')
def interval(x):
 x=np.asarray(x);m=x.mean();s=x.std(ddof=1)/np.sqrt(len(x));return m,m-stats.t.ppf(.975,len(x)-1)*s,m+stats.t.ppf(.975,len(x)-1)*s
# Marginal inference averages repeated measurements within seed first.
agg=[]
for dims in [['num_experts'],['sparsity'],['num_experts','sparsity']]:
 for k,g in f.groupby(dims):
  x=g.groupby('seed').ticket_gap.mean();m,l,u=interval(x);agg.append(dict(grouping='+'.join(dims),group=str(k),n=5,mean_gap=m,ci95_lower=l,ci95_upper=u,passing_strict=int((g.ticket_gap<=0).sum()),denominator=len(g)))
pd.DataFrame(agg).to_csv(O/'marginal_seed_summary.csv',index=False)
# Fixed contrasts averaged over sparsities within each seed.
bound=[]
for e,g in f.groupby('num_experts'):
 p=g.groupby(['seed','router_age'])[['ticket_gap','sparse_final_validation_loss','dense_final_validation_loss']].mean()
 x=p.xs(100,level=1)-p.xs(0,level=1);m,l,u=interval(x.ticket_gap)
 bound.append(dict(num_experts=e,mean_R100_minus_R0_gap=m,ci95_lower=l,ci95_upper=u,mean_sparse_change=x.sparse_final_validation_loss.mean(),mean_dense_change=x.dense_final_validation_loss.mean(),seeds_R100_better=int((x.ticket_gap<0).sum()),p=stats.ttest_1samp(x.ticket_gap,0).pvalue))
def holm(p):
 p=np.asarray(p);i=np.argsort(p);o=np.empty(len(p));o[i]=np.minimum(1,np.maximum.accumulate(p[i]*(len(p)-np.arange(len(p)))));return o
bound=pd.DataFrame(bound);bound['p_holm_5']=holm(bound.p);bound.to_csv(O/'marginal_boundary_contrasts.csv',index=False)
# Descriptive trend shape and selected minimum comparison are explicitly exploratory.
shapes=[]
for (e,sp),g in f.groupby(['num_experts','sparsity']):
 p=g.pivot(index='seed',columns='router_age',values='ticket_gap');m=p.mean();diff=np.diff(m);rho=[stats.spearmanr(p.columns,row).statistic for _,row in p.iterrows()];a=int(m.idxmin());selected=[]
 for b in [0,100]:
  if a==b:continue
  z=ac[(ac.num_experts==e)&(ac.sparsity==sp)&(ac.age_a==min(a,b))&(ac.age_b==max(a,b))].iloc[0];sign=1 if a<b else -1
  selected.append(dict(boundary=b,selected_minus_boundary=sign*z.mean_gap_difference,selected_minus_boundary_sparse=sign*z.mean_sparse_difference,selected_minus_boundary_dense=sign*z.mean_dense_difference,p_holm_1100=z.p_holm_family_1100,seeds_selected_better=int(z.seeds_a_better) if sign==1 else 5-int(z.seeds_a_better)))
 shapes.append(dict(num_experts=e,sparsity=sp,best_gap_age=a,observed_mean_range=m.max()-m.min(),mean_curve_shape='monotonic increasing' if (diff>=0).all() else 'monotonic decreasing' if (diff<=0).all() else 'nonmonotonic',seed_spearman_rho=json.dumps(dict(zip([int(x) for x in p.index],rho))),positive_seed_rho=sum(x>0 for x in rho),negative_seed_rho=sum(x<0 for x in rho),boundary_evidence=json.dumps(selected),interior_better_than_both_holm=a not in [0,100] and all(z['selected_minus_boundary']<0 and z['p_holm_1100']<.05 for z in selected)))
shapes=pd.DataFrame(shapes);shapes.to_csv(O/'age_shape_summary.csv',index=False)
strong=ss.nsmallest(5,'mean_gap');neg=ss[ss.mean_gap<0];positive_loo=int((neg.max_leave_one_seed_out_mean>0).sum());stable=int((neg.max_leave_one_seed_out_mean<0).sum())
strictseed=f.assign(strict=f.ticket_gap<=0).groupby('seed').strict.sum();highest=pd.read_csv(O/'highest_observed_sparsity.csv');hf=highest[(highest.threshold=='ppl_5pct')&(highest.criterion=='at_least_5_seeds')]
sparse_con=[]
for (e,a),g in f.groupby(['num_experts','router_age']):
 p=g.pivot(index='seed',columns='sparsity',values='ticket_gap');x=p[.65]-p[.5];m,l,u=interval(x);sparse_con.append(dict(num_experts=e,router_age=a,mean_gap_65_minus_50=m,ci95_lower=l,ci95_upper=u,seeds_65_worse=int((x>0).sum()),p=stats.ttest_1samp(x,0).pvalue))
sparse_con=pd.DataFrame(sparse_con);sparse_con['p_holm_55']=holm(sparse_con.p);sparse_con.to_csv(O/'paired_65_vs_50.csv',index=False)
old= c[c.analysis_origin=='original_finalized'];origsummary=old.groupby('sparsity').ticket_gap.agg(['mean','min','max']);lowmean=f.groupby('sparsity').ticket_gap.mean()
text=f'''# Numerical interpretation

These conclusions use all five seeds and the validated paired endpoint tables. Marginal confidence intervals average repeated ages/sparsities within seed first (five source-seed means), avoiding 1,100-run pseudoreplication.

50–65% expert sparsity sometimes matches or beats dense (212 strict endpoints) but does not do so across all five seeds at any of the 220 settings. Every architecture's average gap is positive. Strict matches are distributed across seed IDs as follows:

```
{strictseed.to_string()}
```

The largest seed-ID contribution is {int(strictseed.max())}/212 strict matches ({100*strictseed.max()/212:.1f}%), from seed {int(strictseed.idxmax())}; matches span {f.loc[f.ticket_gap<=0,['num_experts','seed']].drop_duplicates().shape[0]}/25 unique source trajectories. These counts quantify concentration across correlated runs. {len(neg)} settings have negative mean gaps; {stable} remain negative after deleting any one seed, while {positive_loo} reverse sign after deleting at least one seed. The strongest selected mean settings are:

```
{strong[['num_experts','router_age','sparsity','mean_gap','sd_gap','upper95_one_sided','strict_passing_seeds','strict_p_one_sided','strict_p_holm','max_leave_one_seed_out_mean']].to_string(index=False)}
```

Architecture and sparsity marginal summaries (pointwise t intervals, n=5 per group):

```
{pd.DataFrame(agg)[pd.DataFrame(agg).grouping.isin(['num_experts','sparsity'])].to_string(index=False)}
```

Expert-count effects are not monotonic: 16E has the smallest average gap, 32E the largest. This is a comparison of the tested architectures under equal global attempted budgets, not equal per-expert exposure or a causal expert-count mechanism.

R100 versus R0, averaging the four sparsities within seed, decomposes as:

```
{bound.to_string(index=False)}
```

A negative gap change means R100 improves relative sparse performance; a negative sparse change alone means better absolute sparse CE. Relative changes must be read with dense changes. The separate five-architecture Holm family above applies only to these fixed marginal boundary comparisons; it does not validate selected interior ages.

Interior minima of the mean occur in {int((~bc.best_gap_age.isin([0,100])).sum())}/20 curves, but only {int(shapes.interior_better_than_both_holm.sum())} beat both endpoints after Holm over all 1,100 age comparisons. {int((bc.seeds_selecting_mean_winner==5).sum())}/20 mean-winning ages are selected by all five seeds. Mean curve shapes are {shapes.mean_curve_shape.value_counts().to_dict()}; nonmonotonic sample means alone are not reproducible optima. A curve can be statistically unresolved without being shown equivalent or truly flat. Seed rank trends and selected age-versus-boundary sparse/dense decompositions are in age_shape_summary.csv; all seed views are plotted. Lowest absolute sparse CE and lowest gap occur at different mean ages in {int((bc.best_gap_age!=bc.best_sparse_age).sum())}/20 curves.

At 65% versus 50%, {int((sparse_con.mean_gap_65_minus_50>0).sum())}/55 architecture/age settings have worse mean gaps; {int((sparse_con.seeds_65_worse==5).sum())}/55 show this in all five seeds. {int((sparse_con.p_holm_55<.05).sum())}/55 differences survive the separate two-sided Holm family. Full paired contrasts are in paired_65_vs_50.csv. Adjacent sparsity contrasts additionally connect all ten measured sparsities.

Original higher-sparsity mean-gap inventory:

```
{origsummary.to_string()}
```

No strict all-five-seed match exists in the lower grid. In the combined grid, strict all-five matches: {int((cs.strict_passing_seeds==5).sum())}/550. Highest observed all-five 5% candidate sparsities by architecture are descriptive maxima over measured cells:

```
{hf.groupby('num_experts').highest_observed_tested_sparsity.agg(['min','max','count']).to_string()}
```

The count column is the number of router ages with any qualifying tested sparsity. The min/max are over those ages, not an interpolated sparsity threshold. highest_observed_sparsity.csv gives the exact architecture/age criterion values, including individual seeds and 3/4/5 seed replication, and adjusted inference. A setting selected by its best observed age is subject to selection bias.

The three settings surviving the joint 880-test Holm family for the exploratory 5% tolerance are 8E/R70/50%, 8E/R70/55%, and 8E/R80/50%. Their mean gaps are −0.002257, −0.001465, and +0.005077 nats/token; adjusted p values are 0.040157, 0.030503, and 0.022485. These support mean performance within the stated exploratory tolerance, not strict superiority. The most negative selected mean, 16E/R60/50%, is −0.015776 with SD 0.038369 and one-sided upper95 +0.020804; only 2/5 seeds strictly match, so it is not a replicated strict ticket.

The smallest warranted follow-up is fixed-setting replication on fresh source seeds and an independent held-out corpus. Prioritize settings with all-five 5% matches and adjusted mean-below-5% evidence; strict negative means have weaker support. Test an interior candidate against both fixed endpoints only if its multiplicity-adjusted contrasts justify it. Do not expand to a full new grid or infer mechanisms from these endpoints.
'''
(O/'scientific_interpretation.md').write_text(text)
with (O/'findings.md').open('a') as h:h.write('\n\n'+text)
print(text[:3500])
