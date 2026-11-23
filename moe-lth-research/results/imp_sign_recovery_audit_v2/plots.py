from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9})
def save(fig,name):
 fig.tight_layout();fig.savefig(OUT/f'{name}.png',dpi=170);fig.savefig(OUT/f'{name}.pdf');plt.close(fig)
for analysis in ('discovery','dense'):
 d=pd.read_csv(OUT/f'{analysis}_by_layer_projection.csv')
 if analysis=='dense':
  extra=pd.read_csv(OUT/'dense_legacy_random_by_layer_projection.csv');d=pd.concat([d,extra],ignore_index=True)
 groups=[('retained','retained'),('newly_pruned','newly pruned'),('random_survivors','random from alive set')] if analysis=='discovery' else [('eventual_imp_survivors','eventual IMP80 survivors'),('eventual_imp_pruned','eventual IMP80 pruned'),('legacy_random_survivors','legacy random survivors')]
 fig,axes=plt.subplots(2,5,figsize=(19,7),sharey=True)
 for col,seed in enumerate(SEEDS):
  for row,projection in enumerate(('fc1','fc2')):
   ax=axes[row,col];z=d[(d.seed==seed)&(d.projection==projection)]
   for j,(group,label) in enumerate(groups):
    v=z[z.group==group].groupby('layer').sign_agreement.agg(['mean','std']).reindex(range(4))
    ax.bar(np.arange(4)+(j-1)*.25,v['mean'],.24,yerr=v['std'].fillna(0),label=label,color=['#176ba0','#e07a24','#7c8c3d'][j])
   ax.set_ylim(0,1.03);ax.set_xticks(range(4));ax.set_xlabel('Layer');ax.set_title(f'Seed {seed}, {projection}')
   if col==0:ax.set_ylabel('Sign agreement')
 axes[0,-1].legend(fontsize=7,loc='lower left')
 fig.suptitle(f'E32/IMP80: {analysis}, round {7 if analysis=="discovery" else 0}; random error bars = realization SD')
 save(fig,f'{analysis}_sign_agreement_by_layer')
# Legacy full-shuffle interactions only. New Phase2 arms are not measured yet.
z=pd.read_csv(OUT/'legacy_order_averaged_realizations.csv');means=z.groupby(['seed','step']).paired_interaction.mean().reset_index();ci=pd.read_csv(OUT/'legacy_interaction_intervals.csv')
fig,axes=plt.subplots(1,2,figsize=(12,4),gridspec_kw={'width_ratios':[3,1]})
for seed in SEEDS:
 t=means[(means.seed==seed)&(means.step>0)];axes[0].plot(t.step,t.paired_interaction,lw=1,alpha=.6,label=f'Seed {seed}')
c=ci[ci.step>0];axes[0].plot(c.step,c.mean_interaction,'k',lw=2,label='Five-seed mean');axes[0].fill_between(c.step,c.bootstrap_low,c.bootstrap_high,color='black',alpha=.12,label='Nested bootstrap 95% interval');axes[0].axvspan(1500,2500,color='#e6bc5c',alpha=.12);axes[0].axhline(0,color='gray',lw=.5);axes[0].set_xlabel('Recovery step');axes[0].set_ylabel('Full-shuffle interaction (nats)');axes[0].legend(fontsize=7,ncol=2)
b=means[means.step==0];axes[1].bar(b.seed.astype(str),b.paired_interaction,color='#777777');axes[1].set_title('Step 0, separate scale');axes[1].set_ylabel('Interaction (nats)');fig.suptitle('E32/IMP80: legacy full shuffle; orders averaged within seed, not independent')
save(fig,'legacy_full_shuffle_interaction')
d=pd.read_csv(OUT/'legacy_absolute_ce.csv');comp=pd.read_csv(OUT/'legacy_unpruned_comparator.csv')
for zoom in (False,True):
 fig,axes=plt.subplots(2,5,figsize=(19,7),sharey='row')
 for col,seed in enumerate(SEEDS):
  for row,family in enumerate(('imp','random')):
   ax=axes[row,col];sub=d[(d.seed==seed)&(d.family==family)]
   if zoom:sub=sub[sub.step>0]
   for placement,color in [('original','#176ba0'),('shuffled','#d85631')]:
    g=sub[sub.placement==placement].groupby('step').cross_entropy.mean();ax.plot(g.index,g.values,color=color,label=placement)
   g=comp[comp.seed==seed].groupby('step').unpruned_CE.mean()
   if zoom:g=g[g.index>0]
   ax.plot(g.index,g.values,color='gray',ls='--',lw=1,label='unpruned comparator')
   if not zoom:ax.set_yscale('log')
   ax.set_title(f'Seed {seed}, {family}');ax.set_xlabel('Recovery step')
   if col==0:ax.set_ylabel('CE (nats)'+('' if zoom else ', log axis'))
 axes[0,-1].legend(fontsize=7);fig.suptitle('E32/IMP80: legacy absolute CE; per-seed means over nested realizations/orders'+('; step 0 omitted from zoom' if zoom else ''))
 save(fig,'legacy_absolute_ce_zoom' if zoom else 'legacy_absolute_ce')
t=pd.read_csv(OUT/'legacy_time_to_threshold.csv');rng=np.random.default_rng(32083)
fig,axes=plt.subplots(3,5,figsize=(19,10),sharey=True)
labels=['IMP original','IMP shuffle','random original','random shuffle']
for col,seed in enumerate(SEEDS):
 for row,anchor in enumerate((1000,2000,5000)):
  ax=axes[row,col];s=t[(t.seed==seed)&(t.target_anchor_step==anchor)]
  for j,(family,placement) in enumerate([('imp','original'),('imp','shuffled'),('random','original'),('random','shuffled')]):
   mask=s.variant.str.startswith('imp' if family=='imp' else 'random')
   mask &= s.variant.str.contains('shuffle') if placement=='shuffled' else ~s.variant.str.contains('shuffle')
   z=s[mask];values=z.time_to_threshold.fillna(5200)
   ax.scatter(j+rng.uniform(-.12,.12,len(z)),values,s=13,color='#176ba0' if placement=='original' else '#d85631',alpha=.7)
   missing=z.time_to_threshold.isna().sum()
   if missing:ax.text(j,5300,str(missing),ha='center',fontsize=7)
  ax.axhline(5200,color='gray',ls=':',lw=.5);ax.set_xticks(range(4),labels,rotation=45,ha='right',fontsize=7);ax.set_ylim(0,5500);ax.set_title(f'Seed {seed}, target from step {anchor}')
  if col==0:ax.set_ylabel('First-crossing step; 5200 = not reached')
fig.suptitle('E32/IMP80: legacy time-to-threshold; dots are dependent orders/nested realizations, not independent seeds')
save(fig,'legacy_time_to_threshold')
# Audit visualization keeps dense and each fixed random-mask baseline visible.
h=pd.read_csv(OUT/'high_ce_audit.csv');fig,ax=plt.subplots(figsize=(9,4))
for j,arm in enumerate(('dense','imp_original','random1_original','random2_original')):
 z=h[h.arm==arm].set_index('seed').loc[list(SEEDS)];ax.bar(np.arange(5)+(j-1.5)*.18,z.CE,.17,label=arm)
ax.axhline(math.log(256),ls='--',color='black',lw=.8,label='Uniform chance');ax.set_xticks(range(5),[str(s) for s in SEEDS]);ax.set_ylabel('Step-0 CE (nats)');ax.set_xlabel('Seed');ax.legend(fontsize=8,ncol=2);ax.set_title('E32/IMP80 audit: all baselines copy the input token with extreme confidence')
save(fig,'step0_ce_audit')
print('plots complete')
