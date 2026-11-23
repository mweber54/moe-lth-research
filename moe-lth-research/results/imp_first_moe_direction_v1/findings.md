# First-MoE per-token direction study

This follow-up and layer 0 were chosen after prior interface/amplitude results; the new configuration and numerical eligibility rule are sealed before intervention losses. Validation has been repeatedly examined and is not an independent test set. Source seeds (n=5) are the replication unit; tokens, sequences, batches, contexts and cells are nested observations. Directions are those of the complete native gated contribution in native residual coordinates, with no learned alignment or transformation. On S only, norm and direction donors are crossed; outside S every variant equals the recipient's own native contribution. Unsupported positions are not tested and are not shown harmless. Other-diagonal hybrids are common-support substitutions, not necessarily full-donor transplants. Cross-recipient contrasts include residual context, downstream parameters and unsupported-position backgrounds; they are not downstream-weight attribution. Each sparse/unpruned trajectory uses its own norms/directions and is not an equal-sized cross-model perturbation. Downstream replay fixes each recipient's own decisions and computes experts on current inputs; mode differences do not isolate one routing mechanism. Algebraic decompositions are not percentages of training recovery. Frozen experts did not learn new weights. Direction effects may reflect changed assignments or different inputs to the same frozen experts; they do not isolate upstream cause, semantic content, all recovery, winning tickets, normal-training attainability, or an online pruning criterion. No subsequent experiment is launched.

Positive direction benefits mean recovered directions lower whole-population CE at fixed per-token norm targets on S, with recipient-native backgrounds elsewhere. Per-seed values precede aggregates.


## Short verdict

A–D: Yes. Recovered directions improve whole-population CE at both fixed per-token norm targets on common eligible positions in all five source seeds, both recipient contexts, and both model types. All signs persist under recipient-baseline downstream replay. Native sparse mean direction benefits at starting/recovered norms are 0.007680/0.013474 CE in the starting recipient and 0.041681/0.068865 in the recovered recipient. These are descriptive source-seed means, not independent cell replications.

E: Common-position coverage is 85.53–90.08% for sparse and 89.07–92.86% for unpruned. Across sparse seeds, S contains 96.66–97.37% of starting and 97.09–98.46% of recovered contribution energy; unpruned ranges are 97.27–98.82% and 97.33–98.97%. Exclusions are exact zeros; numerical-floor exclusions are zero. Unsupported positions retain recipient-native contributions, so their direction effects remain untested.

F: No. Under native downstream routing, sparse direction benefits are smaller than unpruned in all five seeds at starting norms in both recipients, and at recovered norms in the starting recipient. At recovered norms in the recovered recipient, sparse benefits are larger in only two of five seeds. Replay preserves the absence of a consistent sparse advantage.

G: Recovered token-specific norms help under either direction in every seed and context tested, but their benefit depends on the direction. Interaction is positive in all recovered-recipient seeds and all unpruned seeds; sparse starting-recipient seed61 has a small negative interaction in both routing modes. Thus positive norm benefits coexist with norm–direction interaction.

The supported conclusion concerns usefulness of the resulting native-coordinate contribution directions on S. It does not identify their upstream cause or explain all recovery. Validation is repeatedly used, not an independent test set. No follow-up is launched or recommended here.

## NATIVE_DOWNSTREAM


### sparse, start recipient

| Seed | D start norm | D end norm | N start dir | N end dir | B common | I |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.005586 | 0.009483 | 0.112625 | 0.116523 | 0.122109 | 0.003898 |
| 17 | 0.012524 | 0.021288 | 0.126477 | 0.135242 | 0.147765 | 0.008765 |
| 29 | 0.005409 | 0.009272 | 0.109850 | 0.113714 | 0.119122 | 0.003863 |
| 43 | 0.012231 | 0.025508 | 0.128354 | 0.141631 | 0.153862 | 0.013277 |
| 61 | 0.002650 | 0.001818 | 0.097873 | 0.097042 | 0.099692 | -0.000831 |
D_start_norm: mean 0.007680, median 0.005586, range [0.002650, 0.012524], sample SD 0.004445; n=5 source seeds.

D_end_norm: mean 0.013474, median 0.009483, range [0.001818, 0.025508], sample SD 0.009687; n=5 source seeds.

N_start_dir: mean 0.115036, median 0.112625, range [0.097873, 0.128354], sample SD 0.012605; n=5 source seeds.

N_end_dir: mean 0.120830, median 0.116523, range [0.097042, 0.141631], sample SD 0.017857; n=5 source seeds.

B_common: mean 0.128510, median 0.122109, range [0.099692, 0.153862], sample SD 0.022210; n=5 source seeds.

I: mean 0.005794, median 0.003898, range [-0.000831, 0.013277], sample SD 0.005386; n=5 source seeds.

D_start_norm positive in 5/5 seeds, negative in 0/5.

D_end_norm positive in 5/5 seeds, negative in 0/5.

N_start_dir positive in 5/5 seeds, negative in 0/5.

N_end_dir positive in 5/5 seeds, negative in 0/5.

Norm–direction interaction I positive in 4/5 and negative in 1/5; norm allocation is not assumed independent of chosen direction.


### sparse, end recipient

| Seed | D start norm | D end norm | N start dir | N end dir | B common | I |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.028025 | 0.043425 | 0.065375 | 0.080774 | 0.108799 | 0.015399 |
| 17 | 0.051575 | 0.079296 | 0.087930 | 0.115651 | 0.167226 | 0.027721 |
| 29 | 0.033940 | 0.054613 | 0.052918 | 0.073592 | 0.107531 | 0.020673 |
| 43 | 0.052636 | 0.110339 | 0.069818 | 0.127521 | 0.180157 | 0.057702 |
| 61 | 0.042227 | 0.056651 | 0.072172 | 0.086596 | 0.128823 | 0.014424 |
D_start_norm: mean 0.041681, median 0.042227, range [0.028025, 0.052636], sample SD 0.010777; n=5 source seeds.

D_end_norm: mean 0.068865, median 0.056651, range [0.043425, 0.110339], sample SD 0.026591; n=5 source seeds.

N_start_dir: mean 0.069643, median 0.069818, range [0.052918, 0.087930], sample SD 0.012637; n=5 source seeds.

N_end_dir: mean 0.096827, median 0.086596, range [0.073592, 0.127521], sample SD 0.023445; n=5 source seeds.

B_common: mean 0.138507, median 0.128823, range [0.107531, 0.180157], sample SD 0.033524; n=5 source seeds.

I: mean 0.027184, median 0.020673, range [0.014424, 0.057702], sample SD 0.017859; n=5 source seeds.

D_start_norm positive in 5/5 seeds, negative in 0/5.

D_end_norm positive in 5/5 seeds, negative in 0/5.

N_start_dir positive in 5/5 seeds, negative in 0/5.

N_end_dir positive in 5/5 seeds, negative in 0/5.

Norm–direction interaction I positive in 5/5 and negative in 0/5; norm allocation is not assumed independent of chosen direction.


### unpruned, start recipient

| Seed | D start norm | D end norm | N start dir | N end dir | B common | I |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.025440 | 0.032139 | 0.002945 | 0.009645 | 0.035084 | 0.006700 |
| 17 | 0.024433 | 0.034118 | 0.004764 | 0.014449 | 0.038882 | 0.009686 |
| 29 | 0.027142 | 0.032779 | 0.004708 | 0.010345 | 0.037487 | 0.005637 |
| 43 | 0.025633 | 0.033110 | 0.003088 | 0.010566 | 0.036199 | 0.007477 |
| 61 | 0.021634 | 0.033749 | 0.007559 | 0.019675 | 0.041308 | 0.012116 |
D_start_norm: mean 0.024856, median 0.025440, range [0.021634, 0.027142], sample SD 0.002045; n=5 source seeds.

D_end_norm: mean 0.033179, median 0.033110, range [0.032139, 0.034118], sample SD 0.000783; n=5 source seeds.

N_start_dir: mean 0.004613, median 0.004708, range [0.002945, 0.007559], sample SD 0.001859; n=5 source seeds.

N_end_dir: mean 0.012936, median 0.010566, range [0.009645, 0.019675], sample SD 0.004209; n=5 source seeds.

B_common: mean 0.037792, median 0.037487, range [0.035084, 0.041308], sample SD 0.002425; n=5 source seeds.

I: mean 0.008323, median 0.007477, range [0.005637, 0.012116], sample SD 0.002589; n=5 source seeds.

D_start_norm positive in 5/5 seeds, negative in 0/5.

D_end_norm positive in 5/5 seeds, negative in 0/5.

N_start_dir positive in 5/5 seeds, negative in 0/5.

N_end_dir positive in 5/5 seeds, negative in 0/5.

Norm–direction interaction I positive in 5/5 and negative in 0/5; norm allocation is not assumed independent of chosen direction.


### unpruned, end recipient

| Seed | D start norm | D end norm | N start dir | N end dir | B common | I |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.048173 | 0.054575 | 0.007827 | 0.014228 | 0.062402 | 0.006402 |
| 17 | 0.053359 | 0.063363 | 0.008384 | 0.018388 | 0.071747 | 0.010004 |
| 29 | 0.051418 | 0.058057 | 0.005655 | 0.012294 | 0.063712 | 0.006639 |
| 43 | 0.055700 | 0.064298 | 0.006744 | 0.015342 | 0.071041 | 0.008598 |
| 61 | 0.055003 | 0.065395 | 0.007935 | 0.018327 | 0.073330 | 0.010392 |
D_start_norm: mean 0.052731, median 0.053359, range [0.048173, 0.055700], sample SD 0.003036; n=5 source seeds.

D_end_norm: mean 0.061137, median 0.063363, range [0.054575, 0.065395], sample SD 0.004626; n=5 source seeds.

N_start_dir: mean 0.007309, median 0.007827, range [0.005655, 0.008384], sample SD 0.001103; n=5 source seeds.

N_end_dir: mean 0.015716, median 0.015342, range [0.012294, 0.018388], sample SD 0.002646; n=5 source seeds.

B_common: mean 0.068446, median 0.071041, range [0.062402, 0.073330], sample SD 0.005011; n=5 source seeds.

I: mean 0.008407, median 0.008598, range [0.006402, 0.010392], sample SD 0.001849; n=5 source seeds.

D_start_norm positive in 5/5 seeds, negative in 0/5.

D_end_norm positive in 5/5 seeds, negative in 0/5.

N_start_dir positive in 5/5 seeds, negative in 0/5.

N_end_dir positive in 5/5 seeds, negative in 0/5.

Norm–direction interaction I positive in 5/5 and negative in 0/5; norm allocation is not assumed independent of chosen direction.


### Paired sparse-minus-unpruned direction benefits

start recipient D_start_norm: seed order 7,17,29,43,61 = -0.019854, -0.011909, -0.021733, -0.013403, -0.018984; sparse larger in 0/5 seeds. Each model has its own norms, directions and S.

start recipient D_end_norm: seed order 7,17,29,43,61 = -0.022656, -0.012830, -0.023507, -0.007603, -0.031931; sparse larger in 0/5 seeds. Each model has its own norms, directions and S.

end recipient D_start_norm: seed order 7,17,29,43,61 = -0.020148, -0.001784, -0.017478, -0.003063, -0.012776; sparse larger in 0/5 seeds. Each model has its own norms, directions and S.

end recipient D_end_norm: seed order 7,17,29,43,61 = -0.011150, 0.015933, -0.003444, 0.046041, -0.008743; sparse larger in 2/5 seeds. Each model has its own norms, directions and S.


## RECIPIENT_BASELINE_REPLAY


### sparse, start recipient

| Seed | D start norm | D end norm | N start dir | N end dir | B common | I |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.005474 | 0.008988 | 0.112324 | 0.115838 | 0.121312 | 0.003514 |
| 17 | 0.012582 | 0.020708 | 0.126722 | 0.134848 | 0.147430 | 0.008125 |
| 29 | 0.005222 | 0.008877 | 0.108577 | 0.112231 | 0.117454 | 0.003655 |
| 43 | 0.012298 | 0.025035 | 0.128005 | 0.140742 | 0.153040 | 0.012737 |
| 61 | 0.002557 | 0.001542 | 0.097871 | 0.096856 | 0.099414 | -0.001015 |
D_start_norm: mean 0.007627, median 0.005474, range [0.002557, 0.012582], sample SD 0.004541; n=5 source seeds.

D_end_norm: mean 0.013030, median 0.008988, range [0.001542, 0.025035], sample SD 0.009600; n=5 source seeds.

N_start_dir: mean 0.114700, median 0.112324, range [0.097871, 0.128005], sample SD 0.012727; n=5 source seeds.

N_end_dir: mean 0.120103, median 0.115838, range [0.096856, 0.140742], sample SD 0.017776; n=5 source seeds.

B_common: mean 0.127730, median 0.121312, range [0.099414, 0.153040], sample SD 0.022233; n=5 source seeds.

I: mean 0.005403, median 0.003655, range [-0.001015, 0.012737], sample SD 0.005221; n=5 source seeds.

D_start_norm positive in 5/5 seeds, negative in 0/5.

D_end_norm positive in 5/5 seeds, negative in 0/5.

N_start_dir positive in 5/5 seeds, negative in 0/5.

N_end_dir positive in 5/5 seeds, negative in 0/5.

Norm–direction interaction I positive in 4/5 and negative in 1/5; norm allocation is not assumed independent of chosen direction.


### sparse, end recipient

| Seed | D start norm | D end norm | N start dir | N end dir | B common | I |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.027692 | 0.043292 | 0.066057 | 0.081657 | 0.109349 | 0.015600 |
| 17 | 0.051919 | 0.079264 | 0.090623 | 0.117969 | 0.169888 | 0.027346 |
| 29 | 0.034028 | 0.054387 | 0.053249 | 0.073608 | 0.107636 | 0.020359 |
| 43 | 0.056712 | 0.109585 | 0.085401 | 0.138274 | 0.194986 | 0.052874 |
| 61 | 0.042243 | 0.057004 | 0.072671 | 0.087432 | 0.129675 | 0.014761 |
D_start_norm: mean 0.042519, median 0.042243, range [0.027692, 0.056712], sample SD 0.012060; n=5 source seeds.

D_end_norm: mean 0.068707, median 0.057004, range [0.043292, 0.109585], sample SD 0.026317; n=5 source seeds.

N_start_dir: mean 0.073600, median 0.072671, range [0.053249, 0.090623], sample SD 0.015008; n=5 source seeds.

N_end_dir: mean 0.099788, median 0.087432, range [0.073608, 0.138274], sample SD 0.027288; n=5 source seeds.

B_common: mean 0.142307, median 0.129675, range [0.107636, 0.194986], sample SD 0.038677; n=5 source seeds.

I: mean 0.026188, median 0.020359, range [0.014761, 0.052874], sample SD 0.015733; n=5 source seeds.

D_start_norm positive in 5/5 seeds, negative in 0/5.

D_end_norm positive in 5/5 seeds, negative in 0/5.

N_start_dir positive in 5/5 seeds, negative in 0/5.

N_end_dir positive in 5/5 seeds, negative in 0/5.

Norm–direction interaction I positive in 5/5 and negative in 0/5; norm allocation is not assumed independent of chosen direction.


### unpruned, start recipient

| Seed | D start norm | D end norm | N start dir | N end dir | B common | I |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.024868 | 0.031331 | 0.002872 | 0.009334 | 0.034202 | 0.006462 |
| 17 | 0.024338 | 0.033330 | 0.004294 | 0.013287 | 0.037625 | 0.008993 |
| 29 | 0.026337 | 0.032012 | 0.004584 | 0.010258 | 0.036595 | 0.005675 |
| 43 | 0.026866 | 0.033371 | 0.002371 | 0.008876 | 0.035742 | 0.006505 |
| 61 | 0.022214 | 0.034256 | 0.006613 | 0.018655 | 0.040869 | 0.012043 |
D_start_norm: mean 0.024925, median 0.024868, range [0.022214, 0.026866], sample SD 0.001835; n=5 source seeds.

D_end_norm: mean 0.032860, median 0.033330, range [0.031331, 0.034256], sample SD 0.001172; n=5 source seeds.

N_start_dir: mean 0.004147, median 0.004294, range [0.002371, 0.006613], sample SD 0.001664; n=5 source seeds.

N_end_dir: mean 0.012082, median 0.010258, range [0.008876, 0.018655], sample SD 0.004056; n=5 source seeds.

B_common: mean 0.037007, median 0.036595, range [0.034202, 0.040869], sample SD 0.002497; n=5 source seeds.

I: mean 0.007935, median 0.006505, range [0.005675, 0.012043], sample SD 0.002613; n=5 source seeds.

D_start_norm positive in 5/5 seeds, negative in 0/5.

D_end_norm positive in 5/5 seeds, negative in 0/5.

N_start_dir positive in 5/5 seeds, negative in 0/5.

N_end_dir positive in 5/5 seeds, negative in 0/5.

Norm–direction interaction I positive in 5/5 and negative in 0/5; norm allocation is not assumed independent of chosen direction.


### unpruned, end recipient

| Seed | D start norm | D end norm | N start dir | N end dir | B common | I |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.048079 | 0.054170 | 0.007961 | 0.014052 | 0.062131 | 0.006091 |
| 17 | 0.054489 | 0.063483 | 0.008267 | 0.017260 | 0.071749 | 0.008994 |
| 29 | 0.051127 | 0.057498 | 0.005673 | 0.012044 | 0.063171 | 0.006371 |
| 43 | 0.056589 | 0.064629 | 0.007002 | 0.015042 | 0.071631 | 0.008041 |
| 61 | 0.055296 | 0.065169 | 0.007867 | 0.017740 | 0.073036 | 0.009874 |
D_start_norm: mean 0.053116, median 0.054489, range [0.048079, 0.056589], sample SD 0.003465; n=5 source seeds.

D_end_norm: mean 0.060990, median 0.063483, range [0.054170, 0.065169], sample SD 0.004889; n=5 source seeds.

N_start_dir: mean 0.007354, median 0.007867, range [0.005673, 0.008267], sample SD 0.001051; n=5 source seeds.

N_end_dir: mean 0.015228, median 0.015042, range [0.012044, 0.017740], sample SD 0.002345; n=5 source seeds.

B_common: mean 0.068344, median 0.071631, range [0.062131, 0.073036], sample SD 0.005239; n=5 source seeds.

I: mean 0.007874, median 0.008041, range [0.006091, 0.009874], sample SD 0.001637; n=5 source seeds.

D_start_norm positive in 5/5 seeds, negative in 0/5.

D_end_norm positive in 5/5 seeds, negative in 0/5.

N_start_dir positive in 5/5 seeds, negative in 0/5.

N_end_dir positive in 5/5 seeds, negative in 0/5.

Norm–direction interaction I positive in 5/5 and negative in 0/5; norm allocation is not assumed independent of chosen direction.


### Paired sparse-minus-unpruned direction benefits

start recipient D_start_norm: seed order 7,17,29,43,61 = -0.019394, -0.011755, -0.021115, -0.014568, -0.019656; sparse larger in 0/5 seeds. Each model has its own norms, directions and S.

start recipient D_end_norm: seed order 7,17,29,43,61 = -0.022343, -0.012623, -0.023135, -0.008336, -0.032714; sparse larger in 0/5 seeds. Each model has its own norms, directions and S.

end recipient D_start_norm: seed order 7,17,29,43,61 = -0.020387, -0.002570, -0.017099, 0.000123, -0.013053; sparse larger in 1/5 seeds. Each model has its own norms, directions and S.

end recipient D_end_norm: seed order 7,17,29,43,61 = -0.010878, 0.015782, -0.003111, 0.044956, -0.008166; sparse larger in 2/5 seeds. Each model has its own norms, directions and S.


## Common-position and energy coverage

Seed 7 sparse: S 440496/492416 positions (0.894561); S/start nonzero 0.950000, S/end nonzero 0.963749; start/end contribution energy fractions 0.973736/0.984564; exact-zero exclusions 51920, numerical-floor exclusions 0; start-only/end-only/both-zero counts 23184/16569/12167. Denominators are this seed's complete validation positions or native contribution energy, not averages of batch fractions.

Seed 7 unpruned: S 450028/492416 positions (0.913918); S/start nonzero 0.970557, S/end nonzero 0.968852; start/end contribution energy fractions 0.982454/0.989725; exact-zero exclusions 42388, numerical-floor exclusions 0; start-only/end-only/both-zero counts 13652/14468/14268. Denominators are this seed's complete validation positions or native contribution energy, not averages of batch fractions.

Seed 17 sparse: S 435803/492416 positions (0.885030); S/start nonzero 0.945850, S/end nonzero 0.961414; start/end contribution energy fractions 0.972666/0.984426; exact-zero exclusions 56613, numerical-floor exclusions 0; start-only/end-only/both-zero counts 24950/17491/14172. Denominators are this seed's complete validation positions or native contribution energy, not averages of batch fractions.

Seed 17 unpruned: S 448453/492416 positions (0.910720); S/start nonzero 0.973305, S/end nonzero 0.960774; start/end contribution energy fractions 0.988203/0.989052; exact-zero exclusions 43963, numerical-floor exclusions 0; start-only/end-only/both-zero counts 12300/18309/13354. Denominators are this seed's complete validation positions or native contribution energy, not averages of batch fractions.

Seed 29 sparse: S 443588/492416 positions (0.900840); S/start nonzero 0.946858, S/end nonzero 0.971828; start/end contribution energy fractions 0.970641/0.983727; exact-zero exclusions 48828, numerical-floor exclusions 0; start-only/end-only/both-zero counts 24896/12859/11073. Denominators are this seed's complete validation positions or native contribution energy, not averages of batch fractions.

Seed 29 unpruned: S 457235/492416 positions (0.928554); S/start nonzero 0.975989, S/end nonzero 0.974576; start/end contribution energy fractions 0.985501/0.989087; exact-zero exclusions 35181, numerical-floor exclusions 0; start-only/end-only/both-zero counts 11249/11928/12004. Denominators are this seed's complete validation positions or native contribution energy, not averages of batch fractions.

Seed 43 sparse: S 435453/492416 positions (0.884319); S/start nonzero 0.932314, S/end nonzero 0.971397; start/end contribution energy fractions 0.966642/0.981453; exact-zero exclusions 56963, numerical-floor exclusions 0; start-only/end-only/both-zero counts 31614/12822/12527. Denominators are this seed's complete validation positions or native contribution energy, not averages of batch fractions.

Seed 43 unpruned: S 450235/492416 positions (0.914339); S/start nonzero 0.963962, S/end nonzero 0.971261; start/end contribution energy fractions 0.984941/0.986668; exact-zero exclusions 42181, numerical-floor exclusions 0; start-only/end-only/both-zero counts 16832/13322/12027. Denominators are this seed's complete validation positions or native contribution energy, not averages of batch fractions.

Seed 61 sparse: S 421168/492416 positions (0.855309); S/start nonzero 0.927683, S/end nonzero 0.955123; start/end contribution energy fractions 0.967209/0.970900; exact-zero exclusions 71248, numerical-floor exclusions 0; start-only/end-only/both-zero counts 32832/19789/18627. Denominators are this seed's complete validation positions or native contribution energy, not averages of batch fractions.

Seed 61 unpruned: S 438615/492416 positions (0.890741); S/start nonzero 0.966112, S/end nonzero 0.958418; start/end contribution energy fractions 0.972704/0.973327; exact-zero exclusions 53801, numerical-floor exclusions 0; start-only/end-only/both-zero counts 15385/19030/19386. Denominators are this seed's complete validation positions or native contribution energy, not averages of batch fractions.


## Routing and recipient dependence

sparse, start, D_start_norm: native/replay positive-sign agreement in 5/5 seeds. Native-minus-replay seed differences: 0.000112, -0.000059, 0.000186, -0.000068, 0.000092.

sparse, start, D_end_norm: native/replay positive-sign agreement in 5/5 seeds. Native-minus-replay seed differences: 0.000495, 0.000581, 0.000395, 0.000472, 0.000276.

sparse, end, D_start_norm: native/replay positive-sign agreement in 5/5 seeds. Native-minus-replay seed differences: 0.000333, -0.000344, -0.000089, -0.004075, -0.000016.

sparse, end, D_end_norm: native/replay positive-sign agreement in 5/5 seeds. Native-minus-replay seed differences: 0.000132, 0.000032, 0.000226, 0.000753, -0.000353.

unpruned, start, D_start_norm: native/replay positive-sign agreement in 5/5 seeds. Native-minus-replay seed differences: 0.000571, 0.000095, 0.000805, -0.001233, -0.000580.

unpruned, start, D_end_norm: native/replay positive-sign agreement in 5/5 seeds. Native-minus-replay seed differences: 0.000809, 0.000788, 0.000767, -0.000261, -0.000507.

unpruned, end, D_start_norm: native/replay positive-sign agreement in 5/5 seeds. Native-minus-replay seed differences: 0.000094, -0.001130, 0.000291, -0.000889, -0.000293.

unpruned, end, D_end_norm: native/replay positive-sign agreement in 5/5 seeds. Native-minus-replay seed differences: 0.000405, -0.000120, 0.000559, -0.000332, 0.000225.

sparse NATIVE_DOWNSTREAM, recovered-minus-starting recipient D_start_norm: 0.022440, 0.039051, 0.028531, 0.040406, 0.039577. These include residual, downstream-parameter and unsupported-background differences, not downstream weights alone.

sparse NATIVE_DOWNSTREAM, recovered-minus-starting recipient D_end_norm: 0.033941, 0.058007, 0.045341, 0.084831, 0.054833. These include residual, downstream-parameter and unsupported-background differences, not downstream weights alone.

sparse RECIPIENT_BASELINE_REPLAY, recovered-minus-starting recipient D_start_norm: 0.022218, 0.039336, 0.028806, 0.044414, 0.039685. These include residual, downstream-parameter and unsupported-background differences, not downstream weights alone.

sparse RECIPIENT_BASELINE_REPLAY, recovered-minus-starting recipient D_end_norm: 0.034304, 0.058557, 0.045510, 0.084550, 0.055462. These include residual, downstream-parameter and unsupported-background differences, not downstream weights alone.

unpruned NATIVE_DOWNSTREAM, recovered-minus-starting recipient D_start_norm: 0.022734, 0.028927, 0.024276, 0.030067, 0.033369. These include residual, downstream-parameter and unsupported-background differences, not downstream weights alone.

unpruned NATIVE_DOWNSTREAM, recovered-minus-starting recipient D_end_norm: 0.022436, 0.029245, 0.025278, 0.031187, 0.031645. These include residual, downstream-parameter and unsupported-background differences, not downstream weights alone.

unpruned RECIPIENT_BASELINE_REPLAY, recovered-minus-starting recipient D_start_norm: 0.023211, 0.030151, 0.024790, 0.029723, 0.033082. These include residual, downstream-parameter and unsupported-background differences, not downstream weights alone.

unpruned RECIPIENT_BASELINE_REPLAY, recovered-minus-starting recipient D_end_norm: 0.022840, 0.030152, 0.025486, 0.031258, 0.030913. These include residual, downstream-parameter and unsupported-background differences, not downstream weights alone.


## Supported interpretation

sparse, NATIVE_DOWNSTREAM, D_start_norm: positive in 10/10 seed-by-recipient combinations. Where positive: on common positions with defined starting and recovered directions, replacing starting contribution direction with recovered direction improves CE under the tested fixed per-token norm and recipient conditions. Mixed signs remain context/norm-target dependence.

sparse, NATIVE_DOWNSTREAM, D_end_norm: positive in 10/10 seed-by-recipient combinations. Where positive: on common positions with defined starting and recovered directions, replacing starting contribution direction with recovered direction improves CE under the tested fixed per-token norm and recipient conditions. Mixed signs remain context/norm-target dependence.

unpruned, NATIVE_DOWNSTREAM, D_start_norm: positive in 10/10 seed-by-recipient combinations. Where positive: on common positions with defined starting and recovered directions, replacing starting contribution direction with recovered direction improves CE under the tested fixed per-token norm and recipient conditions. Mixed signs remain context/norm-target dependence.

unpruned, NATIVE_DOWNSTREAM, D_end_norm: positive in 10/10 seed-by-recipient combinations. Where positive: on common positions with defined starting and recovered directions, replacing starting contribution direction with recovered direction improves CE under the tested fixed per-token norm and recipient conditions. Mixed signs remain context/norm-target dependence.

sparse, RECIPIENT_BASELINE_REPLAY, D_start_norm: positive in 10/10 seed-by-recipient combinations. Where positive: on common positions with defined starting and recovered directions, replacing starting contribution direction with recovered direction improves CE under the tested fixed per-token norm and recipient conditions. Mixed signs remain context/norm-target dependence.

sparse, RECIPIENT_BASELINE_REPLAY, D_end_norm: positive in 10/10 seed-by-recipient combinations. Where positive: on common positions with defined starting and recovered directions, replacing starting contribution direction with recovered direction improves CE under the tested fixed per-token norm and recipient conditions. Mixed signs remain context/norm-target dependence.

unpruned, RECIPIENT_BASELINE_REPLAY, D_start_norm: positive in 10/10 seed-by-recipient combinations. Where positive: on common positions with defined starting and recovered directions, replacing starting contribution direction with recovered direction improves CE under the tested fixed per-token norm and recipient conditions. Mixed signs remain context/norm-target dependence.

unpruned, RECIPIENT_BASELINE_REPLAY, D_end_norm: positive in 10/10 seed-by-recipient combinations. Where positive: on common positions with defined starting and recovered directions, replacing starting contribution direction with recovered direction improves CE under the tested fixed per-token norm and recipient conditions. Mixed signs remain context/norm-target dependence.

Full-donor CE context is preserved in full_donor_context.csv. No full-minus-common remainder is called a causal support effect. Whole-population loss includes effects transmitted by attention and later layers, including to unsupported positions; it is not an isolated expert-position loss. No subsequent experiment is launched.

## Engineering and instruction-compliance record

Revision 1 became synchronization-bound under concurrent GPU jobs. Only this study's worker was stopped. All source snapshots, seals, eligibility evidence, its completed seed17-sparse cache and partial unpruned pilot log are preserved under attempts/revision_1_before_replay_cache/. Revision 2 validates the same immutable replay metadata on CPU once per batch and hashes it on every reuse, caching only device routing indices/gates/acceptance/order. Current expert inputs, native accepted order and per-forward trace checks are unchanged. Scientific configuration, common-support rule and tolerances are exactly unchanged. All old eligibility/evaluation caches were invalidated for dispatch and retained; revision 2 recomputed them. The completed revision1 pair was compared cell-by-cell under the unchanged CE tolerance, with the difference recorded in numerical_checks.json. No effect sign selected a condition or implementation.

An instruction error occurred in reused numerical tests: the legacy test_top2_routes_to_two_distinct_experts_and_backpropagates fixture called backward on a disposable random synthetic model in a test subprocess. This was contrary to the user's no-backward restriction. No research checkpoint, research backward pass, optimizer step or weight update was involved. The original logs are preserved. The authoritative corrected suite excludes exactly that fixture and passes 29 forward-only tests, with a plugin that fails on backward, autograd.grad or optimizer steps. All checkpoint evaluation was inference-only. This test-only correction does not change scientific losses or require result-cache invalidation. Future resumes must use engineering/forward_only_entrypoint.py. The source-hashed correction record is forward_only_numerical_amendment.json. We do not claim that the earlier numerical suites contained no backward calls.
