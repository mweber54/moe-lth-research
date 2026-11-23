# First-MoE interface experiment

Layer 0 was chosen post hoc after examining the prior contribution-ablation study. This new design and implementation were sealed before new outcomes; this is not project-level preregistration. Validation was repeatedly examined and is not an independent test. Source seeds (n=5) are replication units; batches, tokens, alphas, layers and cells are nested measurements. The recipient context includes its own pre-addition residual representation and subsequent computation, not downstream weights alone. Donor and recipient share seed/model type; no alignment, norm matching, weight transplantation or training is performed. Usefulness is confined to tested recipient contexts, not universally better semantic information. Q is an evaluation compatibility interaction that can reflect coordinate mismatch, scaling, routing and subsequent use. Reciprocal harm does not prove semantic co-specialization. Frozen expert/router weights and biases do not learn. Fractional attenuation has different absolute sizes across donors, but fixed donor/alpha perturbations are identical across recipients. Norms and cosines do not establish usefulness or explain CE causally. Sparse-versus-unpruned donor perturbations can also differ in absolute magnitude. No percentage of recovery, training counterfactual, ticket mechanism, safe pruning rule or generalization beyond this architecture/mask family/window is identified. Nonmonotone curves and negative effects are retained.

All effects below use source seed as the replication unit. Individual seeds appear before descriptive summaries. Positive V means the recovered contribution helps; positive T means increased recipient-context sensitivity to the same donor perturbation.


## NATIVE_RESPONSE


### sparse

Seed 7: native start CE=2.558416, native end CE=2.057034; start<-end CE=2.438349, end<-start CE=2.167966; V(start)=0.120067, V(end)=0.110932, Q=-0.009135; S_start=-0.120067, S_end=0.110932; prior diagonal damage change=0.175739.

Seed 17: native start CE=2.516974, native end CE=2.028336; start<-end CE=2.369854, end<-start CE=2.197419; V(start)=0.147120, V(end)=0.169083, Q=0.021963; S_start=-0.147120, S_end=0.169083; prior diagonal damage change=0.296121.

Seed 29: native start CE=2.536108, native end CE=2.049269; start<-end CE=2.419451, end<-start CE=2.157838; V(start)=0.116657, V(end)=0.108569, Q=-0.008089; S_start=-0.116657, S_end=0.108569; prior diagonal damage change=0.202848.

Seed 43: native start CE=2.595880, native end CE=2.060850; start<-end CE=2.442351, end<-start CE=2.245518; V(start)=0.153530, V(end)=0.184668, Q=0.031138; S_start=-0.153530, S_end=0.184668; prior diagonal damage change=0.355503.

Seed 61: native start CE=2.621438, native end CE=2.043791; start<-end CE=2.519546, end<-start CE=2.179812; V(start)=0.101892, V(end)=0.136021, Q=0.034129; S_start=-0.101892, S_end=0.136021; prior diagonal damage change=0.279105.

V_start: mean 0.127853, median 0.120067, range [0.101892, 0.153530], sample SD 0.021740.

V_end: mean 0.141855, median 0.136021, range [0.108569, 0.184668], sample SD 0.034178.

Q: mean 0.014001, median 0.021963, range [-0.009135, 0.034129], sample SD 0.021127.

Recovered-contribution benefit by seed: both recipients: 5/5, end recipient only: 0/5, start recipient only: 0/5, neither recipient: 0/5. Zero values, if present, mean no measured CE benefit, not evidence of equivalence.

Reciprocal substitutions hurt in both directions for 0/5 seeds. Such compatibility can involve residual/output coordinates, scaling and subsequent use.

Fixed start donor, alpha=0.9: T in seed order 7,17,29,43,61 = -0.007112, 0.001952, -0.002903, 0.001271, 0.004950; mean -0.000368, median 0.001271, range [-0.007112, 0.004950], sample SD 0.004697; increased in 3/5 seeds. This is the prespecified modest-attenuation contrast.

Fixed start donor, alpha=0.75: T in seed order 7,17,29,43,61 = -0.014462, 0.010541, -0.002401, 0.010297, 0.017090; mean 0.004213, median 0.010297, range [-0.014462, 0.017090], sample SD 0.012603; increased in 3/5 seeds. Response-curve point, not an independent replication.

Fixed start donor, alpha=0.5: T in seed order 7,17,29,43,61 = -0.011977, 0.039296, 0.013037, 0.045966, 0.045850; mean 0.026435, median 0.039296, range [-0.011977, 0.045966], sample SD 0.025391; increased in 4/5 seeds. Response-curve point, not an independent replication.

Fixed start donor, alpha=0.0: T in seed order 7,17,29,43,61 = 0.064808, 0.127038, 0.094280, 0.170835, 0.143083; mean 0.120009, median 0.127038, range [0.064808, 0.170835], sample SD 0.041456; increased in 5/5 seeds. Response-curve point, not an independent replication.

Fixed end donor, alpha=0.9: T in seed order 7,17,29,43,61 = -0.015062, -0.014267, -0.014815, -0.014651, -0.011903; mean -0.014140, median -0.014651, range [-0.015062, -0.011903], sample SD 0.001283; increased in 0/5 seeds. This is the prespecified modest-attenuation contrast.

Fixed end donor, alpha=0.75: T in seed order 7,17,29,43,61 = -0.035115, -0.028042, -0.032315, -0.029955, -0.017297; mean -0.028545, median -0.029955, range [-0.035115, -0.017297], sample SD 0.006822; increased in 0/5 seeds. Response-curve point, not an independent replication.

Fixed end donor, alpha=0.5: T in seed order 7,17,29,43,61 = -0.055226, -0.017720, -0.040012, -0.019487, 0.016249; mean -0.023239, median -0.019487, range [-0.055226, 0.016249], sample SD 0.026964; increased in 1/5 seeds. Response-curve point, not an independent replication.

Fixed end donor, alpha=0.0: T in seed order 7,17,29,43,61 = 0.055673, 0.149001, 0.086191, 0.201973, 0.177212; mean 0.134010, median 0.149001, range [0.055673, 0.201973], sample SD 0.061509; increased in 5/5 seeds. Response-curve point, not an independent replication.

Modest attenuation versus removal for fixed start donor: alpha=.9 T is positive in 3/5 seeds (mean -0.000368); alpha=0 T is positive in 5/5 (mean 0.120009). These are different perturbation sizes, not independent replications or a locally linear extrapolation.

Modest attenuation versus removal for fixed end donor: alpha=.9 T is positive in 0/5 seeds (mean -0.014140); alpha=0 T is positive in 5/5 (mean 0.134010). These are different perturbation sizes, not independent replications or a locally linear extrapolation.

Attenuation improves CE (negative A) in 0 of 80 nontrivial attenuation cells; full signed curves are in attenuation_effects.csv. Native alpha=1 is not assumed optimal.


### unpruned

Seed 7: native start CE=2.087976, native end CE=1.989057; start<-end CE=2.056804, end<-start CE=2.051782; V(start)=0.031172, V(end)=0.062725, Q=0.031553; S_start=-0.031172, S_end=0.062725; prior diagonal damage change=0.084111.

Seed 17: native start CE=2.067352, native end CE=1.961325; start<-end CE=2.026807, end<-start CE=2.037449; V(start)=0.040545, V(end)=0.076124, Q=0.035579; S_start=-0.040545, S_end=0.076124; prior diagonal damage change=0.158568.

Seed 29: native start CE=2.070126, native end CE=1.967500; start<-end CE=2.032190, end<-start CE=2.033351; V(start)=0.037935, V(end)=0.065851, Q=0.027915; S_start=-0.037935, S_end=0.065851; prior diagonal damage change=0.113401.

Seed 43: native start CE=2.101044, native end CE=1.984956; start<-end CE=2.065118, end<-start CE=2.057953; V(start)=0.035926, V(end)=0.072997, Q=0.037072; S_start=-0.035926, S_end=0.072997; prior diagonal damage change=0.156207.

Seed 61: native start CE=2.108713, native end CE=1.980973; start<-end CE=2.063578, end<-start CE=2.063134; V(start)=0.045135, V(end)=0.082161, Q=0.037026; S_start=-0.045135, S_end=0.082161; prior diagonal damage change=0.111781.

V_start: mean 0.038143, median 0.037935, range [0.031172, 0.045135], sample SD 0.005201.

V_end: mean 0.071972, median 0.072997, range [0.062725, 0.082161], sample SD 0.007828.

Q: mean 0.033829, median 0.035579, range [0.027915, 0.037072], sample SD 0.003998.

Recovered-contribution benefit by seed: both recipients: 5/5, end recipient only: 0/5, start recipient only: 0/5, neither recipient: 0/5. Zero values, if present, mean no measured CE benefit, not evidence of equivalence.

Reciprocal substitutions hurt in both directions for 0/5 seeds. Such compatibility can involve residual/output coordinates, scaling and subsequent use.

Fixed start donor, alpha=0.9: T in seed order 7,17,29,43,61 = 0.001638, -0.003550, -0.001674, 0.003757, 0.001357; mean 0.000306, median 0.001357, range [-0.003550, 0.003757], sample SD 0.002897; increased in 3/5 seeds. This is the prespecified modest-attenuation contrast.

Fixed start donor, alpha=0.75: T in seed order 7,17,29,43,61 = 0.004606, -0.007351, -0.002467, 0.009839, 0.002877; mean 0.001501, median 0.002877, range [-0.007351, 0.009839], sample SD 0.006617; increased in 3/5 seeds. Response-curve point, not an independent replication.

Fixed start donor, alpha=0.5: T in seed order 7,17,29,43,61 = 0.010333, -0.004366, 0.002726, 0.024689, 0.006301; mean 0.007937, median 0.006301, range [-0.004366, 0.024689], sample SD 0.010812; increased in 4/5 seeds. Response-curve point, not an independent replication.

Fixed start donor, alpha=0.0: T in seed order 7,17,29,43,61 = 0.021386, 0.082443, 0.047550, 0.083209, 0.029620; mean 0.052842, median 0.047550, range [0.021386, 0.083209], sample SD 0.028962; increased in 5/5 seeds. Response-curve point, not an independent replication.

Fixed end donor, alpha=0.9: T in seed order 7,17,29,43,61 = 0.003309, -0.000427, 0.000332, 0.005665, 0.003265; mean 0.002429, median 0.003265, range [-0.000427, 0.005665], sample SD 0.002475; increased in 4/5 seeds. This is the prespecified modest-attenuation contrast.

Fixed end donor, alpha=0.75: T in seed order 7,17,29,43,61 = 0.009060, 0.000363, 0.003139, 0.015815, 0.008771; mean 0.007430, median 0.008771, range [0.000363, 0.015815], sample SD 0.005982; increased in 5/5 seeds. Response-curve point, not an independent replication.

Fixed end donor, alpha=0.5: T in seed order 7,17,29,43,61 = 0.022491, 0.011253, 0.014160, 0.040570, 0.021045; mean 0.021904, median 0.021045, range [0.011253, 0.040570], sample SD 0.011434; increased in 5/5 seeds. Response-curve point, not an independent replication.

Fixed end donor, alpha=0.0: T in seed order 7,17,29,43,61 = 0.052939, 0.118023, 0.075466, 0.120281, 0.066646; mean 0.086671, median 0.075466, range [0.052939, 0.120281], sample SD 0.030729; increased in 5/5 seeds. Response-curve point, not an independent replication.

Modest attenuation versus removal for fixed start donor: alpha=.9 T is positive in 3/5 seeds (mean 0.000306); alpha=0 T is positive in 5/5 (mean 0.052842). These are different perturbation sizes, not independent replications or a locally linear extrapolation.

Modest attenuation versus removal for fixed end donor: alpha=.9 T is positive in 4/5 seeds (mean 0.002429); alpha=0 T is positive in 5/5 (mean 0.086671). These are different perturbation sizes, not independent replications or a locally linear extrapolation.

Attenuation improves CE (negative A) in 1 of 80 nontrivial attenuation cells; full signed curves are in attenuation_effects.csv. Native alpha=1 is not assumed optimal.


### Same-seed sparse-minus-unpruned contrasts

V_start  : seed values 0.088895, 0.106575, 0.078722, 0.117604, 0.056758; mean 0.089711, median 0.088895, range [0.056758, 0.117604], sample SD 0.023821; sparse is larger in 5/5 seeds.

V_end  : seed values 0.048207, 0.092959, 0.042718, 0.111670, 0.053861; mean 0.069883, median 0.053861, range [0.042718, 0.111670], sample SD 0.030591; sparse is larger in 5/5 seeds.

Q  : seed values -0.040688, -0.013616, -0.036004, -0.005933, -0.002897; mean -0.019828, median -0.013616, range [-0.040688, -0.002897], sample SD 0.017429; sparse is larger in 0/5 seeds.

T start alpha=.9: seed values -0.008750, 0.005502, -0.001229, -0.002486, 0.003593; mean -0.000674, median -0.001229, range [-0.008750, 0.005502], sample SD 0.005594; sparse is larger in 2/5 seeds.

T end alpha=.9: seed values -0.018371, -0.013840, -0.015147, -0.020316, -0.015168; mean -0.016568, median -0.015168, range [-0.020316, -0.013840], sample SD 0.002679; sparse is larger in 0/5 seeds.


## RECIPIENT_BASELINE_REPLAY


### sparse

Seed 7: native start CE=2.558416, native end CE=2.057034; start<-end CE=2.439315, end<-start CE=2.168514; V(start)=0.119101, V(end)=0.111481, Q=-0.007620; S_start=-0.119101, S_end=0.111481; prior diagonal damage change=0.181697.

Seed 17: native start CE=2.516974, native end CE=2.028336; start<-end CE=2.370759, end<-start CE=2.200789; V(start)=0.146216, V(end)=0.172453, Q=0.026238; S_start=-0.146216, S_end=0.172453; prior diagonal damage change=0.327460.

Seed 29: native start CE=2.536108, native end CE=2.049269; start<-end CE=2.421744, end<-start CE=2.158960; V(start)=0.114364, V(end)=0.109691, Q=-0.004673; S_start=-0.114364, S_end=0.109691; prior diagonal damage change=0.200979.

Seed 43: native start CE=2.595880, native end CE=2.060850; start<-end CE=2.445889, end<-start CE=2.261041; V(start)=0.149991, V(end)=0.200191, Q=0.050199; S_start=-0.149991, S_end=0.200191; prior diagonal damage change=0.389068.

Seed 61: native start CE=2.621438, native end CE=2.043791; start<-end CE=2.520074, end<-start CE=2.180862; V(start)=0.101364, V(end)=0.137071, Q=0.035707; S_start=-0.101364, S_end=0.137071; prior diagonal damage change=0.283393.

V_start: mean 0.126207, median 0.119101, range [0.101364, 0.149991], sample SD 0.021059.

V_end: mean 0.146177, median 0.137071, range [0.109691, 0.200191], sample SD 0.039452.

Q: mean 0.019970, median 0.026238, range [-0.007620, 0.050199], sample SD 0.025344.

Recovered-contribution benefit by seed: both recipients: 5/5, end recipient only: 0/5, start recipient only: 0/5, neither recipient: 0/5. Zero values, if present, mean no measured CE benefit, not evidence of equivalence.

Reciprocal substitutions hurt in both directions for 0/5 seeds. Such compatibility can involve residual/output coordinates, scaling and subsequent use.

Fixed start donor, alpha=0.9: T in seed order 7,17,29,43,61 = -0.006865, 0.002427, -0.002413, 0.002514, 0.005052; mean 0.000143, median 0.002427, range [-0.006865, 0.005052], sample SD 0.004759; increased in 3/5 seeds. This is the prespecified modest-attenuation contrast.

Fixed start donor, alpha=0.75: T in seed order 7,17,29,43,61 = -0.013594, 0.012015, -0.001237, 0.013785, 0.017205; mean 0.005635, median 0.012015, range [-0.013594, 0.017205], sample SD 0.012826; increased in 3/5 seeds. Response-curve point, not an independent replication.

Fixed start donor, alpha=0.5: T in seed order 7,17,29,43,61 = -0.008923, 0.044546, 0.014837, 0.053885, 0.046354; mean 0.030140, median 0.044546, range [-0.008923, 0.053885], sample SD 0.026431; increased in 4/5 seeds. Response-curve point, not an independent replication.

Fixed start donor, alpha=0.0: T in seed order 7,17,29,43,61 = 0.070216, 0.155007, 0.091288, 0.188877, 0.146322; mean 0.130342, median 0.146322, range [0.070216, 0.188877], sample SD 0.048555; increased in 5/5 seeds. Response-curve point, not an independent replication.

Fixed end donor, alpha=0.9: T in seed order 7,17,29,43,61 = -0.014676, -0.014270, -0.014539, -0.014442, -0.011691; mean -0.013923, median -0.014442, range [-0.014676, -0.011691], sample SD 0.001257; increased in 0/5 seeds. This is the prespecified modest-attenuation contrast.

Fixed end donor, alpha=0.75: T in seed order 7,17,29,43,61 = -0.034295, -0.028003, -0.031634, -0.029101, -0.016578; mean -0.027922, median -0.029101, range [-0.034295, -0.016578], sample SD 0.006791; increased in 0/5 seeds. Response-curve point, not an independent replication.

Fixed end donor, alpha=0.5: T in seed order 7,17,29,43,61 = -0.053159, -0.016481, -0.037271, -0.016548, 0.017284; mean -0.021235, median -0.016548, range [-0.053159, 0.017284], sample SD 0.026479; increased in 1/5 seeds. Response-curve point, not an independent replication.

Fixed end donor, alpha=0.0: T in seed order 7,17,29,43,61 = 0.062596, 0.181245, 0.086615, 0.239077, 0.182029; mean 0.150312, median 0.181245, range [0.062596, 0.239077], sample SD 0.073473; increased in 5/5 seeds. Response-curve point, not an independent replication.

Modest attenuation versus removal for fixed start donor: alpha=.9 T is positive in 3/5 seeds (mean 0.000143); alpha=0 T is positive in 5/5 (mean 0.130342). These are different perturbation sizes, not independent replications or a locally linear extrapolation.

Modest attenuation versus removal for fixed end donor: alpha=.9 T is positive in 0/5 seeds (mean -0.013923); alpha=0 T is positive in 5/5 (mean 0.150312). These are different perturbation sizes, not independent replications or a locally linear extrapolation.

Attenuation improves CE (negative A) in 0 of 80 nontrivial attenuation cells; full signed curves are in attenuation_effects.csv. Native alpha=1 is not assumed optimal.


### unpruned

Seed 7: native start CE=2.087976, native end CE=1.989057; start<-end CE=2.057819, end<-start CE=2.051825; V(start)=0.030157, V(end)=0.062768, Q=0.032611; S_start=-0.030157, S_end=0.062768; prior diagonal damage change=0.081310.

Seed 17: native start CE=2.067352, native end CE=1.961325; start<-end CE=2.028302, end<-start CE=2.039190; V(start)=0.039050, V(end)=0.077865, Q=0.038815; S_start=-0.039050, S_end=0.077865; prior diagonal damage change=0.162072.

Seed 29: native start CE=2.070126, native end CE=1.967500; start<-end CE=2.035949, end<-start CE=2.035062; V(start)=0.034177, V(end)=0.067562, Q=0.033386; S_start=-0.034177, S_end=0.067562; prior diagonal damage change=0.123313.

Seed 43: native start CE=2.101044, native end CE=1.984956; start<-end CE=2.073929, end<-start CE=2.061581; V(start)=0.027115, V(end)=0.076626, Q=0.049510; S_start=-0.027115, S_end=0.076626; prior diagonal damage change=0.112975.

Seed 61: native start CE=2.108713, native end CE=1.980973; start<-end CE=2.065910, end<-start CE=2.063379; V(start)=0.042802, V(end)=0.082407, Q=0.039604; S_start=-0.042802, S_end=0.082407; prior diagonal damage change=0.107232.

V_start: mean 0.034660, median 0.034177, range [0.027115, 0.042802], sample SD 0.006384.

V_end: mean 0.073446, median 0.076626, range [0.062768, 0.082407], sample SD 0.008040.

Q: mean 0.038785, median 0.038815, range [0.032611, 0.049510], sample SD 0.006763.

Recovered-contribution benefit by seed: both recipients: 5/5, end recipient only: 0/5, start recipient only: 0/5, neither recipient: 0/5. Zero values, if present, mean no measured CE benefit, not evidence of equivalence.

Reciprocal substitutions hurt in both directions for 0/5 seeds. Such compatibility can involve residual/output coordinates, scaling and subsequent use.

Fixed start donor, alpha=0.9: T in seed order 7,17,29,43,61 = 0.001470, -0.003973, -0.002085, 0.003452, 0.000739; mean -0.000079, median 0.000739, range [-0.003973, 0.003452], sample SD 0.002946; increased in 3/5 seeds. This is the prespecified modest-attenuation contrast.

Fixed start donor, alpha=0.75: T in seed order 7,17,29,43,61 = 0.003955, -0.008256, -0.003413, 0.009140, 0.001747; mean 0.000634, median 0.001747, range [-0.008256, 0.009140], sample SD 0.006709; increased in 3/5 seeds. Response-curve point, not an independent replication.

Fixed start donor, alpha=0.5: T in seed order 7,17,29,43,61 = 0.009313, -0.007846, 0.000692, 0.023529, 0.004418; mean 0.006021, median 0.004418, range [-0.007846, 0.023529], sample SD 0.011626; increased in 4/5 seeds. Response-curve point, not an independent replication.

Fixed start donor, alpha=0.0: T in seed order 7,17,29,43,61 = 0.018541, 0.084207, 0.055750, 0.036350, 0.024826; mean 0.043935, median 0.036350, range [0.018541, 0.084207], sample SD 0.026594; increased in 5/5 seeds. Response-curve point, not an independent replication.

Fixed end donor, alpha=0.9: T in seed order 7,17,29,43,61 = 0.003243, -0.000602, 0.000202, 0.005936, 0.002769; mean 0.002310, median 0.002769, range [-0.000602, 0.005936], sample SD 0.002605; increased in 4/5 seeds. This is the prespecified modest-attenuation contrast.

Fixed end donor, alpha=0.75: T in seed order 7,17,29,43,61 = 0.008918, -0.000187, 0.002477, 0.015857, 0.007393; mean 0.006891, median 0.007393, range [-0.000187, 0.015857], sample SD 0.006212; increased in 4/5 seeds. Response-curve point, not an independent replication.

Fixed end donor, alpha=0.5: T in seed order 7,17,29,43,61 = 0.021613, 0.008530, 0.012988, 0.040209, 0.018464; mean 0.020361, median 0.018464, range [0.008530, 0.040209], sample SD 0.012181; increased in 5/5 seeds. Response-curve point, not an independent replication.

Fixed end donor, alpha=0.0: T in seed order 7,17,29,43,61 = 0.051153, 0.123023, 0.089136, 0.085860, 0.064430; mean 0.082720, median 0.085860, range [0.051153, 0.123023], sample SD 0.027415; increased in 5/5 seeds. Response-curve point, not an independent replication.

Modest attenuation versus removal for fixed start donor: alpha=.9 T is positive in 3/5 seeds (mean -0.000079); alpha=0 T is positive in 5/5 (mean 0.043935). These are different perturbation sizes, not independent replications or a locally linear extrapolation.

Modest attenuation versus removal for fixed end donor: alpha=.9 T is positive in 4/5 seeds (mean 0.002310); alpha=0 T is positive in 5/5 (mean 0.082720). These are different perturbation sizes, not independent replications or a locally linear extrapolation.

Attenuation improves CE (negative A) in 1 of 80 nontrivial attenuation cells; full signed curves are in attenuation_effects.csv. Native alpha=1 is not assumed optimal.


### Same-seed sparse-minus-unpruned contrasts

V_start  : seed values 0.088943, 0.107166, 0.080188, 0.122876, 0.058562; mean 0.091547, median 0.088943, range [0.058562, 0.122876], sample SD 0.024745; sparse is larger in 5/5 seeds.

V_end  : seed values 0.048712, 0.094588, 0.042129, 0.123565, 0.054665; mean 0.072732, median 0.054665, range [0.042129, 0.123565], sample SD 0.035006; sparse is larger in 5/5 seeds.

Q  : seed values -0.040231, -0.012578, -0.038059, 0.000689, -0.003897; mean -0.018815, median -0.012578, range [-0.040231, 0.000689], sample SD 0.019176; sparse is larger in 1/5 seeds.

T start alpha=.9: seed values -0.008336, 0.006401, -0.000328, -0.000938, 0.004312; mean 0.000222, median -0.000328, range [-0.008336, 0.006401], sample SD 0.005696; sparse is larger in 2/5 seeds.

T end alpha=.9: seed values -0.017918, -0.013668, -0.014741, -0.020378, -0.014460; mean -0.016233, median -0.014741, range [-0.020378, -0.013668], sample SD 0.002827; sparse is larger in 0/5 seeds.


## Interface magnitude and routing interpretation

Seed 7 sparse: contribution start/end RMS 0.600953/1.010949, end/start norm ratio 1.682242, relative L2 change 0.899450, global cosine 0.897887; actual alpha=.9 attenuation displacement RMS start/end donor 0.060095/0.101095. Fixed donor perturbations are identical across recipients; donor-to-donor fractional comparisons have these different absolute sizes. Magnitudes cannot by themselves establish the cause of CE differences.

Seed 17 sparse: contribution start/end RMS 0.677448/1.197933, end/start norm ratio 1.768303, relative L2 change 0.960898, global cosine 0.905832; actual alpha=.9 attenuation displacement RMS start/end donor 0.067745/0.119793. Fixed donor perturbations are identical across recipients; donor-to-donor fractional comparisons have these different absolute sizes. Magnitudes cannot by themselves establish the cause of CE differences.

Seed 29 sparse: contribution start/end RMS 0.691700/1.116531, end/start norm ratio 1.614183, relative L2 change 0.823911, global cosine 0.906576; actual alpha=.9 attenuation displacement RMS start/end donor 0.069170/0.111653. Fixed donor perturbations are identical across recipients; donor-to-donor fractional comparisons have these different absolute sizes. Magnitudes cannot by themselves establish the cause of CE differences.

Seed 43 sparse: contribution start/end RMS 0.625293/1.109983, end/start norm ratio 1.775140, relative L2 change 0.992674, global cosine 0.891682; actual alpha=.9 attenuation displacement RMS start/end donor 0.062529/0.110998. Fixed donor perturbations are identical across recipients; donor-to-donor fractional comparisons have these different absolute sizes. Magnitudes cannot by themselves establish the cause of CE differences.

Seed 61 sparse: contribution start/end RMS 0.765751/1.225850, end/start norm ratio 1.600848, relative L2 change 0.795286, global cosine 0.915213; actual alpha=.9 attenuation displacement RMS start/end donor 0.076575/0.122585. Fixed donor perturbations are identical across recipients; donor-to-donor fractional comparisons have these different absolute sizes. Magnitudes cannot by themselves establish the cause of CE differences.

Seed 7 unpruned: contribution start/end RMS 2.800525/2.875232, end/start norm ratio 1.026676, relative L2 change 0.330801, global cosine 0.947054; actual alpha=.9 attenuation displacement RMS start/end donor 0.280053/0.287523. Fixed donor perturbations are identical across recipients; donor-to-donor fractional comparisons have these different absolute sizes. Magnitudes cannot by themselves establish the cause of CE differences.

Seed 17 unpruned: contribution start/end RMS 3.125696/3.205344, end/start norm ratio 1.025482, relative L2 change 0.326499, global cosine 0.948340; actual alpha=.9 attenuation displacement RMS start/end donor 0.312570/0.320535. Fixed donor perturbations are identical across recipients; donor-to-donor fractional comparisons have these different absolute sizes. Magnitudes cannot by themselves establish the cause of CE differences.

Seed 29 unpruned: contribution start/end RMS 3.269342/3.347521, end/start norm ratio 1.023913, relative L2 change 0.284166, global cosine 0.960847; actual alpha=.9 attenuation displacement RMS start/end donor 0.326934/0.334752. Fixed donor perturbations are identical across recipients; donor-to-donor fractional comparisons have these different absolute sizes. Magnitudes cannot by themselves establish the cause of CE differences.

Seed 43 unpruned: contribution start/end RMS 2.953375/3.026575, end/start norm ratio 1.024785, relative L2 change 0.302757, global cosine 0.955577; actual alpha=.9 attenuation displacement RMS start/end donor 0.295338/0.302658. Fixed donor perturbations are identical across recipients; donor-to-donor fractional comparisons have these different absolute sizes. Magnitudes cannot by themselves establish the cause of CE differences.

Seed 61 unpruned: contribution start/end RMS 3.475888/3.510440, end/start norm ratio 1.009940, relative L2 change 0.328881, global cosine 0.946500; actual alpha=.9 attenuation displacement RMS start/end donor 0.347589/0.351044. Fixed donor perturbations are identical across recipients; donor-to-donor fractional comparisons have these different absolute sizes. Magnitudes cannot by themselves establish the cause of CE differences.

sparse, fixed start donor alpha=.9: native/replay T signs agree in 5/5 seeds. Native-minus-replay differences: -0.000246, -0.000475, -0.000490, -0.001243, -0.000102. Replay fixes each recipient’s own downstream decisions, not a common routing policy across recipients.

sparse, fixed end donor alpha=.9: native/replay T signs agree in 5/5 seeds. Native-minus-replay differences: -0.000386, 0.000003, -0.000276, -0.000209, -0.000213. Replay fixes each recipient’s own downstream decisions, not a common routing policy across recipients.

unpruned, fixed start donor alpha=.9: native/replay T signs agree in 5/5 seeds. Native-minus-replay differences: 0.000168, 0.000424, 0.000411, 0.000305, 0.000618. Replay fixes each recipient’s own downstream decisions, not a common routing policy across recipients.

unpruned, fixed end donor alpha=.9: native/replay T signs agree in 5/5 seeds. Native-minus-replay differences: 0.000066, 0.000175, 0.000130, -0.000271, 0.000495. Replay fixes each recipient’s own downstream decisions, not a common routing policy across recipients.


All five seeds are retained. The per-seed ranges above show concentration and heterogeneity without successful-seed filtering, thresholds, significance tests, or fitted classifiers. The previous diagonal ZERO_L0 result is reproduced independently and must not be substituted for fixed-donor T. Absolute CE matrices and attenuation curves are shown alongside changes in six figures.

Engineering note: the aggregate contrast label initially collided with the summary measure field. The original summary is preserved under attempts/aggregate_label_metadata_v1/. Final summaries retain contrast and measure in separate columns; all numerical values, cached evaluations and prespecified conditions are unchanged.

Pre-seal engineering note: an initial configuration check assumed the endpoint metadata schema for full-state parent checkpoints. Inspection located the parent configuration at identity.config.model (endpoints use config.model). The corrected check requires exact registered configuration equality in each explicit schema, with no fallback or tolerance relaxation. The failed check/source and correction are preserved under attempts/configuration_schema_assumption_v1/. No new interface outcome had been evaluated and no cache existed at that correction.
