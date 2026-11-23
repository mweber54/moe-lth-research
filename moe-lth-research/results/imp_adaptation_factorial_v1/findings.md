# Adaptation factorial findings

Verified conditions: 60/60. New training jobs launched: 45; completed: 45; completed reused full controls: 15. Counts: {'COMPLETE': 45, 'REUSED': 15}.

## Observations and differential effects

Original native full-validation CE is primary; diagnostic FP32 losses do not replace it. All E32 source seeds and both IMP sparsities remain included. The five dense controls are shared across the two sparsity comparisons.
- Seed 7, sparsity 80%, full: sparse CE 1.329516, dense CE 1.365255, G -0.035739, D +0.000000; sparse change +0.000000, dense change +0.000000; original full dense level recovered=True.
- Seed 7, sparsity 80%, expert_only: sparse CE 3.699194, dense CE 3.074618, G +0.624576, D +0.660315; sparse change +2.369678, dense change +1.709364; original full dense level recovered=False.
- Seed 7, sparsity 80%, expert_router: sparse CE 2.471918, dense CE 2.436540, G +0.035378, D +0.071117; sparse change +1.142403, dense change +1.071286; original full dense level recovered=False.
- Seed 7, sparsity 80%, expert_shared: sparse CE 1.454200, dense CE 1.336974, G +0.117226, D +0.152966; sparse change +0.124684, dense change -0.028281; original full dense level recovered=False.
- Seed 7, sparsity 85%, full: sparse CE 1.343951, dense CE 1.365255, G -0.021304, D +0.000000; sparse change +0.000000, dense change +0.000000; original full dense level recovered=True.
- Seed 7, sparsity 85%, expert_only: sparse CE 3.704870, dense CE 3.074618, G +0.630251, D +0.651555; sparse change +2.360919, dense change +1.709364; original full dense level recovered=False.
- Seed 7, sparsity 85%, expert_router: sparse CE 2.673618, dense CE 2.436540, G +0.237077, D +0.258381; sparse change +1.329667, dense change +1.071286; original full dense level recovered=False.
- Seed 7, sparsity 85%, expert_shared: sparse CE 1.490428, dense CE 1.336974, G +0.153455, D +0.174759; sparse change +0.146478, dense change -0.028281; original full dense level recovered=False.
- Seed 17, sparsity 80%, full: sparse CE 1.351047, dense CE 1.329847, G +0.021199, D +0.000000; sparse change +0.000000, dense change +0.000000; original full dense level recovered=False.
- Seed 17, sparsity 80%, expert_only: sparse CE 3.842927, dense CE 2.685605, G +1.157323, D +1.136123; sparse change +2.491881, dense change +1.355757; original full dense level recovered=False.
- Seed 17, sparsity 80%, expert_router: sparse CE 2.470280, dense CE 2.424850, G +0.045429, D +0.024230; sparse change +1.119233, dense change +1.095003; original full dense level recovered=False.
- Seed 17, sparsity 80%, expert_shared: sparse CE 1.467192, dense CE 1.361792, G +0.105400, D +0.084200; sparse change +0.116145, dense change +0.031945; original full dense level recovered=False.
- Seed 17, sparsity 85%, full: sparse CE 1.366170, dense CE 1.329847, G +0.036323, D +0.000000; sparse change +0.000000, dense change +0.000000; original full dense level recovered=False.
- Seed 17, sparsity 85%, expert_only: sparse CE 4.042243, dense CE 2.685605, G +1.356638, D +1.320315; sparse change +2.676072, dense change +1.355757; original full dense level recovered=False.
- Seed 17, sparsity 85%, expert_router: sparse CE 2.477040, dense CE 2.424850, G +0.052189, D +0.015866; sparse change +1.110869, dense change +1.095003; original full dense level recovered=False.
- Seed 17, sparsity 85%, expert_shared: sparse CE 1.496999, dense CE 1.361792, G +0.135207, D +0.098884; sparse change +0.130829, dense change +0.031945; original full dense level recovered=False.
- Seed 29, sparsity 80%, full: sparse CE 1.345155, dense CE 1.341446, G +0.003709, D +0.000000; sparse change +0.000000, dense change +0.000000; original full dense level recovered=False.
- Seed 29, sparsity 80%, expert_only: sparse CE 3.184008, dense CE 2.573839, G +0.610169, D +0.606461; sparse change +1.838854, dense change +1.232393; original full dense level recovered=False.
- Seed 29, sparsity 80%, expert_router: sparse CE 2.469173, dense CE 2.427544, G +0.041629, D +0.037920; sparse change +1.124019, dense change +1.086098; original full dense level recovered=False.
- Seed 29, sparsity 80%, expert_shared: sparse CE 1.480733, dense CE 1.367072, G +0.113661, D +0.109952; sparse change +0.135578, dense change +0.025626; original full dense level recovered=False.
- Seed 29, sparsity 85%, full: sparse CE 1.366216, dense CE 1.341446, G +0.024770, D +0.000000; sparse change +0.000000, dense change +0.000000; original full dense level recovered=False.
- Seed 29, sparsity 85%, expert_only: sparse CE 2.770214, dense CE 2.573839, G +0.196375, D +0.171605; sparse change +1.403998, dense change +1.232393; original full dense level recovered=False.
- Seed 29, sparsity 85%, expert_router: sparse CE 2.544019, dense CE 2.427544, G +0.116474, D +0.091704; sparse change +1.177803, dense change +1.086098; original full dense level recovered=False.
- Seed 29, sparsity 85%, expert_shared: sparse CE 1.677088, dense CE 1.367072, G +0.310016, D +0.285246; sparse change +0.310872, dense change +0.025626; original full dense level recovered=False.
- Seed 43, sparsity 80%, full: sparse CE 1.337740, dense CE 1.339511, G -0.001772, D +0.000000; sparse change +0.000000, dense change +0.000000; original full dense level recovered=True.
- Seed 43, sparsity 80%, expert_only: sparse CE 3.655621, dense CE 3.652096, G +0.003525, D +0.005297; sparse change +2.317881, dense change +2.312584; original full dense level recovered=False.
- Seed 43, sparsity 80%, expert_router: sparse CE 2.485594, dense CE 2.413260, G +0.072334, D +0.074106; sparse change +1.147855, dense change +1.073749; original full dense level recovered=False.
- Seed 43, sparsity 80%, expert_shared: sparse CE 1.466777, dense CE 1.352234, G +0.114543, D +0.116315; sparse change +0.129038, dense change +0.012722; original full dense level recovered=False.
- Seed 43, sparsity 85%, full: sparse CE 1.352644, dense CE 1.339511, G +0.013132, D +0.000000; sparse change +0.000000, dense change +0.000000; original full dense level recovered=False.
- Seed 43, sparsity 85%, expert_only: sparse CE 4.372081, dense CE 3.652096, G +0.719986, D +0.706853; sparse change +3.019437, dense change +2.312584; original full dense level recovered=False.
- Seed 43, sparsity 85%, expert_router: sparse CE 2.480257, dense CE 2.413260, G +0.066997, D +0.053865; sparse change +1.127614, dense change +1.073749; original full dense level recovered=False.
- Seed 43, sparsity 85%, expert_shared: sparse CE 1.503022, dense CE 1.352234, G +0.150788, D +0.137656; sparse change +0.150378, dense change +0.012722; original full dense level recovered=False.
- Seed 61, sparsity 80%, full: sparse CE 1.339290, dense CE 1.366093, G -0.026802, D +0.000000; sparse change +0.000000, dense change +0.000000; original full dense level recovered=True.
- Seed 61, sparsity 80%, expert_only: sparse CE 3.389923, dense CE 3.348531, G +0.041392, D +0.068194; sparse change +2.050633, dense change +1.982438; original full dense level recovered=False.
- Seed 61, sparsity 80%, expert_router: sparse CE 2.466482, dense CE 2.427888, G +0.038594, D +0.065396; sparse change +1.127192, dense change +1.061796; original full dense level recovered=False.
- Seed 61, sparsity 80%, expert_shared: sparse CE 1.546221, dense CE 1.399678, G +0.146543, D +0.173345; sparse change +0.206931, dense change +0.033585; original full dense level recovered=False.
- Seed 61, sparsity 85%, full: sparse CE 1.395800, dense CE 1.366093, G +0.029708, D +0.000000; sparse change +0.000000, dense change +0.000000; original full dense level recovered=False.
- Seed 61, sparsity 85%, expert_only: sparse CE 4.675382, dense CE 3.348531, G +1.326851, D +1.297143; sparse change +3.279581, dense change +1.982438; original full dense level recovered=False.
- Seed 61, sparsity 85%, expert_router: sparse CE 2.484069, dense CE 2.427888, G +0.056181, D +0.026473; sparse change +1.088269, dense change +1.061796; original full dense level recovered=False.
- Seed 61, sparsity 85%, expert_shared: sparse CE 1.565527, dense CE 1.399678, G +0.165849, D +0.136141; sparse change +0.169727, dense change +0.033585; original full dense level recovered=False.

## Equal-seed endpoint summaries

- 80% full, n=5 seeds (complete): leaves the mean gap unchanged; mean D=+0.000000, descriptive 95% t interval [0.0, 0.0], signs +0/-0. Mean sparse change +0.000000, dense change +0.000000: neither shows mean deterioration.
- 80% expert_only, n=5 seeds (complete): increases the sparse-minus-dense gap; mean D=+0.495278, descriptive 95% t interval [-0.08471436668159027, 1.075270469203232], signs +5/-0. Mean sparse change +2.213785, dense change +1.718507: both sparse and dense deteriorate.
- 80% expert_router, n=5 seeds (complete): increases the sparse-minus-dense gap; mean D=+0.054554, descriptive 95% t interval [0.026995780607531394, 0.08211191620728718], signs +5/-0. Mean sparse change +1.132140, dense change +1.077586: both sparse and dense deteriorate.
- 80% expert_shared, n=5 seeds (complete): increases the sparse-minus-dense gap; mean D=+0.127356, descriptive 95% t interval [0.08320208427516337, 0.17150943631921198], signs +5/-0. Mean sparse change +0.142475, dense change +0.015119: both sparse and dense deteriorate.
- 85% full, n=5 seeds (complete): leaves the mean gap unchanged; mean D=+0.000000, descriptive 95% t interval [0.0, 0.0], signs +0/-0. Mean sparse change +0.000000, dense change +0.000000: neither shows mean deterioration.
- 85% expert_only, n=5 seeds (complete): increases the sparse-minus-dense gap; mean D=+0.829494, descriptive 95% t interval [0.22785639063212315, 1.431132230809148], signs +5/-0. Mean sparse change +2.548002, dense change +1.718507: both sparse and dense deteriorate.
- 85% expert_router, n=5 seeds (complete): increases the sparse-minus-dense gap; mean D=+0.089258, descriptive 95% t interval [-0.03364422894279864, 0.2121602993709275], signs +5/-0. Mean sparse change +1.166844, dense change +1.077586: both sparse and dense deteriorate.
- 85% expert_shared, n=5 seeds (complete): increases the sparse-minus-dense gap; mean D=+0.166537, descriptive 95% t interval [0.0776598376304285, 0.2554146006843042], signs +5/-0. Mean sparse change +0.181657, dense change +0.015119: both sparse and dense deteriorate.

## Interpretation

Positive D indicates greater relative sparse impairment. Interpret its sparse and dense components separately: a lower gap driven by poor dense performance is not successful recovery of the original performance level. Similar sparse/dense deterioration provides little explanation of sparse outperformance. Freezing initially untrained shared attention and tied embedding/output limits this protocol; fixed-budget failure can be slower learning. Frozen router parameters do not freeze decisions or exposure. Clipping/optimizer changes are part of the restriction, not a purely routing effect.

Seed is the independent unit (n=5), with equal weights and separate .80/.85 summaries. The sparsities share seeds and dense controls. Descriptive t intervals and sign consistency are not equivalence tests or proof of a unique mechanism, compensation, generality or novelty. No performance-based stopping, extension or additional experiment is scheduled.

Diagnostics use the same complete native-input probe, proposed/accepted assignment counts and explicit gradient denominators. Update norms are actual net parameter displacement across fixed intervals, not summed path length. Historical clipping is reconstructed and labelled; new clipping uses the actual native return. Primary curves and every seed effect precede aggregated summaries in the CSVs. Missing/pending endpoints are not observations.

See protocol.md, manifest.json, reuse_decisions.json, input_hashes.json, parameter_groups.csv, resource_estimate.json, validation/, status/, workers/, and per-attempt diagnostic/training/state files.
