# IMP mask information v2 findings

Status: COMPLETE. Phase 1 verified endpoints: 30/30. Phase 2 completed: 25/25 conditional endpoints.
Job counts: {'COMPLETE': 49, 'REUSED': 6}. Phase 2 gate: ADMITTED.
New jobs actually claimed: 49; reused: 6; recovery failures: 0; recorded pre-worker startup failures: 1 (see launch_failed_*.json and launch_correction.json).

## Direct observations

E16, nominal 80% sparsity, seeds 7/17/29/43/61 were fixed before new outcomes, irrespective of historical dense gaps. Seed17 IMP and five matrix controls are compatible pilot reuse. Global/expert controls were not rerun. All new recoveries use 2500 attempted updates; applied updates and AMP skips appear in results.csv.
- Seed 7 matrix: 5/5 draws; IMP CE 1.591078; mean control CE 1.680889; SD 0.009126; mean minus IMP +0.089810; historical matched dense CE 1.479220.
- Seed 7 degree: 5/5 draws; IMP CE 1.591078; mean control CE 1.686177; SD 0.005477; mean minus IMP +0.095099; historical matched dense CE 1.479220.
- Seed 17 matrix: 5/5 draws; IMP CE 1.458223; mean control CE 1.678239; SD 0.008363; mean minus IMP +0.220016; historical matched dense CE 1.479363.
- Seed 17 degree: 5/5 draws; IMP CE 1.458223; mean control CE 1.680305; SD 0.022812; mean minus IMP +0.222082; historical matched dense CE 1.479363.
- Seed 29 matrix: 5/5 draws; IMP CE 1.488630; mean control CE 1.662958; SD 0.018492; mean minus IMP +0.174328; historical matched dense CE 1.428627.
- Seed 29 degree: 5/5 draws; IMP CE 1.488630; mean control CE 1.671172; SD 0.016488; mean minus IMP +0.182541; historical matched dense CE 1.428627.
- Seed 43 matrix: 5/5 draws; IMP CE 1.484748; mean control CE 1.675915; SD 0.012978; mean minus IMP +0.191167; historical matched dense CE 1.452162.
- Seed 43 degree: 5/5 draws; IMP CE 1.484748; mean control CE 1.689406; SD 0.007454; mean minus IMP +0.204658; historical matched dense CE 1.452162.
- Seed 61 matrix: 5/5 draws; IMP CE 1.608024; mean control CE 1.645488; SD 0.009989; mean minus IMP +0.037463; historical matched dense CE 1.458569.
- Seed 61 degree: 5/5 draws; IMP CE 1.608024; mean control CE 1.633078; SD 0.006041; mean minus IMP +0.025054; historical matched dense CE 1.458569.

## Descriptive uncertainty

Random draws are nested within source seeds. Within-seed intervals describe variation over masks under fixed training seeds and do not increase independent model replication. Across-seed intervals use equal-weight paired effects with a t reference and at most five source seeds; they are descriptive and limited to this architecture/sparsity/protocol.
- matrix: 5 seed effects, mean +0.142557; descriptive 95% t interval [0.04799632469206243, 0.23711746367171427].
- degree: 5 seed effects, mean +0.145887; descriptive 95% t interval [0.042404249278997955, 0.2493696671732497].
- matrix_minus_degree: 5 seed effects, mean -0.003330; descriptive 95% t interval [-0.015436695213283436, 0.008776567124812472].

## Execution decision

The degree phase is admitted only if all five seed-level matrix-minus-IMP means exceed 0.02. This is an execution rule, not a significance test. Finite-chain diagnostics must also pass for every diagnostic and production matrix before any degree recovery. No uniform-sampling or independent-stationary-draw claim is made.

## Interpretation and unresolved alternatives

Any replicated matrix-control deficit is evidence about mask placement beyond matrix survivor counts for these E16/80% seed runs, not a general result or discovered mechanism. If degree-preserving masks recover better, connectivity counts may explain part of the matrix-random deficit; a remaining deficit means these counts are insufficient under this protocol. Neither outcome identifies indispensable edges, proves routed-token information, or explains sparse outperformance by itself. Finite swap-chain mixing, initialization, optimization and the narrow architecture/sparsity scope remain limitations. Do not declare equivalence from nonsignificant differences.

Exact provenance: protocol.md, registration.json, manifest.json, seal.json, reuse_checks.json, host_checks/, per-attempt stage metadata, mask_validation.csv and degree_diagnostics.json when admitted. Every observed endpoint and original dense CE is in results.csv; each canonical curve is referenced by path. No further expansion is scheduled.
