# High sparsity coupling: execution report

Exploratory/post hoc. Completed reruns: 15/15. Verified measured states: 315. Overall complete: True.

Historical strict matches: 4/10, representing 3 independent source seeds. Five distinct dense controls. Historical labels remain separate from reruns.

seed,sparsity,gap,strict_success
7,0.8,-0.0356003672295124,True
7,0.85,-0.0212048150546713,True
17,0.8,0.0211169305081984,False
17,0.85,0.0362705615454908,False
29,0.8,0.0037887048684128,False
29,0.85,0.0247973300030488,False
43,0.8,-0.0017636812040022,True
43,0.85,0.0131984211916982,False
61,0.8,-0.026912756390097,True
61,0.85,0.0295671104052359,False


## Findings and limits

Measured fixed-initial-pair coupling changes (matched valid expert populations):

track,seed,sparsity,initial,final,delta,n_matched_experts
historical,7,0.8,0.1894383984732913,0.3708327336931475,0.18139433521985615,128
historical,7,0.85,0.1707124633689434,0.3746909575477442,0.20397849417880073,128
historical,17,0.8,0.17443489833470913,0.34627040512400814,0.171835506789299,126
historical,17,0.85,0.1589101614953693,0.38190664803243585,0.22299648653706639,126
historical,29,0.8,0.2356331720937549,0.4208337589778042,0.18520058688404925,128
historical,29,0.85,0.20962194102161097,0.3823929350074417,0.17277099398583082,128
historical,43,0.8,0.1545240874005896,0.334519379980854,0.17999529258026442,128
historical,43,0.85,0.12878229191381235,0.36476818190246274,0.2359858899886504,128
historical,61,0.8,0.2401734209898989,0.3507840598565257,0.11061063886662682,128
historical,61,0.85,0.2187874431181303,0.3940043998976456,0.17521695677951526,128
rerun,7,0.8,0.1894383984732913,0.3708327336931475,0.18139433521985615,128
rerun,7,0.85,0.1707124633689434,0.3746909575477442,0.20397849417880073,128
rerun,17,0.8,0.17443489833470913,0.34627040512400814,0.171835506789299,126
rerun,17,0.85,0.1589101614953693,0.38190664803243585,0.22299648653706639,126
rerun,29,0.8,0.2356331720937549,0.4208337589778042,0.18520058688404925,128
rerun,29,0.85,0.20962194102161097,0.3823929350074417,0.17277099398583082,128
rerun,43,0.8,0.1545240874005896,0.334519379980854,0.17999529258026442,128
rerun,43,0.85,0.12878229191381235,0.36476818190246274,0.2359858899886504,128
rerun,61,0.8,0.2401734209898989,0.3507840598565257,0.11061063886662682,128
rerun,61,0.85,0.2187874431181303,0.3940043998976456,0.17521695677951526,128


Masked single-token gradient identity maximum absolute residual: 0. Router identity residual: 0. Unselected task-gradient logits observed: up to 31 per layer. This repository's full-softmax gate differs from selected-logit normalization.

## Answers supported by completed measurements

1–2. historical, sparsity 0.8, fixed_source, all_expert: 5 source seeds; coupling increased in 5, decreased in 0; mean initial 0.1670, final 0.3066, change +0.1396. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. historical, sparsity 0.8, fixed_source, current_accepted: 5 source seeds; coupling increased in 5, decreased in 0; mean initial -0.0052, final 0.1204, change +0.1256. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. historical, sparsity 0.8, fixed_source, fixed_initial: 5 source seeds; coupling increased in 5, decreased in 0; mean initial 0.0195, final 0.2517, change +0.2321. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. historical, sparsity 0.8, native, all_expert: 5 source seeds; coupling increased in 5, decreased in 0; mean initial 0.2621, final 0.3961, change +0.1340. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. historical, sparsity 0.8, native, current_accepted: 5 source seeds; coupling increased in 0, decreased in 5; mean initial 0.1816, final 0.0237, change -0.1579. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. historical, sparsity 0.8, native, fixed_initial: 5 source seeds; coupling increased in 5, decreased in 0; mean initial 0.1988, final 0.3646, change +0.1658. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. historical, sparsity 0.85, fixed_source, all_expert: 5 source seeds; coupling increased in 5, decreased in 0; mean initial 0.1842, final 0.3554, change +0.1712. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. historical, sparsity 0.85, fixed_source, current_accepted: 5 source seeds; coupling increased in 5, decreased in 0; mean initial 0.0162, final 0.1819, change +0.1657. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. historical, sparsity 0.85, fixed_source, fixed_initial: 5 source seeds; coupling increased in 5, decreased in 0; mean initial 0.0342, final 0.3213, change +0.2871. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. historical, sparsity 0.85, native, all_expert: 5 source seeds; coupling increased in 5, decreased in 0; mean initial 0.2545, final 0.3968, change +0.1423. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. historical, sparsity 0.85, native, current_accepted: 5 source seeds; coupling increased in 0, decreased in 5; mean initial 0.1632, final 0.0057, change -0.1574. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. historical, sparsity 0.85, native, fixed_initial: 5 source seeds; coupling increased in 5, decreased in 0; mean initial 0.1774, final 0.3796, change +0.2022. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. rerun, sparsity 0.8, fixed_source, all_expert: 5 source seeds; coupling increased in 5, decreased in 0; mean initial 0.1670, final 0.3066, change +0.1396. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. rerun, sparsity 0.8, fixed_source, current_accepted: 5 source seeds; coupling increased in 5, decreased in 0; mean initial -0.0052, final 0.1204, change +0.1256. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. rerun, sparsity 0.8, fixed_source, fixed_initial: 5 source seeds; coupling increased in 5, decreased in 0; mean initial 0.0195, final 0.2517, change +0.2321. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. rerun, sparsity 0.8, native, all_expert: 5 source seeds; coupling increased in 5, decreased in 0; mean initial 0.2621, final 0.3961, change +0.1340. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. rerun, sparsity 0.8, native, current_accepted: 5 source seeds; coupling increased in 0, decreased in 5; mean initial 0.1816, final 0.0237, change -0.1579. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. rerun, sparsity 0.8, native, fixed_initial: 5 source seeds; coupling increased in 5, decreased in 0; mean initial 0.1988, final 0.3646, change +0.1658. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. rerun, sparsity 0.85, fixed_source, all_expert: 5 source seeds; coupling increased in 5, decreased in 0; mean initial 0.1842, final 0.3554, change +0.1712. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. rerun, sparsity 0.85, fixed_source, current_accepted: 5 source seeds; coupling increased in 5, decreased in 0; mean initial 0.0162, final 0.1819, change +0.1657. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. rerun, sparsity 0.85, fixed_source, fixed_initial: 5 source seeds; coupling increased in 5, decreased in 0; mean initial 0.0342, final 0.3213, change +0.2871. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. rerun, sparsity 0.85, native, all_expert: 5 source seeds; coupling increased in 5, decreased in 0; mean initial 0.2545, final 0.3968, change +0.1423. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. rerun, sparsity 0.85, native, current_accepted: 5 source seeds; coupling increased in 0, decreased in 5; mean initial 0.1632, final 0.0057, change -0.1574. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
1–2. rerun, sparsity 0.85, native, fixed_initial: 5 source seeds; coupling increased in 5, decreased in 0; mean initial 0.1774, final 0.3796, change +0.2022. Pair identities remain fixed only for the fixed selections. See paired_seed_coupling_changes.csv for changes relative to each seed's dense control.
Input and selection contrasts are separately measured on the same token positions. Their differences are descriptive, not an additive causal decomposition. Missing experts are excluded using common initial/final membership within each contrast.

3. Across completed checks, maximum masked expert identity residual is 0; median retained input norm is 0.5893. Signed and absolute directions are separate raw columns; zero-gradient cases are undefined. Up to 31 unselected router logits receive task gradients under full-softmax normalization. Batch-aggregated AdamW changes are separately recorded and are not the single-token identity.

4. Native, fixed-initial-pair effects by historical status (validation labels; the two sparsities within seed are dependent):

```csv
track,seed,sparsity,strict_success,delta
historical,7,0.8,True,0.18139433521985615
rerun,7,0.8,True,0.18139433521985615
historical,7,0.85,True,0.20397849417880073
rerun,7,0.85,True,0.20397849417880073
historical,17,0.8,False,0.171835506789299
rerun,17,0.8,False,0.171835506789299
historical,17,0.85,False,0.22299648653706639
rerun,17,0.85,False,0.22299648653706639
historical,29,0.8,False,0.18520058688404925
rerun,29,0.8,False,0.18520058688404925
historical,29,0.85,False,0.17277099398583082
rerun,29,0.85,False,0.17277099398583082
historical,43,0.8,True,0.17999529258026442
rerun,43,0.8,True,0.17999529258026442
historical,43,0.85,False,0.2359858899886504
rerun,43,0.85,False,0.2359858899886504
historical,61,0.8,True,0.11061063886662682
rerun,61,0.8,True,0.11061063886662682
historical,61,0.85,False,0.17521695677951526
rerun,61,0.85,False,0.17521695677951526
```
Consistency must be judged across source seeds, with rerun status separately evaluated; these nested rows are not independent replications.

5. Embedding and router-input trajectories are cached on identical positions (earliest measured post-initial step: 25). Tied byte embeddings also serve output prediction. The observation grid bounds temporal resolution; these measurements alone do not establish causal precedence.

6. Association, full-spectrum q, Gram/SVD, complete expert-function errors and model CE/KL are exported separately. No cross-metric mechanistic conclusion is claimed automatically from correlation, distance, or a shared time trend. Inspect paired seed effects and exclusions; incomplete families remain incomplete.

7. Candidate subsequent intervention: randomize whether shared representation parameters are frozen during a short continuation with identical initial weights, masks and data. Compare fixed-pair coupling and CE. This would isolate a contribution of representation evolution; it is not performed here, and its priority remains provisional until the complete observational results are reviewed.

No independent test split has been established. No causal mechanism or publication claim is made. Historical missing states are listed in coverage. Exact checkpoint key inventory is in checkpoint_coverage.csv. Failed attempts and live heartbeats remain under reruns/.
