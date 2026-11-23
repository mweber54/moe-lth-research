# High sparsity coupling: execution report

Exploratory/post hoc. Completed reruns: 0/15. Verified measured states: 6. Overall complete: False.

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

Masked single-token gradient identity maximum absolute residual: 0. Router identity residual: 0. Unselected task-gradient logits observed: up to 31 per layer. This repository's full-softmax gate differs from selected-logit normalization.

1. Coupling persistence/increase/decrease: see seed trajectories and fixed expert changes; a cohort-wide conclusion requires complete paired coverage.
2. Input versus pair selection: fixed-source/native and fixed-initial/source/current selections are separately cached. These comparisons are observational and are not additive causal decompositions.
3. Masking changes local expert row gradients to delta*(mask*x); signed directions and retained input norms are reported separately from optimizer updates.
4. Successful versus unsuccessful consistency: historical status alone does not establish rerun status; no success-only selection is used.
5. Embedding/hidden evolution: byte embeddings are tied to the output head. Partial snapshots cannot establish temporal precedence.
6. Robust conclusions require agreement across association, q, Gram/SVD, full expert outputs and model losses; incomplete checks remain incomplete.
7. Subsequent focused intervention: after full observational analysis, randomize whether shared representation parameters are frozen during a short continuation with identical initial weights, masks and data. Compare fixed-pair coupling and validation CE. This is a candidate input-evolution test, not performed here; revise if complete findings favor another explanation.

No independent test split has been established. No causal mechanism or publication claim is made. Historical missing states are listed in coverage. Exact checkpoint key inventory is in checkpoint_coverage.csv. Failed attempts and live heartbeats remain under reruns/.
