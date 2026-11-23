# Initial checkpoint analysis

Exploratory learned-minus-random pruning-only effects; primary rank 16; seed is the replication unit. All seed effects are in pruning_seed_effects.csv. Historical endpoints are separate directories; no trajectories inferred.

- sparsity 0.5, alignment_centered: n=5, mean=0.021466295135667713, seed bootstrap 95% interval [0.02000748466449736, 0.02307986191029009].
- sparsity 0.5, alignment_above_shuffled: n=5, mean=0.01676255850768154, seed bootstrap 95% interval [0.015738279631406626, 0.017692991579028056].
- sparsity 0.5, source_subspace_overlap: n=5, mean=0.25218543527199017, seed bootstrap 95% interval [0.2434148163777583, 0.2609560541662219].
- sparsity 0.65, alignment_centered: n=5, mean=0.03379912728216271, seed bootstrap 95% interval [0.031846618729804724, 0.035588305004286724].
- sparsity 0.65, alignment_above_shuffled: n=5, mean=0.02610279671196094, seed bootstrap 95% interval [0.024701129135168497, 0.027383905951084288].
- sparsity 0.65, source_subspace_overlap: n=5, mean=0.2861401147240878, seed bootstrap 95% interval [0.2795584919168845, 0.29204322274086303].

## All seed effects for primary centered alignment

| Sparsity | Seed 7 | Seed 17 | Seed 29 | Seed 43 | Seed 61 |
|---|---:|---:|---:|---:|---:|
| 50% | 0.02044 | 0.01919 | 0.02289 | 0.02402 | 0.02079 |
| 65% | 0.03680 | 0.03420 | 0.03044 | 0.03278 | 0.03477 |

Learned pruning slightly raises alignment above the original source level; higher alignment is not literal identity with source geometry. Compare alignment and basis preservation separately.
Sparsity 50%: source-subspace overlap learned/random is 0.943/0.690 before rewind and 0.618/0.217 after rewind. Actual R100 rewind alignment is 0.130/0.061. These are distinct reference states, not recovery findings.
Sparsity 65%: source-subspace overlap learned/random is 0.888/0.602 before rewind and 0.667/0.174 after rewind. Actual R100 rewind alignment is 0.162/0.063. These are distinct reference states, not recovery findings.

175 existing states summarized separately in existing_state_seed_summaries.csv. Source-age geometry, unmasked rewind, masked rewind, pruning-only trained states and historical endpoints must not be conflated. All 15 canonical historical dense checkpoints are hash-verified on both hosts and included. Separate table-backed reference directories do not themselves contain model checkpoints.
No causal claim. Rewound initialization and source-age states are separate raw tables. New recovery findings remain pending.
