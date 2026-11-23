# Existing source and pruning-only references

These states are not recovery trajectories or extra seed replications. See existing_gram_seed_summaries.csv and existing_activation_seed_summaries.csv for every source, masked-trained, masked-initialization and historical endpoint state. Pruning-only learned/random comparisons jointly match valid experts across SVD, Gram and activation measurements; all five seeds and paired bootstrap uncertainty are in pruning_joint_paired_summary.csv.

Source-final activation reference (equal eligible-expert Pearson means):

| Mode | Selection | Seed 7 | Seed 17 | Seed 29 | Seed 43 | Seed 61 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fixed | accepted | 0.04949 | 0.10633 | 0.09141 | 0.12586 | 0.11863 |
| fixed | all_expert | -0.03247 | 0.05043 | -0.05190 | 0.03384 | 0.02836 |
| fixed | proposed | 0.05270 | 0.10255 | 0.08768 | 0.11926 | 0.10857 |
| native | accepted | 0.04949 | 0.10633 | 0.09142 | 0.12472 | 0.11863 |
| native | all_expert | -0.03247 | 0.05043 | -0.05190 | 0.03397 | 0.02836 |
| native | proposed | 0.05270 | 0.10255 | 0.08768 | 0.11769 | 0.10857 |

Gram pruning-only source-final shape and energy (all experts, separate from joint paired eligible-expert analysis):

| Mask | Sparsity | Mean shape distance | Mean trace ratio |
| --- | ---: | ---: | ---: |
| learned | 50% | 0.09188 | 0.96200 |
| learned | 65% | 0.14698 | 0.90402 |
| random | 50% | 0.35984 | 0.63984 |
| random | 65% | 0.45845 | 0.51081 |

These projection statistics do not quantify the complete nonlinear expert or imply that pruning preserves its full function. Native and fixed-input activation modes remain distinct. Individual expert distributions are descriptive. The exact eligible sample counts and exclusions are retained in the raw tables.
