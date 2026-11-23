# Prior-work check

**Search date:** 2026-10-20. Searches focused on the only candidate that survived the repository sweep—the existing learned-versus-random expert-local mask comparison—and on adjacent MoE pruning work. This is a narrow overlap check, not a systematic review and not proof of novelty.

## Search record

Queries:

1. “primary paper for lottery ticket hypothesis Mixture of Experts pruning, expert lottery tickets, IMP MoE, official code”
2. “papers mixture of experts router pruning router checkpoint age lottery ticket expert subnetworks evaluation”
3. “MoE-Pruner github.com router hints”

The second search request temporarily returned an upstream 503; primary arXiv pages and the official REAP repository were accessed directly afterward. The 2024 MoE-Pruner paper was accessible, but I could not verify an official paper-linked code repository. A GitHub repository with the same name surfaced as an expert-fusion project, so it is not treated as verified implementation evidence.

## Closest primary work

### Lottery-ticket controls

Frankle and Carbin, **“The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks,”** ICLR 2019, [arXiv:1803.03635](https://arxiv.org/abs/1803.03635). The paper identifies a mask by training and magnitude-pruning, resets surviving weights to their original initialization, and compares the resulting learned subnetwork with randomly sampled sparse subnetworks. This establishes that learned-versus-random and initialization rewind are standard baseline questions, not novel methodological claims by themselves. The paper is not an MoE/router-age study.

### Router-informed MoE pruning

**“MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router,”** [arXiv:2410.12013](https://arxiv.org/abs/2410.12013), 2024. The accessible primary-paper text studies MoE compression using router-derived signals, including expert activation frequency/load balance, and relates pruning strategy to how the MoE was initialized/trained. It is concerned with router-informed pruning/compression, not a five-seed comparison of R0/R100 initialization-rewound, exact per-tensor-count-matched sparse weight masks. An official code link was not verified.

### Whole-expert pruning

**“REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression,”** ICLR 2026, [arXiv:2510.13999](https://arxiv.org/abs/2510.13999) and the authors’ [CerebrasResearch/reap repository](https://github.com/CerebrasResearch/reap). The paper introduces Router-weighted Expert Activation Pruning, combining router gate values and expert activation norms to select experts for pruning/merging in large pretrained MoEs; the released code includes calibration and compression workflows. Its unit of pruning is principally whole experts in models ranging from tens of billions of parameters upward. It does not answer whether learned within-expert sparse coordinates improve the small-model R0 recovery endpoint over a random mask with exactly the same per-tensor keep counts.

## Assessment

The only credible remaining question is a bounded analysis of existing data: **within this E32 setup, does a learned expert-local mask outperform one per-tensor-count-matched random mask at R0 and R100 over 50-65% target sparsity?** The comparison is ordinary lottery-ticket control logic applied to MoE experts; its specific contribution could be a carefully matched, seed-complete MoE result, not a new pruning principle. The result is validation-based, uses one random draw per source/sparsity, and reuses five already-seen seeds, so it is exploratory. A fresh confirmation would need new source seeds and a genuinely untouched evaluation split; neither is presently established.

I found no basis to say that no one has studied this question. Whether the narrow protocol-specific result is sufficiently distinct for a paper remains unresolved, and it does not justify more training by itself.
