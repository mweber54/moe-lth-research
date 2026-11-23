# Optimization Geometry Preflight

## Existing Pairing

All 18 required 85% sparse/dense pairs are present: e8 R0/R20/R100 and e12 R20/R60/R100 across seeds 7, 17, and 29. Metadata confirms matched seed, router age/step, shared-state hash, training/validation sequence hashes, recovery budget, optimizer, scheduler, and precision. Sparse masks are expert-local magnitude masks and existing integrity checks passed.

## Falsified Retrospective Path

Only `final_recovered.pt` (step 2500) is saved for recovery runs. There are no full sparse/dense recovery checkpoints at 0, 100, 250, 500, 750, 1000, 1500, or 2000, and no optimizer states. Historical `gradient_stats.jsonl` provides norms only, not gradient vectors. Therefore the pre-specified primary metrics cannot be reconstructed across time; norms must not be substituted for cosines.
