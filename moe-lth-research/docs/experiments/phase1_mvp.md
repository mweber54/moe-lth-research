# Phase 1: MVP

Train the routing conditions, then compare routing stability, expert usage,
expert-local loss, pruning curves, random masks, and mask overlap.

## Fast Smoke Run

Use this pair first to verify Phase 1 locally:

```powershell
python -m moe_lth.experiments.run_suite --config configs/smoke.yaml --with-pruning
python -m moe_lth.experiments.analyze --suite-dir results/smoke_suite
```

The smoke configuration writes to `results/smoke_suite`.

## Full Baseline Run

Use this matching pair for the full Phase 1 experiment:

```powershell
python -m moe_lth.experiments.run_suite --config configs/baseline.yaml --with-pruning
python -m moe_lth.experiments.analyze --suite-dir results/runs/baseline_suite
```

The baseline configuration writes to `results/runs/baseline_suite`. Do not use
this analysis path after running only the smoke configuration.

## CPU TinyStories Pilot

This configuration reads the local Hugging Face dataset at `data/TinyStories`
and uses a smaller architecture suitable for a CPU-only machine:

```powershell
python -m moe_lth.experiments.run_suite --config configs/tinystories_cpu.yaml --with-pruning
python -m moe_lth.experiments.analyze --suite-dir results/tinystories_cpu_suite
```

## RTX 3080 TinyStories Run

Use the installed CUDA environment. The GPU configuration processes about
41 million tokens per condition using FP16, batch size 128, and the original
4-layer/256-wide/8-expert MVP architecture:

```powershell
conda activate torch-gpu
python -m moe_lth.experiments.run_suite --config configs/tinystories_gpu.yaml --with-pruning
python -m moe_lth.experiments.analyze --suite-dir results/tinystories_gpu_suite
```

The deterministic text subsets used by CUDA training can be regenerated with:

```powershell
python data/export_tinystories.py --dataset-dir data/TinyStories --output-dir data/processed --train-examples 50000 --validation-examples 2000
```

## RTX 3080 WikiText-103 Subset Run

This matching GPU configuration uses the local text splits under
`data/wikitext103_subset`:

```powershell
conda activate torch-gpu
python -m moe_lth.experiments.run_suite --config configs/wikitext103_gpu.yaml --with-pruning
python -m moe_lth.experiments.analyze --suite-dir results/wikitext103_gpu_suite
python -m moe_lth.experiments.analyze_checkpoint --config results/wikitext103_gpu_suite/normal/resolved_config.yaml --checkpoint results/wikitext103_gpu_suite/normal/checkpoints/step_2500.pt
```

The completed run is summarized in `results/wikitext103_gpu_results.md`.

## Three-Seed WikiText Replication

This resumable command reuses the completed seed-7 suite, runs seeds 17 and 29,
and aggregates normal, random-every-step, replay, swapped, pruning,
specialization, and 50%/80% rewind results:

```powershell
conda activate torch-gpu
python -m moe_lth.experiments.run_multiseed `
  --configs configs/wikitext103_gpu.yaml configs/wikitext103_gpu_seed17.yaml configs/wikitext103_gpu_seed29.yaml `
  --sparsities 0.5 0.8 `
  --output-dir results/wikitext103_gpu_multiseed
```

Completed stages are detected from their result artifacts and skipped. The
aggregate report is written to
`results/wikitext103_gpu_multiseed/multiseed_results.md`.

## Fixed-Random and Shuffled-Usage Causal Controls

After the three-seed routing suites are complete, append the two controls that
separate stable random router geometry and expert usage counts from routed-token
identity:

```powershell
conda activate torch-gpu
python -m moe_lth.experiments.run_causal_extensions `
  --configs configs/wikitext103_gpu.yaml configs/wikitext103_gpu_seed17.yaml configs/wikitext103_gpu_seed29.yaml `
  --output-dir results/wikitext103_gpu_multiseed
```

This command preserves the validated `aux_loss_weight=0.1`, reuses each seed's
normal route history, evaluates pruning controls, regenerates suite analysis,
and updates the multi-seed aggregate.

## Cross-Initialization Replay

Use seed 7 as the source routing trajectory while independently initializing
target models with seeds 17 and 29. The target runs use seed 7's data order so
the replayed routes remain aligned to the same training tokens:

```powershell
conda activate torch-gpu
python -m moe_lth.experiments.run_cross_init_replay `
  --source-suite results/wikitext103_gpu_suite `
  --target-configs configs/wikitext103_gpu_seed17.yaml configs/wikitext103_gpu_seed29.yaml `
  --output-dir results/wikitext103_cross_init_replay `
  --sparsities 0.5 0.8
```
