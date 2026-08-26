# Routing-Conditioned Lottery Tickets in MoE Experts

This repository executes the research plan in `moe_lth_execution_plan.md`.
It is a controlled, small-scale PyTorch framework for testing whether
training-time routing trajectories causally induce sparse, lottery-ticket-like
subnetworks inside individual Mixture-of-Experts (MoE) experts.

The default research configuration is a 4-layer decoder-only Transformer with
4 attention heads, `d_model=256`, 8 Top-1 experts per layer, expert hidden size
1024, capacity factor 1.25, and load-balancing weight 0.01. A deterministic
built-in byte-level story corpus makes every command runnable without a
download. Set `data.path` to a TinyStories, WikiText, or OpenWebText text file
for the intended language-model experiments.

## What Is Implemented

- Learned Top-1 and Top-2 MoE routing, capacity constraints, and auxiliary
  balancing loss
- Fixed random router, random-every-step, replay, swap, usage-preserving
  shuffle, and strong-load-balancing interventions
- Per-step expert usage, entropy, margin, dropped-token, context-sample, and
  compressed route-history logs
- Fixed validation routing snapshots at checkpoints
- Expert-local and expert-layer-global magnitude pruning
- Random-mask, other-expert-mask, and random-reinitialization controls
- Initialization/early-checkpoint rewind with masked retraining
- Expert-local loss and expert-substitution specificity evaluation
- Usage entropy/CV/dead experts, routing stability, token-distribution JS
  divergence, mask Jaccard, router geometry, and hidden-state separability
- Minimum viable, all-counterfactual, rewind, and robustness experiment runners
- Figure generation for routing, usage, masks, pruning, rewind, specificity,
  and routing-history-versus-mask similarity

The completed Phase 4 runner covers 4/8/16 experts, top-1/top-2 routing, 4/8
layers, three seeds, and balanced TinyStories/WikiText validation.

## Setup

From this directory:

```powershell
python -m pip install -e .
python -m pytest
```

Run the fast end-to-end suite:

```powershell
python -m moe_lth.experiments.run_suite --config configs/smoke.yaml --with-pruning
python -m moe_lth.experiments.analyze --suite-dir results/smoke_suite
```

Keep each training command paired with its matching analysis path:

| Training config | Generated suite directory |
|---|---|
| `configs/smoke.yaml` | `results/smoke_suite` |
| `configs/baseline.yaml` | `results/runs/baseline_suite` |
| `configs/tinystories_cpu.yaml` | `results/tinystories_cpu_suite` |
| `configs/tinystories_gpu.yaml` | `results/tinystories_gpu_suite` |

### RTX 3080 Environment

The base and `torch-gpu` Conda environments contain CPU-only PyTorch on this
machine. Use the existing CUDA-capable `thgnn` environment for GPU experiments:

```powershell
conda activate thgnn
$env:PYTHONPATH="src"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m moe_lth.experiments.run_suite --config configs/tinystories_gpu.yaml --with-pruning
```

`configs/tinystories_gpu.yaml` uses FP16 with gradient scaling, TF32-enabled
matrix operations, batch size 128, and approximately 41 million training tokens
per condition. The local Arrow dataset is exported to deterministic text files
before CUDA training to avoid a Windows Arrow/PyTorch OpenMP runtime conflict.

## Main Experiments

Train one baseline:

```powershell
python -m moe_lth.training.train --config configs/baseline.yaml
```

Run the minimum viable paper suite: normal, random every step, replay, and
swapped histories:

```powershell
python -m moe_lth.experiments.run_suite --config configs/baseline.yaml --with-pruning
python -m moe_lth.experiments.analyze --suite-dir results/runs/baseline_suite
```

Run every causal routing condition:

```powershell
python -m moe_lth.experiments.run_all_conditions --config configs/baseline.yaml
```

Analyze expert specialization, router geometry, hidden-state separability, and
expert substitutions for one checkpoint:

```powershell
python -m moe_lth.experiments.analyze_checkpoint `
  --config configs/baseline.yaml `
  --checkpoint results/runs/baseline/checkpoints/step_50000.pt
```

Generate pruning and rewind figures after a run (kept in a separate process to
avoid Windows PyTorch/matplotlib OpenMP conflicts):

```powershell
python -m moe_lth.visualization.generate_figures --run-dir results/runs/baseline
```

Run lottery-ticket controls after the baseline:

```powershell
python -m moe_lth.experiments.run_rewind_suite `
  --config configs/baseline.yaml `
  --final-checkpoint results/runs/baseline_suite/normal/checkpoints/step_50000.pt `
  --sparsity 0.8
```

Run or resume the Phase 4 architecture and dataset robustness grid:

```powershell
python -m moe_lth.experiments.run_phase4_robustness `
  --architecture-configs configs/wikitext103_gpu.yaml configs/wikitext103_gpu_seed17.yaml configs/wikitext103_gpu_seed29.yaml `
  --dataset-configs configs/tinystories_gpu.yaml configs/tinystories_gpu_seed17.yaml configs/tinystories_gpu_seed29.yaml configs/multidomain_gpu.yaml configs/multidomain_gpu_seed17.yaml configs/multidomain_gpu_seed29.yaml `
  --output-dir results/phase4_robustness
```

Use `--report-only` with the same arguments to regenerate the aggregate report
and figures without rerunning training.

## Conditions

| Condition | Causal question |
|---|---|
| `normal` | Do learned-router experts form sparse subnetworks? |
| `fixed_random` | Is fixed router geometry enough? |
| `random_every_step` | Does destroying stable history weaken masks? |
| `replay` | Does the same route history reproduce similar masks? |
| `swapped` | Do masks follow history rather than expert identity? |
| `shuffled_usage` | Are matched update counts sufficient? |
| `strong_balance` | Does specialization survive balanced usage? |

## Lottery-Ticket Decision Rule

Use the configured 80% or higher sparsity results and compare them with the
dense final checkpoint. A ticket supports the plan's criterion when full and
expert-local loss degrade by no more than 2-5%, learned masks beat random
masks, and rewound weights beat random reinitialization. Initialization rewind
supports strict LTH; an early checkpoint supports practical modern LTH.

## Outputs

Every run writes a resolved config, checkpoints, JSONL training/usage/context
logs, compressed validation routes, optional compressed training-route history,
masks, JSON result tables, and PNG figures below its configured `output_dir`.
Starting a run resets that run directory's generated training logs and
checkpoints because training resumption is not currently supported.

The full experimental logic and artifact-to-claim mapping are documented in
[`docs/experimental_protocol.md`](docs/experimental_protocol.md).

The completed 500-step CPU TinyStories pilot and balanced causal control are
summarized in [`results/tinystories_cpu_results.md`](results/tinystories_cpu_results.md).

RTX 3080 environment details, benchmarks, and run commands are documented in
[`docs/gpu_setup.md`](docs/gpu_setup.md).
