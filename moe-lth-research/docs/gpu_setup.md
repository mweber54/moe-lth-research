# RTX 3080 GPU Workflow

## Environment

The system's base Conda environment has CPU-only PyTorch. The existing
`torch-gpu` environment contains PyTorch 2.5.1 with CUDA 12.4 and detects the
16GB NVIDIA GeForce RTX 3080 Laptop GPU.

```powershell
conda activate torch-gpu
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

Expected device output includes:

```text
NVIDIA GeForce RTX 3080 Laptop GPU
```

The project and its dependencies have already been installed editable into
this environment:

```powershell
python -m pip install -e ".[dev]"
```

## Dataset Export

CUDA training reads deterministic UTF-8 subsets rather than loading Arrow in
the same process. This avoids a Windows OpenMP runtime collision between the
local Hugging Face Arrow dataset and CUDA PyTorch.

```powershell
python data/export_tinystories.py `
  --dataset-dir data/TinyStories `
  --output-dir data/processed `
  --train-examples 50000 `
  --validation-examples 2000
```

## GPU Experiment

```powershell
python -m moe_lth.experiments.run_suite `
  --config configs/tinystories_gpu.yaml `
  --with-pruning

python -m moe_lth.experiments.analyze `
  --suite-dir results/tinystories_gpu_suite
```

The configuration uses:

- Original MVP architecture: 4 layers, width 256, 8 experts, hidden size 1024
- FP16 autocast and gradient scaling
- TF32 matrix operations
- Batch size 128 and sequence length 128
- 2,500 steps, approximately 41 million tokens per condition
- Strong load balancing (`aux_loss_weight=0.1`)
- Checkpoints at initialization, 1%, 5%, 10%, and later training points

## Measured Benchmark

On this system:

| Metric | Result |
|---|---:|
| Model parameters | 17,981,952 |
| Peak allocated VRAM | 1.1 GB |
| Measured throughput | approximately 25,100 tokens/s |
| Route-history dtype | `uint8` |

The model is memory-light because Top-1 dispatch evaluates only selected expert
tokens. Throughput is limited more by Python-level expert dispatch than VRAM.

