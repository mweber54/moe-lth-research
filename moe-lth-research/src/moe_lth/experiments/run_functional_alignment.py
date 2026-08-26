from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from moe_lth.config import load_config
from moe_lth.data import build_dataloaders
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.masks import mask_jaccard
from moe_lth.training.checkpoint import load_checkpoint
from moe_lth.utils import configure_device, resolve_device


SignatureDict = dict[tuple[int, int], dict[str, torch.Tensor]]


def _final_checkpoint(run_dir: Path) -> Path:
    config = load_config(run_dir / "resolved_config.yaml")
    return run_dir / "checkpoints" / f"step_{config['training']['steps']}.pt"


def _load_model(checkpoint: Path, device: torch.device) -> TinyMoELanguageModel:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = TinyMoELanguageModel(payload["config"]["model"]).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    return model


@torch.no_grad()
def collect_functional_signatures(
    checkpoint: Path,
    config: dict,
    sample_tokens: int,
    device: torch.device,
) -> SignatureDict:
    _, validation_loader = build_dataloaders(
        config["data"], int(config["training"]["batch_size"]), int(config["seed"])
    )
    model = _load_model(checkpoint, device)
    collected: dict[tuple[int, int], dict[str, list[torch.Tensor]]] = {}
    total = 0
    for token_ids, _ in validation_loader:
        if total >= sample_tokens:
            break
        sequences = min(token_ids.shape[0], max(1, (sample_tokens - total + token_ids.shape[1] - 1) // token_ids.shape[1]))
        token_ids = token_ids[:sequences].to(device)
        output = model(token_ids)
        take = min(sample_tokens - total, token_ids.numel())
        for layer_id, hidden_states in enumerate(output.pre_router_hidden_states):
            inputs = hidden_states.reshape(-1, hidden_states.shape[-1])[:take]
            for expert_id, expert in enumerate(model.blocks[layer_id].moe.experts):
                hidden = F.gelu(expert.fc1(inputs))
                expert_output = expert.fc2(hidden)
                record = collected.setdefault((layer_id, expert_id), {"hidden": [], "output": []})
                record["hidden"].append(hidden.float().cpu())
                record["output"].append(expert_output.float().cpu())
        total += take
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if total < sample_tokens:
        raise ValueError(f"Requested {sample_tokens} tokens but validation data supplied only {total}.")
    return {
        key: {name: torch.cat(chunks, dim=0) for name, chunks in values.items()}
        for key, values in collected.items()
    }


def linear_cka(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.float() - first.float().mean(dim=0, keepdim=True)
    second = second.float() - second.float().mean(dim=0, keepdim=True)
    cross = first.T @ second
    first_cov = first.T @ first
    second_cov = second.T @ second
    denominator = torch.linalg.norm(first_cov) * torch.linalg.norm(second_cov)
    return float((torch.linalg.norm(cross).square() / denominator.clamp_min(1e-12)).item())


def match_experts(
    source: SignatureDict,
    target: SignatureDict,
    layer_id: int,
    num_experts: int,
) -> tuple[dict[int, int], torch.Tensor]:
    similarities = torch.empty(num_experts, num_experts)
    for source_expert in range(num_experts):
        for target_expert in range(num_experts):
            similarities[source_expert, target_expert] = linear_cka(
                source[(layer_id, source_expert)]["output"],
                target[(layer_id, target_expert)]["output"],
            )
    rows, columns = linear_sum_assignment(-similarities.numpy())
    return {int(row): int(column) for row, column in zip(rows, columns, strict=True)}, similarities


def match_neurons(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, float]:
    source = source.float()
    target = target.float()
    source = (source - source.mean(dim=0)) / source.std(dim=0).clamp_min(1e-6)
    target = (target - target.mean(dim=0)) / target.std(dim=0).clamp_min(1e-6)
    correlations = (source.T @ target / source.shape[0]).abs()
    rows, columns = linear_sum_assignment(-correlations.cpu().numpy())
    permutation = torch.empty(source.shape[1], dtype=torch.long)
    permutation[torch.as_tensor(rows)] = torch.as_tensor(columns)
    matched = correlations[
        torch.as_tensor(rows, device=correlations.device),
        torch.as_tensor(columns, device=correlations.device),
    ]
    return permutation, float(matched.mean().item())


def build_functional_alignment(
    source_signatures: SignatureDict,
    target_signatures: SignatureDict,
    num_layers: int,
    num_experts: int,
    compute_device: torch.device = torch.device("cpu"),
) -> dict:
    source_signatures = {
        key: {name: tensor.to(compute_device) for name, tensor in values.items()}
        for key, values in source_signatures.items()
    }
    target_signatures = {
        key: {name: tensor.to(compute_device) for name, tensor in values.items()}
        for key, values in target_signatures.items()
    }
    identity_expert_cka = []
    matched_expert_cka = []
    matched_neuron_correlation = []
    expert_mappings = {}
    neuron_permutations = {}

    for layer_id in range(num_layers):
        mapping, similarities = match_experts(
            source_signatures, target_signatures, layer_id, num_experts
        )
        expert_mappings[f"layer_{layer_id}"] = {
            str(source_expert): target_expert for source_expert, target_expert in mapping.items()
        }
        identity_expert_cka.extend(similarities.diag().tolist())
        matched_expert_cka.extend(
            similarities[source_expert, target_expert].item()
            for source_expert, target_expert in mapping.items()
        )
        for source_expert, target_expert in mapping.items():
            permutation, correlation = match_neurons(
                source_signatures[(layer_id, source_expert)]["hidden"],
                target_signatures[(layer_id, target_expert)]["hidden"],
            )
            matched_neuron_correlation.append(correlation)
            neuron_permutations[(layer_id, source_expert)] = permutation

    return {
        "identity_expert_output_cka": mean(identity_expert_cka),
        "matched_expert_output_cka": mean(matched_expert_cka),
        "matched_neuron_abs_correlation": mean(matched_neuron_correlation),
        "expert_mappings": expert_mappings,
        "neuron_permutations": neuron_permutations,
    }


def aligned_mask_jaccard(
    source_masks: dict[str, torch.Tensor],
    target_masks: dict[str, torch.Tensor],
    alignment: dict,
) -> float:
    intersection = 0
    union = 0
    for layer_name, mapping in alignment["expert_mappings"].items():
        layer_id = int(layer_name.split("_")[-1])
        for source_token, target_expert in mapping.items():
            source_expert = int(source_token)
            permutation = alignment["neuron_permutations"][(layer_id, source_expert)]
            prefix = f"blocks.{layer_id}.moe.experts"
            source_fc1 = source_masks[f"{prefix}.{source_expert}.fc1.weight"].bool()
            source_fc2 = source_masks[f"{prefix}.{source_expert}.fc2.weight"].bool()
            target_fc1 = target_masks[f"{prefix}.{target_expert}.fc1.weight"].bool()[permutation]
            target_fc2 = target_masks[f"{prefix}.{target_expert}.fc2.weight"].bool()[:, permutation]
            for first, second in ((source_fc1, target_fc1), (source_fc2, target_fc2)):
                intersection += torch.logical_and(first, second).sum().item()
                union += torch.logical_or(first, second).sum().item()
    return float(intersection / union)


def run_functional_alignment(
    source_run: str,
    cross_init_root: str,
    output_dir: str,
    sparsities: list[float],
    sample_tokens: int,
    requested_device: str,
) -> dict:
    device = resolve_device(requested_device)
    configure_device(device)
    source_dir = Path(source_run)
    source_config = load_config(source_dir / "resolved_config.yaml")
    source_checkpoint = _final_checkpoint(source_dir)
    source_signatures = collect_functional_signatures(
        source_checkpoint, source_config, sample_tokens, device
    )
    source_model = _load_model(source_checkpoint, torch.device("cpu"))
    source_masks = {
        sparsity: expert_local_magnitude_masks(source_model, sparsity) for sparsity in sparsities
    }
    del source_model

    results = []
    for target_root in sorted(Path(cross_init_root).glob("source_*_target_*")):
        target_seed = int(target_root.name.rsplit("_", maxsplit=1)[-1])
        for condition in ("matched_data_learned", "cross_init_replay"):
            run_dir = target_root / condition
            target_config = load_config(run_dir / "resolved_config.yaml")
            target_checkpoint = _final_checkpoint(run_dir)
            target_signatures = collect_functional_signatures(
                target_checkpoint, target_config, sample_tokens, device
            )
            target_model = _load_model(target_checkpoint, torch.device("cpu"))
            row = {"target_seed": target_seed, "condition": condition, "sparsities": {}}
            alignment = build_functional_alignment(
                source_signatures,
                target_signatures,
                int(source_config["model"]["num_layers"]),
                int(source_config["model"]["num_experts"]),
                device,
            )
            for sparsity in sparsities:
                target_masks = expert_local_magnitude_masks(target_model, sparsity)
                row["sparsities"][str(sparsity)] = {
                    "raw_mask_jaccard": mask_jaccard(source_masks[sparsity], target_masks),
                    "aligned_mask_jaccard": aligned_mask_jaccard(
                        source_masks[sparsity], target_masks, alignment
                    ),
                    "identity_expert_output_cka": alignment["identity_expert_output_cka"],
                    "matched_expert_output_cka": alignment["matched_expert_output_cka"],
                    "matched_neuron_abs_correlation": alignment[
                        "matched_neuron_abs_correlation"
                    ],
                    "expert_mappings": alignment["expert_mappings"],
                }
            del target_model
            results.append(row)

    report = {
        "source_run": source_run,
        "cross_init_root": cross_init_root,
        "sample_tokens": sample_tokens,
        "sparsities": sparsities,
        "results": results,
    }
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "functional_alignment_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_report(report, root)
    return report


def _write_report(report: dict, root: Path) -> None:
    rows = []
    for row in report["results"]:
        for sparsity, values in row["sparsities"].items():
            rows.append(
                f"| {row['target_seed']} | {row['condition']} | {float(sparsity):.0%} | "
                f"{values['raw_mask_jaccard']:.4f} | {values['aligned_mask_jaccard']:.4f} | "
                f"{values['identity_expert_output_cka']:.4f} | "
                f"{values['matched_expert_output_cka']:.4f} | "
                f"{values['matched_neuron_abs_correlation']:.4f} |"
            )

    summary_rows = []
    for condition in ("matched_data_learned", "cross_init_replay"):
        for sparsity in report["sparsities"]:
            selected = [
                row["sparsities"][str(sparsity)]
                for row in report["results"]
                if row["condition"] == condition
            ]
            summary_rows.append(
                f"| {condition} | {sparsity:.0%} | "
                f"{mean(value['raw_mask_jaccard'] for value in selected):.4f} | "
                f"{mean(value['aligned_mask_jaccard'] for value in selected):.4f} | "
                f"{mean(value['matched_expert_output_cka'] for value in selected):.4f} | "
                f"{mean(value['matched_neuron_abs_correlation'] for value in selected):.4f} |"
            )

    markdown = f"""# Functional Cross-Initialization Alignment

Experts are matched within each layer by linear CKA of their outputs on the
same {report["sample_tokens"]} validation-token positions. Internal expert
neurons are then matched by activation correlation before recomputing mask
Jaccard.

## Mean Results

| Condition | Sparsity | Raw Jaccard | Functionally aligned Jaccard | Matched expert CKA | Matched neuron correlation |
|---|---:|---:|---:|---:|---:|
{chr(10).join(summary_rows)}

## Per-Target Results

| Target seed | Condition | Sparsity | Raw Jaccard | Aligned Jaccard | Identity expert CKA | Matched expert CKA | Matched neuron correlation |
|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Raw results: [`functional_alignment_summary.json`](functional_alignment_summary.json)

![Functional alignment mask similarity](functional_alignment_mask_similarity.png)
"""
    (root / "functional_alignment_results.md").write_text(markdown, encoding="utf-8")

    figure, axes = plt.subplots(
        1, len(report["sparsities"]), figsize=(6 * len(report["sparsities"]), 4), squeeze=False
    )
    conditions = ("matched_data_learned", "cross_init_replay")
    for axis, sparsity in zip(axes[0], report["sparsities"], strict=True):
        raw = []
        aligned = []
        for condition in conditions:
            selected = [
                row["sparsities"][str(sparsity)]
                for row in report["results"]
                if row["condition"] == condition
            ]
            raw.append(mean(value["raw_mask_jaccard"] for value in selected))
            aligned.append(mean(value["aligned_mask_jaccard"] for value in selected))
        positions = range(len(conditions))
        axis.bar([position - 0.18 for position in positions], raw, 0.36, label="raw")
        axis.bar([position + 0.18 for position in positions], aligned, 0.36, label="aligned")
        axis.set_xticks(list(positions), ["learned", "replay"])
        axis.set_ylabel("Mask Jaccard with source")
        axis.set_title(f"{sparsity:.0%} masks")
        axis.legend()
    figure.tight_layout()
    figure.savefig(root / "functional_alignment_mask_similarity.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Functionally align experts and neurons across initializations."
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--cross-init-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sparsities", nargs="+", type=float, default=[0.5, 0.8])
    parser.add_argument("--sample-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = run_functional_alignment(
        args.source_run,
        args.cross_init_root,
        args.output_dir,
        args.sparsities,
        args.sample_tokens,
        args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
