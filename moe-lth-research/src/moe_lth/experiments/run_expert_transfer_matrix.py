from __future__ import annotations

import argparse
import json
from pathlib import Path

from moe_lth.config import load_config
from moe_lth.data import build_dataloaders
from moe_lth.models import TinyMoELanguageModel
from moe_lth.training.checkpoint import load_checkpoint
from moe_lth.training.evaluate import evaluate_expert_substitution_matrix
from moe_lth.utils import resolve_data_seed, resolve_device, seed_everything


def summarize_transfer_matrix(matrix: dict) -> dict:
    """Summarize expert-transfer results by layer as diagonal vs. off-diagonal means."""
    if "results" not in matrix:
        raise KeyError("Expected an expert-transfer matrix with a 'results' list.")

    by_layer: dict[str, dict] = {}
    whole: dict[str, float] = {}
    for row in matrix["results"]:
        layer = str(row["layer"])
        source_expert = int(row["source_expert"])
        target_expert = int(row["target_expert"])
        value = float(row["loss"])
        entry = by_layer.setdefault(layer, {"values": []})
        entry["values"].append((source_expert, target_expert, value))

    for layer, entry in by_layer.items():
        values = entry["values"]
        diagonal = [value for source, target, value in values if source == target]
        off_diagonal = [value for source, target, value in values if source != target]
        diagonal_mean = sum(diagonal) / len(diagonal) if diagonal else float("nan")
        off_diagonal_mean = sum(off_diagonal) / len(off_diagonal) if off_diagonal else float("nan")
        penalty = (
            off_diagonal_mean - diagonal_mean
            if diagonal_mean == diagonal_mean and off_diagonal_mean == off_diagonal_mean
            else 0.0
        )
        by_layer[layer] = {
            "diagonal_mean": diagonal_mean,
            "off_diagonal_mean": off_diagonal_mean,
            "mean_transfer_penalty": penalty,
            "diagonal_count": len(diagonal),
            "off_diagonal_count": len(off_diagonal),
        }
        whole[layer] = penalty

    return {
        "layers": list(by_layer),
        "by_layer": by_layer,
        "mean_transfer_penalty_overall": sum(whole.values()) / len(whole) if whole else 0.0,
    }


def generate_expert_transfer_matrix(
    config: dict,
    checkpoint: str,
    output_dir: str | Path,
    max_batches: int | None = None,
) -> dict:
    """Compute the full source-expert × target-expert substitution matrix and write raw + summary outputs."""
    seed_everything(int(config["seed"]))
    device = resolve_device(config["device"])
    _, validation_loader = build_dataloaders(
        config["data"], int(config["training"]["batch_size"]), resolve_data_seed(config)
    )
    model = TinyMoELanguageModel(config["model"]).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    matrix = evaluate_expert_substitution_matrix(
        model,
        validation_loader,
        device,
        max_batches=max_batches or int(config["data"].get("validation_blocks", 1)),
    )

    rows = []
    for layer_name, grid in matrix.items():
        for source_expert, row in enumerate(grid):
            for target_expert, loss in enumerate(row):
                rows.append(
                    {
                        "layer": layer_name,
                        "source_expert": source_expert,
                        "target_expert": target_expert,
                        "loss": float(loss),
                    }
                )

    report = {
        "checkpoint": checkpoint,
        "num_layers": len(matrix),
        "num_experts": len(next(iter(matrix.values()), [])),
        "results": rows,
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "expert_transfer_matrix.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = summarize_transfer_matrix(report)
    summary["checkpoint"] = checkpoint
    (destination / "expert_transfer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    for layer, stats in summary["by_layer"].items():
        lines.append(
            f"| {layer} | {stats['diagonal_mean']:.4f} | {stats['off_diagonal_mean']:.4f} | {stats['mean_transfer_penalty']:.4f} |"
        )
    markdown = f'''# Expert Transfer Matrix Summary

| Layer | Diagonal mean | Off-diagonal mean | Mean transfer penalty |
|---|---:|---:|---:|
{chr(10).join(lines)}

Overall mean penalty: **{summary['mean_transfer_penalty_overall']:.4f}**
'''
    (destination / "expert_transfer_summary.md").write_text(markdown, encoding="utf-8")
    return report


def run_expert_transfer_matrix(summary_path: str | Path, output_dir: str | Path) -> dict:
    source = Path(summary_path)
    matrix = json.loads(source.read_text(encoding="utf-8"))
    report = summarize_transfer_matrix(matrix)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "expert_transfer_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = []
    for layer, stats in report["by_layer"].items():
        lines.append(
            f"| {layer} | {stats['diagonal_mean']:.4f} | {stats['off_diagonal_mean']:.4f} | {stats['mean_transfer_penalty']:.4f} |"
        )
    markdown = f'''# Expert Transfer Matrix Summary

| Layer | Diagonal mean | Off-diagonal mean | Mean transfer penalty |
|---|---:|---:|---:|
{chr(10).join(lines)}

Overall mean penalty: **{report['mean_transfer_penalty_overall']:.4f}**
'''
    (destination / "expert_transfer_summary.md").write_text(markdown, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute a full expert transfer matrix and summarize the diagonal vs off-diagonal performance gap."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    report = generate_expert_transfer_matrix(config, args.checkpoint, args.output_dir, max_batches=args.max_batches)
    print(json.dumps({"checkpoint": args.checkpoint, "layers": report["num_layers"], "num_experts": report["num_experts"]}, indent=2))


if __name__ == "__main__":
    main()
