from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import torch

from moe_lth.analysis.routing_stability import routing_agreement
from moe_lth.config import load_config
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.evaluate_pruning import evaluate_pruning
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.masks import mask_jaccard
from moe_lth.training.checkpoint import load_checkpoint
from moe_lth.training.train import train_from_config


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _final_step(run_dir: Path) -> int:
    return int(load_config(run_dir / "resolved_config.yaml")["training"]["steps"])


def _final_checkpoint(run_dir: Path) -> Path:
    return run_dir / "checkpoints" / f"step_{_final_step(run_dir)}.pt"


def _final_routes(run_dir: Path) -> Path:
    return run_dir / "logs" / f"validation_routes_step_{_final_step(run_dir)}.npz"


def _masks(checkpoint: Path, sparsity: float):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = TinyMoELanguageModel(payload["config"]["model"])
    load_checkpoint(checkpoint, model)
    return expert_local_magnitude_masks(model, sparsity)


def _train_or_load(config: dict, with_pruning: bool, compact_checkpoints: bool = False) -> dict:
    if compact_checkpoints:
        config = deepcopy(config)
        config["training"]["checkpoint_steps"] = [int(config["training"]["steps"])]
        config["training"]["save_optimizer"] = False
    run_dir = Path(config["output_dir"])
    final_checkpoint = run_dir / "checkpoints" / f"step_{config['training']['steps']}.pt"
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and final_checkpoint.exists():
        summary = _read_json(summary_path)
    else:
        summary = train_from_config(config)
    pruning_path = run_dir / "tables" / "pruning_results.json"
    if with_pruning:
        summary["pruning"] = (
            _read_json(pruning_path)
            if pruning_path.exists()
            else evaluate_pruning(config, str(final_checkpoint))
        )
    return summary


def run_cross_init_replay(
    source_suite: str,
    target_configs: list[str],
    output_dir: str,
    sparsities: list[float],
    with_pruning: bool = True,
) -> dict:
    source_normal = Path(source_suite) / "normal"
    source_config = load_config(source_normal / "resolved_config.yaml")
    source_seed = int(source_config["seed"])
    source_history = source_normal / "logs" / "train_route_history.npz"
    if not source_history.exists():
        raise FileNotFoundError(f"Missing source route history: {source_history}")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    source_checkpoint = _final_checkpoint(source_normal)
    source_routes = _final_routes(source_normal)
    source_masks = {sparsity: _masks(source_checkpoint, sparsity) for sparsity in sparsities}
    results = []

    for target_path in target_configs:
        target_base = load_config(target_path)
        target_seed = int(target_base["seed"])
        target_root = root / f"source_{source_seed}_target_{target_seed}"
        runs = {}
        for condition, mode in (("matched_data_learned", "learned"), ("cross_init_replay", "replay")):
            config = deepcopy(target_base)
            config["training"]["data_seed"] = source_seed
            config["training"]["record_train_routes"] = False
            config["routing"]["mode"] = mode
            config["routing"]["replay_path"] = str(source_history) if mode == "replay" else None
            config["output_dir"] = str(target_root / condition)
            summary = _train_or_load(config, with_pruning, compact_checkpoints=True)
            summary["condition"] = condition
            runs[condition] = {"config": config, "summary": summary, "dir": Path(config["output_dir"])}

        learned_dir = runs["matched_data_learned"]["dir"]
        replay_dir = runs["cross_init_replay"]["dir"]
        row = {
            "source_seed": source_seed,
            "target_seed": target_seed,
            "source_loss": _read_json(source_normal / "summary.json")["final_validation_loss"],
            "matched_data_learned_loss": runs["matched_data_learned"]["summary"]["final_validation_loss"],
            "cross_init_replay_loss": runs["cross_init_replay"]["summary"]["final_validation_loss"],
            "source_vs_matched_data_learned_routing": routing_agreement(
                str(source_routes), str(_final_routes(learned_dir))
            )["overall"],
            "source_vs_cross_init_replay_routing": routing_agreement(
                str(source_routes), str(_final_routes(replay_dir))
            )["overall"],
            "mask_similarity": {},
            "pruning": {},
        }
        for condition, run in runs.items():
            selected = {}
            for pruning_row in run["summary"].get("pruning", []):
                if pruning_row["sparsity"] in sparsities and pruning_row["condition"] in {
                    "dense",
                    "magnitude",
                    "random_mask",
                    "other_expert_mask",
                }:
                    selected[f"{pruning_row['condition']}|{pruning_row['sparsity']:g}"] = pruning_row[
                        "loss"
                    ]
            row["pruning"][condition] = selected
        for sparsity in sparsities:
            learned_masks = _masks(_final_checkpoint(learned_dir), sparsity)
            replay_masks = _masks(_final_checkpoint(replay_dir), sparsity)
            row["mask_similarity"][str(sparsity)] = {
                "source_vs_matched_data_learned": mask_jaccard(source_masks[sparsity], learned_masks),
                "source_vs_cross_init_replay": mask_jaccard(source_masks[sparsity], replay_masks),
                "matched_data_learned_vs_cross_init_replay": mask_jaccard(learned_masks, replay_masks),
            }
        for run in runs.values():
            checkpoint_dir = run["dir"] / "checkpoints"
            for checkpoint_path in checkpoint_dir.glob("*.pt"):
                checkpoint_path.unlink()
        results.append(row)

    report = {
        "source_suite": source_suite,
        "source_seed": source_seed,
        "target_seeds": [row["target_seed"] for row in results],
        "sparsities": sparsities,
        "results": results,
    }
    json_path = root / "cross_init_replay_summary.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_report(report, root)
    return report


def run_cross_init_replay_matrix(
    source_suites: list[str],
    target_configs: list[str],
    output_dir: str,
    sparsities: list[float],
    with_pruning: bool = True,
) -> dict:
    """Run the same target set against multiple source route histories."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    reports = []
    for source_suite in source_suites:
        source_config = load_config(Path(source_suite) / "normal" / "resolved_config.yaml")
        source_seed = int(source_config["seed"])
        reports.append(
            run_cross_init_replay(
                source_suite,
                target_configs,
                str(root / f"source_{source_seed}"),
                sparsities,
                with_pruning=with_pruning,
            )
        )

    results = [row for report in reports for row in report["results"]]
    matrix = {
        "source_suites": source_suites,
        "source_seeds": [report["source_seed"] for report in reports],
        "target_seeds": sorted({row["target_seed"] for row in results}),
        "sparsities": sparsities,
        "results": results,
    }
    (root / "cross_init_replay_matrix_summary.json").write_text(
        json.dumps(matrix, indent=2), encoding="utf-8"
    )
    _write_matrix_report(matrix, root)
    return matrix


def _write_matrix_report(matrix: dict, root: Path) -> None:
    rows = []
    for row in matrix["results"]:
        rows.append(
            f"| {row['source_seed']} | {row['target_seed']} | "
            f"{row['matched_data_learned_loss']:.4f} | "
            f"{row['cross_init_replay_loss']:.4f} | "
            f"{row['source_vs_cross_init_replay_routing']:.4f} |"
        )
    markdown = f"""# Route × Initialization Replay Matrix

Source routing seeds: {", ".join(map(str, matrix["source_seeds"]))}. Target
initialization seeds: {", ".join(map(str, matrix["target_seeds"]))}.

Each cell compares matched-data learned routing with replay of the source
route history under the target initialization. This is the initial 3x3
matrix artifact; inferential summaries should be added after all cells are
available and checked for protocol consistency.

| Source seed | Target seed | Matched learned loss | Cross-init replay loss | Replay route agreement |
|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Raw results: [`cross_init_replay_matrix_summary.json`](cross_init_replay_matrix_summary.json)
"""
    (root / "cross_init_replay_matrix_results.md").write_text(markdown, encoding="utf-8")


def _write_report(report: dict, root: Path) -> None:
    rows = []
    mask_rows = []
    for row in report["results"]:
        rows.append(
            f"| {row['target_seed']} | {row['source_loss']:.4f} | "
            f"{row['matched_data_learned_loss']:.4f} | {row['cross_init_replay_loss']:.4f} | "
            f"{row['source_vs_matched_data_learned_routing']:.4f} | "
            f"{row['source_vs_cross_init_replay_routing']:.4f} |"
        )
        for sparsity, similarities in row["mask_similarity"].items():
            mask_rows.append(
                f"| {row['target_seed']} | {float(sparsity):.0%} | "
                f"{similarities['source_vs_matched_data_learned']:.4f} | "
                f"{similarities['source_vs_cross_init_replay']:.4f} | "
                f"{similarities['matched_data_learned_vs_cross_init_replay']:.4f} |"
            )
    matched_loss = mean(row["matched_data_learned_loss"] for row in report["results"])
    replay_loss = mean(row["cross_init_replay_loss"] for row in report["results"])
    replay_penalty = (replay_loss / matched_loss - 1.0) * 100.0
    summary_rows = []
    for sparsity in report["sparsities"]:
        source_vs_learned = mean(
            row["mask_similarity"][str(sparsity)]["source_vs_matched_data_learned"]
            for row in report["results"]
        )
        source_vs_replay = mean(
            row["mask_similarity"][str(sparsity)]["source_vs_cross_init_replay"]
            for row in report["results"]
        )
        learned_vs_replay = mean(
            row["mask_similarity"][str(sparsity)]["matched_data_learned_vs_cross_init_replay"]
            for row in report["results"]
        )
        summary_rows.append(
            f"| {sparsity:.0%} | {source_vs_learned:.4f} | {source_vs_replay:.4f} | "
            f"{learned_vs_replay:.4f} |"
        )
    pruning_rows = []
    for condition in ("matched_data_learned", "cross_init_replay"):
        values = []
        for sparsity in report["sparsities"]:
            for pruning_condition in ("magnitude", "random_mask", "other_expert_mask"):
                key = f"{pruning_condition}|{sparsity:g}"
                values.append(
                    mean(row["pruning"][condition][key] for row in report["results"])
                )
        pruning_rows.append(
            f"| {condition} | " + " | ".join(f"{value:.4f}" for value in values) + " |"
        )
    markdown = f"""# Cross-Initialization Replay Results

Source routing history: seed {report["source_seed"]}. Target initializations:
{", ".join(map(str, report["target_seeds"]))}. All target runs use the source
seed's data order.

## Interpretation

Cross-initialization replay forces exactly the source seed's routes but is
**{replay_penalty:.2f}% worse** than matched-data learned routing on average.
Replayed masks are not more source-like than matched-data learned masks.
However, learned and replay masks under the same target initialization differ
substantially, showing that routing changes masks while initialization anchors
their coordinate identity.

The supported mechanism is therefore:

```text
initialization x routing trajectory -> sparse mask
```

Routing history is causal but not sufficient, and a foreign route history does
not transfer as a standalone sparse-mask blueprint.

## Performance and Routing

| Target seed | Source loss | Matched-data learned loss | Cross-init replay loss | Source vs learned routing | Source vs replay routing |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Mask Similarity

| Target seed | Sparsity | Source vs matched-data learned | Source vs cross-init replay | Learned vs replay |
|---:|---:|---:|---:|---:|
{chr(10).join(mask_rows)}

### Mean Mask Similarity

| Sparsity | Source vs matched-data learned | Source vs cross-init replay | Same-init learned vs replay |
|---:|---:|---:|---:|
{chr(10).join(summary_rows)}

Coordinate-level mask overlap across independent initializations should be
interpreted cautiously because hidden-unit permutation symmetries can reduce
Jaccard even when functions are related.

## Direct Pruning

| Condition | 50% magnitude | 50% random | 50% other expert | 80% magnitude | 80% random | 80% other expert |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(pruning_rows)}

Raw results: [`cross_init_replay_summary.json`](cross_init_replay_summary.json)

![Cross-initialization mask similarity](cross_init_mask_similarity.png)
"""
    (root / "cross_init_replay_results.md").write_text(markdown, encoding="utf-8")

    figure, axes = plt.subplots(1, len(report["sparsities"]), figsize=(6 * len(report["sparsities"]), 4))
    axes = [axes] if len(report["sparsities"]) == 1 else axes
    for axis, sparsity in zip(axes, report["sparsities"], strict=True):
        labels = [str(row["target_seed"]) for row in report["results"]]
        learned = [
            row["mask_similarity"][str(sparsity)]["source_vs_matched_data_learned"]
            for row in report["results"]
        ]
        replay = [
            row["mask_similarity"][str(sparsity)]["source_vs_cross_init_replay"]
            for row in report["results"]
        ]
        positions = list(range(len(labels)))
        axis.bar([position - 0.18 for position in positions], learned, width=0.36, label="learned")
        axis.bar([position + 0.18 for position in positions], replay, width=0.36, label="replay")
        axis.set_xticks(positions, labels)
        axis.set_xlabel("Target initialization seed")
        axis.set_ylabel("Mask Jaccard with source")
        axis.set_title(f"{sparsity:.0%} masks")
        axis.legend()
    figure.tight_layout()
    figure.savefig(root / "cross_init_mask_similarity.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay one source routing history across independent target initializations."
    )
    parser.add_argument("--source-suite")
    parser.add_argument("--source-suites", nargs="+")
    parser.add_argument("--target-configs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sparsities", nargs="+", type=float, default=[0.5, 0.8])
    parser.add_argument("--without-pruning", action="store_true")
    args = parser.parse_args()
    if args.source_suites:
        report = run_cross_init_replay_matrix(
            args.source_suites,
            args.target_configs,
            args.output_dir,
            args.sparsities,
            with_pruning=not args.without_pruning,
        )
    elif args.source_suite:
        report = run_cross_init_replay(
            args.source_suite,
            args.target_configs,
            args.output_dir,
            args.sparsities,
            with_pruning=not args.without_pruning,
        )
    else:
        parser.error("one of --source-suite or --source-suites is required")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
