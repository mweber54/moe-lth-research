from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean, pstdev

from moe_lth.config import load_config, save_config
from moe_lth.data import build_dataloaders
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.train_ticket import train_ticket
from moe_lth.training.checkpoint import load_checkpoint
from moe_lth.training.evaluate import evaluate_language_model
from moe_lth.training.train import build_controller, build_validation_overrides
from moe_lth.utils import configure_device, resolve_data_seed, resolve_device, seed_everything

from .run_long_best_checkpoint_rewinds import (
    _best_saved_normal_record,
    _closest_checkpoint,
    _condition_spec,
    _prepare_masks,
)


CORE_CONDITIONS = ("learned_mask", "random_mask", "random_reinit")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _sample_evenly(text: str, target_chars: int, chunk_chars: int) -> str:
    if target_chars >= len(text):
        return text
    chunk_count = max(1, target_chars // chunk_chars)
    usable_chars = chunk_count * chunk_chars
    max_start = len(text) - chunk_chars
    starts = [
        round(index * max_start / max(1, chunk_count - 1))
        for index in range(chunk_count)
    ]
    return "".join(text[start : start + chunk_chars] for start in starts)[:usable_chars]


def _interleave(first: str, second: str, chunk_chars: int) -> str:
    chunks = []
    limit = max(len(first), len(second))
    for start in range(0, limit, chunk_chars):
        if start < len(first):
            chunks.append(first[start : start + chunk_chars])
        if start < len(second):
            chunks.append(second[start : start + chunk_chars])
    return "\n\n".join(chunks)


def build_large_validation(
    output_dir: Path,
    tiny_validation: Path,
    wiki_validation: Path,
    wiki_test: Path,
    chunk_chars: int,
) -> dict:
    validation_path = output_dir / "multidomain_validation_large.txt"
    metadata_path = output_dir / "metadata.json"
    if validation_path.exists() and metadata_path.exists():
        return _read_json(metadata_path)

    tiny = tiny_validation.read_text(encoding="utf-8")
    wiki = (
        wiki_validation.read_text(encoding="utf-8")
        + "\n\n"
        + wiki_test.read_text(encoding="utf-8")
    )
    target_chars = min(len(tiny), len(wiki))
    tiny_sample = _sample_evenly(tiny, target_chars, chunk_chars)
    wiki_sample = _sample_evenly(wiki, target_chars, chunk_chars)
    combined = _interleave(tiny_sample, wiki_sample, chunk_chars)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(combined, encoding="utf-8")
    seq_len = 128
    batch_size = 128
    token_count = len(combined.encode("utf-8"))
    examples = max(0, (token_count - 1) // seq_len)
    metadata = {
        "output_path": str(validation_path),
        "tiny_validation": str(tiny_validation),
        "wiki_validation": str(wiki_validation),
        "wiki_test": str(wiki_test),
        "tiny_chars": len(tiny_sample),
        "wiki_chars": len(wiki_sample),
        "output_chars": len(combined),
        "byte_tokens": token_count,
        "seq_len": seq_len,
        "examples": examples,
        "batches_at_128": (examples + batch_size - 1) // batch_size,
        "chunk_chars": chunk_chars,
    }
    _write_json(metadata_path, metadata)
    return metadata


def _with_large_validation(config: dict, validation_path: str, validation_blocks: int) -> dict:
    updated = deepcopy(config)
    updated["data"]["validation_path"] = validation_path
    updated["data"]["validation_blocks"] = validation_blocks
    return updated


def _evaluate_dense_if_needed(config: dict, checkpoint: str, output_dir: Path) -> dict:
    result_path = output_dir / "tables" / "dense_result.json"
    if result_path.exists():
        return _read_json(result_path)

    seed_everything(int(config["seed"]))
    device = resolve_device(config["device"])
    configure_device(device)
    _, validation_loader = build_dataloaders(
        config["data"],
        int(config["training"]["batch_size"]),
        resolve_data_seed(config),
    )
    model = TinyMoELanguageModel(config["model"]).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    controller = build_controller(config)
    checkpoint_step = int(Path(checkpoint).stem.split("_")[-1])
    overrides = build_validation_overrides(config, checkpoint_step, device, controller)
    metrics = evaluate_language_model(
        model,
        validation_loader,
        device,
        controller=controller if config["routing"]["mode"] in {"fixed_random", "random_every_step"} else None,
        override_batches=overrides,
        max_batches=int(config["data"]["validation_blocks"]),
        route_step_offset=checkpoint_step * 100003,
    )
    result = {
        "condition": "dense_best_saved",
        "checkpoint": checkpoint,
        "loss": metrics["loss"],
        "perplexity": metrics["perplexity"],
        "expert_local_loss": metrics["expert_local_loss"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "resolved_config.yaml")
    _write_json(result_path, result)
    return result


def _run_ticket_if_needed(
    config: dict,
    output_dir: Path,
    rewind_checkpoint: Path,
    mask_path: str,
    random_reinit: bool,
    routing_mode: str,
) -> dict:
    result_path = output_dir / "tables" / "ticket_result.json"
    if result_path.exists():
        return _read_json(result_path)

    ticket_config = deepcopy(config)
    ticket_config["routing"]["mode"] = routing_mode
    ticket_config["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(ticket_config, output_dir / "resolved_config.yaml")
    return train_ticket(ticket_config, str(rewind_checkpoint), mask_path, random_reinit)


def _summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _aggregate(records: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        grouped.setdefault(record["condition"], []).append(float(record["loss"]))
    return {condition: _summary(values) for condition, values in sorted(grouped.items())}


def _write_report(result: dict, output_dir: Path) -> Path:
    report_path = output_dir / "long_validation_extension_results.md"
    summary_path = output_dir / "long_validation_extension_summary.json"
    _write_json(summary_path, result)

    dense_mean = result["aggregate"]["dense_best_saved"]["mean"]
    rows = [
        f"| {condition} | {stats['mean']:.4f} | {stats['std']:.4f} | "
        f"{(stats['mean'] / dense_mean - 1.0) * 100.0:+.2f}% |"
        for condition, stats in result["aggregate"].items()
    ]
    seed_rows = [
        f"| {row['seed']} | {row['condition']} | {row['loss']:.4f} | {row['perplexity']:.4f} |"
        for row in result["records"]
    ]
    metadata = result["validation_metadata"]
    markdown = f"""# Long-Budget Larger-Validation Extension

This suite reruns the closest long-budget 80% initialization-rewind comparison
on a larger held-out multi-domain validation file.

Validation file: `{metadata['output_path']}`

Validation coverage: `{metadata['examples']}` sequence examples, approximately
`{metadata['batches_at_128']}` batches at batch size 128.

## Aggregate

| Condition | Mean loss | Std | Delta vs dense best-saved |
|---|---:|---:|---:|
{chr(10).join(rows)}

## Per-Seed Rows

| Seed | Condition | Loss | Perplexity |
|---:|---|---:|---:|
{chr(10).join(seed_rows)}

Raw aggregate: [long_validation_extension_summary.json](long_validation_extension_summary.json)
"""
    report_path.write_text(markdown, encoding="utf-8")
    return report_path


def run_long_validation_extension(
    config_paths: list[str],
    output_dir: str,
    validation_output_dir: str,
    validation_blocks: int,
    sparsity: float,
    conditions: list[str],
) -> dict:
    invalid = sorted(set(conditions) - set(CORE_CONDITIONS))
    if invalid:
        raise ValueError(f"Unsupported core condition(s): {', '.join(invalid)}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "long_validation_extension_status.json"
    validation_metadata = build_large_validation(
        Path(validation_output_dir),
        Path("data/processed/tinystories_validation_2k.txt"),
        Path("data/wikitext103_subset/wikitext103_validation.txt"),
        Path("data/wikitext103_subset/wikitext103_test.txt"),
        chunk_chars=16384,
    )
    selected = [_best_saved_normal_record(config_path) for config_path in config_paths]
    existing = _read_json(status_path) if status_path.exists() else {}
    records_by_key = {
        (int(row["seed"]), row["condition"]): row
        for row in existing.get("records", [])
    }

    for checkpoint_record in selected:
        seed = int(checkpoint_record["seed"])
        source_run_dir = Path(checkpoint_record["run_dir"])
        base_config = load_config(source_run_dir / "resolved_config.yaml")
        config = _with_large_validation(
            base_config,
            validation_metadata["output_path"],
            validation_blocks,
        )
        seed_dir = output / f"seed_{seed}" / "normal"
        save_config(config, seed_dir / "source_resolved_config.yaml")

        dense_dir = seed_dir / "dense_best_saved"
        print(f"[long-validation] seed={seed} condition=dense_best_saved", flush=True)
        dense = _evaluate_dense_if_needed(config, checkpoint_record["checkpoint"], dense_dir)
        records_by_key[(seed, "dense_best_saved")] = {
            "seed": seed,
            "condition": "dense_best_saved",
            "output_dir": str(dense_dir),
            "checkpoint": checkpoint_record["checkpoint"],
            "loss": float(dense["loss"]),
            "perplexity": float(dense["perplexity"]),
        }

        masks = _prepare_masks(config, checkpoint_record["checkpoint"], seed_dir, sparsity)
        rewind_checkpoint = _closest_checkpoint(source_run_dir / "checkpoints", 0)
        for condition in conditions:
            mask_path, random_reinit, routing_mode = _condition_spec(
                condition,
                masks,
                base_config["routing"]["mode"],
            )
            run_dir = seed_dir / "rewind" / f"sparsity_{sparsity:g}" / f"{condition}_fraction_0"
            print(f"[long-validation] seed={seed} condition={condition}", flush=True)
            result = _run_ticket_if_needed(
                config,
                run_dir,
                rewind_checkpoint,
                mask_path,
                random_reinit,
                routing_mode,
            )
            records_by_key[(seed, condition)] = {
                "seed": seed,
                "condition": condition,
                "output_dir": str(run_dir),
                "rewind_checkpoint": str(rewind_checkpoint),
                "mask_path": mask_path,
                "random_reinitialize_experts": random_reinit,
                "routing_mode": routing_mode,
                "loss": float(result["loss"]),
                "perplexity": float(result["perplexity"]),
            }

            records = sorted(records_by_key.values(), key=lambda row: (row["seed"], row["condition"]))
            partial = {
                "validation_metadata": validation_metadata,
                "validation_blocks": validation_blocks,
                "sparsity": sparsity,
                "conditions": ["dense_best_saved", *conditions],
                "selected_checkpoints": selected,
                "records": records,
                "aggregate": _aggregate(records),
                "partial": True,
            }
            _write_json(status_path, partial)

    records = sorted(records_by_key.values(), key=lambda row: (row["seed"], row["condition"]))
    result = {
        "validation_metadata": validation_metadata,
        "validation_blocks": validation_blocks,
        "sparsity": sparsity,
        "conditions": ["dense_best_saved", *conditions],
        "selected_checkpoints": selected,
        "records": records,
        "aggregate": _aggregate(records),
    }
    report = _write_report(result, output)
    result["report"] = str(report)
    _write_json(status_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun close long-budget comparisons on a larger validation file."
    )
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="results/long_validation_extension")
    parser.add_argument("--validation-output-dir", default="data/multidomain_validation_extension")
    parser.add_argument("--validation-blocks", type=int, default=100000)
    parser.add_argument("--sparsity", type=float, default=0.8)
    parser.add_argument("--conditions", nargs="+", default=list(CORE_CONDITIONS))
    args = parser.parse_args()
    result = run_long_validation_extension(
        args.configs,
        args.output_dir,
        args.validation_output_dir,
        args.validation_blocks,
        args.sparsity,
        args.conditions,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
