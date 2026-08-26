from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from moe_lth.analysis.expert_specificity import (
    context_embedding_cosine_similarity,
    expert_token_histograms,
    pairwise_token_distribution_js,
)
from moe_lth.analysis.router_geometry import hidden_state_separability, router_vector_similarities
from moe_lth.config import load_config
from moe_lth.data import build_dataloaders
from moe_lth.models import TinyMoELanguageModel
from moe_lth.training.checkpoint import load_checkpoint
from moe_lth.training.evaluate import evaluate_expert_substitution_matrix, evaluate_language_model
from moe_lth.utils import resolve_data_seed, resolve_device, seed_everything
from moe_lth.visualization.plot_results import plot_expert_specificity_heatmap


@torch.no_grad()
def analyze_checkpoint(config: dict, checkpoint: str) -> dict:
    seed_everything(int(config["seed"]))
    device = resolve_device(config["device"])
    _, validation_loader = build_dataloaders(
        config["data"], int(config["training"]["batch_size"]), resolve_data_seed(config)
    )
    model = TinyMoELanguageModel(config["model"]).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    max_batches = int(config["data"]["validation_blocks"])
    hidden_by_layer: dict[int, list[torch.Tensor]] = defaultdict(list)
    routes_by_layer: dict[int, list[torch.Tensor]] = defaultdict(list)
    tokens_by_layer: dict[int, list[np.ndarray]] = defaultdict(list)

    for batch_id, (token_ids, _) in enumerate(validation_loader):
        if batch_id >= max_batches:
            break
        token_ids = token_ids.to(device)
        output = model(token_ids)
        for layer_id, (hidden, trace) in enumerate(zip(output.pre_router_hidden_states, output.routes, strict=True)):
            hidden_by_layer[layer_id].append(hidden.cpu())
            routes_by_layer[layer_id].append(trace.selected_experts.cpu())
            tokens_by_layer[layer_id].append(token_ids.cpu().numpy())

    layer_analysis = {}
    num_experts = int(config["model"]["num_experts"])
    vocab_size = int(config["model"]["vocab_size"])
    for layer_id in hidden_by_layer:
        hidden = torch.cat(hidden_by_layer[layer_id], dim=0)
        routes = torch.cat(routes_by_layer[layer_id], dim=0)
        histograms = expert_token_histograms(
            tokens_by_layer[layer_id],
            [values.numpy() for values in routes_by_layer[layer_id]],
            num_experts,
            vocab_size,
        )
        layer_analysis[f"layer_{layer_id}"] = {
            "hidden_state_separability": hidden_state_separability(hidden, routes),
            "context_embedding_cosine_similarity": context_embedding_cosine_similarity(
                hidden, routes, num_experts
            ).tolist(),
            "token_distribution_jensen_shannon": pairwise_token_distribution_js(histograms),
        }

    substitution = evaluate_expert_substitution_matrix(model, validation_loader, device, max_batches=max_batches)
    report = {
        "checkpoint": checkpoint,
        "validation": evaluate_language_model(model, validation_loader, device, max_batches=max_batches),
        "router_vector_similarity": router_vector_similarities(model),
        "layers": layer_analysis,
        "expert_substitution_loss": substitution,
    }
    report["validation"].pop("routing_batches")
    destination = Path(config["output_dir"]) / "tables" / "checkpoint_analysis.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for layer, matrix in substitution.items():
        plot_expert_specificity_heatmap(
            matrix,
            str(Path(config["output_dir"]) / "figures" / f"{layer}_expert_specificity.png"),
            f"{layer} expert substitution loss",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze_checkpoint(load_config(args.config), args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
