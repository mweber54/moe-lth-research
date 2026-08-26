from __future__ import annotations

from pathlib import Path

import torch

from moe_lth.data import ByteTokenizer
from moe_lth.models.transformer import LanguageModelOutput
from moe_lth.utils import append_jsonl


class RoutingLogger:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.usage_log = self.output_dir / "logs" / "expert_usage.jsonl"

    def log_step(self, step: int, model_output: LanguageModelOutput) -> None:
        for layer_id, trace in enumerate(model_output.routes):
            for expert_id, usage_fraction in enumerate(trace.usage.detach().cpu().tolist()):
                append_jsonl(
                    self.usage_log,
                    {
                        "step": step,
                        "layer_id": layer_id,
                        "expert_id": expert_id,
                        "token_count": int(
                            (trace.selected_expert_indices == expert_id)
                            .sum()
                            .detach()
                            .cpu()
                            .item()
                        ),
                        "usage_fraction": usage_fraction,
                        "router_entropy_mean": float(trace.entropy.mean().detach().cpu()),
                        "router_margin_mean": float(trace.margin.mean().detach().cpu()),
                        "dropped_fraction": float(trace.dropped_fraction.detach().cpu()),
                    },
                )


def sampled_context_records(
    token_ids: torch.Tensor,
    model_output: LanguageModelOutput,
    step: int,
    max_per_expert: int = 2,
) -> list[dict]:
    records: list[dict] = []
    for layer_id, trace in enumerate(model_output.routes):
        for expert_id in range(trace.usage.numel()):
            assigned = (trace.selected_expert_indices == expert_id).any(dim=-1)
            positions = torch.nonzero(assigned, as_tuple=False)[:max_per_expert]
            for batch_id, position in positions.tolist():
                start = max(0, position - 12)
                context = token_ids[batch_id, start : position + 1].detach().cpu().tolist()
                records.append(
                    {
                        "step": step,
                        "layer_id": layer_id,
                        "expert_id": expert_id,
                        "token_ids": context,
                        "decoded_context": ByteTokenizer.decode(context),
                    }
                )
    return records
