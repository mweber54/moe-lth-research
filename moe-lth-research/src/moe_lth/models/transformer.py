from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .moe_layer import MoEFeedForward, RouteTrace


class MoETransformerBlock(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        d_model = int(config["d_model"])
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            int(config["num_heads"]),
            dropout=float(config["dropout"]),
            batch_first=True,
        )
        self.moe_norm = nn.LayerNorm(d_model)
        self.moe = MoEFeedForward(
            d_model=d_model,
            hidden_size=int(config["expert_hidden_size"]),
            num_experts=int(config["num_experts"]),
            top_k=int(config.get("top_k", 1)),
            capacity_factor=float(config["capacity_factor"]),
            dropout=float(config["dropout"]),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        causal_mask: torch.Tensor,
        override_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RouteTrace, torch.Tensor]:
        normalized = self.attention_norm(inputs)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask, 
        )
        hidden = inputs + attended
        pre_router_hidden = self.moe_norm(hidden)
        expert_output, auxiliary_loss, trace = self.moe(pre_router_hidden, override_ids)
        return hidden + expert_output, auxiliary_loss, trace, pre_router_hidden


@dataclass
class LanguageModelOutput:
    logits: torch.Tensor
    auxiliary_loss: torch.Tensor
    routes: list[RouteTrace]
    pre_router_hidden_states: list[torch.Tensor]


class TinyMoELanguageModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        top_k = int(config.get("top_k", 1))
        num_experts = int(config["num_experts"])
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts.")
        self.config = config
        vocab_size = int(config["vocab_size"])
        d_model = int(config["d_model"])
        max_seq_len = int(config["max_seq_len"])
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            MoETransformerBlock(config) for _ in range(int(config["num_layers"]))
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        token_ids: torch.Tensor,
        route_overrides: list[torch.Tensor | None] | None = None,
    ) -> LanguageModelOutput:
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        causal_mask = torch.full(
            (seq_len, seq_len),
            float("-inf"),
            device=token_ids.device,
        ).triu(1)
        routes: list[RouteTrace] = []
        pre_router_hidden_states: list[torch.Tensor] = []
        auxiliary_losses = []

        for layer_id, block in enumerate(self.blocks):
            override = None if route_overrides is None else route_overrides[layer_id]
            hidden, auxiliary_loss, trace, pre_router_hidden = block(hidden, causal_mask, override)
            pre_router_hidden_states.append(pre_router_hidden.detach())
            auxiliary_losses.append(auxiliary_loss)
            routes.append(trace)

        logits = self.lm_head(self.final_norm(hidden))
        return LanguageModelOutput(
            logits=logits,
            auxiliary_loss=torch.stack(auxiliary_losses).mean(),
            routes=routes,
            pre_router_hidden_states=pre_router_hidden_states,
        )
