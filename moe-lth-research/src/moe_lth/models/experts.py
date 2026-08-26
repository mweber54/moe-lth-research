from __future__ import annotations

import torch
from torch import nn


class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.activation(self.fc1(inputs))))

