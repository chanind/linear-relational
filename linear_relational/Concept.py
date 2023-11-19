from __future__ import annotations

from typing import Any

import torch
from torch import nn


class Concept(nn.Module):
    layer: int
    vector: torch.Tensor
    object: str
    relation: str
    metadata: dict[str, Any]

    def __init__(
        self,
        layer: int,
        vector: torch.Tensor,
        object: str,
        relation: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.vector = vector
        self.object = object
        self.relation = relation
        self.metadata = metadata or {}

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        vector = self.vector.to(activations.device)
        if len(activations.shape) == 1:
            return vector @ activations
        return vector @ activations.T

    @property
    def name(self) -> str:
        return f" {self.relation}: {self.object}"
