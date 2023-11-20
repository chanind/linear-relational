from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn


class Concept(nn.Module):
    layer: int
    vector: torch.Tensor
    object: str
    relation: str
    name: str
    metadata: dict[str, Any]

    def __init__(
        self,
        layer: int,
        vector: torch.Tensor,
        object: str,
        relation: str,
        metadata: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.vector = vector
        self.object = object
        self.relation = relation
        self.metadata = metadata or {}
        self.name = name or f"{self.relation}: {self.object}"

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        vector = self.vector.to(activations.device)
        if len(activations.shape) == 1:
            return vector @ activations
        return vector @ activations.T
