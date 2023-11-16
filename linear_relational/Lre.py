from typing import Any, Literal

import torch
from torch import nn


class InvertedLre(nn.Module):
    """Low-rank inverted LRE, used for calculating subject activations from object activations"""

    relation: str
    subject_layer: int
    object_layer: int
    # store u, v, s, and bias separately to avoid storing the full weight matrix
    u: nn.Parameter
    s: nn.Parameter
    v: nn.Parameter
    bias: nn.Parameter
    object_aggregation: Literal["mean", "first_token"]
    metadata: dict[str, Any] | None = None

    def __init__(
        self,
        relation: str,
        subject_layer: int,
        object_layer: int,
        object_aggregation: Literal["mean", "first_token"],
        u: torch.Tensor,
        s: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.relation = relation
        self.subject_layer = subject_layer
        self.object_layer = object_layer
        self.object_aggregation = object_aggregation
        self.u = nn.Parameter(u, requires_grad=False)
        self.s = nn.Parameter(s, requires_grad=False)
        self.v = nn.Parameter(v, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)
        self.metadata = metadata

    @property
    def rank(self) -> int:
        return self.s.shape[0]

    def w_inv_times_vec(self, vec: torch.Tensor) -> torch.Tensor:
        # group u.T @ vec to avoid calculating larger matrices than needed
        return self.v @ torch.diag(1 / self.s) @ (self.u.T @ vec)

    def forward(
        self,
        subject_activations: torch.Tensor,  # a tensor of shape (num_activations, hidden_activation_size)
        normalize: bool = False,
    ) -> torch.Tensor:
        return self.calculate_object_activation(
            subject_activations=subject_activations,
            normalize=normalize,
        )

    def calculate_subject_activation(
        self,
        object_activations: torch.Tensor,  # a tensor of shape (num_activations, hidden_activation_size)
        normalize: bool = False,
    ) -> torch.Tensor:
        # match precision of weight_inverse and bias
        unbiased_acts = object_activations - self.bias.unsqueeze(0)
        vec = self.w_inv_times_vec(unbiased_acts.T).mean(dim=1)

        if normalize:
            vec = vec / vec.norm()
        return vec


class LowRankLre(nn.Module):
    """Low-rank approximation of a LRE"""

    relation: str
    subject_layer: int
    object_layer: int
    # store u, v, s, and bias separately to avoid storing the full weight matrix
    u: nn.Parameter
    s: nn.Parameter
    v: nn.Parameter
    bias: nn.Parameter
    object_aggregation: Literal["mean", "first_token"]
    metadata: dict[str, Any] | None = None

    def __init__(
        self,
        relation: str,
        subject_layer: int,
        object_layer: int,
        object_aggregation: Literal["mean", "first_token"],
        u: torch.Tensor,
        s: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.relation = relation
        self.subject_layer = subject_layer
        self.object_layer = object_layer
        self.object_aggregation = object_aggregation
        self.u = nn.Parameter(u, requires_grad=False)
        self.s = nn.Parameter(s, requires_grad=False)
        self.v = nn.Parameter(v, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)
        self.metadata = metadata

    @property
    def rank(self) -> int:
        return self.s.shape[0]

    def w_times_vec(self, vec: torch.Tensor) -> torch.Tensor:
        # group v.T @ vec to avoid calculating larger matrices than needed
        return self.u @ torch.diag(self.s) @ (self.v.T @ vec)

    def forward(
        self,
        subject_activations: torch.Tensor,  # a tensor of shape (num_activations, hidden_activation_size)
        normalize: bool = False,
    ) -> torch.Tensor:
        return self.calculate_object_activation(
            subject_activations=subject_activations,
            normalize=normalize,
        )

    def calculate_object_activation(
        self,
        subject_activations: torch.Tensor,  # a tensor of shape (num_activations, hidden_activation_size)
        normalize: bool = False,
    ) -> torch.Tensor:
        # match precision of weight_inverse and bias
        ws = self.w_times_vec(subject_activations.T)
        vec = (ws + self.bias.unsqueeze(-1)).mean(dim=1)
        if normalize:
            vec = vec / vec.norm()
        return vec


class Lre(nn.Module):
    """Linear Relational Embedding"""

    relation: str
    subject_layer: int
    object_layer: int
    weight: nn.Parameter
    bias: nn.Parameter
    object_aggregation: Literal["mean", "first_token"]
    metadata: dict[str, Any] | None = None

    def __init__(
        self,
        relation: str,
        subject_layer: int,
        object_layer: int,
        object_aggregation: Literal["mean", "first_token"],
        weight: torch.Tensor,
        bias: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.relation = relation
        self.subject_layer = subject_layer
        self.object_layer = object_layer
        self.object_aggregation = object_aggregation
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)
        self.metadata = metadata

    def invert(self, rank: int) -> InvertedLre:
        """Invert this LRE using a low-rank approximation"""
        u, s, v = self._low_rank_svd(rank)
        return InvertedLre(
            relation=self.relation,
            subject_layer=self.subject_layer,
            object_layer=self.object_layer,
            object_aggregation=self.object_aggregation,
            u=u.detach().clone(),
            s=s.detach().clone(),
            v=v.detach().clone(),
            bias=self.bias.detach().clone(),
            metadata=self.metadata,
        )

    def to_low_rank(self, rank: int) -> LowRankLre:
        """Create a low-rank approximation of this LRE"""
        u, s, v = self._low_rank_svd(rank)
        return LowRankLre(
            relation=self.relation,
            subject_layer=self.subject_layer,
            object_layer=self.object_layer,
            object_aggregation=self.object_aggregation,
            u=u.detach().clone(),
            s=s.detach().clone(),
            v=v.detach().clone(),
            bias=self.bias.detach().clone(),
            metadata=self.metadata,
        )

    @torch.no_grad()
    def _low_rank_svd(
        self, rank: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # use a float for the svd, then convert back to the original dtype
        u, s, v = torch.svd(self.weight.float())
        low_rank_u: torch.Tensor = u[:, :rank].to(self.weight.dtype)
        low_rank_v: torch.Tensor = v[:, :rank].to(self.weight.dtype)
        low_rank_s: torch.Tensor = s[:rank].to(self.weight.dtype)
        return low_rank_u, low_rank_s, low_rank_v
