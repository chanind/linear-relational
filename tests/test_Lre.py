import torch

from linear_relational.Lre import InvertedLre, LowRankLre, Lre


def test_Lre_invert() -> None:
    bias = torch.tensor([1.0, 0.0, 0.0])
    lre = Lre(
        relation="test",
        subject_layer=5,
        object_layer=10,
        object_aggregation="mean",
        bias=bias,
        weight=torch.eye(3),
    )
    inv_lre = lre.invert(rank=2)
    assert inv_lre.relation == "test"
    assert inv_lre.subject_layer == 5
    assert inv_lre.object_layer == 10
    assert inv_lre.object_aggregation == "mean"
    assert torch.allclose(inv_lre.bias, bias)
    assert inv_lre.u.shape == (3, 2)
    assert inv_lre.s.shape == (2,)
    assert inv_lre.v.shape == (3, 2)
    assert inv_lre.rank == 2


def test_Lre_to_low_rank() -> None:
    bias = torch.tensor([1.0, 0.0, 0.0])
    lre = Lre(
        relation="test",
        subject_layer=5,
        object_layer=10,
        object_aggregation="mean",
        bias=bias,
        weight=torch.eye(3),
    )
    low_rank_lre = lre.to_low_rank(rank=2)
    assert low_rank_lre.relation == "test"
    assert low_rank_lre.subject_layer == 5
    assert low_rank_lre.object_layer == 10
    assert low_rank_lre.object_aggregation == "mean"
    assert torch.allclose(low_rank_lre.bias, bias)
    assert low_rank_lre.u.shape == (3, 2)
    assert low_rank_lre.s.shape == (2,)
    assert low_rank_lre.v.shape == (3, 2)
    assert low_rank_lre.rank == 2


def test_LowRankLre_calculate_object_activation_unnormalized() -> None:
    acts = torch.stack(
        [
            torch.tensor([2.0, 1.0, 1.0]),
            torch.tensor([1.0, 0.0, 0.0]),
        ]
    )
    bias = torch.tensor([-1.0, 0.0, 0.0])
    lre = LowRankLre(
        relation="test",
        subject_layer=0,
        object_layer=0,
        # this u,s,v makes W_inv the identity matrix
        u=torch.eye(3),
        s=torch.ones(3),
        v=torch.eye(3),
        object_aggregation="mean",
        bias=bias,
    )
    vec = lre.calculate_object_activation(acts, normalize=False)
    assert torch.allclose(vec, torch.tensor([0.5, 0.5, 0.5]))
    assert torch.allclose(vec, lre(acts, normalize=False))
    assert lre.rank == 3


def test_InvertedLre_calculate_subject_activation_unnormalized() -> None:
    acts = torch.stack(
        [
            torch.tensor([2.0, 1.0, 1.0]),
            torch.tensor([1.0, 0.0, 0.0]),
        ]
    )
    bias = torch.tensor([1.0, 0.0, 0.0])
    inv_lre = InvertedLre(
        relation="test",
        subject_layer=0,
        object_layer=0,
        # this u,s,v makes W_inv the identity matrix
        u=torch.eye(3),
        s=torch.ones(3),
        v=torch.eye(3),
        object_aggregation="mean",
        bias=bias,
    )
    vec = inv_lre.calculate_subject_activation(acts, normalize=False)
    assert torch.allclose(vec, torch.tensor([0.5, 0.5, 0.5]))
    assert torch.allclose(vec, inv_lre(acts, normalize=False))
    assert inv_lre.rank == 3


def test_InvertedLre_calculate_subject_activation_normalized() -> None:
    acts = torch.stack(
        [
            torch.tensor([2.0, 1.0, 1.0]),
            torch.tensor([1.0, 0.0, 0.0]),
        ]
    )
    bias = torch.tensor([1.0, 0.0, 0.0])
    inv_lre = InvertedLre(
        relation="test",
        subject_layer=0,
        object_layer=0,
        # this u,s,v makes W_inv the identity matrix
        u=torch.eye(3),
        s=torch.ones(3),
        v=torch.eye(3),
        object_aggregation="mean",
        bias=bias,
    )
    vec = inv_lre.calculate_subject_activation(acts, normalize=True)
    raw_target = torch.tensor([0.5, 0.5, 0.5])
    target = raw_target / raw_target.norm()
    assert torch.allclose(vec, target)
    assert torch.allclose(vec, inv_lre(acts, normalize=True))
