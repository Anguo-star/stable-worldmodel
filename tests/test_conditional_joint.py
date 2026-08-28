from __future__ import annotations

import pytest
import torch

from stable_worldmodel.wm.conditional_joint import (
    conditional_joint_alignment,
    conditional_joint_group_rows,
    conditional_joint_loss_terms,
)


def test_binary_exact_targets_are_stationary() -> None:
    target = torch.tensor(
        [[[0.0, 0.0]], [[2.0, 0.0]], [[10.0, 1.0]], [[10.0, 3.0]]],
        requires_grad=True,
    )
    prediction = target.detach().clone().requires_grad_(True)
    groups = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

    result = conditional_joint_alignment(prediction, target, groups)
    result["loss"].backward()

    assert result["loss"].item() == pytest.approx(0.0, abs=1.0e-7)
    assert torch.allclose(prediction.grad, torch.zeros_like(prediction.grad))
    assert target.grad is None


def test_binary_history_blind_predictions_receive_response_gradient() -> None:
    target = torch.tensor([[[0.0]], [[2.0]]])
    prediction = torch.tensor([[[1.0]], [[1.0]]], requires_grad=True)
    groups = torch.tensor([[0, 1]], dtype=torch.long)

    result = conditional_joint_alignment(prediction, target, groups)
    result["loss"].backward()

    assert result["loss"].item() > 0.0
    assert prediction.grad is not None
    assert prediction.grad[0].item() > 0.0
    assert prediction.grad[1].item() < 0.0


def test_triplet_assignment_prefers_matched_futures() -> None:
    target = torch.tensor([[[0.0]], [[2.0]], [[5.0]]])
    groups = torch.tensor([[0, 1, 2]], dtype=torch.long)
    exact = conditional_joint_alignment(target.clone(), target, groups)
    collapsed = conditional_joint_alignment(
        torch.full_like(target, target.mean()), target, groups
    )

    assert exact["loss"] < collapsed["loss"]
    assert exact["assignment_margin"] > 0.0


def test_groups_must_be_disjoint() -> None:
    value = torch.randn(3, 1, 4)
    with pytest.raises(ValueError, match="globally disjoint"):
        conditional_joint_alignment(
            value, value, torch.tensor([[0, 1], [1, 2]])
        )


def test_group_labels_build_disjoint_rows_and_ignore_negative_labels() -> None:
    labels = torch.tensor([7, -1, 7, 2, 2], dtype=torch.int32)

    groups = conditional_joint_group_rows(
        labels,
        expected_width=2,
        batch_size=5,
        device=torch.device("cpu"),
    )

    assert torch.equal(groups, torch.tensor([[3, 4], [0, 2]]))
    assert groups.dtype == torch.long


def test_loss_terms_accept_patch_latents_without_changing_the_objective() -> None:
    target = torch.tensor(
        [
            [[[0.0, 0.0], [1.0, 0.0]]],
            [[[2.0, 0.0], [1.0, 2.0]]],
        ]
    )
    prediction = torch.full_like(target, 0.5, requires_grad=True)
    labels = torch.tensor([0, 0])

    patch_result = conditional_joint_loss_terms(
        prediction, target, labels, group_width=2
    )
    flat_result = conditional_joint_alignment(
        prediction.flatten(start_dim=2),
        target.flatten(start_dim=2),
        torch.tensor([[0, 1]]),
    )

    assert torch.equal(
        patch_result["conditional_joint_loss"], flat_result["loss"]
    )
