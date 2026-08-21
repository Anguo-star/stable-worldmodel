from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    centered_conditional_response_matching_v1 as ccrm,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    pair_normalized_exact_future_residual_v1 as exact_future,
)


def _target_and_groups() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(14321)
    target = torch.randn(8, 3, 5, requires_grad=True)
    groups = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.long)
    return target, groups


def test_exact_target_has_zero_loss_and_no_target_gradient():
    target, groups = _target_and_groups()
    prediction = target.detach().clone().requires_grad_(True)

    result = exact_future.pair_normalized_exact_future_residual(
        prediction,
        target,
        groups,
    )
    prediction_gradient, target_gradient = torch.autograd.grad(
        result["loss"],
        (prediction, target),
        allow_unused=True,
    )

    assert torch.allclose(result["loss"], torch.zeros_like(result["loss"]))
    assert torch.allclose(
        prediction_gradient, torch.zeros_like(prediction_gradient)
    )
    assert target_gradient is None
    assert torch.count_nonzero(result["identity_absolute_error_by_group"]) == 0


def test_zero_response_has_positive_unit_loss_when_centers_match():
    target, groups = _target_and_groups()
    prediction = target.detach().clone()
    for group in groups:
        prediction[group, -1] = target.detach()[group, -1].mean(dim=0)
    prediction.requires_grad_(True)

    result = exact_future.pair_normalized_exact_future_residual(
        prediction,
        target,
        groups,
    )

    assert result["loss"] > 0
    assert torch.allclose(
        result["normalized_exact_future_residual_by_group"],
        torch.ones_like(result["normalized_exact_future_residual_by_group"]),
    )
    assert torch.allclose(
        result["ccrm_normalized_error_by_group"],
        torch.ones_like(result["ccrm_normalized_error_by_group"]),
    )
    assert torch.allclose(
        result["normalized_common_center_mse_by_group"],
        torch.zeros_like(result["normalized_common_center_mse_by_group"]),
    )


def test_group_and_within_group_permutations_are_invariant():
    target, groups = _target_and_groups()
    prediction = torch.randn_like(target, requires_grad=True)
    baseline = exact_future.pair_normalized_exact_future_residual(
        prediction,
        target,
        groups,
    )
    group_order = torch.tensor([3, 1, 0, 2])
    reordered = exact_future.pair_normalized_exact_future_residual(
        prediction,
        target,
        groups[group_order],
    )
    within_group = groups.clone()
    within_group[0] = within_group[0].flip(0)
    within_group[2] = within_group[2].flip(0)
    within = exact_future.pair_normalized_exact_future_residual(
        prediction,
        target,
        within_group,
    )

    assert torch.allclose(baseline["loss"], reordered["loss"])
    assert torch.allclose(
        reordered["normalized_exact_future_residual_by_group"],
        baseline["normalized_exact_future_residual_by_group"][group_order],
    )
    assert torch.allclose(baseline["loss"], within["loss"])
    assert torch.allclose(
        baseline["normalized_exact_future_residual_by_group"],
        within["normalized_exact_future_residual_by_group"],
    )


def test_exact_future_equals_ccrm_plus_normalized_common_center_mse():
    target, groups = _target_and_groups()
    prediction = torch.randn_like(target, requires_grad=True)
    result = exact_future.pair_normalized_exact_future_residual(
        prediction,
        target,
        groups,
    )
    ccrm_result = ccrm.centered_conditional_response_matching(
        prediction,
        target,
        groups,
    )

    assert torch.allclose(
        result["ccrm_normalized_error_by_group"],
        ccrm_result["normalized_error_by_group"],
        rtol=exact_future.IDENTITY_RTOL,
        atol=exact_future.IDENTITY_ATOL,
    )
    assert torch.allclose(
        result["normalized_exact_future_residual_by_group"],
        result["ccrm_normalized_error_by_group"]
        + result["normalized_common_center_mse_by_group"],
        rtol=exact_future.IDENTITY_RTOL,
        atol=exact_future.IDENTITY_ATOL,
    )


def test_invalid_inputs_fail_closed():
    target, groups = _target_and_groups()
    prediction = target.detach().clone()
    with pytest.raises(ValueError, match="share \\(B,T,D\\) shape"):
        exact_future.pair_normalized_exact_future_residual(
            prediction[:, :-1], target, groups
        )
    with pytest.raises(TypeError, match="floating-point"):
        exact_future.pair_normalized_exact_future_residual(
            prediction.to(torch.int64), target.to(torch.int64), groups
        )
    with pytest.raises(TypeError, match="torch.long"):
        exact_future.pair_normalized_exact_future_residual(
            prediction, target, groups.to(torch.int32)
        )
    with pytest.raises(ValueError, match="globally disjoint"):
        exact_future.pair_normalized_exact_future_residual(
            prediction, target, torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        )
    with pytest.raises(ValueError, match="out-of-range"):
        exact_future.pair_normalized_exact_future_residual(
            prediction, target, torch.tensor([[0, 8]], dtype=torch.long)
        )
    with pytest.raises(ValueError, match="N >= 2"):
        exact_future.pair_normalized_exact_future_residual(
            prediction, target, torch.tensor([[0]], dtype=torch.long)
        )


def test_nonfinite_collapsed_and_device_mismatched_inputs_fail_closed():
    target, groups = _target_and_groups()
    prediction = target.detach().clone()
    nonfinite = prediction.clone()
    nonfinite[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="prediction contains nonfinite"):
        exact_future.pair_normalized_exact_future_residual(
            nonfinite, target, groups
        )
    collapsed = torch.zeros_like(target)
    with pytest.raises(ValueError, match="group-centered target energy"):
        exact_future.pair_normalized_exact_future_residual(
            prediction, collapsed, groups
        )
    with pytest.raises(ValueError, match="share a device"):
        exact_future.pair_normalized_exact_future_residual(
            prediction, torch.empty_like(target, device="meta"), groups
        )
