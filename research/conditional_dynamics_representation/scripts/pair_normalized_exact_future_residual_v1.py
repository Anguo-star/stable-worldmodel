"""Training-unwired pair-normalized exact-future residual.

For each disjoint matched group, this objective compares terminal predictions
with their detached terminal targets, normalized by the group-centered target
energy.  It contains no model, data-loader, optimizer, or runner integration.
"""

from __future__ import annotations

from typing import Any

import torch


MINIMUM_TARGET_CENTERED_ENERGY = 1.0e-8
IDENTITY_ATOL = 1.0e-6
IDENTITY_RTOL = 1.0e-5


def pair_normalized_exact_future_residual(
    prediction: Any,
    target: Any,
    groups: Any,
) -> dict[str, Any]:
    """Return a normalized exact terminal-future residual for matched groups.

    For a group ``g`` of size ``N``, this is

    ``MSE(prediction[g, -1] - stopgrad(target[g, -1])) /
    MSE(stopgrad(target[g, -1]) - mean_g(target[g, -1]))``.

    The returned per-group decomposition verifies the exact identity

    ``exact_future = CCRM_response + normalized_common_center_MSE``.
    """

    if not torch.is_tensor(prediction) or not torch.is_tensor(target):
        raise TypeError("prediction and target must be torch tensors")
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share (B,T,D) shape")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating-point tensors")
    if prediction.device != target.device:
        raise ValueError("prediction and target must share a device")
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("prediction contains nonfinite values")
    if not bool(torch.isfinite(target).all()):
        raise FloatingPointError("target contains nonfinite values")
    if not torch.is_tensor(groups):
        raise TypeError("groups must be a torch tensor")
    if groups.ndim != 2:
        raise ValueError("groups must have shape (P,N)")
    if groups.dtype != torch.long:
        raise TypeError("groups must use torch.long")
    if groups.device != prediction.device:
        raise ValueError("groups and prediction must share a device")
    if groups.size(0) < 1:
        raise ValueError("at least one matched group is required")
    if groups.size(1) < 2:
        raise ValueError("matched groups require N >= 2")
    if int(groups.min()) < 0 or int(groups.max()) >= prediction.size(0):
        raise ValueError("groups contain an out-of-range row")
    if torch.unique(groups).numel() != groups.numel():
        raise ValueError("matched groups must be globally disjoint")

    predicted = prediction[groups, -1].float()
    detached_target = target[groups, -1].detach().float()
    prediction_center = predicted.mean(dim=1, keepdim=True)
    target_center = detached_target.mean(dim=1, keepdim=True)
    centered_prediction = predicted - prediction_center
    centered_target = detached_target - target_center
    target_centered_energy_by_group = centered_target.square().mean(dim=(1, 2))
    collapsed = target_centered_energy_by_group <= MINIMUM_TARGET_CENTERED_ENERGY
    if bool(collapsed.any()):
        indices = torch.nonzero(collapsed, as_tuple=False).flatten().tolist()
        raise ValueError(
            "group-centered target energy must be strictly above "
            f"{MINIMUM_TARGET_CENTERED_ENERGY}: collapsed groups {indices[:8]}"
        )

    exact_residual_energy_by_group = (
        (predicted - detached_target).square().mean(dim=(1, 2))
    )
    ccrm_response_energy_by_group = (
        (centered_prediction - centered_target).square().mean(dim=(1, 2))
    )
    common_center_mse_by_group = (
        (prediction_center - target_center).square().mean(dim=(1, 2))
    )
    normalized_exact_future_residual_by_group = (
        exact_residual_energy_by_group / target_centered_energy_by_group
    )
    ccrm_normalized_error_by_group = (
        ccrm_response_energy_by_group / target_centered_energy_by_group
    )
    normalized_common_center_mse_by_group = (
        common_center_mse_by_group / target_centered_energy_by_group
    )
    decomposition_by_group = (
        ccrm_normalized_error_by_group
        + normalized_common_center_mse_by_group
    )
    identity_absolute_error_by_group = (
        normalized_exact_future_residual_by_group - decomposition_by_group
    ).abs()
    if not bool(
        torch.allclose(
            normalized_exact_future_residual_by_group,
            decomposition_by_group,
            rtol=IDENTITY_RTOL,
            atol=IDENTITY_ATOL,
        )
    ):
        raise RuntimeError("exact-future/CCRM-center decomposition identity failed")
    loss = normalized_exact_future_residual_by_group.mean()
    finite_values = (
        exact_residual_energy_by_group,
        ccrm_response_energy_by_group,
        common_center_mse_by_group,
        normalized_exact_future_residual_by_group,
        ccrm_normalized_error_by_group,
        normalized_common_center_mse_by_group,
        identity_absolute_error_by_group,
        loss,
    )
    if not all(bool(torch.isfinite(value).all()) for value in finite_values):
        raise FloatingPointError("nonfinite pair-normalized exact-future residual")

    return {
        "loss": loss,
        "normalized_exact_future_residual_by_group": (
            normalized_exact_future_residual_by_group
        ),
        "exact_residual_energy_by_group": exact_residual_energy_by_group,
        "target_centered_energy_by_group": target_centered_energy_by_group,
        "ccrm_normalized_error_by_group": ccrm_normalized_error_by_group,
        "normalized_common_center_mse_by_group": (
            normalized_common_center_mse_by_group
        ),
        "common_center_mse_by_group": common_center_mse_by_group,
        "identity_absolute_error_by_group": identity_absolute_error_by_group,
        "prediction_center": prediction_center,
        "detached_target_center": target_center,
        "group_width": int(groups.size(1)),
    }


__all__ = [
    "IDENTITY_ATOL",
    "IDENTITY_RTOL",
    "MINIMUM_TARGET_CENTERED_ENERGY",
    "pair_normalized_exact_future_residual",
]
