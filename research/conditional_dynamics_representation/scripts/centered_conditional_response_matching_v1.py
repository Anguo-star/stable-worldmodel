"""Training-only centered conditional response matching.

For every matched group, predictions and detached real targets share the same
query and action but differ in history.  The objective matches only the
within-group conditional response; native prediction MSE remains responsible
for the common center.  It adds no parameters or inference path.
"""

from __future__ import annotations

from typing import Any

import torch


MINIMUM_TARGET_SCALE = 1.0e-8


def centered_conditional_response_matching(
    prediction: Any,
    target: Any,
    groups: Any,
) -> dict[str, Any]:
    """Match the centered prediction configuration to real target response.

    Args:
        prediction: Live predictions with shape ``(B, T, D)``.
        target: Target latents with the same shape. Targets are detached inside
            this auxiliary objective; the caller owns the native live route.
        groups: Disjoint row indices with shape ``(P, N)``, where every group
            has one shared query/action and ``N >= 2`` histories/futures.

    In the binary case, the returned loss is exactly

    ``mse((p1-p0) - (t1-t0)) / mse(t1-t0)``.

    Therefore exact conditional response is a zero-loss, zero-gradient
    solution, while a history-independent zero response has loss one.
    """

    if not torch.is_tensor(prediction) or not torch.is_tensor(target):
        raise TypeError("prediction and target must be torch tensors")
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share (B,T,D) shape")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating-point tensors")
    if prediction.device != target.device:
        raise ValueError("prediction and target must share a device")
    if not torch.is_tensor(groups):
        raise TypeError("groups must be a torch tensor")
    if groups.ndim != 2:
        raise ValueError("groups must have shape (P,N)")
    if groups.dtype != torch.long:
        raise TypeError("groups must use torch.long")
    if groups.device != prediction.device:
        raise ValueError("groups and prediction must share a device")
    if groups.size(0) < 1:
        raise ValueError("at least one counterfactual group is required")
    group_width = int(groups.size(1))
    if group_width < 2:
        raise ValueError("counterfactual groups require N >= 2")
    if int(groups.min()) < 0 or int(groups.max()) >= prediction.size(0):
        raise ValueError("groups contain an out-of-range row")
    if torch.unique(groups).numel() != groups.numel():
        raise ValueError("counterfactual groups must be globally disjoint")

    predicted = prediction[groups, -1].float()
    real = target[groups, -1].detach().float()
    predicted_center = predicted.mean(dim=1, keepdim=True)
    target_center = real.mean(dim=1, keepdim=True)
    predicted_response = predicted - predicted_center
    target_response = real - target_center
    response_error = predicted_response - target_response
    response_error_by_group = response_error.square().mean(dim=(1, 2))
    target_scale_by_group = target_response.square().mean(dim=(1, 2))

    finite_values = (
        predicted,
        real,
        predicted_response,
        target_response,
        response_error_by_group,
        target_scale_by_group,
    )
    if not all(bool(torch.isfinite(value).all()) for value in finite_values):
        raise FloatingPointError("nonfinite centered response component")
    collapsed = target_scale_by_group <= MINIMUM_TARGET_SCALE
    if bool(collapsed.any()):
        indices = torch.nonzero(collapsed, as_tuple=False).flatten().tolist()
        raise ValueError(
            "target response scale must be strictly above "
            f"{MINIMUM_TARGET_SCALE}: collapsed groups {indices[:8]}"
        )

    normalized_error_by_group = (
        response_error_by_group / target_scale_by_group
    )
    loss = normalized_error_by_group.mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite centered response loss")

    return {
        "loss": loss,
        "normalized_error_by_group": normalized_error_by_group,
        "response_error_by_group": response_error_by_group,
        "target_scale": target_scale_by_group.mean(),
        "target_scale_by_group": target_scale_by_group,
        "predicted_response": predicted_response,
        "target_response": target_response,
        "group_width": group_width,
    }


__all__ = [
    "MINIMUM_TARGET_SCALE",
    "centered_conditional_response_matching",
]
