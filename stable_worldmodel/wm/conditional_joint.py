"""Parameter-free conditional-overlap objectives for world-model training.

The objectives in this module consume groups of trajectories that share the
same visible query and action but differ in their visible histories and true
futures.  Group metadata is a training-time sampling relation only; it is
never passed to the world model.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as functional


MINIMUM_TARGET_ENERGY = 1.0e-8
CANONICAL_BINARY_MARGIN = 0.5
CONDITIONAL_JOINT_BATCH_KEY = "conditional_joint_group"


@contextmanager
def temporary_eval_modules(*modules):
    """Disable stochastic/stateful layers while preserving trainability."""

    states = {
        child: child.training
        for module in modules
        for child in module.modules()
    }
    for module in modules:
        module.eval()
    try:
        yield
    finally:
        for child, training in states.items():
            child.training = training


def conditional_joint_group_rows(
    group_labels: Any,
    *,
    expected_width: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert per-row group labels into disjoint objective row groups.

    Negative labels mark ordinary, unpaired rows. Non-negative labels identify
    rows belonging to the same conditional intervention. The relation remains
    training metadata and is never supplied to the model itself.
    """

    if not torch.is_tensor(group_labels) or group_labels.ndim != 1:
        raise ValueError(
            f"{CONDITIONAL_JOINT_BATCH_KEY} must have shape (B,)"
        )
    if group_labels.size(0) != batch_size:
        raise ValueError(
            f"{CONDITIONAL_JOINT_BATCH_KEY} row count must match the batch"
        )
    if group_labels.dtype == torch.bool or group_labels.is_floating_point():
        raise TypeError(
            f"{CONDITIONAL_JOINT_BATCH_KEY} must use an integer dtype"
        )
    if expected_width not in {2, 3}:
        raise ValueError("conditional group width must be two or three")

    labels = group_labels.to(device=device, dtype=torch.long)
    active_ids = torch.unique(labels[labels >= 0], sorted=True)
    groups = [
        torch.nonzero(labels == group_id, as_tuple=False).flatten()
        for group_id in active_ids
    ]
    if not groups:
        raise ValueError("conditional joint batch contains no active group")
    if any(rows.numel() != expected_width for rows in groups):
        observed = [int(rows.numel()) for rows in groups]
        raise ValueError(
            "conditional joint groups changed width: "
            f"expected={expected_width} observed={observed[:8]}"
        )
    return torch.stack(groups)


def _validate(
    prediction: Any,
    target: Any,
    groups: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(prediction) or not torch.is_tensor(target):
        raise TypeError("prediction and target must be torch tensors")
    if prediction.shape != target.shape or prediction.ndim < 3:
        raise ValueError(
            "prediction and target must share (B,T,...) shape"
        )
    if prediction.device != target.device:
        raise ValueError("prediction and target must share a device")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating-point tensors")
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("prediction contains nonfinite values")
    if not bool(torch.isfinite(target).all()):
        raise FloatingPointError("target contains nonfinite values")
    if not torch.is_tensor(groups) or groups.ndim != 2:
        raise ValueError("groups must be a tensor with shape (P,G)")
    if groups.dtype != torch.long:
        raise TypeError("groups must use torch.long")
    if groups.device != prediction.device:
        raise ValueError("groups and prediction must share a device")
    if groups.size(0) < 1 or groups.size(1) not in {2, 3}:
        raise ValueError("conditional groups must have width two or three")
    if int(groups.min()) < 0 or int(groups.max()) >= prediction.size(0):
        raise ValueError("groups contain an out-of-range row")
    if torch.unique(groups).numel() != groups.numel():
        raise ValueError("conditional groups must be globally disjoint")

    flat_prediction = prediction.flatten(start_dim=2)
    flat_target = target.flatten(start_dim=2)
    predicted = flat_prediction[groups, -1].float()
    detached_target = flat_target[groups, -1].detach().float()
    return predicted, detached_target, groups


def _binary_response_alignment(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Centered response fit plus the target-stationary assignment barrier."""

    prediction_center = predicted.mean(dim=1, keepdim=True)
    target_center = target.mean(dim=1, keepdim=True)
    centered_prediction = predicted - prediction_center
    centered_target = target - target_center
    target_energy = centered_target.square().mean(dim=(1, 2))
    if bool((target_energy <= MINIMUM_TARGET_ENERGY).any()):
        raise ValueError(
            "binary conditional targets need nonzero centered energy"
        )

    response_error = (
        (centered_prediction - centered_target).square().mean(dim=(1, 2))
        / target_energy
    )
    target_axis = target[:, 1] - target[:, 0]
    prediction_axis = predicted[:, 1] - predicted[:, 0]
    axis_energy = target_axis.square().sum(dim=-1)
    alpha = (prediction_axis * target_axis).sum(dim=-1) / axis_energy
    beta = (
        ((prediction_center[:, 0] - target_center[:, 0]) * target_axis)
        .sum(dim=-1)
        / axis_energy
    )
    assignment_margins = torch.stack(
        (0.5 * alpha - beta, 0.5 * alpha + beta), dim=1
    )
    deficits = torch.relu(CANONICAL_BINARY_MARGIN - assignment_margins)
    margin_loss = deficits.square().mean()
    response_loss = response_error.mean()
    loss = response_loss + margin_loss
    return {
        "loss": loss,
        "response_loss": response_loss,
        "assignment_loss": margin_loss,
        "response_gain": alpha.mean(),
        "response_center_bias": beta.abs().mean(),
        "target_energy": target_energy.mean(),
    }


def _triplet_assignment(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Symmetric three-way future assignment normalized by target scale."""

    distances = (
        predicted.unsqueeze(2) - target.unsqueeze(1)
    ).square().mean(dim=-1)
    off_diagonal = ~torch.eye(
        3, dtype=torch.bool, device=predicted.device
    ).unsqueeze(0)
    target_distances = (
        target.unsqueeze(2) - target.unsqueeze(1)
    ).square().mean(dim=-1)
    target_scale = target_distances.masked_select(off_diagonal).reshape(
        target.size(0), -1
    ).mean(dim=1).clamp_min(MINIMUM_TARGET_ENERGY)
    normalized = distances / target_scale[:, None, None]
    labels = torch.arange(3, device=predicted.device).expand(
        target.size(0), -1
    )
    prediction_to_target = functional.cross_entropy(
        -normalized.reshape(-1, 3), labels.reshape(-1)
    )
    target_to_prediction = functional.cross_entropy(
        -normalized.transpose(1, 2).reshape(-1, 3), labels.reshape(-1)
    )
    loss = 0.5 * (prediction_to_target + target_to_prediction)
    matched = distances.diagonal(dim1=1, dim2=2).mean()
    counterfactual = distances.masked_select(off_diagonal).mean()
    return {
        "loss": loss,
        "prediction_to_target_loss": prediction_to_target,
        "target_to_prediction_loss": target_to_prediction,
        "matched_distance": matched,
        "counterfactual_distance": counterfactual,
        "assignment_margin": counterfactual - matched,
        "target_energy": target_scale.mean(),
    }


def conditional_joint_alignment(
    prediction: Any,
    target: Any,
    groups: Any,
) -> dict[str, torch.Tensor]:
    """Fit the history-conditioned joint relation for binary or triplet data.

    Binary groups use the center-free response-and-assignment objective that
    was validated on continuous hidden dynamics.  Three-way ActionDelay
    groups use symmetric conditional future assignment.  Targets are detached
    inside this auxiliary; native prediction MSE and SIGReg retain their
    ordinary encoder gradients.
    """

    predicted, detached_target, validated_groups = _validate(
        prediction, target, groups
    )
    if validated_groups.size(1) == 2:
        result = _binary_response_alignment(predicted, detached_target)
        kind = "binary_response"
    else:
        result = _triplet_assignment(predicted, detached_target)
        kind = "triplet_assignment"
    if not all(bool(value.isfinite().all()) for value in result.values()):
        raise FloatingPointError("conditional joint objective is nonfinite")
    return {
        **result,
        "group_count": torch.as_tensor(
            validated_groups.size(0),
            dtype=prediction.dtype,
            device=prediction.device,
        ),
        "group_width": torch.as_tensor(
            validated_groups.size(1),
            dtype=prediction.dtype,
            device=prediction.device,
        ),
        "kind_binary": torch.as_tensor(
            float(kind == "binary_response"),
            dtype=prediction.dtype,
            device=prediction.device,
        ),
    }


def conditional_joint_loss_terms(
    prediction: Any,
    target: Any,
    group_labels: Any,
    *,
    group_width: int,
) -> dict[str, torch.Tensor]:
    """Build consistently named COJA loss terms from batch row labels."""

    if not torch.is_tensor(prediction) or prediction.ndim < 1:
        raise TypeError("prediction must be a torch tensor")
    groups = conditional_joint_group_rows(
        group_labels,
        expected_width=group_width,
        batch_size=prediction.size(0),
        device=prediction.device,
    )
    result = conditional_joint_alignment(prediction, target, groups)
    return {
        "conditional_joint_loss": result["loss"],
        "conditional_joint_response_loss": result.get(
            "response_loss", result["loss"] * 0.0
        ),
        "conditional_joint_assignment_loss": result.get(
            "assignment_loss", result["loss"]
        ),
    }


__all__ = [
    "CANONICAL_BINARY_MARGIN",
    "CONDITIONAL_JOINT_BATCH_KEY",
    "MINIMUM_TARGET_ENERGY",
    "conditional_joint_alignment",
    "conditional_joint_group_rows",
    "conditional_joint_loss_terms",
    "temporary_eval_modules",
]
