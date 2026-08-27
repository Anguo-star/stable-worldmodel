"""Proper exact-future fit with a canonical binary assignment barrier.

The exact-future term has its unique optimum at the two matched targets.  The
additional squared hinge supplies a finite assignment gradient while the two
predictions are history-independent, but is exactly zero once both normalized
target-cell margins reach the margins of the real targets themselves (1/2).
Consequently it does not keep pushing the response past the real target pair.
"""

from __future__ import annotations

from typing import Any

import torch

from research.conditional_dynamics_representation.scripts.pair_normalized_exact_future_residual_v1 import (
    pair_normalized_exact_future_residual as _base_exact_future_residual,
)


CANONICAL_ASSIGNMENT_MARGIN = 0.5


def canonical_margin_exact_future(
    prediction: Any,
    target: Any,
    groups: Any,
) -> dict[str, Any]:
    """Return normalized exact-future fit plus a target-stationary barrier.

    For binary matched targets ``t0,t1``, let ``alpha`` be predicted response
    gain along ``t1-t0`` and ``beta`` the normalized common-center error along
    that axis.  The two correct-assignment margins are ``alpha/2 +/- beta``.
    The real targets have ``alpha=1,beta=0`` and hence margin ``1/2`` on both
    sides.  Penalizing only deficits below that canonical value gives a strong
    gradient at ``alpha=beta=0`` while remaining zero at exact prediction.
    """

    result = _base_exact_future_residual(
        prediction,
        target,
        groups,
    )
    if result["group_width"] != 2:
        raise ValueError("canonical-margin exact future requires binary groups")

    predicted = prediction[groups, -1].float()
    detached_target = target[groups, -1].detach().float()
    target_axis = detached_target[:, 1] - detached_target[:, 0]
    prediction_axis = predicted[:, 1] - predicted[:, 0]
    target_axis_squared_norm = target_axis.square().sum(dim=-1)
    if bool((target_axis_squared_norm <= 0.0).any()):
        raise ValueError("canonical-margin target axes must be nonzero")

    prediction_center = predicted.mean(dim=1)
    target_center = detached_target.mean(dim=1)
    alpha_by_group = (
        (prediction_axis * target_axis).sum(dim=-1)
        / target_axis_squared_norm
    )
    beta_by_group = (
        ((prediction_center - target_center) * target_axis).sum(dim=-1)
        / target_axis_squared_norm
    )
    assignment_margins = torch.stack(
        (
            0.5 * alpha_by_group - beta_by_group,
            0.5 * alpha_by_group + beta_by_group,
        ),
        dim=1,
    )
    margin_deficits = torch.relu(
        float(CANONICAL_ASSIGNMENT_MARGIN) - assignment_margins
    )
    canonical_margin_loss_by_group = margin_deficits.square().mean(dim=1)
    canonical_margin_loss = canonical_margin_loss_by_group.mean()
    loss = result["loss"] + canonical_margin_loss

    finite = (
        target_axis,
        prediction_axis,
        alpha_by_group,
        beta_by_group,
        assignment_margins,
        margin_deficits,
        canonical_margin_loss_by_group,
        canonical_margin_loss,
        loss,
    )
    if not all(bool(value.isfinite().all()) for value in finite):
        raise FloatingPointError("nonfinite canonical-margin exact-future value")

    return {
        **result,
        "loss": loss,
        "exact_future_loss": result["loss"],
        "canonical_margin_loss": canonical_margin_loss,
        "canonical_margin_loss_by_group": canonical_margin_loss_by_group,
        "assignment_margins": assignment_margins,
        "margin_deficits": margin_deficits,
        "both_canonical_margins_satisfied_by_group": (
            assignment_margins >= float(CANONICAL_ASSIGNMENT_MARGIN)
        ).all(dim=1),
        "alpha_by_group": alpha_by_group,
        "beta_by_group": beta_by_group,
        "canonical_assignment_margin": CANONICAL_ASSIGNMENT_MARGIN,
    }


__all__ = [
    "CANONICAL_ASSIGNMENT_MARGIN",
    "canonical_margin_exact_future",
]
