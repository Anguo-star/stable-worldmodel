"""Canonical conditional-response objective without direct center regression."""

from __future__ import annotations

from typing import Any

import torch

from research.conditional_dynamics_representation.scripts import (
    canonical_margin_exact_future_v1 as canonical,
)


def canonical_response_only(
    prediction: Any,
    target: Any,
    groups: Any,
) -> dict[str, Any]:
    """Fit centered response plus the target-stationary assignment barrier.

    Native per-row MSE remains responsible for absolute future fit in the
    runner.  This paired auxiliary contains no normalized common-center MSE;
    the canonical barrier intentionally still depends on center displacement
    along the target response axis whenever that displacement breaks correct
    assignment.
    """

    result = canonical.canonical_margin_exact_future(
        prediction,
        target,
        groups,
    )
    response_loss = result["ccrm_normalized_error_by_group"].mean()
    loss = response_loss + result["canonical_margin_loss"]
    if not all(
        bool(value.isfinite().all())
        for value in (response_loss, result["canonical_margin_loss"], loss)
    ):
        raise FloatingPointError("nonfinite canonical response-only objective")
    return {
        **result,
        "loss": loss,
        "response_loss": response_loss,
        "direct_common_center_mse_included": False,
    }


__all__ = ["canonical_response_only"]
