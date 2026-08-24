#!/usr/bin/env python3
"""Build a contact-rich real History x Action Motion overlay.

This is the single support-matched falsification of the zero-contact Cartesian
overlay.  The observed branch remains the frozen zero query action.  The
counterfactual branch uses a 0.45-scale action from the query agent position
toward the query block position.  Both hidden modes are simulated, so every
future is real and the deployed LeWM remains unchanged.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    build_pusht_motion_damping_cartesian_action_overlay_v1 as base,
)


ACTION_AMPLITUDE = 0.45
ACTION_BRANCHES = ("observed_zero", "toward_block_0p45")
PLAYFIELD_MIN = 5.0
PLAYFIELD_MAX = 506.0


def contact_query_actions(template: Any) -> np.ndarray:
    """Return five legal 0.45-scale actions aimed at the query block."""

    snapshot = np.asarray(
        template.expected_natural_query_snapshot, dtype=np.float64
    )
    if snapshot.shape != (12,) or not np.all(np.isfinite(snapshot)):
        raise ValueError("query snapshot must be finite 12-D state")
    direction = snapshot[6:8] - snapshot[0:2]
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("query agent and block positions must differ")
    action = ACTION_AMPLITUDE * direction / norm
    result = np.repeat(action[None, :], 5, axis=0)
    if (
        result.shape != (5, 2)
        or not np.all(np.isfinite(result))
        or float(np.max(np.abs(result))) > 1.0
        or not np.allclose(
            np.linalg.norm(result, axis=1), ACTION_AMPLITUDE
        )
    ):
        raise RuntimeError("contact action left the frozen action contract")
    return result


def _bounds_inside_playfield(bounds: dict[str, list[float]]) -> bool:
    return bool(
        all(
            box[0] >= PLAYFIELD_MIN
            and box[1] >= PLAYFIELD_MIN
            and box[2] <= PLAYFIELD_MAX
            and box[3] <= PLAYFIELD_MAX
            for box in bounds.values()
        )
    )


def main() -> int:
    native_simulate = base.simulate_motion_damping_clip
    native_build = base._build
    observations: dict[tuple[str, str, str], np.ndarray] = {}
    contact_steps: list[int] = []
    interaction_norms: list[float] = []
    model_bounds_inside: list[bool] = []

    def tracked_simulate(template: Any, *, mode: str, resolution: int) -> Any:
        result = native_simulate(template, mode=mode, resolution=resolution)
        query = np.asarray(template.query_actions, dtype=np.float64)
        branch = "contact" if bool(np.any(np.abs(query) > 1.0e-12)) else "zero"
        key = (str(template.template_id), branch, str(mode))
        if key in observations:
            raise RuntimeError(f"duplicate simulator branch {key}")
        observations[key] = np.asarray(
            result["natural_future_snapshot"], dtype=np.float64
        )
        if branch == "contact":
            contact_steps.append(int(result["query_contact_steps"]))
            model_bounds_inside.extend(
                _bounds_inside_playfield(bounds)
                for bounds in result["body_bounds"]
            )

        template_id = str(template.template_id)
        keys = [
            (template_id, action_branch, hidden_mode)
            for action_branch in ("zero", "contact")
            for hidden_mode in base.ENDPOINT_MODES
        ]
        if all(candidate in observations for candidate in keys):
            zero_delta = observations[keys[0]] - observations[keys[1]]
            contact_delta = observations[keys[2]] - observations[keys[3]]
            interaction_norms.append(
                float(np.linalg.norm(contact_delta - zero_delta))
            )
            for candidate in keys:
                del observations[candidate]
        return result

    def tracked_build(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        payload, receipt = native_build(**kwargs)
        template_count = int(kwargs["template_count"])
        expected_contact_rollouts = 2 * template_count
        if observations:
            raise RuntimeError("incomplete contact Cartesian simulator grid")
        if (
            len(contact_steps) != expected_contact_rollouts
            or len(interaction_norms) != template_count
            or len(model_bounds_inside) != 4 * expected_contact_rollouts
        ):
            raise RuntimeError("contact-support accounting changed")
        receipt.update(
            {
                "action_rule": "0p45_scale_unit_vector_from_query_agent_to_query_block",
                "action_amplitude": ACTION_AMPLITUDE,
                "minimum_alternate_query_contact_steps": int(
                    min(contact_steps)
                ),
                "mean_alternate_query_contact_steps": float(
                    np.mean(contact_steps)
                ),
                "all_alternate_model_bounds_inside_playfield": bool(
                    all(model_bounds_inside)
                ),
                "minimum_hidden_action_interaction_norm": float(
                    min(interaction_norms)
                ),
                "median_hidden_action_interaction_norm": float(
                    np.median(interaction_norms)
                ),
                "interaction_definition": "norm((future_low_a1-future_high_a1)-(future_low_a0-future_high_a0))",
            }
        )
        hard_checks = {
            "every_alternate_rollout_has_query_contact": (
                receipt["minimum_alternate_query_contact_steps"] >= 1
            ),
            "all_model_bounds_inside_playfield": receipt[
                "all_alternate_model_bounds_inside_playfield"
            ],
            "every_template_has_nonzero_hidden_action_interaction": (
                receipt["minimum_hidden_action_interaction_norm"] > 1.0e-8
            ),
        }
        receipt["contact_support_hard_checks"] = hard_checks
        if not all(hard_checks.values()):
            raise RuntimeError(f"contact-support gate failed: {hard_checks}")
        return payload, receipt

    base.THIS_SOURCE = THIS_SOURCE
    base.ACTION_BRANCHES = ACTION_BRANCHES
    base.alternate_query_actions = contact_query_actions
    base.simulate_motion_damping_clip = tracked_simulate
    base._build = tracked_build
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
