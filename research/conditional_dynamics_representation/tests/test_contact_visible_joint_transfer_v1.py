from __future__ import annotations

import json
from pathlib import Path

import torch

from research.conditional_dynamics_representation.scripts import (
    canonical_response_only_v1 as center_free,
)
from research.conditional_dynamics_representation.scripts import (
    run_pusht_contact_friction_visible_joint_absolute_single_stage_step4096_v1
    as step4096,
)
from research.conditional_dynamics_representation.scripts import (
    run_pusht_contact_friction_visible_joint_absolute_single_stage_step8192_v1
    as step8192,
)
from research.conditional_dynamics_representation.scripts import (
    run_pusht_contact_friction_visible_joint_exact_future_single_stage_v1
    as exact,
)


ROOT = Path(__file__).resolve().parents[1]


def test_exact_center_arm_is_one_additive_term() -> None:
    prediction = torch.tensor(
        [[[2.0, 0.0]], [[2.0, 0.0]]], requires_grad=True
    )
    target = torch.tensor([[[-1.0, 0.0]], [[1.0, 0.0]]])
    groups = torch.tensor([[0, 1]], dtype=torch.long)

    without_center = center_free.canonical_response_only(
        prediction, target, groups
    )
    with_center = exact._exact_future_auxiliary(prediction, target, groups)
    center = with_center["normalized_common_center_mse_by_group"].mean()

    assert bool(center > 0)
    torch.testing.assert_close(
        with_center["loss"], without_center["loss"] + center
    )
    assert with_center["direct_common_center_mse_included"] is True


def test_budget_extensions_change_only_registered_step_identity() -> None:
    assert step4096.OPTIMIZER_STEPS == 4096
    assert step8192.OPTIMIZER_STEPS == 8192
    assert step4096.parent is step8192.parent
    assert step4096.CANDIDATE != step8192.CANDIDATE


def test_compact_summary_keeps_method_and_claim_boundaries() -> None:
    path = (
        ROOT
        / "artifacts/pusht_contact_friction_visible_joint_transfer_v1/summary.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    method = payload["method"]
    decision = payload["decision"]

    assert method["learned_parameters_added"] == 0
    assert method["model_modules_added"] == 0
    assert method["inference_compute_added"] == 0
    assert method["teacher"] is False
    assert method["hidden_label_used_by_model_or_loss"] is False
    assert decision["retained_discovery_checkpoint"] == "center_free_step4096"
    assert decision["formal_contact_release_gate_passed"] is False
    assert decision["public_test_opened"] is False
    assert decision["additional_seed_opened"] is False
    assert decision["stop_budget_or_schedule_search_after_8192"] is True
