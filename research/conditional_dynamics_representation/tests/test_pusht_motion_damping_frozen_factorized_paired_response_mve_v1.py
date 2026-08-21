from __future__ import annotations

import torch
from torch import nn

from research.conditional_dynamics_representation.scripts import (
    run_pusht_motion_damping_frozen_factorized_paired_response_mve_v1 as runner,
)


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 3)
        self.projector = nn.Linear(3, 3)
        self.predictor = nn.Sequential(nn.Dropout(0.5), nn.Linear(3, 3))
        self.action_encoder = nn.Linear(3, 3)
        self.pred_proj = nn.Linear(3, 3)


def test_native_model_is_frozen_and_forced_eval() -> None:
    model = DummyModel()
    receipt = runner.freeze_native_lewm_before_response_attachment(model)
    assert receipt["native_trainable_parameter_count"] == 0
    assert not any(parameter.requires_grad for parameter in model.parameters())
    model.train(True)
    assert model.training is True
    assert all(not getattr(model, name).training for name in runner.BASE_MODULE_NAMES)


def test_response_attached_after_freeze_is_only_trainable_module() -> None:
    model = DummyModel()
    runner.freeze_native_lewm_before_response_attachment(model)
    response = runner.routed.factorized.FactorizedContextResponse(rank=64)
    model.predictor.add_module("episode_context_response", response)
    model.train(True)
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    assert trainable
    assert all("episode_context_response" in name for name in trainable)
    assert sum(value.numel() for value in response.parameters()) == 36992
    assert response.training is True


def test_static_contract_and_cli_defaults() -> None:
    assert all(runner.validate_static_contract().values())
    args = runner.parse_args(
        ["--arm", "oracle", "--output", "/tmp/frozen-factorized-dry", "--dry-run"]
    )
    assert args.seed == 14321
    assert args.optimizer_steps == 256
    assert args.dry_run is True


def test_trajectory_addendum_authorizes_only_oracle_1024() -> None:
    checks = runner.validate_trajectory_addendum(
        runner.TRAJECTORY_ADDENDUM,
        arm="oracle",
        steps=1024,
    )
    assert all(checks.values())
