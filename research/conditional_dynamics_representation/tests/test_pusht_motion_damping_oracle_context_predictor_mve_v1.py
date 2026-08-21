from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from research.conditional_dynamics_representation.scripts import (
    run_pusht_motion_damping_oracle_context_predictor_mve_v1 as runner,
)


class DummyPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))

    def forward(self, history: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        return history + conditioning


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = DummyPredictor()


def test_context_tensors_have_exact_frozen_order_and_norm() -> None:
    oracle = runner.training_context(
        "oracle",
        batch_size=128,
        original_batch_size=64,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    constant = runner.training_context(
        "constant",
        batch_size=128,
        original_batch_size=64,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.equal(oracle[:64], torch.zeros(64, 2))
    assert torch.equal(oracle[64::2], torch.tensor([[1.0, 0.0]]).expand(32, -1))
    assert torch.equal(oracle[65::2], torch.tensor([[0.0, 1.0]]).expand(32, -1))
    assert torch.equal(constant[:64], torch.zeros(64, 2))
    assert torch.allclose(constant[64:].norm(dim=1), torch.ones(64))
    assert torch.equal(constant[64::2], constant[65::2])


def test_explicit_evaluation_context_never_uses_a_cursor() -> None:
    controller = runner.EpisodeContextController("oracle")
    conditioning = torch.zeros(7, 3, 192)
    with controller.evaluating("faster_decay"):
        first = controller.for_conditioning(conditioning)
        second = controller.for_conditioning(conditioning[:2])
    assert torch.equal(first, torch.tensor([[1.0, 0.0]]).expand(7, -1))
    assert torch.equal(second, torch.tensor([[1.0, 0.0]]).expand(2, -1))
    assert controller.evaluation_condition is None
    with controller.evaluating("no_extra_decay"):
        high = controller.for_conditioning(conditioning[:3])
    assert torch.equal(high, torch.tensor([[0.0, 1.0]]).expand(3, -1))


@pytest.mark.parametrize("arm", ["oracle", "constant"])
def test_zero_start_is_exact_and_optimizer_sees_all_384_parameters(arm: str) -> None:
    torch.manual_seed(0)
    model = DummyModel()
    history = torch.randn(128, 3, 192)
    conditioning = torch.randn(128, 3, 192)
    before = model.predictor(history, conditioning)
    weight = runner.attach_episode_context_projection(
        model, runner.EpisodeContextController(arm), {}
    )
    after = model.predictor(history, conditioning)
    assert weight.numel() == 384
    assert weight.requires_grad
    assert torch.equal(weight, torch.zeros_like(weight))
    assert torch.equal(before, after)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert id(weight) in optimizer_ids


def test_screen_thresholds_are_strict_where_registered() -> None:
    metrics = {
        "two_real_future_target_selection_rate": 0.55,
        "correct_history_preference_rate": 0.55,
        "correct_rule_switch_rate": 0.75,
        "worst_mode_target_selection_rate": 0.10,
        "latent_response": {
            "response_gain": 0.10,
            "normalized_response_error": np.nextafter(0.95, 0.0),
            "target_latent_separation": {"zero_separation_pair_count": 0},
        },
    }
    bootstrap = {
        "paired_query_balanced_macro": {"lower_95": 0.50},
        "target_energy_weighted_normalized_response_error": {
            "upper_95": np.nextafter(1.0, 0.0)
        },
    }
    assert runner.screen_decision(metrics, bootstrap)["passed"]
    metrics = copy.deepcopy(metrics)
    metrics["latent_response"]["normalized_response_error"] = 0.95
    assert not runner.screen_decision(metrics, bootstrap)["passed"]
    metrics["latent_response"]["normalized_response_error"] = 0.94
    bootstrap["target_energy_weighted_normalized_response_error"]["upper_95"] = 1.0
    assert not runner.screen_decision(metrics, bootstrap)["passed"]


def test_release_overlay_is_narrow_and_in_memory() -> None:
    base = {
        "release_id": runner.EXPECTED_RELEASE_ID,
        "training": {
            "reference_matrix": {
                "status": "failed_development",
                "completed_development_seeds": [14321, 99999],
            }
        },
    }
    motion = SimpleNamespace(load_motion_damping_icl_release=lambda _path: base)
    audit = runner.install_release_authorization_overlay(
        motion, arm="oracle", steps=256
    )
    overlaid = motion.load_motion_damping_icl_release(Path("unused"))
    assert base["training"]["reference_matrix"]["completed_development_seeds"] == [14321, 99999]
    assert overlaid["training"]["reference_matrix"]["completed_development_seeds"] == [99999]
    assert overlaid["training"]["learnability_followup"]["public_test_open"] is False
    assert audit["base_release_mutated_on_disk"] is False


def test_oracle_disclosure_and_constant_control_boundary_are_explicit() -> None:
    prereg = runner.yaml.safe_load(runner.PREREG.read_text(encoding="utf-8"))
    assert prereg["arms"]["oracle"]["privileged_label_diagnostic_only"] is True
    assert prereg["arms"]["constant"]["contains_condition_identity"] is False
    assert prereg["objective"]["additional_loss"] is None
    assert prereg["evidence_boundary"]["public_test_open"] is False
    assert prereg["evidence_boundary"]["cem_open"] is False


def test_cli_defaults_and_dry_run_surface() -> None:
    args = runner.parse_args(
        ["--arm", "oracle", "--output", "/tmp/oracle-context-dry", "--dry-run"]
    )
    assert args.arm == "oracle"
    assert args.seed == 14321
    assert args.optimizer_steps == 256
    assert args.dry_run is True
    checks = runner.validate_static_contract()
    assert all(checks.values())
