from __future__ import annotations

import torch
from torch import nn

from research.conditional_dynamics_representation.scripts import (
    run_pusht_motion_damping_factorized_context_response_mve_v1 as runner,
)


class DummyPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return history + action


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = DummyPredictor()

    def predict(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predictor(history, action)


def test_factorized_branch_has_expected_size_and_exact_zero_start() -> None:
    branch = runner.FactorizedContextResponse()
    assert branch.trainable_parameter_count == 4624
    history = torch.randn(5, 3, 192)
    action = torch.randn(5, 3, 192)
    context = torch.tensor([[1.0, 0.0]]).expand(5, -1)
    assert torch.equal(branch(history, action, context), torch.zeros(5, 192))


def test_attached_step0_is_bitwise_native_and_optimizer_includes_branch() -> None:
    torch.manual_seed(5)
    model = DummyModel()
    history = torch.randn(128, 3, 192)
    action = torch.randn(128, 3, 192)
    before = model.predict(history, action)
    branch = runner.attach_factorized_context_response(
        model, runner.additive.EpisodeContextController("oracle"), {}
    )
    after = model.predict(history, action)
    assert torch.equal(before, after)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(id(parameter) in optimizer_ids for parameter in branch.parameters())


def test_original_rows_are_structurally_zero_even_after_branch_changes() -> None:
    torch.manual_seed(7)
    branch = runner.FactorizedContextResponse()
    nn.init.normal_(branch.W_o.weight)
    history = torch.randn(64, 3, 192)
    action = torch.randn(64, 3, 192)
    zero_context = torch.zeros(64, 2)
    assert torch.equal(branch(history, action, zero_context), torch.zeros(64, 192))


def test_oracle_and_constant_use_same_parameterized_branch() -> None:
    oracle = runner.FactorizedContextResponse()
    constant = runner.FactorizedContextResponse()
    assert oracle.trainable_parameter_count == constant.trainable_parameter_count
    assert set(oracle.state_dict()) == set(constant.state_dict())
    assert runner.ARMS.keys() == {"oracle", "constant"}


def test_static_contract_and_cli_defaults() -> None:
    assert runner.ADDED_PARAMETER_COUNT == 4624
    assert all(runner.validate_static_contract().values())
    args = runner.parse_args(
        ["--arm", "constant", "--output", "/tmp/factorized-context-dry", "--dry-run"]
    )
    assert args.seed == 14321
    assert args.optimizer_steps == 256
    assert args.dry_run is True
