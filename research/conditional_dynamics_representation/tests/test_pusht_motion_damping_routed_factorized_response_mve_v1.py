from __future__ import annotations

import torch
from torch import nn

from research.conditional_dynamics_representation.scripts import (
    run_pusht_motion_damping_routed_factorized_response_mve_v1 as runner,
)


class DummyPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.scale * (history + action)


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = DummyPredictor()

    def predict(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predictor(history, action)


def test_rank_covers_observed_response_rank_and_stays_small() -> None:
    branch = runner.factorized.FactorizedContextResponse(rank=runner.RANK)
    assert runner.RANK == 64
    assert runner.RANK >= 44
    assert branch.trainable_parameter_count == runner.ADDED_PARAMETER_COUNT == 36992


def test_step0_values_are_native_but_hidden_gradient_is_branch_only() -> None:
    torch.manual_seed(11)
    model = DummyModel()
    controller = runner.additive.EpisodeContextController("oracle")
    history = torch.randn(128, 3, 192, requires_grad=True)
    action = torch.randn(128, 3, 192, requires_grad=True)
    before = model.predict(history, action)
    branch = runner.attach_routed_factorized_response(model, controller, {})
    after = model.predict(history, action)
    assert torch.equal(before, after)
    target = torch.randn(64, 192)
    hidden_loss = (after[64:, -1] - target).square().mean()
    hidden_loss.backward()
    # torch.cat may materialize an exact-zero gradient for the unused native
    # prefix; both None and bitwise zero mean no hidden-loss contribution.
    assert model.predictor.scale.grad is None or torch.equal(
        model.predictor.scale.grad, torch.zeros_like(model.predictor.scale.grad)
    )
    assert history.grad is None or torch.equal(
        history.grad, torch.zeros_like(history.grad)
    )
    assert action.grad is None or torch.equal(
        action.grad, torch.zeros_like(action.grad)
    )
    assert branch.W_o.weight.grad is not None
    assert float(branch.W_o.weight.grad.norm()) > 0.0


def test_routed_loss_keeps_original_graph_and_detaches_hidden_target() -> None:
    prediction = torch.randn(128, 3, 192, requires_grad=True)
    embeddings = torch.randn(128, 4, 192, requires_grad=True)
    loss = runner.routed_prediction_loss(
        prediction=prediction,
        embeddings=embeddings,
        original_batch_size=64,
        conditional_population="identifiable_future_only",
    )
    loss.backward()
    assert float(embeddings.grad[:64].abs().sum()) > 0.0
    assert torch.equal(embeddings.grad[64:], torch.zeros_like(embeddings.grad[64:]))


def test_static_contract_and_cli_defaults() -> None:
    assert all(runner.validate_static_contract().values())
    args = runner.parse_args(
        ["--arm", "oracle", "--output", "/tmp/routed-factorized-dry", "--dry-run"]
    )
    assert args.seed == 14321
    assert args.optimizer_steps == 256
    assert args.dry_run is True
