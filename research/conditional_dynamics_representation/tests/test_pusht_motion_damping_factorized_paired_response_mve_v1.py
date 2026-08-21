from __future__ import annotations

import torch

from research.conditional_dynamics_representation.scripts import (
    run_pusht_motion_damping_factorized_paired_response_mve_v1 as runner,
)


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    prediction = torch.zeros(128, 3, 4, requires_grad=True)
    embeddings = torch.zeros(128, 4, 4)
    target_low = torch.tensor([0.0, 0.0, 0.0, 0.0])
    target_high = torch.tensor([1.0, -1.0, 2.0, -2.0])
    embeddings[64::2, -1] = target_low
    embeddings[65::2, -1] = target_high
    return prediction, embeddings


def test_paired_response_is_zero_for_exact_response_despite_arbitrary_center() -> None:
    prediction, embeddings = _batch()
    center = torch.tensor([5.0, -3.0, 2.0, 7.0])
    response = embeddings[65, -1] - embeddings[64, -1]
    with torch.no_grad():
        prediction[64::2, -1] = center
        prediction[65::2, -1] = center + response
    loss, metrics = runner.paired_response_loss(prediction, embeddings)
    assert torch.equal(loss, torch.zeros_like(loss))
    assert float(metrics["positive_alignment_rate"]) == 1.0


def test_paired_response_has_direct_nonzero_gradient() -> None:
    prediction, embeddings = _batch()
    loss, _ = runner.paired_response_loss(prediction, embeddings)
    loss.backward()
    assert float(prediction.grad[64:].norm()) > 0.0
    assert torch.equal(prediction.grad[:64], torch.zeros_like(prediction.grad[:64]))


def test_constant_center_shift_cancels_from_pair_loss() -> None:
    prediction, embeddings = _batch()
    shifted = prediction.detach().clone().requires_grad_(True)
    with torch.no_grad():
        prediction[64:, -1] += 9.0
        shifted[64:, -1] -= 4.0
    first, _ = runner.paired_response_loss(prediction, embeddings)
    second, _ = runner.paired_response_loss(shifted, embeddings)
    assert torch.equal(first, second)


def test_static_contract_and_cli_defaults() -> None:
    assert runner.RESPONSE_WEIGHT == 1.0
    assert all(runner.validate_static_contract().values())
    args = runner.parse_args(
        ["--arm", "oracle", "--output", "/tmp/factorized-paired-dry", "--dry-run"]
    )
    assert args.seed == 14321
    assert args.optimizer_steps == 256
    assert args.dry_run is True


def test_shared_trainer_trace_field_is_loss_trace() -> None:
    source = runner.Path(runner.__file__).read_text(encoding="utf-8")
    assert 'result["loss_trace"]' in source
    assert 'result["trace"]' not in source
