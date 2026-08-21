from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_factorized_residual_mve_v1 as runner,
)


def _history(batch: int = 6, latent_dim: int = 5, action_dim: int = 3):
    torch.manual_seed(19)
    return torch.randn(batch, 3, latent_dim), torch.randn(batch, 3, action_dim)


def test_factorized_residual_shape_and_zero_initial_output() -> None:
    latents, actions = _history()
    head = runner.FactorizedResidualHead(
        latent_dim=latents.shape[-1],
        action_dim=actions.shape[-1],
        rank=3,
        variant="learned",
    )
    output = head(latents, actions)
    assert output.shape == (latents.shape[0], latents.shape[-1])
    assert torch.equal(output, torch.zeros_like(output))


def test_oracle_mlp_has_exact_zero_start_and_output_shape() -> None:
    latents, actions = _history(batch=4, latent_dim=5, action_dim=3)
    labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    head = runner.OracleMLPResidualHead(
        latent_dim=latents.shape[-1],
        action_dim=actions.shape[-1],
        hidden_dim=11,
    )

    output = head(latents, actions, labels)

    assert output.shape == (latents.shape[0], latents.shape[-1])
    assert torch.equal(output, torch.zeros_like(output))
    assert torch.equal(head.fc2.weight, torch.zeros_like(head.fc2.weight))
    assert torch.equal(head.fc2.bias, torch.zeros_like(head.fc2.bias))
    assert head.fc1.in_features == head.query_dim + 2
    assert head.fc1.out_features == 11


def test_oracle_mlp_requires_labels() -> None:
    latents, actions = _history()
    head = runner.OracleMLPResidualHead(
        latent_dim=latents.shape[-1],
        action_dim=actions.shape[-1],
        hidden_dim=7,
    )

    with pytest.raises(ValueError, match="requires damping labels"):
        head(latents, actions)


def test_oracle_mlp_parameters_are_trainable_and_cli_default_is_exposed() -> None:
    latents, actions = _history(latent_dim=4, action_dim=2)
    head = runner.OracleMLPResidualHead(
        latent_dim=latents.shape[-1],
        action_dim=actions.shape[-1],
        hidden_dim=13,
    )

    assert all(parameter.requires_grad for parameter in head.parameters())
    assert head.trainable_parameter_count == sum(
        parameter.numel() for parameter in head.parameters()
    )
    args = runner.parse_args(["--variant", "oracle_mlp", "--output", "report.json"])
    assert args.mlp_hidden_dim == 256


def test_legacy_factorized_variants_keep_their_label_semantics() -> None:
    latents, actions = _history(batch=4, latent_dim=4, action_dim=2)
    labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    learned = runner.FactorizedResidualHead(
        latent_dim=latents.shape[-1],
        action_dim=actions.shape[-1],
        rank=3,
        variant="learned",
    )
    oracle = runner.FactorizedResidualHead(
        latent_dim=latents.shape[-1],
        action_dim=actions.shape[-1],
        rank=3,
        variant="oracle",
    )

    assert learned.context_dim == 2 * latents.shape[-1] + 2 * actions.shape[-1]
    assert oracle.context_dim == 2
    assert torch.equal(learned(latents, actions), torch.zeros(4, 4))
    assert torch.equal(oracle(latents, actions, labels), torch.zeros(4, 4))
    with pytest.raises(ValueError, match="cannot consume oracle labels"):
        learned(latents, actions, labels)
    with pytest.raises(ValueError, match="requires damping labels"):
        oracle(latents, actions)
    with pytest.raises(ValueError, match="variant"):
        runner.FactorizedResidualHead(
            latent_dim=latents.shape[-1],
            action_dim=actions.shape[-1],
            rank=3,
            variant="oracle_mlp",
        )


def test_context_swap_changes_output_after_output_weights_are_nonzero() -> None:
    latents, actions = _history()
    head = runner.FactorizedResidualHead(
        latent_dim=latents.shape[-1],
        action_dim=actions.shape[-1],
        rank=4,
        variant="learned",
    )
    with torch.no_grad():
        head.W_q.weight.fill_(0.07)
        head.W_c.weight.fill_(0.11)
        head.W_o.weight.copy_(torch.arange(head.latent_dim * head.rank, dtype=torch.float32).reshape(head.latent_dim, head.rank) / 10.0)
    context = head.learned_context(latents, actions)
    original = head(latents, actions, context=context)
    swapped = head(latents, actions, context=context.flip(0))
    assert not torch.allclose(original, swapped)


def test_oracle_context_label_ordering_is_explicit() -> None:
    labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    expected = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
    )
    assert torch.equal(runner.oracle_context(labels), expected)
    assert runner.ORACLE_CLASS_NAMES == (
        "faster_decay_0p2",
        "no_extra_decay_1p0",
    )


def test_known_rank_response_svd_reports_entropy_and_energy_ranks() -> None:
    response = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    diagnostics = runner.target_response_svd_diagnostics(response)
    assert diagnostics["entropy_effective_rank"] == pytest.approx(2.0, abs=1e-6)
    assert diagnostics["r90"] == 2
    assert diagnostics["r95"] == 2
    assert diagnostics["r99"] == 2


def test_ridge_probe_folds_isolate_forward_reverse_twin_groups() -> None:
    latents, actions = _history(batch=12, latent_dim=3, action_dim=2)
    labels = torch.tensor([0, 1, 0, 1] * 3, dtype=torch.long)
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    first = runner.leave_forward_reverse_twin_group_out_ridge_probes(
        latents, actions, labels, group_ids=groups, seed=3073
    )
    second = runner.leave_forward_reverse_twin_group_out_ridge_probes(
        latents, actions, labels, group_ids=groups, seed=3073
    )
    assert first == second
    assert first["fold_isolation_passed"] is True
    for fold in first["folds"]:
        assert set(fold["train_indices"]).isdisjoint(fold["held_out_indices"])
        assert set(fold["train_groups"]).isdisjoint(fold["held_out_groups"])
        assert fold["group_isolation_passed"] is True


def test_query_only_probe_excludes_history_differences() -> None:
    latents, actions = _history(batch=4, latent_dim=3, action_dim=2)
    first = runner.make_probe_features(latents, actions)["query_only"]
    changed = latents.clone()
    changed[:, :2] += 100.0
    second = runner.make_probe_features(changed, actions)["query_only"]
    assert torch.equal(first, second)
    assert first.shape[-1] == latents.shape[-1] + actions.shape[-1]
