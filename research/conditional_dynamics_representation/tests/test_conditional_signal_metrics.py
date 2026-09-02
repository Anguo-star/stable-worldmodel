from __future__ import annotations

import math

import pytest
import torch

from research.conditional_dynamics_representation.scripts import (
    conditional_signal_metrics as metrics,
)


def test_random_pair_identities_hold() -> None:
    generator = torch.Generator().manual_seed(17)
    prediction = torch.randn(31, 2, 3, 5, generator=generator)
    target = torch.randn(31, 2, 3, 5, generator=generator)
    parts = metrics.paired_signal_components(prediction, target)
    assert torch.allclose(
        parts["correct_loss"],
        parts["center_loss"] + parts["response_loss"],
        atol=2e-6,
        rtol=0,
    )
    assert torch.allclose(parts["g_swap"], parts["cross_energy"], atol=2e-6)


def test_zero_response_has_unit_nre_and_zero_g_swap() -> None:
    target = torch.tensor([[[0.0, 0.0], [2.0, -1.0]]])
    prediction = torch.zeros_like(target)
    summary = metrics.paired_signal_summary(prediction, target)
    geometry = summary["response_geometry"]
    assert geometry["gain"] == 0.0
    assert geometry["prediction_to_target_energy_ratio"] == 0.0
    assert geometry["normalized_response_error"] == 1.0
    assert summary["g_swap"]["distribution"]["mean"] == 0.0


def test_exact_response_has_zero_nre() -> None:
    target = torch.tensor(
        [[[1.0, 4.0], [2.0, 7.0]], [[-3.0, 0.0], [1.0, 2.0]]]
    )
    summary = metrics.paired_signal_summary(target.clone(), target)
    geometry = summary["response_geometry"]
    assert geometry["gain"] == pytest.approx(1.0)
    assert geometry["prediction_to_target_energy_ratio"] == pytest.approx(1.0)
    assert geometry["normalized_response_error"] == pytest.approx(0.0)
    assert geometry["orthogonal_residual"] == pytest.approx(0.0)
    assert geometry["scale_error"] == pytest.approx(0.0)


def test_seven_times_response_is_pure_scale_drift() -> None:
    target = torch.tensor([[[2.0, -1.0], [4.0, 3.0]]])
    center = target.mean(dim=1, keepdim=True)
    prediction = center + 7.0 * (target - center)
    geometry = metrics.paired_signal_summary(prediction, target)[
        "response_geometry"
    ]
    assert geometry["gain"] == pytest.approx(7.0)
    assert geometry["prediction_to_target_energy_ratio"] == pytest.approx(49.0)
    assert geometry["normalized_response_error"] == pytest.approx(36.0)
    assert geometry["orthogonal_residual"] == pytest.approx(0.0)
    assert geometry["scale_error"] == pytest.approx(36.0)


def test_rho_cond_matches_target_variance_decomposition() -> None:
    target = torch.tensor(
        [[[0.0], [2.0]], [[10.0], [14.0]], [[-4.0], [-2.0]]]
    )
    prediction = torch.zeros_like(target)
    variance = metrics.paired_signal_summary(prediction, target)[
        "target_variance"
    ]
    assert variance["total"] == pytest.approx(
        variance["within_pair_conditional"]
        + variance["between_pair_centers"]
    )
    assert variance["decomposition_absolute_error"] < 1e-12
    assert 0.0 < variance["rho_cond_data"] < 1.0
    assert metrics.paired_target_variance_summary(target) == variance


def test_gradient_population_aligned_and_opposed() -> None:
    aligned = metrics.gradient_population_summary(
        [(torch.tensor([1.0, 0.0]),), (torch.tensor([2.0, 0.0]),)]
    )["scopes"]["all"]
    assert aligned["coherence"] == pytest.approx(0.9)
    assert aligned["cancellation_ratio"] == pytest.approx(1.0)

    opposed = metrics.gradient_population_summary(
        [(torch.tensor([1.0, 0.0]),), (torch.tensor([-1.0, 0.0]),)]
    )["scopes"]["all"]
    assert opposed["mean_gradient_norm"] == 0.0
    assert opposed["coherence"] == 0.0
    assert opposed["critical_batch_size"] is None
    assert opposed["cancellation_ratio"] == 0.0


def test_gradient_none_slots_and_group_relation() -> None:
    samples = [
        (torch.tensor([1.0, 0.0]), None, torch.tensor([0.0, 1.0])),
        (torch.tensor([1.0, 0.0]), None, torch.tensor([0.0, 1.0])),
    ]
    report = metrics.gradient_population_summary(
        samples, parameter_groups=("predictor", "unused", "pred_proj")
    )
    assert set(report["scopes"]) == {
        "all",
        "predictor",
        "unused",
        "pred_proj",
    }
    assert report["scopes"]["unused"]["mean_gradient_norm"] == 0.0

    relation = metrics.gradient_relation_summary(
        samples[0],
        (torch.tensor([2.0, 0.0]), None, torch.tensor([0.0, -1.0])),
        parameter_groups=("predictor", "unused", "pred_proj"),
    )
    assert relation["scopes"]["predictor"]["cosine"] == pytest.approx(1.0)
    assert relation["scopes"]["pred_proj"]["cosine"] == pytest.approx(-1.0)
    assert math.isfinite(relation["scopes"]["all"]["cosine"])


def test_streaming_gradient_population_matches_materialized_summary() -> None:
    samples = [
        (torch.tensor([1.0, 2.0]), torch.tensor([-1.0])),
        (torch.tensor([3.0, -2.0]), None),
        (torch.tensor([-2.0, 1.0]), torch.tensor([4.0])),
    ]
    groups = ("predictor", "pred_proj")
    expected = metrics.gradient_population_summary(
        samples, parameter_groups=groups, batch_sizes=(1, 3, 32)
    )
    accumulator = metrics.GradientPopulationAccumulator(
        parameter_groups=groups, batch_sizes=(1, 3, 32)
    )
    for sample in samples:
        accumulator.add(sample)
    observed = accumulator.summary()
    for scope in expected["scopes"]:
        for key, value in expected["scopes"][scope].items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    assert observed["scopes"][scope][key][subkey] == pytest.approx(
                        subvalue
                    )
            else:
                assert observed["scopes"][scope][key] == pytest.approx(value)
    means = accumulator.mean_gradients()
    assert torch.allclose(
        means[0], torch.tensor([2.0 / 3.0, 1.0 / 3.0], dtype=torch.float64)
    )
    assert torch.allclose(means[1], torch.tensor([1.0], dtype=torch.float64))


@pytest.mark.parametrize(
    ("prediction", "target"),
    [
        (torch.zeros(2, 3, 4), torch.zeros(2, 3, 4)),
        (torch.zeros(2, 2, 4), torch.zeros(3, 2, 4)),
        (torch.tensor([[[float("nan")], [0.0]]]), torch.zeros(1, 2, 1)),
    ],
)
def test_bad_pair_inputs_are_rejected(
    prediction: torch.Tensor, target: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        metrics.paired_signal_summary(prediction, target)


def test_nonfinite_gradient_is_rejected() -> None:
    with pytest.raises(ValueError):
        metrics.gradient_population_summary(
            [(torch.tensor([float("inf")]),), (torch.tensor([0.0]),)]
        )
