import math

import numpy as np
import pytest
import torch

from research.conditional_dynamics_representation.scripts import (
    analyze_pusht_motion_damping_d1_latent_gradient_zero_step_v1 as audit,
)


def test_audit_schedule_indices_are_frozen_midpoints() -> None:
    report = audit.audit_schedule_indices()
    assert report["indices"] == [256 + 512 * index for index in range(16)]
    assert all(report["checks"].values())


def test_validate_twin_rows_accepts_complete_unique_twins() -> None:
    rows = torch.tensor(
        [4 * twin + offset for twin in range(16) for offset in range(4)]
    )
    assert audit.validate_twin_rows(rows) == list(range(16))


@pytest.mark.parametrize(
    "rows",
    [
        torch.arange(63),
        torch.tensor([value for twin in range(16) for value in [4 * twin] * 4]),
        torch.tensor(
            [4 * (twin % 15) + offset for twin in range(16) for offset in range(4)]
        ),
    ],
)
def test_validate_twin_rows_fails_closed(rows: torch.Tensor) -> None:
    with pytest.raises((RuntimeError, ValueError)):
        audit.validate_twin_rows(rows)


def test_device_gradient_accumulator_reports_population_snr() -> None:
    parameters = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(1))]
    accumulator = audit.DeviceGradientAccumulator(
        parameters, ["predictor", "pred_proj"]
    )
    accumulator.add((torch.tensor([1.0, 0.0]), torch.tensor([1.0])))
    accumulator.add((torch.tensor([1.0, 0.0]), torch.tensor([-1.0])))
    report = accumulator.summary(snr_batch_sizes=(1, 2))["scopes"]
    assert report["predictor"]["critical_batch_size"] == 0.0
    assert report["pred_proj"]["mean_gradient_norm"] == 0.0
    assert report["all"]["mean_gradient_norm"] == 1.0
    assert math.isclose(report["all"]["critical_batch_size"], 1.0)
    assert math.isclose(report["all"]["snr_by_batch_size"]["2"], math.sqrt(2.0))


def test_gradient_relation_uses_population_means() -> None:
    parameters = [torch.nn.Parameter(torch.zeros(2))]
    left = audit.DeviceGradientAccumulator(parameters, ["predictor"])
    right = audit.DeviceGradientAccumulator(parameters, ["predictor"])
    left.add((torch.tensor([1.0, 0.0]),))
    left.add((torch.tensor([1.0, 0.0]),))
    right.add((torch.tensor([0.0, 1.0]),))
    report = audit.gradient_relation(left, right)
    assert report["all"]["cosine"] == 0.0
    assert report["predictor"]["left_mean_gradient_norm"] == 1.0


def test_latent_local_energy_reuses_supplied_graph(monkeypatch) -> None:
    monkeypatch.setattr(audit, "EXPECTED_TWINS", 2)
    monkeypatch.setattr(audit, "EXPECTED_PAIRS", 4)
    monkeypatch.setattr(audit, "LATENT_NEIGHBOUR_SCALES", (1,))
    # Rows are [pair, mode].  D1 emphasizes twin 1, whose response is larger.
    query = torch.zeros(8, 1)
    future = torch.tensor([[0.0], [2.0], [1.0], [3.0], [0.0], [4.0], [1.0], [5.0]])
    neighbours = np.asarray([[2], [3], [0], [1]], dtype=np.int64)
    report = audit.latent_local_energy(
        query,
        future,
        neighbours,
        {
            "D0": np.asarray([0.5, 0.5]),
            "D1-MS50": np.asarray([0.25, 0.75]),
        },
    )
    d0 = report["arms"]["D0"]["by_k"]["1"]
    d1 = report["arms"]["D1-MS50"]["by_k"]["1"]
    assert d1["weighted_conditional_energy"] > d0["weighted_conditional_energy"]
    assert report["physical_neighbour_graph_reused"] is True


def _fake_gradient(norm_delta: float | None, snr_delta: float | None) -> dict:
    return {
        "batch_count_per_arm": 16,
        "twin_gradient_units_per_arm": 256,
        "comparison": {
            "all": {
                "response_mean_gradient_norm_relative_delta": norm_delta,
                "response_snr16_relative_delta": snr_delta,
            }
        },
        "state_restoration": {
            "parameters_unchanged": True,
            "parameter_grad_slots_remain_none": True,
            "buffers_restored": True,
            "module_modes_restored": True,
        },
    }


def _fake_latent(delta: float) -> dict:
    return {
        "comparison": {
            str(scale): {"rho_lat_absolute_delta": delta}
            for scale in (32, 64, 128)
        }
    }


def test_decision_requires_latent_and_gradient_visibility() -> None:
    passed = audit._decision(_fake_latent(0.01), _fake_gradient(-0.1, 0.1))
    assert passed["status"] == "passed_go_for_single_d1_native_training"
    assert all(passed["checks"].values())

    failed_latent = audit._decision(_fake_latent(0.0), _fake_gradient(0.1, 0.1))
    assert failed_latent["status"] == "failed_no_go"

    failed_gradient = audit._decision(_fake_latent(0.01), _fake_gradient(-0.1, -0.1))
    assert failed_gradient["status"] == "failed_no_go"


def test_static_identity_is_training_only() -> None:
    report = audit.static_identity(
        checkpoint=audit.DEFAULT_CHECKPOINT,
        catalog=audit.DEFAULT_CATALOG,
        multiplicity=audit.DEFAULT_MULTIPLICITY,
    )
    assert report["catalog"]["directed_neighbour_shape"] == [8192, 128]
    assert report["multiplicity"]["positive_support"] is True
    assert report["authority"] == {
        "training_only": True,
        "optimizer_steps": 0,
        "development_opened": False,
        "public_test_opened": False,
        "full_training_authorized": False,
    }
