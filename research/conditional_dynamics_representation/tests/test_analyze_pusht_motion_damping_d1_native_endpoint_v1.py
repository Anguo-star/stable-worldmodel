from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from research.conditional_dynamics_representation.scripts import (
    analyze_pusht_motion_damping_d1_native_endpoint_v1 as endpoint,
)


def _diagnostic(*, checkpoint: str = "checkpoint", public: bool = False) -> dict:
    records = [
        {
            "pair_id": f"pair-{index}",
            "g_swap": float(index - 1),
            "response_gain": float(index) / 10.0,
            "prediction_delta_energy": 2.0 + index,
            "target_delta_energy": 2.0,
        }
        for index in range(4)
    ]
    return {
        "schema_version": 1,
        "status": "completed_zero_training_development_diagnostic",
        "data": {
            "public_test_opened": public,
            "description": {
                "split": "Development",
                "split_name": "loader_validation",
                "pair_count": 4,
            },
            "development_identity": {
                "split": "loader_validation",
                "pair_count": 4,
                "passed": True,
                "data_manifest_sha256": endpoint.EXPECTED_MANIFEST_SHA256,
                "observed_data_manifest_sha256": endpoint.EXPECTED_MANIFEST_SHA256,
                "lance_table": "loader_validation.lance",
                "lance_table_sha256": endpoint.EXPECTED_LANCE_SHA256,
                "observed_lance_table_sha256": endpoint.EXPECTED_LANCE_SHA256,
            },
        },
        "model": {
            "checkpoint_sha256": checkpoint,
            "adapter": {
                "checkpoint_sha256": checkpoint,
                "protocol": {"native_target_encoder": True},
            },
        },
        "source": {"sha256": endpoint.EXPECTED_DIAGNOSTIC_SOURCE_SHA256},
        "random_pairing_baselines": {
            "within_pair_sign_flip": {"two_sided_monte_carlo_p": 0.01},
            "cross_query_pairing": {"one_sided_monte_carlo_p": 0.01},
        },
        "records": records,
    }


def _result(*, checkpoint: str = "checkpoint") -> dict:
    return {
        "metrics": {
            "correct_history_preference_rate": 0.5,
            "correct_rule_switch_rate": 0.5,
            "two_real_future_target_selection_rate": 0.5,
            "worst_mode_target_selection_rate": 0.5,
            "latent_response": {
                "response_gain": 0.1,
                "normalized_response_error": 0.9,
                "aggregate_cosine_alignment": 0.2,
                "mean_pair_cosine_alignment": 0.1,
            },
        },
        "checkpoint_sha256": checkpoint,
    }


@pytest.fixture
def small_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(endpoint, "PAIR_COUNT", 4)


def test_statistics_are_deterministic_and_clustered_by_pair(small_catalog: None) -> None:
    values = np.asarray([1.0, -1.0, 2.0, 4.0])
    first = endpoint.paired_bootstrap(values, seed=17, replicates=500)
    second = endpoint.paired_bootstrap(values, seed=17, replicates=500)
    assert first == second
    sign_first = endpoint.paired_sign_flip(values, seed=17, replicates=500)
    assert sign_first == endpoint.paired_sign_flip(values, seed=17, replicates=500)
    clustered = endpoint.paired_bootstrap(
        values,
        seed=17,
        replicates=500,
        cluster=["a", "a", "b", "b"],
    )
    assert clustered["cluster"] == "pair"
    assert clustered["statistics"]["mean"]["point"] == 1.5


def test_hash_guard_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "payload.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setitem(endpoint.EXPECTED_INPUT_SHA256, "fixture", "bad")
    with pytest.raises(RuntimeError, match="hash changed"):
        endpoint._load_frozen("fixture", path)
    monkeypatch.setitem(endpoint.EXPECTED_INPUT_SHA256, "fixture", hashlib.sha256(path.read_bytes()).hexdigest())
    assert endpoint._load_frozen("fixture", path) == {}


def test_public_test_identity_is_rejected(small_catalog: None) -> None:
    with pytest.raises(RuntimeError, match="Public Test"):
        endpoint._record_arrays(_diagnostic(public=True), "fixture")


def test_pair_order_is_rejected(small_catalog: None) -> None:
    payloads = {
        "d0_diagnostic": _diagnostic(),
        "coja_diagnostic": _diagnostic(),
        "d1_diagnostic": _diagnostic(),
        "d0_result": _result(),
        "coja_result": _result(),
        "d1_result": _result(),
    }
    payloads["d1_diagnostic"]["records"][0]["pair_id"] = "pair-reordered"
    with pytest.raises(RuntimeError, match="pair_id/order"):
        endpoint.analyze(payloads)


def test_analyze_reports_directional_effect_and_not_measured_history_ablation(
    small_catalog: None,
) -> None:
    d0 = _diagnostic()
    d1 = copy.deepcopy(d0)
    for record in d1["records"]:
        record["g_swap"] += 0.5
        record["response_gain"] += 0.5
        record["prediction_delta_energy"] = 1.0
    payloads = {
        "d0_diagnostic": d0,
        "coja_diagnostic": _diagnostic(),
        "d1_diagnostic": d1,
        "d0_result": _result(),
        "coja_result": _result(),
        "d1_result": _result(),
    }
    report = endpoint.analyze(payloads)
    assert report["decision"]["removed_history"] == "not_measured"
    assert report["deltas"]["g_swap"]["point"]["mean"] == 0.5
    assert report["deltas"]["response_gain"]["point"]["mean"] == 0.5
