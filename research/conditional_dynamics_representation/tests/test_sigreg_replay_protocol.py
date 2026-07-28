from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_sigreg_replay_protocol.py"
)
SPEC = importlib.util.spec_from_file_location("sigreg_replay_protocol", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _training_base() -> dict:
    return {
        "schema_version": 1,
        "benchmark": "base_training",
        "status": "frozen",
        "data": {},
        "training_protocol": {
            "synthetic_only": True,
            "paired_training_seeds": [3072, 4096, 5120],
            "group_sampling": {},
            "fairness_contract": {},
        },
        "models": [],
        "comparison": {},
    }


def _validation_base() -> dict:
    return {
        "schema_version": 2,
        "benchmark": "tworoom_hidden_passage_history3_validation_v2",
        "status": "frozen",
        "comparison": {
            "required_results": {},
            "checkpoint_training_model_id": {},
        },
        "training_provenance": {"original_baseline": {}},
    }


def test_training_protocol_has_exact_replay_and_one_intended_difference(
    tmp_path: Path,
) -> None:
    protocol = MODULE.build_training_protocol(
        _training_base(), source_path=tmp_path / "base.yaml"
    )

    assert protocol["training_protocol"]["synthetic_only"] is False
    assert protocol["training_protocol"]["paired_training_seeds"] == [3072]
    assert len(protocol["models"]) == 3
    for model in protocol["models"]:
        model_id = model["model_id"]
        assert model["training_groups"] == ["original", "passage_mixed"]
        assert protocol["training_protocol"]["group_sampling"][model_id] == {
            "original": 0.5,
            "passage_mixed": 0.5,
        }
    assert (
        protocol["training_protocol"]["fairness_contract"][
            "intentional_difference"
        ]
        == "lewm_sigreg_weight_only"
    )


def test_validation_protocol_binds_replay_and_noninferiority_gate(
    tmp_path: Path,
) -> None:
    training_path = tmp_path / "training.yaml"
    protocol = MODULE.build_validation_protocol(
        _validation_base(),
        source_path=tmp_path / "validation-base.yaml",
        training_protocol_path=training_path,
    )

    required = protocol["comparison"]["required_results"]
    assert set(required) == set(MODULE.MODEL_IDS.values())
    assert all(seeds == [3072] for seeds in required.values())
    contracts = protocol["training_provenance"]["passage_formal_by_model"]
    assert set(contracts) == set(required)
    assert {
        contract["lewm_sigreg_weight"] for contract in contracts.values()
    } == {0.09, 0.20, 0.30}
    assert all(
        contract["group_weights"]
        == {"original": 0.5, "passage_mixed": 0.5}
        for contract in contracts.values()
    )
    assert (
        protocol["falsification_gate"][
            "original_cem_noninferiority_margin_absolute"
        ]
        == 0.05
    )


def test_serialization_is_deterministic(tmp_path: Path) -> None:
    payload = MODULE.build_training_protocol(
        _training_base(), source_path=tmp_path / "base.yaml"
    )
    first = MODULE._serialized(payload)
    second = MODULE._serialized(payload)

    assert first == second
    assert yaml.safe_load(first) == payload
